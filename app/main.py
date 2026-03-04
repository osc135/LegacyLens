import json
import logging
import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from langfuse import get_client
from app.config import MAX_QUERY_LENGTH, VALID_MODES, PATTERNS_TOP_K, DEPS_MAX_DEPTH, DEPS_MAX_NODES, FEEDBACK_DIR, MAX_FEEDBACK_COMMENT, VERIFICATION_LOG, PRECISION_LOG
from app.retrieval import search, resolve_dependency_graph
from app.generation import generate_answer_stream, generate_followups, generate_deps_stream, verify_citations, score_retrieval_precision

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    get_client().flush()


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="LegacyLens", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    mode: str = Field(default="ask")
    context: str = Field(default="", max_length=1000)

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return v.strip()

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in VALID_MODES:
            return "ask"
        return v


class FeedbackRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    mode: str = Field(default="ask")
    feedback: str = Field(...)
    comment: str = Field(default="", max_length=MAX_FEEDBACK_COMMENT)
    trace_id: str = Field(default="")

    @field_validator("feedback")
    @classmethod
    def validate_feedback(cls, v: str) -> str:
        if v not in ("up", "down"):
            raise ValueError("feedback must be 'up' or 'down'")
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in VALID_MODES:
            return "ask"
        return v


templates = Jinja2Templates(directory="frontend/templates")


def _log_verification(query: str, mode: str, citations: list[dict]):
    """Append a verification result to the JSONL log file."""
    try:
        os.makedirs(os.path.dirname(VERIFICATION_LOG), exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "mode": mode,
            "citations": citations,
            "all_grounded": all(c.get("grounded", False) for c in citations),
        }
        with open(VERIFICATION_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("Failed to write verification log: %s", e)


def _log_precision(query: str, mode: str, precision_result: dict):
    """Append a retrieval precision result to the JSONL log file."""
    try:
        os.makedirs(os.path.dirname(PRECISION_LOG), exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "mode": mode,
            "precision": precision_result["precision"],
            "precision_pct": f"{precision_result['precision'] * 100:.0f}%",
            "scores": precision_result["scores"],
        }
        with open(PRECISION_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("Failed to write precision log: %s", e)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the web UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/query")
@limiter.limit("20/minute")
async def query(request: Request):
    """
    Handle a user query:
    1. Search Pinecone for relevant code
    2. Stream an LLM-generated answer
    3. Return both the answer and the retrieved chunks
    """
    try:
        body = await request.json()
        req = QueryRequest(**body)
    except Exception as e:
        logger.warning("Invalid request: %s", e)
        raise HTTPException(status_code=400, detail="Invalid request. Provide a 'query' string.")

    def stream_response():
        # Wrap entire pipeline in a single Langfuse trace
        with get_client().start_as_current_span(
            name="query",
            input={"query": req.query, "mode": req.mode},
        ):
            get_client().update_current_trace(
                tags=["legacylens", req.mode],
            )
            trace_id = get_client().get_current_trace_id()

            # Step 1: Retrieve relevant code chunks (or dependency graph)
            graph = None
            try:
                if req.mode == "deps":
                    graph = resolve_dependency_graph(req.query, max_depth=DEPS_MAX_DEPTH, max_nodes=DEPS_MAX_NODES)
                    results = [
                        {
                            "name": name,
                            "file_path": info.get("file_path", ""),
                            "start_line": info.get("start_line", 0),
                            "end_line": info.get("end_line", 0),
                            "dependencies": ", ".join(info.get("dependencies", [])),
                            "text": info.get("text", ""),
                            "score": 1.0,
                        }
                        for name, info in graph["nodes"].items()
                        if info.get("found")
                    ]
                elif req.mode == "patterns":
                    results = search(req.query, top_k=PATTERNS_TOP_K, code_search=True)
                else:
                    # Prepend conversation context for better follow-up retrieval
                    search_query = f"{req.context}\n{req.query}" if req.context else req.query
                    results = search(search_query, top_k=5)
            except Exception as e:
                logger.error("Retrieval failed: %s", e)
                yield f"data: {json.dumps({'type': 'error', 'content': 'Search service unavailable.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # For deps mode, send the graph structure first
            if graph:
                try:
                    yield f"data: {json.dumps({'type': 'graph', 'data': graph})}\n\n"
                except Exception as e:
                    logger.error("Graph serialization failed: %s", e)

            # Step 2: Stream the generated answer
            full_answer = []
            try:
                gen = generate_deps_stream(req.query, graph) if graph else generate_answer_stream(req.query, results, mode=req.mode)
                for text_chunk in gen:
                    full_answer.append(text_chunk)
                    yield f"data: {json.dumps({'type': 'answer', 'content': text_chunk})}\n\n"
            except ValueError as e:
                # Langfuse/OpenTelemetry context detach error — answer already streamed, safe to continue
                logger.debug("Context detach during streaming (non-fatal): %s", e)
            except Exception as e:
                logger.error("Generation failed: %s", e)
                yield f"data: {json.dumps({'type': 'error', 'content': 'Answer generation failed.'})}\n\n"

            # Send retrieved code chunks
            try:
                chunks = [
                    {
                        "name": r["name"],
                        "file_path": r["file_path"],
                        "start_line": r["start_line"],
                        "end_line": r["end_line"],
                        "dependencies": r["dependencies"],
                        "score": float(r["score"]),
                        "text": r["text"][:2000],
                    }
                    for r in results
                ]
                sources_json = json.dumps({'type': 'sources', 'chunks': chunks})
                logger.info("Sources payload: mode=%s, chunks=%d, bytes=%d", req.mode, len(chunks), len(sources_json))
                yield f"data: {sources_json}\n\n"
            except Exception as e:
                logger.error("Source serialization failed: %s", e)

            # Score retrieval precision (skip for deps mode)
            if req.mode != "deps":
                try:
                    if results:
                        precision_result = score_retrieval_precision(req.query, results)
                        yield f"data: {json.dumps({'type': 'precision', 'precision': precision_result['precision'], 'scores': precision_result['scores']})}\n\n"
                        _log_precision(req.query, req.mode, precision_result)
                except Exception as e:
                    logger.warning("Retrieval precision scoring failed: %s", e)

            # Verify citation grounding
            try:
                full_text = "".join(full_answer)
                if full_text and results:
                    citation_checks = verify_citations(full_text, results)
                    if citation_checks:
                        yield f"data: {json.dumps({'type': 'verification', 'citations': citation_checks})}\n\n"
                        # Log to file for review
                        _log_verification(req.query, req.mode, citation_checks)
            except Exception as e:
                logger.warning("Citation verification failed: %s", e)

            # Send trace ID so frontend can attach feedback to the right trace
            if trace_id:
                yield f"data: {json.dumps({'type': 'trace', 'trace_id': trace_id})}\n\n"

            yield "data: [DONE]\n\n"

            # Generate follow-up suggestions after [DONE] (non-blocking)
            try:
                followups = generate_followups(req.query, "".join(full_answer))
                yield f"data: {json.dumps({'type': 'followups', 'questions': followups})}\n\n"
            except Exception as e:
                logger.warning("Follow-up generation failed: %s", e)

    return StreamingResponse(stream_response(), media_type="text/event-stream")


@app.post("/feedback")
@limiter.limit("30/minute")
async def feedback(request: Request):
    """Store user feedback (thumbs up/down with optional comment)."""
    try:
        body = await request.json()
        req = FeedbackRequest(**body)
    except Exception as e:
        logger.warning("Invalid feedback request: %s", e)
        raise HTTPException(status_code=400, detail="Invalid feedback request.")

    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": req.query,
        "mode": req.mode,
        "feedback": req.feedback,
        "comment": req.comment,
    }
    feedback_path = os.path.join(FEEDBACK_DIR, "feedback.jsonl")
    try:
        with open(feedback_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error("Failed to write feedback: %s", e)
        raise HTTPException(status_code=500, detail="Failed to store feedback.")

    # Attach feedback as a Langfuse score on the trace
    if req.trace_id:
        try:
            get_client().create_score(
                trace_id=req.trace_id,
                name="user_feedback",
                value=1.0 if req.feedback == "up" else 0.0,
                comment=req.comment or None,
            )
        except Exception as e:
            logger.warning("Failed to send feedback to Langfuse: %s", e)

    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
