import json
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from app.config import MAX_QUERY_LENGTH, VALID_MODES, PATTERNS_TOP_K, DEPS_MAX_DEPTH, DEPS_MAX_NODES, FEEDBACK_DIR, MAX_FEEDBACK_COMMENT, VERIFICATION_LOG, PRECISION_LOG, SAMPLE_QUERIES
from app.retrieval import search, resolve_dependency_graph
from app.generation import generate_answer_stream, generate_followups, generate_deps_stream, verify_citations, score_retrieval_precision

logger = logging.getLogger(__name__)

_query_cache: dict = {}
_post_stream_pool = ThreadPoolExecutor(max_workers=3)


def _graph_to_results(graph: dict) -> list[dict]:
    """Convert dependency graph nodes into the standard results format."""
    return [
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


def _format_source_chunks(results: list[dict]) -> list[dict]:
    """Format retrieval results for the SSE sources payload."""
    return [
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


def _warm_cache():
    """Pre-run sample queries so they can be served instantly."""
    for query, mode in SAMPLE_QUERIES:
        try:
            logger.info("Caching sample query: %s [%s]", query, mode)
            graph = None
            if mode == "deps":
                graph = resolve_dependency_graph(query, max_depth=DEPS_MAX_DEPTH, max_nodes=DEPS_MAX_NODES)
                results = _graph_to_results(graph)
                gen = generate_deps_stream(query, graph)
            else:
                results = search(query, top_k=5)
                gen = generate_answer_stream(query, results, mode=mode)

            full_answer = []
            for chunk in gen:
                full_answer.append(chunk)

            answer_text = "".join(full_answer)

            try:
                followups = generate_followups(query, results)
            except Exception:
                followups = []

            precision_result = None
            if mode != "deps" and results:
                try:
                    precision_result = score_retrieval_precision(query, results)
                except Exception:
                    pass

            citation_checks = None
            if answer_text and results:
                try:
                    citation_checks = verify_citations(answer_text, results)
                except Exception:
                    pass

            _query_cache[(query, mode)] = {
                "results": results,
                "answer": answer_text,
                "graph": graph,
                "followups": followups,
                "precision": precision_result,
                "citations": citation_checks,
            }
            logger.info("Cached: %s [%s]", query, mode)
        except Exception as e:
            logger.warning("Failed to cache query '%s' [%s]: %s", query, mode, e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warm_cache, daemon=True).start()
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

            # Serve from cache if this is a pre-cached sample query
            cached = _query_cache.get((req.query, req.mode))
            if cached:
                logger.info("Serving cached response: %s [%s]", req.query, req.mode)

                # Stream the graph if present
                if cached["graph"]:
                    try:
                        yield f"data: {json.dumps({'type': 'graph', 'data': cached['graph']})}\n\n"
                    except Exception as e:
                        logger.error("Graph serialization failed: %s", e)

                # Stream the answer in small chunks with pacing to look natural
                answer = cached["answer"]
                chunk_size = 20
                for i in range(0, len(answer), chunk_size):
                    yield f"data: {json.dumps({'type': 'answer', 'content': answer[i:i+chunk_size]})}\n\n"
                    time.sleep(0.008)

                # Send sources
                try:
                    chunks = _format_source_chunks(cached["results"])
                    yield f"data: {json.dumps({'type': 'sources', 'chunks': chunks})}\n\n"
                except Exception as e:
                    logger.error("Source serialization failed: %s", e)

                # Serve cached precision and citation results
                if cached.get("precision"):
                    yield f"data: {json.dumps({'type': 'precision', 'precision': cached['precision']['precision'], 'scores': cached['precision']['scores']})}\n\n"

                if cached.get("citations"):
                    yield f"data: {json.dumps({'type': 'verification', 'citations': cached['citations']})}\n\n"

                if trace_id:
                    yield f"data: {json.dumps({'type': 'trace', 'trace_id': trace_id})}\n\n"

                yield "data: [DONE]\n\n"

                if cached.get("followups"):
                    yield f"data: {json.dumps({'type': 'followups', 'questions': cached['followups']})}\n\n"

                return

            # Step 1: Retrieve relevant code chunks (or dependency graph)
            graph = None
            try:
                if req.mode == "deps":
                    graph = resolve_dependency_graph(req.query, max_depth=DEPS_MAX_DEPTH, max_nodes=DEPS_MAX_NODES)
                    results = _graph_to_results(graph)
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

            # Start followup generation now (runs in parallel with answer streaming)
            followup_future = _post_stream_pool.submit(generate_followups, req.query, results)

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
                chunks = _format_source_chunks(results)
                sources_json = json.dumps({'type': 'sources', 'chunks': chunks})
                logger.info("Sources payload: mode=%s, chunks=%d, bytes=%d", req.mode, len(chunks), len(sources_json))
                yield f"data: {sources_json}\n\n"
            except Exception as e:
                logger.error("Source serialization failed: %s", e)

            # Yield followups immediately if ready, before [DONE]
            try:
                followup_result = followup_future.result(timeout=0.1)
                yield f"data: {json.dumps({'type': 'followups', 'questions': followup_result})}\n\n"
                followup_future = None
            except Exception:
                pass  # not ready yet, will yield after [DONE]

            # Send trace ID and [DONE] immediately so the answer feels complete
            if trace_id:
                yield f"data: {json.dumps({'type': 'trace', 'trace_id': trace_id})}\n\n"

            yield "data: [DONE]\n\n"

            # Run post-stream ops (precision, verification) in parallel
            full_text = "".join(full_answer)
            futures = {}
            if req.mode != "deps" and results:
                futures[_post_stream_pool.submit(score_retrieval_precision, req.query, results)] = "precision"
            if full_text and results:
                futures[_post_stream_pool.submit(verify_citations, full_text, results)] = "verification"
            if followup_future is not None:
                futures[followup_future] = "followups"

            for future in as_completed(futures):
                kind = futures[future]
                try:
                    result = future.result()
                    if kind == "precision":
                        yield f"data: {json.dumps({'type': 'precision', 'precision': result['precision'], 'scores': result['scores']})}\n\n"
                        _log_precision(req.query, req.mode, result)
                    elif kind == "verification" and result:
                        yield f"data: {json.dumps({'type': 'verification', 'citations': result})}\n\n"
                        _log_verification(req.query, req.mode, result)
                    elif kind == "followups":
                        yield f"data: {json.dumps({'type': 'followups', 'questions': result})}\n\n"
                except Exception as e:
                    logger.warning("%s failed: %s", kind, e)

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
