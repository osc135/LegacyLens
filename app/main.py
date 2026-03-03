import json
import logging
import os
from datetime import datetime, timezone
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from app.config import MAX_QUERY_LENGTH, VALID_MODES, PATTERNS_TOP_K, DEPS_MAX_DEPTH, DEPS_MAX_NODES, FEEDBACK_DIR, MAX_FEEDBACK_COMMENT
from app.retrieval import search, resolve_dependency_graph
from app.generation import generate_answer_stream, generate_followups, generate_deps_stream

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="LegacyLens")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    mode: str = Field(default="ask")

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

    # Step 1: Retrieve relevant code chunks (or dependency graph)
    graph = None
    try:
        if req.mode == "deps":
            graph = resolve_dependency_graph(req.query, max_depth=DEPS_MAX_DEPTH, max_nodes=DEPS_MAX_NODES)
            # Build results from found graph nodes for source cards
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
            results = search(req.query, top_k=5)
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        raise HTTPException(status_code=502, detail="Search service unavailable.")

    # Step 2: Stream the LLM answer, then append the retrieved chunks
    def stream_response():
        # For deps mode, send the graph structure first
        if graph:
            try:
                yield f"data: {json.dumps({'type': 'graph', 'data': graph})}\n\n"
            except Exception as e:
                logger.error("Graph serialization failed: %s", e)

        # Stream the generated answer and collect the full text
        full_answer = []
        try:
            gen = generate_deps_stream(req.query, graph) if graph else generate_answer_stream(req.query, results, mode=req.mode)
            for text_chunk in gen:
                full_answer.append(text_chunk)
                yield f"data: {json.dumps({'type': 'answer', 'content': text_chunk})}\n\n"
        except Exception as e:
            logger.error("Generation failed: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': 'Answer generation failed.'})}\n\n"

        # Then send the retrieved code chunks
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

        # Generate and send follow-up suggestions
        try:
            followups = generate_followups(req.query, "".join(full_answer))
            yield f"data: {json.dumps({'type': 'followups', 'questions': followups})}\n\n"
        except Exception as e:
            logger.warning("Follow-up generation failed: %s", e)

        yield "data: [DONE]\n\n"

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

    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
