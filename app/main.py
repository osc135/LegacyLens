import json
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from app.config import MAX_QUERY_LENGTH, VALID_MODES
from app.retrieval import search
from app.generation import generate_answer_stream, generate_followups

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

    # Step 1: Retrieve relevant code chunks
    try:
        results = search(req.query, top_k=5)
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        raise HTTPException(status_code=502, detail="Search service unavailable.")

    # Step 2: Stream the LLM answer, then append the retrieved chunks
    def stream_response():
        # First, stream the generated answer and collect the full text
        full_answer = []
        try:
            for text_chunk in generate_answer_stream(req.query, results, mode=req.mode):
                full_answer.append(text_chunk)
                yield f"data: {json.dumps({'type': 'answer', 'content': text_chunk})}\n\n"
        except Exception as e:
            logger.error("Generation failed: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': 'Answer generation failed.'})}\n\n"

        # Then send the retrieved code chunks
        chunks = [
            {
                "name": r["name"],
                "file_path": r["file_path"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "dependencies": r["dependencies"],
                "score": r["score"],
                "text": r["text"],
            }
            for r in results
        ]
        yield f"data: {json.dumps({'type': 'sources', 'chunks': chunks})}\n\n"

        # Generate and send follow-up suggestions
        try:
            followups = generate_followups(req.query, "".join(full_answer))
            yield f"data: {json.dumps({'type': 'followups', 'questions': followups})}\n\n"
        except Exception as e:
            logger.warning("Follow-up generation failed: %s", e)

        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
