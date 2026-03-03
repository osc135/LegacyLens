import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.retrieval import search
from app.generation import generate_answer_stream, generate_followups

app = FastAPI(title="LegacyLens")

templates = Jinja2Templates(directory="frontend/templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the web UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/query")
async def query(request: Request):
    """
    Handle a user query:
    1. Search Pinecone for relevant code
    2. Stream an LLM-generated answer
    3. Return both the answer and the retrieved chunks
    """
    body = await request.json()
    user_query = body.get("query", "")
    mode = body.get("mode", "ask")

    if mode not in ("ask", "explain", "docs"):
        mode = "ask"

    if not user_query.strip():
        return {"error": "Empty query"}

    # Step 1: Retrieve relevant code chunks
    results = search(user_query, top_k=5)

    # Step 2: Stream the LLM answer, then append the retrieved chunks
    def stream_response():
        # First, stream the generated answer and collect the full text
        full_answer = []
        for text_chunk in generate_answer_stream(user_query, results, mode=mode):
            full_answer.append(text_chunk)
            yield f"data: {json.dumps({'type': 'answer', 'content': text_chunk})}\n\n"

        # Then send the retrieved code chunks
        chunks = []
        for r in results:
            chunks.append({
                "name": r["name"],
                "file_path": r["file_path"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "dependencies": r["dependencies"],
                "score": r["score"],
                "text": r["text"],
            })
        yield f"data: {json.dumps({'type': 'sources', 'chunks': chunks})}\n\n"

        # Generate and send follow-up suggestions
        try:
            followups = generate_followups(user_query, "".join(full_answer))
            yield f"data: {json.dumps({'type': 'followups', 'questions': followups})}\n\n"
        except Exception:
            pass

        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
