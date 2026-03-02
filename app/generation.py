from openai import OpenAI
from app.config import OPENAI_API_KEY, LLM_MODEL

openai_client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """You are an expert Fortran developer specializing in the LAPACK linear algebra library.

When answering questions:
- Cite specific subroutine names and file references using [filename:lines X-Y] format
- Explain code in plain English that a developer unfamiliar with Fortran can understand
- Mention parameter names and their purposes when relevant
- If the retrieved code doesn't fully answer the question, say so clearly
- Keep answers concise but thorough"""


def build_context(results: list[dict]) -> str:
    """Format retrieved code chunks into context for the LLM."""
    context_parts = []
    for r in results:
        header = f"[{r['file_path']}:lines {r['start_line']}-{r['end_line']}] — {r['name']}"
        if r["dependencies"]:
            header += f" (calls: {r['dependencies']})"
        context_parts.append(f"{header}\n{r['text']}")
    return "\n\n---\n\n".join(context_parts)


def generate_answer(query: str, results: list[dict]) -> str:
    """Generate a complete answer (non-streaming) from retrieved code chunks."""
    context = build_context(results)

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {query}\n\nRelevant code from LAPACK:\n\n{context}",
            },
        ],
    )
    return response.choices[0].message.content


def generate_answer_stream(query: str, results: list[dict]):
    """Generate a streaming answer from retrieved code chunks. Yields text chunks."""
    context = build_context(results)

    stream = openai_client.chat.completions.create(
        model=LLM_MODEL,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {query}\n\nRelevant code from LAPACK:\n\n{context}",
            },
        ],
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
