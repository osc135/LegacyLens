from openai import OpenAI
from app.config import OPENAI_API_KEY, LLM_MODEL

openai_client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """You are an expert Fortran developer specializing in the LAPACK linear algebra library.

Rules:
- Be concise — answer what was asked without padding. Do NOT list every similar subroutine — focus on the most relevant 1-2.
- Use markdown: ## headings, **bold**, `inline code`, ```fortran code blocks, and bullet lists.
- You are given numbered sources [Source 1] through [Source 5]. IMPORTANT: When you use information from a source, you MUST cite it inline using ONLY the numbers [1], [2], [3], [4], or [5] — matching the source number. Place the citation immediately after the sentence that uses that source's information. Every claim from a source needs a citation. Do NOT use any other numbers. Example: "The `DGESV` subroutine solves general linear systems [1]. It uses LU factorization with partial pivoting [2]."
- Explain in plain English for developers unfamiliar with Fortran.
- Wrap Fortran identifiers in backticks (e.g. `DGESV`, `INFO`, `LDA`).
- If the retrieved context doesn't answer the question, say so clearly.
- Do NOT repeat the same information about multiple nearly-identical subroutines. Mention the primary one, then briefly note variants exist."""


def build_context(results: list[dict]) -> str:
    """Format retrieved code chunks into numbered context for the LLM."""
    context_parts = []
    for i, r in enumerate(results, 1):
        header = f"[Source {i}] {r['file_path']}:lines {r['start_line']}-{r['end_line']} — {r['name']}"
        if r["dependencies"]:
            header += f" (calls: {r['dependencies']})"
        context_parts.append(f"{header}\n{r['text']}")
    return "\n\n---\n\n".join(context_parts)


def generate_answer(query: str, results: list[dict]) -> str:
    """Generate a complete answer (non-streaming) from retrieved code chunks."""
    context = build_context(results)

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.3,
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
        temperature=0.3,
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


def generate_followups(query: str, answer: str) -> list[str]:
    """Generate 3 relevant follow-up questions based on the Q&A."""
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate follow-up questions about the LAPACK Fortran codebase. "
                    "Given a user question and the answer they received, suggest exactly 3 "
                    "short follow-up questions they might want to ask next. "
                    "Each question should be specific, useful, and different from each other. "
                    "Return ONLY the 3 questions, one per line, no numbering or bullets."
                ),
            },
            {
                "role": "user",
                "content": f"User asked: {query}\n\nAnswer given:\n{answer[:1000]}",
            },
        ],
    )
    lines = response.choices[0].message.content.strip().split("\n")
    return [l.strip() for l in lines if l.strip()][:3]
