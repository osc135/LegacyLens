import logging
from openai import OpenAI
from app.config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, MAX_FOLLOWUP_ANSWER_CHARS

logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=OPENAI_API_KEY)

CITATION_RULES = """- You are given numbered sources [Source 1] through [Source N]. IMPORTANT: When you use information from a source, you MUST cite it inline using the bracketed number matching the source (e.g. [1], [2], [3]). Place the citation immediately after the sentence that uses that source's information. Every claim from a source needs a citation. Example: "The `DGESV` subroutine solves general linear systems [1]. It uses LU factorization with partial pivoting [2].\""""

SYSTEM_PROMPT = f"""You are an expert Fortran developer specializing in the LAPACK linear algebra library.

Rules:
- Be concise — answer what was asked without padding. Do NOT list every similar subroutine — focus on the most relevant 1-2.
- Use markdown: ## headings, **bold**, `inline code`, ```fortran code blocks, and bullet lists.
{CITATION_RULES}
- Explain in plain English for developers unfamiliar with Fortran.
- Wrap Fortran identifiers in backticks (e.g. `DGESV`, `INFO`, `LDA`).
- If the retrieved context doesn't answer the question, say so clearly.
- Do NOT repeat the same information about multiple nearly-identical subroutines. Mention the primary one, then briefly note variants exist."""

EXPLAIN_PROMPT = f"""You are an expert Fortran developer specializing in the LAPACK linear algebra library. Your job is to explain code clearly for developers unfamiliar with Fortran.

Given source code from LAPACK, provide a clear plain-English explanation with this structure:

## Purpose
One-sentence summary of what this subroutine/function does.

## How It Works
Step-by-step walkthrough of the algorithm in plain English. Reference specific lines or variables from the code using `backticks`.

## Parameters
A table or list of the key input/output parameters and what they mean.

## Key Details
Any important edge cases, error conditions, or performance notes.

Rules:
- Use markdown: ## headings, **bold**, `inline code`, ```fortran code blocks, and bullet lists.
{CITATION_RULES}
- Wrap all Fortran identifiers in backticks.
- Explain in plain English — assume the reader does not know Fortran syntax.
- Focus on the primary subroutine. If variants exist, mention them briefly at the end."""

DOCS_PROMPT = f"""You are an expert Fortran developer specializing in the LAPACK linear algebra library. Your job is to generate clean, structured documentation.

Given source code from LAPACK, generate documentation in this exact format:

## `SUBROUTINE_NAME`

**Purpose:** One-sentence description.

**Type:** (e.g., Driver routine, Computational routine, Auxiliary routine)

### Parameters

| Name | Intent | Type | Description |
|------|--------|------|-------------|
| ... | IN/OUT/INOUT | ... | ... |

### Algorithm
Numbered step-by-step description of what the subroutine does internally.

### Error Codes
- `INFO = 0`: Successful exit
- `INFO < 0`: If `INFO = -i`, the i-th argument had an illegal value
- (list other error codes if present)

### Dependencies
List subroutines called by this routine and what each does.

### Example Usage
```fortran
(short example call with typical arguments)
```

Rules:
- Use markdown tables, headings, code blocks.
{CITATION_RULES}
- Wrap all Fortran identifiers in backticks.
- Be precise — extract parameter info directly from the source code comments.
- If info is not available in the sources, omit that section rather than guessing."""


PATTERNS_PROMPT = f"""You are an expert Fortran developer specializing in the LAPACK linear algebra library. Your job is to find and explain code patterns.

The user has pasted a code snippet. You are given similar code chunks retrieved from the LAPACK codebase. Analyze them and respond with:

## Snippet Analysis
Briefly describe what the pasted code snippet does.

## Similar Code Found
For each retrieved chunk that is meaningfully similar, explain what it does and how it relates to the snippet.

## Common Patterns
Highlight shared patterns across the snippet and retrieved code:
- Naming conventions (e.g., prefix conventions like D/S/Z/C)
- Structural patterns (e.g., argument checking, workspace queries, algorithm flow)
- Error handling (e.g., INFO parameter usage)
- Algorithm patterns (e.g., blocking, recursion, factorization steps)

## Notable Differences
Point out meaningful differences between the snippet and the retrieved code — different algorithms, optimizations, or edge-case handling.

Rules:
- Use markdown: ## headings, **bold**, `inline code`, ```fortran code blocks, and bullet lists.
{CITATION_RULES}
- Wrap all Fortran identifiers in backticks.
- Focus on patterns that help the user understand LAPACK's design philosophy.
- If the retrieved code is not related to the snippet, say so clearly."""

PROMPTS = {
    "ask": SYSTEM_PROMPT,
    "explain": EXPLAIN_PROMPT,
    "docs": DOCS_PROMPT,
    "patterns": PATTERNS_PROMPT,
}


def get_system_prompt(mode: str) -> str:
    """Get the system prompt for the given mode."""
    return PROMPTS.get(mode, SYSTEM_PROMPT)


def build_context(results: list[dict]) -> str:
    """Format retrieved code chunks into numbered context for the LLM."""
    context_parts = []
    for i, r in enumerate(results, 1):
        header = f"[Source {i}] {r['file_path']}:lines {r['start_line']}-{r['end_line']} — {r['name']}"
        if r["dependencies"]:
            header += f" (calls: {r['dependencies']})"
        context_parts.append(f"{header}\n{r['text']}")
    return "\n\n---\n\n".join(context_parts)


def _build_user_message(query: str, context: str, mode: str) -> str:
    """Build the user message, labeling input appropriately for the mode."""
    if mode == "patterns":
        return f"Code snippet:\n```\n{query}\n```\n\nSimilar code from LAPACK:\n\n{context}"
    return f"Question: {query}\n\nRelevant code from LAPACK:\n\n{context}"


def generate_answer(query: str, results: list[dict], mode: str = "ask") -> str:
    """Generate a complete answer (non-streaming) from retrieved code chunks."""
    context = build_context(results)

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": get_system_prompt(mode)},
            {"role": "user", "content": _build_user_message(query, context, mode)},
        ],
    )
    return response.choices[0].message.content


def generate_answer_stream(query: str, results: list[dict], mode: str = "ask"):
    """Generate a streaming answer from retrieved code chunks. Yields text chunks."""
    context = build_context(results)

    stream = openai_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        stream=True,
        messages=[
            {"role": "system", "content": get_system_prompt(mode)},
            {"role": "user", "content": _build_user_message(query, context, mode)},
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
                "content": f"User asked: {query}\n\nAnswer given:\n{answer[:MAX_FOLLOWUP_ANSWER_CHARS]}",
            },
        ],
    )
    lines = response.choices[0].message.content.strip().split("\n")
    return [l.strip() for l in lines if l.strip()][:3]
