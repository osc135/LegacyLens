import json
import logging
import re
import tiktoken
from langfuse.openai import OpenAI
from app.config import (
    OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, MAX_FOLLOWUP_ANSWER_CHARS,
    MAX_CONTEXT_TOKENS_PER_CHUNK,
)

logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
_encoding = tiktoken.encoding_for_model(LLM_MODEL)

CITATION_RULES = """- You are given numbered sources [Source 1] through [Source N]. IMPORTANT: You MUST cite EVERY source at least once. When you use information from a source, cite it inline using the bracketed number (e.g. [1], [2], [3]). Place the citation immediately after the sentence that uses that source's information. When multiple sources contain related code (e.g. precision variants), group them in a single citation like [1][2][3][4]. Every source number from [1] through [N] must appear at least once in your response."""

SYSTEM_PROMPT = f"""You are an expert Fortran developer specializing in the LAPACK linear algebra library.

{CITATION_RULES}

Rules:
- Be concise — answer what was asked without padding. Do NOT list every similar subroutine — focus on the most relevant 1-2.
- Use markdown: ## headings, **bold**, `inline code`, ```fortran code blocks, and bullet lists. NEVER use LaTeX math notation (no \\( \\), \\[ \\], or $ $) — write math in plain text like `A = P * L * U`.
- Explain in plain English for developers unfamiliar with Fortran.
- Wrap Fortran identifiers in backticks (e.g. `DGESV`, `INFO`, `LDA`).
- If the retrieved context doesn't answer the question, say so clearly.
- Synthesize information from ALL provided sources. Different sources may cover different aspects — cite each one where relevant.

{CITATION_RULES}"""

EXPLAIN_PROMPT = f"""You are an expert Fortran developer specializing in the LAPACK linear algebra library. Your job is to explain code clearly for developers unfamiliar with Fortran.

{CITATION_RULES}

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
- Use markdown: ## headings, **bold**, `inline code`, ```fortran code blocks, and bullet lists. NEVER use LaTeX math notation (no \\( \\), \\[ \\], or $ $) — write math in plain text like `A = P * L * U`.
- Wrap all Fortran identifiers in backticks.
- Explain in plain English — assume the reader does not know Fortran syntax.
- Synthesize information from ALL provided sources. Different sources may cover different aspects — cite each one where relevant.

{CITATION_RULES}"""

DOCS_PROMPT = f"""You are an expert Fortran developer specializing in the LAPACK linear algebra library. Your job is to generate clean, structured documentation.

{CITATION_RULES}

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
- If info is not available in the sources, omit that section rather than guessing.

{CITATION_RULES}"""


PATTERNS_PROMPT = f"""You are an expert Fortran developer specializing in the LAPACK linear algebra library. Your job is to find and explain code patterns.

{CITATION_RULES}

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
- Use markdown: ## headings, **bold**, `inline code`, ```fortran code blocks, and bullet lists. NEVER use LaTeX math notation (no \\( \\), \\[ \\], or $ $) — write math in plain text like `A = P * L * U`.
{CITATION_RULES}
- Wrap all Fortran identifiers in backticks.
- Focus on patterns that help the user understand LAPACK's design philosophy.
- If the retrieved code is not related to the snippet, say so clearly.

{CITATION_RULES}"""

DEPS_PROMPT = f"""You are an expert Fortran developer specializing in the LAPACK linear algebra library. Your job is to visualize and explain subroutine dependency chains.

{CITATION_RULES}

Given a dependency graph and the source code of each resolved subroutine, respond with:

## Call Graph

```mermaid
graph TD
    A[ROOT] --> B[DEP1]
    A --> C[DEP2]
    B --> D[DEP3]
```

(Generate a Mermaid `graph TD` flowchart showing the full call graph. Use the actual subroutine names. Mark external/unresolved routines with `:::external` class.)

## Overview
A 2-3 sentence summary of what the root subroutine does and how its dependencies fit together.

## Subroutine Roles
For each node in the graph, one bullet describing its role in the call chain.

## Data Flow
Describe how data flows through the call chain — what the root passes to its callees and what they return.

Rules:
- Use markdown: ## headings, **bold**, `inline code`, ```mermaid code blocks, and bullet lists.
{CITATION_RULES}
- Wrap all Fortran identifiers in backticks.
- Keep the Mermaid diagram clean — no duplicate edges, use short labels.
- If a dependency was not found in the codebase, note it as an external routine (e.g., BLAS).

{CITATION_RULES}"""

PROMPTS = {
    "ask": SYSTEM_PROMPT,
    "explain": EXPLAIN_PROMPT,
    "docs": DOCS_PROMPT,
    "patterns": PATTERNS_PROMPT,
    "deps": DEPS_PROMPT,
}


def get_system_prompt(mode: str) -> str:
    """Get the system prompt for the given mode."""
    return PROMPTS.get(mode, SYSTEM_PROMPT)


def _truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to max_tokens using tiktoken."""
    tokens = _encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _encoding.decode(tokens[:max_tokens]) + "\n... [truncated]"


def build_context(results: list[dict]) -> str:
    """Format retrieved code chunks into numbered context for the LLM."""
    context_parts = []
    for i, r in enumerate(results, 1):
        header = f"[Source {i}] {r['file_path']}:lines {r['start_line']}-{r['end_line']} — {r['name']}"
        if r["dependencies"]:
            header += f" (calls: {r['dependencies']})"
        text = _truncate_text(r['text'], MAX_CONTEXT_TOKENS_PER_CHUNK)
        context_parts.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(context_parts)


def build_deps_context(graph: dict) -> str:
    """Format a dependency graph into numbered context for the LLM."""
    parts = []
    parts.append(f"Dependency graph for {graph['root']}:")
    parts.append(f"Edges: {graph['edges']}")
    parts.append("")

    source_num = 1
    for name, info in graph["nodes"].items():
        if info.get("found"):
            header = f"[Source {source_num}] {info.get('file_path', '')}:lines {info.get('start_line', 0)}-{info.get('end_line', 0)} — {name}"
            deps = info.get("dependencies", [])
            if deps:
                header += f" (calls: {', '.join(deps)})"
            text = _truncate_text(info.get('text', ''), MAX_CONTEXT_TOKENS_PER_CHUNK)
            parts.append(f"{header}\n{text}")
            parts.append("---")
            source_num += 1
        else:
            parts.append(f"[{name}] — external routine (not found in codebase)")

    return "\n\n".join(parts)


def _build_user_message(query: str, context: str, mode: str) -> str:
    """Build the user message, labeling input appropriately for the mode."""
    if mode == "patterns":
        return f"Code snippet:\n```\n{query}\n```\n\nSimilar code from LAPACK:\n\n{context}"
    if mode == "deps":
        return f"Subroutine: {query}\n\n{context}"
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


def generate_deps_stream(query: str, graph: dict):
    """Generate a streaming answer for dependency mapping mode. Yields text chunks."""
    context = build_deps_context(graph)

    stream = openai_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        stream=True,
        messages=[
            {"role": "system", "content": get_system_prompt("deps")},
            {"role": "user", "content": _build_user_message(query, context, "deps")},
        ],
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def generate_followups(query: str, answer: str) -> list[str]:
    """Generate 3 relevant follow-up questions based on the Q&A."""
    response = openai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate follow-up questions for LegacyLens, a tool that lets users "
                    "explore the LAPACK Fortran source code. Given a user question and the answer "
                    "they received, suggest exactly 3 short follow-up questions they might ask next. "
                    "Questions MUST be answerable by reading LAPACK source code — ask about specific "
                    "subroutines, algorithms, parameters, call chains, or implementation details. "
                    "NEVER ask about external tools, compilers, installation, benchmarks, or anything "
                    "outside the LAPACK source code itself. "
                    "Each question must be a question (end with ?), be specific, and differ from each other. "
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


def verify_citations(answer: str, results: list[dict]) -> list[dict]:
    """Verify that each citation in the answer is grounded in its referenced source.

    Returns a list of dicts: [{"citation": N, "grounded": bool, "reason": "..."}]
    """
    # Find all [N] citations and extract the sentence around each
    citation_pattern = re.compile(r'\[(\d+)\]')
    citations_found = {}  # {citation_num: claim_sentence}

    # Split answer into sentences (rough split on ., !, ? followed by space or newline)
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    for sentence in sentences:
        for match in citation_pattern.finditer(sentence):
            num = int(match.group(1))
            if num not in citations_found and 1 <= num <= len(results):
                # Clean the sentence of citation markers for readability
                clean = citation_pattern.sub('', sentence).strip()
                if clean:
                    citations_found[num] = clean

    if not citations_found:
        return []

    # Build verification prompt with all claims and their sources
    claims_text = ""
    for num, claim in sorted(citations_found.items()):
        source = results[num - 1]
        source_snippet = source.get("text", "")[:1500]
        claims_text += f"Citation [{num}]:\n"
        claims_text += f"  Claim: \"{claim}\"\n"
        claims_text += f"  Source ({source.get('name', 'unknown')}):\n  {source_snippet}\n\n"

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You verify whether cited claims are supported by their sources. "
                    "For each citation, determine if the claim is grounded in the source text. "
                    "Respond with ONLY a JSON array, no other text. Each element: "
                    '{"citation": N, "grounded": true/false, "reason": "brief explanation"}. '
                    "ALWAYS provide a reason, even for grounded citations — explain what in the source supports the claim. "
                    "A claim is grounded if the source reasonably supports the statement. "
                    "Be lenient — minor paraphrasing is fine. Only flag clearly unsupported claims."
                ),
            },
            {"role": "user", "content": claims_text},
        ],
    )

    raw = response.choices[0].message.content.strip()
    # Extract JSON array from response (handle markdown code blocks)
    json_match = re.search(r'\[.*\]', raw, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return []


def score_retrieval_precision(query: str, results: list[dict]) -> dict:
    """Score how relevant each retrieved chunk is to the query using LLM-as-judge.

    Uses a graduated 0-3 relevance scale instead of binary to produce a
    continuous precision score between 0.0 and 1.0.

    Returns {"scores": [{"chunk": N, "relevance": 0-3, "relevant": bool, "reason": "..."}], "precision": float}
    """
    if not results:
        return {"scores": [], "precision": 0.0}

    chunks_text = ""
    for i, r in enumerate(results, 1):
        snippet = r.get("text", "")[:1500]
        chunks_text += f"Chunk {i} ({r.get('name', 'unknown')}):\n{snippet}\n\n"

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a relevance judge for LegacyLens, a code exploration tool that helps "
                    "developers navigate and understand the LAPACK Fortran linear algebra library.\n\n"
                    "A developer has asked a question, and the retrieval system returned source code chunks. "
                    "Your job: would a developer find each chunk USEFUL while exploring this topic?\n\n"
                    "This is a code exploration tool, not a search engine. Developers use it to:\n"
                    "- Understand how algorithms are implemented in LAPACK\n"
                    "- Find relevant subroutines and see their source code\n"
                    "- Learn patterns, conventions, and structure of the codebase\n"
                    "- Trace dependencies and relationships between routines\n\n"
                    "JUDGING GUIDELINES:\n"
                    "- ALL queries are about the LAPACK codebase, even if vaguely worded.\n"
                    "- A chunk is useful if a developer exploring the topic would benefit from seeing it.\n"
                    "- Type variants (SGESV/DGESV/CGESV/ZGESV) are useful — they show the same algorithm "
                    "for different data types, which is a core LAPACK pattern.\n"
                    "- A subroutine that DEMONSTRATES a pattern the user asked about is useful, even if "
                    "the subroutine's primary purpose is something else. Example: if the user asks about "
                    "error handling, any routine that contains INFO parameter checking is useful.\n"
                    "- LAPACK is a library, not a standalone program — it has no main(). When a user asks "
                    "about 'entry points' or 'main routines', they mean driver routines like xGESV, xPOSV, "
                    "xSYEV, xGEEV, etc. These ARE the entry points that users call.\n"
                    "- If the query topic doesn't map cleanly to LAPACK (e.g. 'file I/O'), judge whether "
                    "the chunks represent a reasonable interpretation of the query within what LAPACK "
                    "actually contains.\n\n"
                    "Score each chunk on a 0-3 scale:\n"
                    "  3 = Directly useful (exact subroutine asked about, or a core example of the requested topic)\n"
                    "  2 = Useful (same algorithm family, shows the requested pattern, or provides helpful context)\n"
                    "  1 = Marginally useful (loosely related, a developer might glance at it but it's not what they need)\n"
                    "  0 = Not useful (completely different topic, would waste the developer's time)\n\n"
                    "Respond with ONLY a JSON array, no other text. Each element:\n"
                    '{"chunk": N, "relevance": 0-3, "reason": "brief explanation"}'
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\n{chunks_text}",
            },
        ],
    )

    raw = response.choices[0].message.content.strip()
    json_match = re.search(r'\[.*\]', raw, re.DOTALL)
    scores = json.loads(json_match.group()) if json_match else []

    # Add "relevant" field: relevance >= 2 means the chunk is useful to a developer
    for s in scores:
        s["relevant"] = s.get("relevance", 0) >= 2

    # Precision = fraction of retrieved chunks that are relevant (matches rubric:
    # ">70% relevant chunks in top-5")
    relevant_count = sum(1 for s in scores if s["relevant"])
    precision = relevant_count / len(results) if results else 0.0

    return {"scores": scores, "precision": precision}
