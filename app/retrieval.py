from openai import OpenAI
from pinecone import Pinecone
from app.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
)

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


def expand_query(query: str) -> list[str]:
    """
    Use GPT-4o-mini to rephrase the user's question into
    3 technical variations that will match LAPACK code better.
    """
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search query expander for the LAPACK Fortran linear algebra library. "
                    "Given a user question, generate exactly 3 alternative search queries that would "
                    "match LAPACK subroutine names, comment headers, and documentation. "
                    "Use technical Fortran/LAPACK terminology. "
                    "Return ONLY the 3 queries, one per line, no numbering or extra text."
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    expanded = response.choices[0].message.content.strip().split("\n")
    # Include the original query too
    return [query] + [q.strip() for q in expanded if q.strip()]


def embed_query(query: str) -> list[float]:
    """Turn a user's question into a vector."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    return response.data[0].embedding


def search(query: str, top_k: int = TOP_K, use_expansion: bool = True) -> list[dict]:
    """
    Search Pinecone for code chunks that match the query.
    Uses query expansion to improve results by default.
    Returns a list of results with metadata and similarity scores.
    """
    if use_expansion:
        queries = expand_query(query)
    else:
        queries = [query]

    # Search with each query variation and collect all results
    seen_ids = set()
    all_results = []

    for q in queries:
        query_vector = embed_query(q)
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )

        for match in results.matches:
            if match.id not in seen_ids:
                seen_ids.add(match.id)
                all_results.append({
                    "name": match.metadata.get("name", ""),
                    "file_path": match.metadata.get("file_path", ""),
                    "start_line": match.metadata.get("start_line", 0),
                    "end_line": match.metadata.get("end_line", 0),
                    "dependencies": match.metadata.get("dependencies", ""),
                    "text": match.metadata.get("text", ""),
                    "score": match.score,
                })
            else:
                # If we've seen this result before, keep the higher score
                for r in all_results:
                    if r["name"] == match.metadata.get("name", ""):
                        r["score"] = max(r["score"], match.score)
                        break

    # Sort by score (best first) and return top_k
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]
