import logging
from concurrent.futures import ThreadPoolExecutor
from langfuse.openai import OpenAI
from pinecone import Pinecone
from collections import deque
from app.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
    FETCH_MULTIPLIER,
    QUERY_EXPANSION_TEMPERATURE,
    DEPS_MAX_DEPTH,
    DEPS_MAX_NODES,
)

logger = logging.getLogger(__name__)

# Initialize clients — langfuse.openai.OpenAI auto-traces all LLM/embedding calls
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

EXCLUDED_PREFIXES = ("TESTING/", "INSTALL/", "CMAKE/", "CBLAS/", "LAPACKE/", "BLAS/")

# Reuse a single thread pool across requests to avoid creation overhead
_executor = ThreadPoolExecutor(max_workers=6)


def expand_query(query: str) -> list[str]:
    """
    Use GPT-4o-mini to rephrase the user's question into
    3 technical variations that will match LAPACK code better.
    """
    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            temperature=QUERY_EXPANSION_TEMPERATURE,
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
        return [query] + [q.strip() for q in expanded if q.strip()]
    except Exception as e:
        logger.warning("Query expansion failed, using original query: %s", e)
        return [query]


def describe_code_for_search(code: str) -> list[str]:
    """
    Given a Fortran code snippet, generate natural language descriptions
    that will match LAPACK library source files when embedded.
    """
    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            temperature=QUERY_EXPANSION_TEMPERATURE,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert on the LAPACK Fortran library. "
                        "Given a Fortran code snippet, describe what it does in 3 different ways "
                        "that would match LAPACK subroutine source code, comment headers, and documentation. "
                        "Focus on the pattern (e.g. 'input argument validation with XERBLA', "
                        "'LU factorization with partial pivoting', 'workspace query for optimal LWORK'). "
                        "Include specific LAPACK subroutine names if you can identify them. "
                        "Return ONLY the 3 descriptions, one per line, no numbering or extra text."
                    ),
                },
                {"role": "user", "content": code},
            ],
        )
        descriptions = response.choices[0].message.content.strip().split("\n")
        return [q.strip() for q in descriptions if q.strip()]
    except Exception as e:
        logger.warning("Code description failed, using raw code: %s", e)
        return [code]


def embed_queries(queries: list[str]) -> list[list[float]]:
    """Embed multiple queries in a single API call."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=queries,
    )
    return [item.embedding for item in response.data]


def embed_query(query: str) -> list[float]:
    """Turn a single query into a vector."""
    return embed_queries([query])[0]


def _pinecone_query(vector: list[float], fetch_k: int) -> list:
    """Run a single Pinecone query and return matches."""
    return index.query(vector=vector, top_k=fetch_k, include_metadata=True).matches


def _collect_matches(all_matches: list, seen_ids: set, all_results: list):
    """Deduplicate and collect Pinecone matches into results list."""
    for match in all_matches:
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
            for r in all_results:
                if r["name"] == match.metadata.get("name", ""):
                    r["score"] = max(r["score"], match.score)
                    break


def search(query: str, top_k: int = TOP_K, use_expansion: bool = True, code_search: bool = False, fetch_multiplier: int = FETCH_MULTIPLIER) -> list[dict]:
    """
    Search Pinecone for code chunks that match the query.
    Uses query expansion and parallel execution for speed.
    Returns a list of results with metadata and similarity scores.
    """
    fetch_k = top_k * fetch_multiplier
    seen_ids = set()
    all_results = []

    # Phase 1: Fire expansion AND original query embedding concurrently
    if code_search:
        expansion_future = _executor.submit(describe_code_for_search, query)
    elif use_expansion:
        expansion_future = _executor.submit(expand_query, query)
    else:
        expansion_future = None

    original_embed_future = _executor.submit(embed_queries, [query])

    # Phase 2: As soon as original embedding is ready, search Pinecone
    original_vector = original_embed_future.result()[0]
    original_search_future = _executor.submit(_pinecone_query, original_vector, fetch_k)

    # Phase 3: When expansion is ready, batch-embed expanded queries (as future)
    expanded_search_futures = []
    if expansion_future is not None:
        expanded_queries = expansion_future.result()
        extra_queries = [q for q in expanded_queries if q != query]

        if extra_queries:
            # Submit batch embedding as a future so original Pinecone can finish in parallel
            expanded_embed_future = _executor.submit(embed_queries, extra_queries)
            expanded_vectors = expanded_embed_future.result()

            # Phase 4: Fire all expanded Pinecone queries in parallel
            expanded_search_futures = [
                _executor.submit(_pinecone_query, vec, fetch_k)
                for vec in expanded_vectors
            ]

    # Collect original results (likely already done by now)
    _collect_matches(original_search_future.result(), seen_ids, all_results)

    # Collect expanded results
    for future in expanded_search_futures:
        _collect_matches(future.result(), seen_ids, all_results)

    # Filter out test files — only keep actual library source code
    all_results = [
        r for r in all_results
        if not r["file_path"].startswith(EXCLUDED_PREFIXES)
    ]

    # Sort by score (best first) and return top_k
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


def search_by_name(name: str) -> dict | None:
    """
    Find a subroutine by exact name using Pinecone metadata filter.
    Handles _PART suffixes by trying a prefix match as fallback.
    Returns a single result dict or None.
    """
    name_upper = name.strip().upper()

    # Try exact match first
    try:
        # We need a vector for Pinecone query even with metadata filter,
        # so embed the name itself
        query_vector = embed_query(name_upper)
        results = index.query(
            vector=query_vector,
            top_k=1,
            include_metadata=True,
            filter={"name": {"$eq": name_upper}},
        )
        if results.matches:
            m = results.matches[0]
            return {
                "name": m.metadata.get("name", ""),
                "file_path": m.metadata.get("file_path", ""),
                "start_line": m.metadata.get("start_line", 0),
                "end_line": m.metadata.get("end_line", 0),
                "dependencies": m.metadata.get("dependencies", ""),
                "text": m.metadata.get("text", ""),
                "score": m.score,
            }
    except Exception as e:
        logger.warning("Exact name search failed for %s: %s", name_upper, e)

    # Fallback: try _PART1 suffix (chunked subroutines)
    try:
        query_vector = embed_query(name_upper)
        results = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True,
            filter={"name": {"$eq": f"{name_upper}_PART1"}},
        )
        if results.matches:
            m = results.matches[0]
            return {
                "name": name_upper,
                "file_path": m.metadata.get("file_path", ""),
                "start_line": m.metadata.get("start_line", 0),
                "end_line": m.metadata.get("end_line", 0),
                "dependencies": m.metadata.get("dependencies", ""),
                "text": m.metadata.get("text", ""),
                "score": m.score,
            }
    except Exception as e:
        logger.warning("Part-suffix search failed for %s: %s", name_upper, e)

    return None


def resolve_dependency_graph(
    root_name: str,
    max_depth: int = DEPS_MAX_DEPTH,
    max_nodes: int = DEPS_MAX_NODES,
) -> dict:
    """
    BFS traversal of subroutine dependencies.
    Returns {"root": str, "nodes": {name: {found, file_path, dependencies, text}}, "edges": [[from, to], ...]}.
    """
    root_name = root_name.strip().upper()
    nodes = {}
    edges = []
    visited = set()
    queue = deque()  # (name, depth)

    queue.append((root_name, 0))

    while queue and len(nodes) < max_nodes:
        name, depth = queue.popleft()
        if name in visited:
            continue
        visited.add(name)

        result = search_by_name(name)
        if result:
            deps_str = result["dependencies"]
            dep_list = [d.strip() for d in deps_str.split(",") if d.strip()] if deps_str else []
            nodes[name] = {
                "found": True,
                "file_path": result["file_path"],
                "dependencies": dep_list,
                "text": result["text"][:2000],
                "start_line": result["start_line"],
                "end_line": result["end_line"],
            }
            for dep in dep_list:
                if dep == name:
                    continue  # skip self-referencing (recursive) calls
                edges.append([name, dep])
                if depth < max_depth and dep not in visited and len(nodes) < max_nodes:
                    queue.append((dep, depth + 1))
        else:
            nodes[name] = {"found": False}

    return {"root": root_name, "nodes": nodes, "edges": edges}
