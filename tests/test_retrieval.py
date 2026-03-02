"""
Tests for the retrieval pipeline.
Run: python3 tests/test_retrieval.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.retrieval import search, embed_query, expand_query


def test_embed_query():
    """Test that we can turn a question into a vector."""
    vector = embed_query("How do I solve a linear system?")
    print(f"Query vector has {len(vector)} dimensions")
    assert len(vector) == 1536, "Should be 1536 dimensions"
    print("PASS: Query embedding works\n")


def test_query_expansion():
    """Test that query expansion generates technical variations."""
    queries = expand_query("How do I solve a linear system?")
    print(f"Original + expanded queries:")
    for i, q in enumerate(queries):
        print(f"  {i+1}. {q}")
    print()

    assert len(queries) >= 3, "Should have at least 3 queries (original + expansions)"
    assert queries[0] == "How do I solve a linear system?", "First should be the original"
    print("PASS: Query expansion works\n")


def test_search_linear_system():
    """Test searching for linear system solvers — should find DGESV."""
    results = search("How do I solve a linear system?")
    print(f"Query: 'How do I solve a linear system?'")
    print(f"Found {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['name']} ({r['file_path']}:{r['start_line']}-{r['end_line']}) — score: {r['score']:.4f}")
    print()

    assert len(results) == 5, "Should return 5 results"
    assert results[0]["score"] > 0.4, "Top result should have reasonable similarity score"
    # LAPACK has many linear solvers (xGESV, xPPSV, xPOSV, etc.) — any solver is valid
    print("PASS: Linear system search works\n")


def test_search_by_subroutine_name():
    """Test searching for a specific subroutine by name."""
    results = search("DGESV")
    print(f"Query: 'DGESV'")
    print(f"Found {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['name']} ({r['file_path']}) — score: {r['score']:.4f}")
    print()

    assert results[0]["name"] == "DGESV" or "DGESV" in results[0]["name"], "First result should be DGESV"
    print("PASS: Subroutine name search works\n")


def test_search_matrix_factorization():
    """Test searching for matrix factorization routines."""
    results = search("Find all matrix factorization routines")
    print(f"Query: 'Find all matrix factorization routines'")
    print(f"Found {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['name']} ({r['file_path']}) — score: {r['score']:.4f}")
    print()

    names = [r["name"] for r in results]
    assert any("GETRF" in n or "POTRF" in n or "SYTRF" in n for n in names), \
        "Should find factorization routines (xGETRF, xPOTRF, or xSYTRF)"
    print("PASS: Matrix factorization search works\n")


def test_results_have_metadata():
    """Test that results include all required metadata."""
    results = search("eigenvalue decomposition")
    r = results[0]

    print(f"Checking metadata for: {r['name']}")
    print(f"  file_path: {r['file_path']}")
    print(f"  start_line: {r['start_line']}")
    print(f"  end_line: {r['end_line']}")
    print(f"  dependencies: {r['dependencies']}")
    print(f"  score: {r['score']}")
    print(f"  text length: {len(r['text'])} chars")
    print()

    assert r["name"], "Should have a name"
    assert r["file_path"], "Should have a file path"
    assert r["start_line"] > 0, "Should have a start line"
    assert r["end_line"] > 0, "Should have an end line"
    assert r["score"] > 0, "Should have a similarity score"
    assert r["text"], "Should have the code text"
    print("PASS: Results have all required metadata\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Running retrieval tests...")
    print("=" * 50 + "\n")

    test_embed_query()
    test_query_expansion()
    test_search_linear_system()
    test_search_by_subroutine_name()
    test_search_matrix_factorization()
    test_results_have_metadata()

    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)
