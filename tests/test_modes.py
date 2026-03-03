"""
Tests for the mode system (Ask/Explain/Docs) and follow-up generation.
Run: python3 tests/test_modes.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.retrieval import search
from app.generation import (
    generate_answer,
    generate_answer_stream,
    generate_followups,
    build_context,
    get_system_prompt,
    SYSTEM_PROMPT,
    EXPLAIN_PROMPT,
    DOCS_PROMPT,
)


def test_prompt_routing():
    """Test that each mode returns the correct system prompt."""
    assert get_system_prompt("ask") == SYSTEM_PROMPT, "Ask mode should use SYSTEM_PROMPT"
    assert get_system_prompt("explain") == EXPLAIN_PROMPT, "Explain mode should use EXPLAIN_PROMPT"
    assert get_system_prompt("docs") == DOCS_PROMPT, "Docs mode should use DOCS_PROMPT"
    assert get_system_prompt("invalid") == SYSTEM_PROMPT, "Invalid mode should fall back to SYSTEM_PROMPT"
    print("PASS: Prompt routing works\n")


def test_build_context_numbering():
    """Test that build_context numbers sources correctly."""
    results = search("DGESV", top_k=3)
    context = build_context(results)

    assert "[Source 1]" in context, "Should have [Source 1]"
    assert "[Source 2]" in context, "Should have [Source 2]"
    assert "[Source 3]" in context, "Should have [Source 3]"
    assert "---" in context, "Should have section separators"
    print("PASS: Context numbering works\n")


def test_explain_mode():
    """Test that explain mode produces structured output."""
    results = search("DGESV solve linear system Ax=b", top_k=5)
    answer = generate_answer("DGESV solve linear system Ax=b", results, mode="explain")

    print(f"Explain mode answer ({len(answer)} chars):")
    print(answer[:600])
    print("...\n")

    assert len(answer) > 100, "Answer should be substantial"
    # Explain mode should have structured sections
    has_structure = any(heading in answer for heading in ["Purpose", "How It Works", "Parameters", "Key Details"])
    assert has_structure, "Explain mode should produce structured output with headings"
    print("PASS: Explain mode works\n")


def test_docs_mode():
    """Test that docs mode produces documentation format."""
    results = search("DGETRF LU factorization subroutine", top_k=5)
    answer = generate_answer("DGETRF LU factorization subroutine", results, mode="docs")

    print(f"Docs mode answer ({len(answer)} chars):")
    print(answer[:600])
    print("...\n")

    assert len(answer) > 100, "Answer should be substantial"
    # Docs mode should have parameter table and algorithm section
    has_docs_format = any(heading in answer for heading in ["Parameters", "Algorithm", "Purpose"])
    assert has_docs_format, "Docs mode should produce documentation with Parameters/Algorithm sections"
    print("PASS: Docs mode works\n")


def test_followup_generation():
    """Test that follow-up questions are generated correctly."""
    query = "What does DGESV do?"
    answer = "DGESV solves a system of linear equations Ax=b using LU factorization."
    followups = generate_followups(query, answer)

    print(f"Follow-up questions for '{query}':")
    for i, q in enumerate(followups, 1):
        print(f"  {i}. {q}")
    print()

    assert len(followups) == 3, "Should generate exactly 3 follow-ups"
    assert all(len(q) > 10 for q in followups), "Each follow-up should be a real question"
    print("PASS: Follow-up generation works\n")


def test_retrieval_excludes_test_files():
    """Test that search results exclude TESTING/ and other non-library directories."""
    results = search("solve linear equations", top_k=5)

    print("Checking that results exclude test files:")
    excluded = ("TESTING/", "INSTALL/", "CMAKE/", "CBLAS/", "LAPACKE/", "BLAS/")
    for r in results:
        print(f"  {r['name']} — {r['file_path']}")
        assert not r["file_path"].startswith(excluded), \
            f"Result {r['name']} from {r['file_path']} should be filtered out"

    print("\nPASS: Test files excluded from results\n")


def test_streaming_with_mode():
    """Test that streaming works with mode parameter."""
    results = search("DGESV", top_k=3)

    full_answer = ""
    chunk_count = 0
    for text_chunk in generate_answer_stream("DGESV", results, mode="explain"):
        full_answer += text_chunk
        chunk_count += 1

    print(f"Streamed explain mode: {chunk_count} chunks, {len(full_answer)} chars")
    assert len(full_answer) > 50, "Streamed answer should be substantial"
    assert chunk_count > 5, "Should receive multiple chunks"
    print("PASS: Streaming with mode works\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Running mode & feature tests...")
    print("=" * 50 + "\n")

    test_prompt_routing()
    test_build_context_numbering()
    test_explain_mode()
    test_docs_mode()
    test_followup_generation()
    test_retrieval_excludes_test_files()
    test_streaming_with_mode()

    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)
