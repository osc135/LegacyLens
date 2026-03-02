"""
Tests for the answer generation pipeline.
Run: python3 tests/test_generation.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.retrieval import search
from app.generation import generate_answer, generate_answer_stream, build_context


def test_build_context():
    """Test that context is formatted correctly for the LLM."""
    results = search("DGESV", top_k=3)
    context = build_context(results)

    print("Built context preview (first 500 chars):")
    print(context[:500])
    print("...\n")

    assert "DGESV" in context, "Context should contain DGESV"
    assert "lines" in context, "Context should have line references"
    print("PASS: Context building works\n")


def test_generate_answer():
    """Test full answer generation (non-streaming)."""
    results = search("What does DGESV do?", top_k=5)
    answer = generate_answer("What does DGESV do?", results)

    print(f"Question: 'What does DGESV do?'")
    print(f"Answer ({len(answer)} chars):")
    print(answer)
    print()

    assert len(answer) > 50, "Answer should be substantial"
    assert "DGESV" in answer.upper(), "Answer should mention DGESV"
    print("PASS: Answer generation works\n")


def test_generate_answer_stream():
    """Test streaming answer generation."""
    results = search("Find matrix factorization routines", top_k=5)

    print("Streaming answer for 'Find matrix factorization routines':")
    full_answer = ""
    chunk_count = 0
    for text_chunk in generate_answer_stream("Find matrix factorization routines", results):
        full_answer += text_chunk
        chunk_count += 1
        print(text_chunk, end="", flush=True)
    print(f"\n\n(Received {chunk_count} chunks, {len(full_answer)} total chars)")

    assert len(full_answer) > 50, "Streamed answer should be substantial"
    assert chunk_count > 5, "Should receive multiple chunks"
    print("PASS: Streaming generation works\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Running generation tests...")
    print("=" * 50 + "\n")

    test_build_context()
    test_generate_answer()
    test_generate_answer_stream()

    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)
