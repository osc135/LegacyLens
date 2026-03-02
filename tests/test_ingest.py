"""
Quick smoke tests for the ingestion pipeline.
Tests each stage independently before running the full pipeline.
"""
import sys
import os

# Add project root to path so we can import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, LAPACK_DATA_DIR
from app.ingest import find_fortran_files, parse_subroutines, count_tokens, split_oversized_chunk


def test_find_files():
    """Test that we can find Fortran files in LAPACK."""
    files = find_fortran_files(LAPACK_DATA_DIR)
    print(f"Found {len(files)} Fortran files")
    assert len(files) > 50, "Should find at least 50 Fortran files"
    assert any(f.endswith(".f") for f in files), "Should find .f files"
    print("PASS: File discovery works\n")


def test_parse_subroutine():
    """Test that we can parse a known LAPACK file (dgesv.f)."""
    dgesv_path = os.path.join(LAPACK_DATA_DIR, "SRC", "dgesv.f")
    chunks = parse_subroutines(dgesv_path)

    print(f"Parsed {len(chunks)} chunk(s) from dgesv.f")
    for chunk in chunks:
        print(f"  Name: {chunk['name']}")
        print(f"  File: {chunk['file_path']}")
        print(f"  Lines: {chunk['start_line']}-{chunk['end_line']}")
        print(f"  Dependencies: {chunk['dependencies']}")
        print(f"  Tokens: {count_tokens(chunk['text'])}")
        print(f"  First 200 chars: {chunk['text'][:200]}")
        print()

    assert len(chunks) >= 1, "Should find at least one subroutine"
    assert chunks[0]["name"] == "DGESV", "First subroutine should be DGESV"
    print("PASS: Subroutine parsing works\n")


def test_recursive_subroutine():
    """Test that RECURSIVE SUBROUTINE files get parsed correctly."""
    path = os.path.join(LAPACK_DATA_DIR, "SRC", "zgelqt3.f")
    chunks = parse_subroutines(path)

    print(f"Parsed {len(chunks)} chunk(s) from zgelqt3.f")
    for chunk in chunks:
        print(f"  Name: {chunk['name']}")
        print(f"  Tokens: {count_tokens(chunk['text'])}")

    assert len(chunks) >= 1, "Should find at least one subroutine"
    assert chunks[0]["name"] == "ZGELQT3", "Should parse RECURSIVE SUBROUTINE name"
    print("PASS: RECURSIVE SUBROUTINE parsing works\n")


def test_no_subroutine_fallback():
    """Test that files without SUBROUTINE/FUNCTION keywords still get ingested."""
    path = os.path.join(LAPACK_DATA_DIR, "INSTALL", "LAPACK_version.f")
    chunks = parse_subroutines(path)

    print(f"Parsed {len(chunks)} chunk(s) from LAPACK_version.f")
    for chunk in chunks:
        print(f"  Name: {chunk['name']}")
        print(f"  Lines: {chunk['start_line']}-{chunk['end_line']}")

    assert len(chunks) >= 1, "Should fall back to whole-file chunk"
    assert chunks[0]["text"], "Chunk text should not be empty"
    print("PASS: No-subroutine fallback works\n")


def test_oversized_chunk_splitting():
    """Test that oversized chunks get split with overlap."""
    # Create a fake chunk that's too big
    fake_chunk = {
        "name": "BIG_SUB",
        "text": "x " * 7000,  # ~7000 tokens, over the 6000 limit
        "file_path": "test/fake.f",
        "start_line": 1,
        "end_line": 100,
        "dependencies": ["FOO"],
    }

    pieces = split_oversized_chunk(fake_chunk)
    print(f"Split oversized chunk into {len(pieces)} pieces")
    for p in pieces:
        print(f"  {p['name']}: {count_tokens(p['text'])} tokens")

    assert len(pieces) > 1, "Should split into multiple pieces"
    assert all("PART" in p["name"] for p in pieces), "Pieces should have PART in name"
    assert all(p["file_path"] == "test/fake.f" for p in pieces), "Metadata should be preserved"
    print("PASS: Oversized chunk splitting works\n")


def test_comment_header_attached():
    """Test that comment headers stay attached to their subroutine."""
    dgesv_path = os.path.join(LAPACK_DATA_DIR, "SRC", "dgesv.f")
    chunks = parse_subroutines(dgesv_path)

    # The DGESV chunk should start with comment lines (starting with *)
    first_lines = chunks[0]["text"].split("\n")[:5]
    has_comments = any(line.strip().startswith("*") for line in first_lines)

    print(f"First 5 lines of DGESV chunk:")
    for line in first_lines:
        print(f"  {line.rstrip()}")

    assert has_comments, "Comment header should be attached to subroutine chunk"
    print("PASS: Comment headers attached correctly\n")


def test_openai_connection():
    """Test that we can connect to OpenAI and generate an embedding."""
    from app.ingest import generate_embeddings

    assert OPENAI_API_KEY, "OPENAI_API_KEY is not set"

    embeddings = generate_embeddings(["test embedding for LAPACK subroutine"])
    print(f"Generated embedding with {len(embeddings[0])} dimensions")
    assert len(embeddings[0]) == 1536, "Embedding should be 1536 dimensions"
    print("PASS: OpenAI embedding works\n")


def test_pinecone_connection():
    """Test that we can connect to Pinecone."""
    from pinecone import Pinecone

    assert PINECONE_API_KEY, "PINECONE_API_KEY is not set"

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    print(f"Pinecone index '{PINECONE_INDEX_NAME}' stats:")
    print(f"  Total vectors: {stats.total_vector_count}")
    print(f"  Dimension: {stats.dimension}")
    assert stats.dimension == 1536, "Index dimension should be 1536"
    print("PASS: Pinecone connection works\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Running ingestion smoke tests...")
    print("=" * 50 + "\n")

    test_find_files()
    test_parse_subroutine()
    test_recursive_subroutine()
    test_no_subroutine_fallback()
    test_oversized_chunk_splitting()
    test_comment_header_attached()
    test_openai_connection()
    test_pinecone_connection()

    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)
