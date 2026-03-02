import os
import re
import tiktoken
from openai import OpenAI
from pinecone import Pinecone
from app.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    LAPACK_DATA_DIR,
)

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Token counter for text-embedding-3-small
encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
MAX_TOKENS = 6000  # Leave buffer below the 8191 limit
CHUNK_TOKENS = 4000
OVERLAP_TOKENS = 500


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


def find_fortran_files(base_dir: str) -> list[str]:
    """Recursively find all .f and .f90 files."""
    fortran_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".f") or f.endswith(".f90"):
                fortran_files.append(os.path.join(root, f))
    return fortran_files


def parse_subroutines(filepath: str) -> list[dict]:
    """Parse a Fortran file into subroutine/function chunks with metadata."""
    with open(filepath, "r", errors="replace") as f:
        lines = f.readlines()

    # Get path relative to LAPACK root for cleaner metadata
    rel_path = os.path.relpath(filepath, LAPACK_DATA_DIR)

    chunks = []
    current_chunk_lines = []
    current_name = None
    start_line = 1

    for i, line in enumerate(lines, 1):
        upper = line.upper().strip()

        # Detect subroutine/function start (including RECURSIVE SUBROUTINE, etc.)
        match = re.match(
            r"^\s*(?:RECURSIVE\s+|PURE\s+|ELEMENTAL\s+)?(SUBROUTINE|(?:[\w*]+\s+)?FUNCTION)\s+(\w+)",
            line,
            re.IGNORECASE,
        )

        if match and current_name is None:
            # Starting a new subroutine/function
            current_name = match.group(2).upper()
            # Include any comment block that came before (already accumulated)
            start_line = start_line if current_chunk_lines else i
            current_chunk_lines.append(line)

        elif current_name and (
            upper.startswith("END SUBROUTINE")
            or upper.startswith("END FUNCTION")
            or upper == "END"
        ):
            # End of subroutine/function
            current_chunk_lines.append(line)

            # Extract CALL dependencies
            chunk_text = "".join(current_chunk_lines)
            calls = re.findall(r"CALL\s+(\w+)", chunk_text, re.IGNORECASE)
            dependencies = list(set(c.upper() for c in calls))

            chunks.append({
                "name": current_name,
                "text": chunk_text,
                "file_path": rel_path,
                "start_line": start_line,
                "end_line": i,
                "dependencies": dependencies,
            })

            current_chunk_lines = []
            current_name = None
            start_line = i + 1

        else:
            current_chunk_lines.append(line)

    # If no subroutines were found, treat the whole file as one chunk
    if not chunks and lines:
        full_text = "".join(lines)
        # Try to extract a name from the filename
        name = os.path.splitext(os.path.basename(filepath))[0].upper()
        calls = re.findall(r"CALL\s+(\w+)", full_text, re.IGNORECASE)
        dependencies = list(set(c.upper() for c in calls))

        chunks.append({
            "name": name,
            "text": full_text,
            "file_path": rel_path,
            "start_line": 1,
            "end_line": len(lines),
            "dependencies": dependencies,
        })

    return chunks


def split_oversized_chunk(chunk: dict) -> list[dict]:
    """Split a chunk that exceeds MAX_TOKENS into overlapping pieces."""
    tokens = encoding.encode(chunk["text"])
    if len(tokens) <= MAX_TOKENS:
        return [chunk]

    pieces = []
    start = 0
    piece_num = 1

    while start < len(tokens):
        end = start + CHUNK_TOKENS
        piece_tokens = tokens[start:end]
        piece_text = encoding.decode(piece_tokens)

        pieces.append({
            "name": f"{chunk['name']}_PART{piece_num}",
            "text": piece_text,
            "file_path": chunk["file_path"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "dependencies": chunk["dependencies"],
        })

        start = end - OVERLAP_TOKENS
        piece_num += 1

    print(f"  Split {chunk['name']} into {len(pieces)} pieces ({len(tokens)} tokens)")
    return pieces


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts using OpenAI."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def upsert_to_pinecone(vectors: list[dict], batch_size: int = 100):
    """Upload vectors with metadata to Pinecone in batches."""
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"  Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")


def run_ingestion():
    """Main ingestion pipeline."""
    print("Step 1: Finding Fortran files...")
    files = find_fortran_files(LAPACK_DATA_DIR)
    print(f"  Found {len(files)} Fortran files")

    print("\nStep 2: Parsing subroutines...")
    all_chunks = []
    for filepath in files:
        chunks = parse_subroutines(filepath)
        for chunk in chunks:
            all_chunks.extend(split_oversized_chunk(chunk))

    print(f"  Parsed {len(all_chunks)} total chunks")

    print("\nStep 3: Generating embeddings...")
    batch_size = 100
    all_vectors = []

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        embeddings = generate_embeddings(texts)

        for chunk, embedding in zip(batch, embeddings):
            vector_id = f"{chunk['file_path']}::{chunk['name']}"
            all_vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "name": chunk["name"],
                    "file_path": chunk["file_path"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "dependencies": ",".join(chunk["dependencies"]),
                    "text": chunk["text"][:40000],  # Pinecone metadata limit
                },
            })

        print(f"  Embedded batch {i // batch_size + 1}/{(len(all_chunks) + batch_size - 1) // batch_size}")

    print(f"\nStep 4: Uploading {len(all_vectors)} vectors to Pinecone...")
    upsert_to_pinecone(all_vectors)

    print(f"\nDone! Ingested {len(all_vectors)} chunks into Pinecone.")


if __name__ == "__main__":
    run_ingestion()
