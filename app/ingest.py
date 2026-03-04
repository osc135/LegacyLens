import os
import re
import tiktoken
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
import logging
from app.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    LAPACK_DATA_DIR,
    MAX_TOKENS,
    CHUNK_TOKENS,
    OVERLAP_TOKENS,
    EMBEDDING_BATCH_SIZE,
    UPSERT_BATCH_SIZE,
    PINECONE_METADATA_CHAR_LIMIT,
    MIN_EMBEDDING_TOKENS,
)

logger = logging.getLogger(__name__)

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

BM25_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bm25_model.json")

# Token counter for text-embedding-3-small
encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


def clean_header_text(name: str, raw_text: str, dependencies: list[str]) -> str:
    """Extract and clean the documentation header from a subroutine for embedding.

    Strips Fortran comment markers, Doxygen tags, HTML, URLs, and separator
    lines, producing clean English text that embeds closer to natural language
    queries. Falls back to lightly-cleaned full text when the header is too short.
    """
    lines = raw_text.split("\n")

    # Collect comment-header lines (lines starting with *> or * before code begins)
    header_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("*>") or stripped.startswith("*"):
            header_lines.append(stripped)
        elif re.match(r"^\s*(?:SUBROUTINE|FUNCTION|RECURSIVE|PURE|ELEMENTAL|INTEGER|DOUBLE|REAL|COMPLEX|LOGICAL|CHARACTER)", stripped, re.IGNORECASE):
            break  # reached code
        elif stripped == "":
            header_lines.append("")

    def clean_lines(text_lines):
        cleaned = []
        for l in text_lines:
            # Strip comment markers
            l = re.sub(r"^\*>\s?", "", l)
            l = re.sub(r"^\*\s?", "", l)
            # Remove Doxygen tags
            l = re.sub(r"\\(verbatim|endverbatim|brief|par\b|param\[(in|out|in,out)\]|addtogroup\s+\w*|ingroup\s+\w*)", "", l)
            l = re.sub(r"\\b\b", "", l)
            # Remove HTML tags
            l = re.sub(r"<[^>]+>", "", l)
            # Remove URLs
            l = re.sub(r"https?://\S+", "", l)
            # Remove separator lines (===, ---, ...)
            if re.match(r"^[\s=\-\.*]+$", l):
                continue
            # Remove DOCUMENTATION banner
            if "documentation" in l.lower() and re.match(r"^[\s=\-]*documentation[\s=\-]*$", l, re.IGNORECASE):
                continue
            # Remove download/link lines
            if re.match(r"^\s*\[(TGZ|ZIP|TXT)\]\s*$", l):
                continue
            if re.match(r"^\s*download\s+", l, re.IGNORECASE):
                continue
            # Remove "Online html documentation" lines
            if "online html documentation" in l.lower():
                continue
            # Remove "Definition:" section header and Fortran signature lines
            if re.match(r"^\s*Definition:\s*$", l):
                continue
            if re.match(r"^\s*(SUBROUTINE|FUNCTION|INTEGER|DOUBLE\s+PRECISION|REAL|COMPLEX|LOGICAL|CHARACTER)\b", l, re.IGNORECASE):
                continue
            if re.match(r"^\s*\.\.\s*(Scalar|Array|Local)\s+Arguments\s*\.\.", l, re.IGNORECASE):
                continue
            # Remove author/institution boilerplate
            if re.match(r"^\s*(Univ\.|University|-- LAPACK|September|November|December|January|June|Modified)", l, re.IGNORECASE):
                continue
            cleaned.append(l)
        return "\n".join(cleaned)

    header_text = clean_lines(header_lines)
    # Collapse multiple blank lines
    header_text = re.sub(r"\n{3,}", "\n\n", header_text).strip()

    # Build structured embedding text
    parts = [f"LAPACK subroutine {name}."]
    if header_text:
        parts.append(header_text)
    if dependencies:
        parts.append(f"Calls: {', '.join(sorted(dependencies))}.")

    embedding_text = "\n\n".join(parts)

    # Fallback: if cleaned header is too short, use lightly-cleaned full text
    if count_tokens(embedding_text) < MIN_EMBEDDING_TOKENS:
        fallback = raw_text
        # Light cleaning: strip comment markers and HTML
        fallback = re.sub(r"^\*>\s?", "", fallback, flags=re.MULTILINE)
        fallback = re.sub(r"^\*\s?", "", fallback, flags=re.MULTILINE)
        fallback = re.sub(r"<[^>]+>", "", fallback)
        fallback = re.sub(r"https?://\S+", "", fallback)
        fallback = re.sub(r"\n{3,}", "\n\n", fallback).strip()

        parts = [f"LAPACK subroutine {name}."]
        parts.append(fallback)
        if dependencies:
            parts.append(f"Calls: {', '.join(sorted(dependencies))}.")
        embedding_text = "\n\n".join(parts)

    # Truncate to stay within embedding model token limit
    tokens = encoding.encode(embedding_text)
    if len(tokens) > MAX_TOKENS:
        embedding_text = encoding.decode(tokens[:MAX_TOKENS])

    return embedding_text


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

            embedding_text = clean_header_text(current_name, chunk_text, dependencies)

            chunks.append({
                "name": current_name,
                "text": chunk_text,
                "embedding_text": embedding_text,
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

        embedding_text = clean_header_text(name, full_text, dependencies)

        chunks.append({
            "name": name,
            "text": full_text,
            "embedding_text": embedding_text,
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

        piece_name = f"{chunk['name']}_PART{piece_num}"
        # First piece gets the parent's cleaned header; subsequent pieces
        # get their own cleaned text (header only appears once)
        if piece_num == 1:
            piece_embedding = chunk.get("embedding_text", piece_text)
        else:
            piece_embedding = clean_header_text(piece_name, piece_text, chunk["dependencies"])

        pieces.append({
            "name": piece_name,
            "text": piece_text,
            "embedding_text": piece_embedding,
            "file_path": chunk["file_path"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "dependencies": chunk["dependencies"],
        })

        start = end - OVERLAP_TOKENS
        piece_num += 1

    print(f"  Split {chunk['name']} into {len(pieces)} pieces ({len(tokens)} tokens)")
    return pieces


def generate_embeddings(texts: list[str], max_retries: int = 5) -> list[list[float]]:
    """Generate embeddings for a batch of texts using OpenAI."""
    import time
    for attempt in range(max_retries):
        try:
            response = openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["rate_limit", "429", "connection", "timeout", "disconnect"]):
                wait = 2 ** attempt + 5
                print(f"  Retryable error ({type(e).__name__}), waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")


def upsert_to_pinecone(vectors: list[dict], batch_size: int = UPSERT_BATCH_SIZE, index=None):
    """Upload vectors with metadata to Pinecone in batches."""
    if index is None:
        index = pc.Index(PINECONE_INDEX_NAME)
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"  Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")


def recreate_index():
    """Delete and recreate the Pinecone index with dotproduct metric for hybrid search."""
    import time

    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME in existing:
        print(f"  Deleting existing index '{PINECONE_INDEX_NAME}'...")
        pc.delete_index(PINECONE_INDEX_NAME)
        time.sleep(5)

    print(f"  Creating index '{PINECONE_INDEX_NAME}' with dotproduct metric...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSIONS,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    # Wait for index to be ready
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        print("  Waiting for index to be ready...")
        time.sleep(5)

    print("  Index ready.")
    return pc.Index(PINECONE_INDEX_NAME)


def run_ingestion():
    """Main ingestion pipeline with hybrid (dense + sparse BM25) vectors."""
    print("Step 1: Recreating Pinecone index with dotproduct metric...")
    index = recreate_index()

    print("\nStep 2: Finding Fortran files...")
    files = find_fortran_files(LAPACK_DATA_DIR)
    print(f"  Found {len(files)} Fortran files")

    print("\nStep 3: Parsing subroutines...")
    all_chunks = []
    for filepath in files:
        chunks = parse_subroutines(filepath)
        for chunk in chunks:
            all_chunks.extend(split_oversized_chunk(chunk))

    print(f"  Parsed {len(all_chunks)} total chunks")

    print("\nStep 4: Fitting BM25 on corpus...")
    corpus = [chunk["embedding_text"] for chunk in all_chunks]
    bm25 = BM25Encoder()
    bm25.fit(corpus)
    os.makedirs(os.path.dirname(BM25_MODEL_PATH), exist_ok=True)
    bm25.dump(BM25_MODEL_PATH)
    print(f"  BM25 model saved to {BM25_MODEL_PATH}")

    print("\nStep 5: Generating dense embeddings and sparse vectors...")
    all_vectors = []

    for i in range(0, len(all_chunks), EMBEDDING_BATCH_SIZE):
        batch = all_chunks[i : i + EMBEDDING_BATCH_SIZE]
        texts = [chunk["embedding_text"] for chunk in batch]
        embeddings = generate_embeddings(texts)
        sparse_vectors = [bm25.encode_documents(t) for t in texts]

        for chunk, embedding, sparse in zip(batch, embeddings, sparse_vectors):
            vector_id = f"{chunk['file_path']}::{chunk['name']}"
            all_vectors.append({
                "id": vector_id,
                "values": embedding,
                "sparse_values": {
                    "indices": sparse["indices"],
                    "values": sparse["values"],
                },
                "metadata": {
                    "name": chunk["name"],
                    "file_path": chunk["file_path"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "dependencies": ",".join(chunk["dependencies"]),
                    "text": chunk["text"][:PINECONE_METADATA_CHAR_LIMIT],
                },
            })

        print(f"  Embedded batch {i // EMBEDDING_BATCH_SIZE + 1}/{(len(all_chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE}")

    print(f"\nStep 6: Uploading {len(all_vectors)} vectors to Pinecone...")
    upsert_to_pinecone(all_vectors, index=index)

    print(f"\nDone! Ingested {len(all_vectors)} hybrid vectors into Pinecone.")


if __name__ == "__main__":
    run_ingestion()
