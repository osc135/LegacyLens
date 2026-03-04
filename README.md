# LegacyLens

AI-powered exploration tool for the LAPACK Fortran linear algebra library. Ask questions in plain English, get answers grounded in actual source code.

**Live demo:** [legacylens.up.railway.app](https://legacylens.up.railway.app)

## Features

LegacyLens provides five interaction modes for navigating LAPACK's 1,700+ subroutines:

| Mode | Description |
|------|-------------|
| **Ask** | Natural language Q&A with cited source code |
| **Explain** | Deep-dive explanation of a specific subroutine |
| **Docs** | Generate documentation for any routine |
| **Patterns** | Find common coding patterns across the library |
| **Deps** | Interactive dependency graph visualization |

All answers include citation grounding — source references are verified against retrieved chunks and visually flagged when unverifiable.

## Tech Stack

- **Backend:** Python 3.11, FastAPI, Uvicorn
- **LLM:** GPT-4o-mini (generation, query expansion, reranking)
- **Embeddings:** text-embedding-3-small (1536-dim)
- **Vector DB:** Pinecone (hybrid dense + BM25 sparse search)
- **Observability:** Langfuse (full LLM trace logging)
- **Frontend:** Jinja2 templates, vanilla JS
- **Deployment:** Docker, Railway

## Architecture

```
User Query
    │
    ├─→ Query Expansion (GPT-4o-mini, 2 variants)
    │
    ├─→ Subroutine Name Detection (regex)
    │       │
    │       └─→ Name-targeted Pinecone queries (metadata filter)
    │
    ├─→ Embedding (text-embedding-3-small + BM25)
    │       │
    │       └─→ Hybrid Pinecone search (dense + sparse)
    │
    └─→ Candidate Pool (deduplicated, score-boosted)
            │
            ├─→ LLM Reranker (top 20 → top 5)
            │
            ├─→ Streaming Answer Generation (GPT-4o-mini)
            │
            └─→ Citation Verification + Precision Scoring
```

All retrieval phases run in parallel using a shared thread pool for minimal latency.

## Setup

### Prerequisites

- Python 3.11+
- Pinecone account with a `legacylens` index (1536 dimensions, dotproduct metric)
- OpenAI API key

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...

# Optional: Langfuse observability
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Install

```bash
pip install -r requirements.txt
```

### Ingest LAPACK Source

Download the LAPACK source into `data/lapack/`, then run ingestion to chunk and embed:

```bash
python -m app.ingest
```

This parses Fortran files, extracts subroutine boundaries and dependencies, chunks large routines, and upserts embeddings into Pinecone.

### Run

```bash
uvicorn app.main:app --reload
```

The app serves at `http://localhost:8000`.

### Docker

```bash
docker build -t legacylens .
docker run -p 8000:8000 --env-file .env legacylens
```

## Performance

| Metric | Value |
|--------|-------|
| Retrieval precision (named subroutine queries) | >70% |
| Query latency (p50) | ~2.5s |
| Embedding model | text-embedding-3-small |
| Reranker | GPT-4o-mini (zero-shot) |
| Search strategy | Hybrid dense+sparse, query expansion, name-targeted boost |

## Testing

```bash
# Run retrieval precision evaluation
python tests/test_generation.py

# Check precision logs
cat data/precision_log.jsonl
```

## License

MIT
