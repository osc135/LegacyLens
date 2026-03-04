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
- **LLM:** GPT-4o-mini (generation, query expansion)
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
    └─→ Candidate Pool (deduplicated, score-boosted, top 5)
            │
            ├─→ Streaming Answer Generation (GPT-4o-mini)
            │
            └─→ Citation Verification + Precision Scoring
```

All retrieval phases run in parallel using a shared thread pool for minimal latency. The 5 sample queries shown on the landing page are pre-cached at startup for instant responses.

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

| Metric | Target | Actual |
|--------|--------|--------|
| Query latency (time to first token) | <3s | ~2–3s |
| Retrieval precision (top-5) | >70% relevant | >70% (LLM-as-judge, 0–3 scale) |
| Codebase coverage | 100% files indexed | 3,586 / 3,586 Fortran files (100%) |
| Ingestion scale | 10,000+ LOC | 1,553,667 LOC across 3,586 files → 5,520 vectors |

### Retrieval pipeline

| Component | Detail |
|-----------|--------|
| Embedding model | text-embedding-3-small (1536-dim) |
| Search strategy | Hybrid dense + BM25 sparse, query expansion, name-targeted boost |
| Scoring | Pinecone hybrid score with +1.0 boost for exact subroutine name matches |

## Testing

```bash
# Run retrieval precision evaluation
python tests/test_generation.py

# Check precision logs
cat data/precision_log.jsonl
```

## License

MIT
