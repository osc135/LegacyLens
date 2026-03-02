# LegacyLens — Pre-Search Document

**Building RAG Systems for Legacy Enterprise Codebases** | G4 Week 3 | Pre-Search Checklist Output

---

## Technology Stack Summary

| Layer | Choice |
|-------|--------|
| Target Codebase | LAPACK (Fortran) |
| Vector Database | Pinecone (managed cloud, free tier) |
| Embedding Model | OpenAI text-embedding-3-small (1536 dim) |
| LLM (Answer Generation) | GPT-4o-mini |
| Framework | LangChain (Python) |
| Backend | Python / FastAPI |
| Frontend | Web UI with streaming responses |
| Deployment | Railway (app) + Pinecone (vectors) |

---

## Phase 1: Define Your Constraints

### 1. Scale & Load Profile

**Decision:** LAPACK — hundreds of thousands of lines of Fortran across 1,000+ files

Well over the 10,000 LOC / 50+ file minimum. LAPACK is a pure Fortran library with clean structure — each subroutine is typically its own file with detailed comment headers. Query volume is low (demo/testing scale, dozens of queries). Batch ingestion: ingest once, query many times. Target latency: under 3 seconds per query.

### 2. Budget & Cost Ceiling

| Resource | Estimated Cost |
|----------|---------------|
| Embeddings (text-embedding-3-small @ $0.02/1M tokens) | $0.10–$0.20 |
| LLM — GPT-4o-mini (@ $0.15/1M input tokens) | ~$0.50 |
| Pinecone | Free tier (100K vectors) |
| Railway hosting | Free tier |
| **Total estimated dev spend** | **Under $2** |

GPT-4o-mini chosen over GPT-4o for 93% cost savings. The answer generation task (summarizing Fortran code snippets) does not require frontier-model reasoning, making 4o-mini the right cost/performance tradeoff.

**Trading money for time:** Spending on managed services (Pinecone, OpenAI APIs) and frameworks (LangChain) everywhere to maximize shipping speed on a 24-hour MVP. No self-hosted vector databases, no local embedding models — every infrastructure choice prioritizes fast setup over cost optimization.

### 3. Time to Ship

**MVP Timeline:** 24 hours

**Must-have (MVP):** Ingestion pipeline, embedding generation, vector storage, semantic search, natural language query interface, code snippet retrieval with file/line references, basic answer generation, deployed publicly.

**Nice-to-have (post-MVP):** LLM-based reranking, advanced code understanding features, polished UI, evaluation metrics.

### 4. Data Sensitivity

**Decision:** No restrictions — LAPACK is open source (BSD license)

Code can be freely sent to external APIs (OpenAI, Pinecone). No data residency requirements. No proprietary concerns.

### 5. Team & Skill Constraints

Python-based stack chosen for maximum comfort. Familiar with the LangChain ecosystem through LangSmith (evals, tracing) — chose LangChain for retrieval so that tracing, evaluation, and the RAG pipeline all live in one ecosystem, simplifying debugging and observability. No prior Fortran expertise required — LAPACK's detailed comment headers make the code self-describing, and GPT-4o-mini handles code explanation. New to Pinecone but chose it for zero-config managed hosting and as a learning opportunity with a production-grade tool.

---

## Phase 2: Architecture Discovery

### 6. Vector Database Selection — Pinecone

**Decision:** Pinecone (managed cloud, free tier)

**Why Pinecone:** Managed hosting eliminates persistence and infrastructure concerns. Free tier supports up to 100K vectors, far more than needed for LAPACK. Supports hybrid search (vector + keyword) natively, which is important for exact subroutine name matching. No disk persistence issues on Railway since vectors are stored in Pinecone's cloud.

**Tradeoffs considered:** ChromaDB (simple, embedded, but Railway's ephemeral filesystem would wipe embeddings on every redeploy). Qdrant (powerful filtering, but self-hosting adds complexity). pgvector (familiar SQL interface, but overkill for this use case).

**Why not ChromaDB:** Originally considered for simplicity, but the Railway deployment problem (ephemeral disk = lost embeddings on redeploy) made it impractical without extra workarounds. Pinecone solves this cleanly.

### 7. Embedding Strategy — text-embedding-3-small

**Decision:** OpenAI text-embedding-3-small (1536 dimensions)

**Why this model:** Low cost ($0.02/1M tokens), good quality, widely supported, and 1536 dimensions is a solid balance between expressiveness and storage efficiency.

**Tradeoff:** Voyage Code 2 is optimized for code and might yield slightly better retrieval precision, but requires a separate API key and provider. The marginal improvement is not worth the added complexity for MVP. Can benchmark against Voyage Code 2 post-MVP if retrieval quality is insufficient.

**Batch processing:** Embed all chunks during ingestion in batches. Pinecone stores embeddings persistently, so ingestion is a one-time cost.

### 8. Chunking Approach — One Chunk Per Subroutine

**Decision:** One chunk per subroutine/function, including comment header + code body

**Why this approach:** LAPACK has a clean structure where each subroutine is typically its own file with a detailed comment header describing purpose, parameters, and algorithm. Keeping the header and code together preserves full context for both retrieval and answer generation.

**Metadata per chunk:** File path, subroutine name, line numbers, parameter list (extracted from header).

**Fallback for oversized subroutines:** text-embedding-3-small has an 8,191 token limit. If a subroutine exceeds ~6,000 tokens (leaving buffer), split it into overlapping chunks of ~4,000 tokens with 500 token overlap. Log which subroutines get split to track edge cases. Expected to affect fewer than 5% of subroutines.

**Tradeoffs:** Hierarchical chunking (file → subroutine → comment block) would allow more granular retrieval but adds significant parsing complexity. Can add hierarchical chunking as a post-MVP enhancement.

### 9. Retrieval Pipeline

**Decision:** Hybrid search (vector + keyword) → query expansion → LLM-based reranking (post-MVP) → context assembly → streaming answer generation

**Hybrid search:** Pinecone's native hybrid search combines vector similarity with keyword matching. This is critical for a codebase where users often search by exact subroutine names (e.g., "DGESV"). Pure vector search might miss exact name matches; hybrid search ensures they surface.

**Top-k:** Start with k=5 (aligns with the 70% precision in top-5 target). For reranking, retrieve top-10 initially.

**Query expansion:** Always active. Use GPT-4o-mini to generate 3–4 specific query variations from every input, search all variations in parallel, deduplicate results. Example: "show me the important stuff" → "main entry point subroutines," "most frequently called functions," "core matrix operation routines." Running searches in parallel keeps latency comparable to a single search.

**Re-ranking (post-MVP):** Send top-10 chunks to GPT-4o-mini, have it select and order the 5 most relevant. Cheaper and simpler than deploying a cross-encoder model.

**Context window:** With one-subroutine-per-chunk, most chunks are a few hundred lines max. Five chunks fit comfortably in 4o-mini's context window.

### 10. Answer Generation

**Decision:** GPT-4o-mini with streaming, structured prompt template, citation formatting

**Prompt template:** System prompt defining role ("You are an expert Fortran developer specializing in the LAPACK linear algebra library"), instructions to cite specific subroutine names and line numbers, and formatting guidance. User message contains the query followed by retrieved chunks labeled with `[filename.f:lines X-Y]`.

**Citation format:** Each referenced code snippet labeled with `[filename.f:lines X-Y]` so users can locate source code.

**Streaming:** Yes — streaming responses for better UX in the web interface.

### 11. Framework Selection — LangChain

**Decision:** LangChain (Python)

**Why LangChain:** Already familiar with the LangChain ecosystem through LangSmith (evals, tracing). Choosing LangChain for the RAG pipeline keeps everything in one ecosystem — tracing, evaluation, and retrieval all in one place, making debugging significantly easier. Built-in Pinecone integration, OpenAI embeddings wrapper, and retrieval QA chain provide the fastest path to a working pipeline.

**Tradeoffs:** LlamaIndex is more document-focused but has a steeper learning curve. Custom pipeline offers full control but takes longer to build. Haystack is production-grade but overkill for this timeline.

---

## Phase 3: Post-Stack Refinement

### 12. Failure Mode Analysis

**Decision:** Return results with confidence/relevance scores; let the user judge quality

**No relevant results:** Display retrieved chunks with their similarity scores. Low scores signal to the user that results may not be relevant. The LLM answer will also caveat when context seems insufficient.

**Ambiguous queries:** Query expansion generates multiple search variations to improve coverage.

**Rate limiting:** Simple in-memory rate limit of 10 queries per minute per user to prevent API cost overruns.

**Error handling:** Graceful degradation — if the embedding API is down, return a clear error message. If the LLM is down, still return raw retrieval results without a generated answer. If Pinecone is unreachable, display a clear service unavailable message.

### 13. Evaluation Strategy

**Decision:** Formal test set (10–15 queries) + manual exploration

**Test set queries** (adapted from project spec for LAPACK):

| # | Query | Expected Behavior |
|---|-------|-------------------|
| 1 | "Where is the main entry point for solving linear systems?" | Returns DGESV or similar driver routines |
| 2 | "What subroutines modify the pivot array?" | Returns routines with IPIV parameter |
| 3 | "Explain what DGESV does" | Returns DGESV with clear explanation |
| 4 | "Find all matrix factorization routines" | Returns xGETRF, xPOTRF, xSYTRF, etc. |
| 5 | "What are the dependencies of DGETRF?" | Returns DGETRF and routines it calls |
| 6 | "Show me error handling patterns in LAPACK" | Returns INFO parameter checking patterns |

Additional queries to be added during manual exploration. Precision measured as relevant chunks in top-5 / 5. Target: >70%.

**User feedback:** Thumbs up/down + optional text feedback on each answer. Stored in JSON log for analysis and architecture doc.

### 14. Performance Optimization

**Embedding cache:** Pinecone stores embeddings persistently in the cloud — ingestion is a one-time operation. No re-embedding unless the codebase changes.

**Query preprocessing:** Strip filler words and normalize whitespace before embedding queries.

**Index optimization:** Pinecone handles indexing internally. Default settings are sufficient at this scale.

### 15. Observability

**Logging:** JSON-structured logs for every query, capturing: query text, retrieved chunk IDs and similarity scores, generated answer, latency (end-to-end, retrieval, generation), and user feedback. LangSmith integration for tracing the full retrieval → generation pipeline, enabling easy debugging of retrieval quality issues.

**Metrics to track:** Query latency (p50, p95), similarity score distribution, queries per day, feedback scores (thumbs up/down ratio).

**Alerting:** Not needed at demo scale. LangSmith traces + JSON logs are sufficient for debugging retrieval issues and documenting failure modes.

### 16. Deployment & DevOps

**Decision:** Railway (application) + Pinecone (vector storage), environment variables for secrets, manual re-ingestion

**Deployment:** Railway for the FastAPI application — push repo, set environment variables, get public URL. Pinecone for vector storage — embeddings persist in Pinecone's cloud independently of Railway deploys.

**Secrets:** `OPENAI_API_KEY` and `PINECONE_API_KEY` as environment variables on Railway, `.env` file locally (gitignored). Never committed to version control.

**CI/CD for index updates:** Manual re-ingestion via script. Run `python ingest.py` to rebuild the Pinecone index when codebase changes. Automated CI/CD is a post-MVP enhancement.

---

## Code Understanding Features (4 Selected)

- **Code Explanation** — Low effort. Leverages existing answer generation pipeline. Feed subroutine to 4o-mini, ask for plain English explanation.
- **Documentation Generation** — Low effort. Structured prompt to 4o-mini for formatted docs (parameters, return values, algorithm description).
- **Pattern Detection** — Medium effort. "Query by code" using vector similarity — embed a code snippet, find the most similar chunks across the codebase.
- **Dependency Mapping** — Medium effort. Parse CALL statements in Fortran subroutines, build and display a call graph showing what calls what.

---

## Build Priority Order

| Priority | Task | Time Estimate |
|----------|------|---------------|
| 1 | Download LAPACK, verify file count and LOC | 30 min |
| 2 | Set up Pinecone index and API connection | 30 min |
| 3 | Build ingestion pipeline (file scan → chunking → embedding → Pinecone) | 3–4 hours |
| 4 | Build retrieval + answer generation pipeline | 2–3 hours |
| 5 | Web UI with streaming responses | 2–3 hours |
| 6 | Deploy to Railway | 1 hour |
| 7 | Test with evaluation set, document results | 1–2 hours |
| 8 | Implement 4 code understanding features | 4–6 hours |
| 9 | Add reranking, query expansion, feedback UI | 2–3 hours |
| 10 | Polish, architecture doc, cost analysis, demo video | 2–3 hours |
