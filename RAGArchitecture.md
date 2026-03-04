# RAG Architecture Documentation — LegacyLens

## Vector DB Selection

**Choice:** Pinecone (managed cloud, serverless, free tier)

Pinecone was selected for three reasons: persistent cloud storage, native hybrid search, and zero infrastructure overhead.

The deployment target (Railway) uses an ephemeral filesystem — any embedded data stored locally is lost on every redeploy. Pinecone eliminates this entirely since vectors live in Pinecone's cloud independently. ChromaDB was initially considered for its simplicity but rejected for exactly this persistence issue. Qdrant (self-hosted) would have added operational complexity with no benefit at this scale. pgvector was overkill — we don't need SQL joins on vector data.

Pinecone's native hybrid search (dense + sparse vectors in a single query) was critical. LAPACK users frequently search by exact subroutine name (e.g., "DGESV"), and pure semantic search can miss these. Hybrid search combines OpenAI embeddings for semantic understanding with BM25 sparse vectors for exact keyword matching, giving us both.

**Current usage:** 5,520 vectors, 1 index (dotproduct metric, 1536 dimensions). Well within the free tier's 100K vector limit.

---

## Embedding Strategy

**Choice:** OpenAI `text-embedding-3-small` (1536 dimensions) + BM25 sparse vectors

This model was chosen for its cost ($0.02/1M tokens), quality, and broad compatibility. The entire LAPACK codebase (1.5M lines, 3,586 files) was embedded for ~$0.15.

Voyage Code 2 was considered as a code-optimized alternative but rejected — it requires a separate API provider, and the marginal quality gain was not worth the added complexity. In practice, `text-embedding-3-small` performs well because our embedding text is primarily cleaned English (extracted from Fortran comment headers describing each subroutine's purpose, parameters, and algorithm), not raw code.

**Embedding text preparation:** Rather than embedding raw Fortran source, we clean each chunk's comment header — stripping `*>` comment markers, Doxygen tags, HTML, and separator lines — to produce clean English prose like: `"LAPACK subroutine DGESV. Computes the solution to a real system of linear equations A * X = B. Calls: DGETRF, DGETRS, XERBLA."` This gives the embedding model natural language to work with, improving retrieval quality significantly over embedding raw code.

---

## Chunking Approach

**Strategy:** One chunk per subroutine/function, with oversized splitting as fallback.

LAPACK's structure makes this natural — each file typically contains one subroutine with a detailed comment header. We keep the header and code together so retrieval returns complete context.

**Parsing:** Regex-based detection of `SUBROUTINE`, `FUNCTION`, and `RECURSIVE SUBROUTINE` boundaries. Dependencies extracted by parsing `CALL` statements. Files without subroutine markers fall back to whole-file chunks.

**Oversized handling:** If a chunk exceeds 6,000 tokens (buffer below the 8,191 embedding limit), it's split into ~4,000-token pieces with 500-token overlap. Parts are named `DGESV_PART1`, `DGESV_PART2`, etc. In practice, fewer than 5% of subroutines require splitting.

**Metadata per chunk:** Subroutine name, file path, start/end line numbers, dependencies (from CALL parsing), and the full text (up to 40KB for Pinecone's metadata limit).

---

## Retrieval Pipeline

```
User Query
  |
  ├─> Query Expansion (GPT-4o-mini generates 2 variants)
  ├─> Subroutine Name Detection (regex: [SDCZ][A-Z]{3,})
  |     └─> Name-targeted Pinecone queries (metadata filter)
  ├─> Embed original query (dense + BM25 sparse)
  └─> Embed expanded queries (dense + BM25 sparse)
        |
        └─> All searches run in PARALLEL (ThreadPoolExecutor, 6 workers)
              |
              ├─> Original: hybrid Pinecone search (top 20)
              ├─> Expanded: hybrid Pinecone search (top 20 each)
              └─> Name-targeted: metadata-filtered search per name
                    |
                    └─> Deduplicate → Boost name matches (+1.0) → Filter test files → Top 5
                          |
                          └─> Build context ([Source 1]...[Source 5], 1000 tokens each)
                                |
                                └─> Stream answer via GPT-4o-mini (mode-specific prompt)
```

**Query expansion** generates 2 technical LAPACK-specific query variations, improving recall for vague queries. All searches (original, expanded, name-targeted) execute in parallel, keeping latency under control.

**Name detection** uses regex (`[SDCZ]?[A-Z]{3,}` with post-filtering to require a valid type prefix) to extract likely LAPACK subroutine names and runs metadata-filtered Pinecone queries for each, with a +1.0 score boost for exact matches. This ensures "Explain DGESV" always returns DGESV regardless of semantic similarity.

**Re-ranking:** LLM-based reranking was implemented and then removed. It added 2-3 seconds of latency for marginal precision improvement. Hybrid search + name boost achieves >70% precision without it.

**Context assembly:** Top 5 chunks formatted as numbered sources with file path, line range, subroutine name, and dependencies. Each truncated to 1,000 tokens to fit 5 sources comfortably in GPT-4o-mini's context window.

---

## Failure Modes

**Vague queries without subroutine names:** Queries like "how do I do math" return loosely related results. Query expansion helps but can't fully compensate when the user intent doesn't map to specific LAPACK functionality. Precision drops to 40-60% on these queries.

**Sparse matrix queries:** LAPACK is a dense linear algebra library. Questions about sparse matrices return irrelevant dense-matrix routines. The system has no way to know LAPACK doesn't cover sparse operations — it returns the closest matches, which are misleading. Precision: 0%.

**Cross-subroutine questions:** "What's the difference between DGESV and DGESVX?" requires retrieving both routines and comparing them. The system retrieves both but the LLM comparison quality depends on whether both appear in the top 5 results.

**Very long subroutines:** Chunked subroutines (the `_PART1`, `_PART2` splits) can lose context when only one part is retrieved. The overlap helps but doesn't fully solve this for subroutines split into 3+ parts.

**BM25 cold start:** The BM25 model is fitted on the full corpus during ingestion. If the corpus changes significantly, the model needs refitting. Currently manual.

---

## Performance Results

### Latency
| Metric | Target | Actual |
|--------|--------|--------|
| Time to first token | <3s | ~2-3s |
| Full answer stream | — | ~5-8s |
| Follow-up questions available | — | ~0-1s after answer (parallelized) |

### Retrieval Precision
| Query Type | Precision (relevant in top-5) |
|------------|-------------------------------|
| Named subroutine ("Explain DGESV") | 80-100% |
| Algorithm family ("QR factorization") | 80-100% |
| Broad conceptual ("eigenvalue algorithms") | 60-80% |
| Vague natural language ("solve linear system") | 60-80% |
| Out-of-scope ("sparse matrix") | 0% |

**Evaluation suite:** 21 test cases across 3 match modes (exact, family, any) in `tests/eval_precision.py`. Target: >70% pass rate.

### Example Results

**Query:** "How does LAPACK solve a system of linear equations?" (Ask mode)
- Retrieved: DGESV, DGETRF, DGETRS, DGESVX, DGELS — all relevant
- Precision: 80%

**Query:** "Generate docs for DPOTRF" (Docs mode)
- Retrieved: DPOTRF as top result
- Precision: 100%

**Query:** "Walk me through the DGESVD singular value decomposition" (Explain mode)
- Retrieved: DGESVD and related routines
- Precision: 80%

### Codebase Coverage
| Metric | Value |
|--------|-------|
| Files indexed | 3,586 / 3,586 (100%) |
| Lines of code | 1,553,667 |
| Vectors stored | 5,520 |
| Ingestion time | ~4 minutes |
