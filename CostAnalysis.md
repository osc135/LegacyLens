# AI Cost Analysis — LegacyLens

## Development & Testing Costs

### Claude Code (Development Agent)
| Metric | Value |
|--------|-------|
| API list-price equivalent | $263.29 (not billed — Max subscription) |
| Sessions | 22 |
| Total prompts | 1,903 |
| Output tokens | 415,653 |
| Cache read share | 96.9% |
| Active time | 3h 40m |

### OpenAI API (Embeddings + LLM)
| Component | Model | Cost |
|-----------|-------|------|
| Embedding ingestion | text-embedding-3-small | ~$0.15 |
| Query expansion + answer generation | gpt-4o-mini | ~$2.68 |
| Precision scoring + citation verification | gpt-4o-mini | ~$0.50 |
| **Total OpenAI spend** | | **$3.33** |

**Embedding breakdown:**
- 5,520 chunks embedded (1,553,667 LOC across 3,586 Fortran files)
- Model: `text-embedding-3-small` at $0.02 / 1M tokens
- Estimated ~7.5M tokens embedded = ~$0.15

**LLM breakdown:**
- Model: `gpt-4o-mini` at $0.15 / 1M input tokens, $0.60 / 1M output tokens
- Used for: answer generation, query expansion, follow-up generation, precision scoring, citation verification
- ~50 test queries during development + 5 warm-cache queries per deploy

### Vector Database (Pinecone)
| Item | Cost |
|------|------|
| Plan | Free (Starter) |
| Vectors stored | 5,520 |
| Index | 1 (dotproduct, 1536 dimensions) |
| Monthly cost | $0.00 |

### Hosting (Railway)
| Item | Cost |
|------|------|
| Plan | Free tier |
| Monthly cost | $0.00 |

### Observability (Langfuse)
| Item | Cost |
|------|------|
| Plan | Free (Hobby) |
| Monthly cost | $0.00 |

### Total Development Spend
| Category | Cost |
|----------|------|
| Claude Code (Max subscription) | $0 actual (covered by subscription) |
| OpenAI API | $3.33 |
| Pinecone | $0.00 |
| Railway | $0.00 |
| Langfuse | $0.00 |
| **Total** | **$3.33** |

---

## Production Cost Projections

### Assumptions
- **Queries per user per day:** 5
- **Average tokens per query:**
  - Embedding: ~50 tokens (query text)
  - Query expansion: ~300 input + ~100 output tokens
  - Answer generation: ~4,000 input + ~800 output tokens (5 source chunks at ~1,000 tokens each + system prompt)
  - Follow-ups: ~200 input + ~60 output tokens
  - Precision scoring: ~4,000 input + ~200 output tokens
  - Citation verification: ~4,000 input + ~200 output tokens
- **Total per query:** ~12,500 input tokens, ~1,360 output tokens
- **Days per month:** 30

### Per-Query Cost (gpt-4o-mini)
| Component | Input tokens | Output tokens | Cost |
|-----------|-------------|---------------|------|
| Query embedding | 50 | — | $0.000001 |
| Query expansion | 300 | 100 | $0.000105 |
| Answer generation | 4,000 | 800 | $0.001080 |
| Follow-ups | 200 | 60 | $0.000066 |
| Precision scoring | 4,000 | 200 | $0.000720 |
| Citation verification | 4,000 | 200 | $0.000720 |
| **Total per query** | **12,550** | **1,360** | **$0.002692** |

### Monthly Projections

| Scale | Users | Queries/month | OpenAI API | Pinecone | Railway | **Total** |
|-------|-------|---------------|-----------|----------|---------|-----------|
| Small | 100 | 15,000 | $40 | $0 (free) | $0 (free) | **~$40/mo** |
| Medium | 1,000 | 150,000 | $404 | $70 (Standard) | $5 | **~$479/mo** |
| Large | 10,000 | 1,500,000 | $4,038 | $70 (Standard) | $20 | **~$4,128/mo** |
| Enterprise | 100,000 | 15,000,000 | $40,380 | $230 (Enterprise) | $100+ | **~$40,710/mo** |

### Scaling Notes
- **Pinecone:** Free tier supports up to 100K vectors (we use 5,520). At 1,000+ users, Standard plan ($70/mo) adds replicas for throughput. Enterprise tier at 100K users for higher QPS.
- **Railway:** Free tier handles low traffic. At scale, $5-20/mo for dedicated compute. At 100K users, horizontal scaling with multiple instances needed.
- **OpenAI API:** Dominates cost at every scale. Primary optimization lever is reducing post-stream LLM calls (precision scoring and citation verification could be made optional or sampled).
- **Embedding costs are negligible** — ingestion is one-time ($0.15), and query embeddings cost $0.001/1K queries.

### Cost Optimization Strategies
1. **Disable precision scoring in production** — saves ~27% of per-query LLM cost (only needed for eval)
2. **Sample citation verification** — run on 10% of queries instead of 100%, saves ~24% of per-query cost
3. **Cache frequent queries** — already implemented for 5 sample queries; expanding cache hit rate to 30% would cut costs proportionally
4. **Batch follow-up generation** — could be skipped for repeat/similar queries
5. **Switch to GPT-4o-mini-2025** — future model releases typically reduce cost at same quality
6. **With optimizations 1-3 applied:** projected costs drop ~40%, bringing 10K users to ~$2,500/mo
