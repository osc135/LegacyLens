import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Langfuse observability
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Pinecone
PINECONE_INDEX_NAME = "legacylens"

# Embedding
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# LLM
LLM_MODEL = "gpt-4o-mini"
FOLLOWUPS_MODEL = "gpt-4o-mini"

# Retrieval
TOP_K = 5
FETCH_MULTIPLIER = 4  # Over-fetch ratio to compensate for filtering
RERANK_CANDIDATES = 20  # Number of candidates to send to LLM reranker
NAME_MATCH_BOOST = 1.0  # Score boost for exact subroutine name matches

# Generation
LLM_TEMPERATURE = 0.3
MAX_CONTEXT_TOKENS_PER_CHUNK = 1000
QUERY_EXPANSION_TEMPERATURE = 0.5
MAX_FOLLOWUP_ANSWER_CHARS = 1000

# Ingestion
MAX_TOKENS = 6000       # Leave buffer below the 8191 embedding limit
CHUNK_TOKENS = 4000     # Target chunk size when splitting
OVERLAP_TOKENS = 500    # Sliding window overlap
EMBEDDING_BATCH_SIZE = 100
UPSERT_BATCH_SIZE = 100
PINECONE_METADATA_CHAR_LIMIT = 40000
MIN_EMBEDDING_TOKENS = 50

# Query validation
MAX_QUERY_LENGTH = 2000
VALID_MODES = ("ask", "explain", "docs", "patterns", "deps")
DEPS_MAX_DEPTH = 2
DEPS_MAX_NODES = 30
PATTERNS_TOP_K = 5

# Feedback
FEEDBACK_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "feedback")
VERIFICATION_LOG = os.path.join(os.path.dirname(__file__), "..", "data", "verification_log.jsonl")
PRECISION_LOG = os.path.join(os.path.dirname(__file__), "..", "data", "precision_log.jsonl")
MAX_FEEDBACK_COMMENT = 500

# Paths
LAPACK_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "lapack")

# Sample queries to pre-cache on startup for instant responses
SAMPLE_QUERIES = [
    ("How does LAPACK solve a system of linear equations?", "ask"),
    ("What eigenvalue algorithms does LAPACK provide?", "ask"),
    ("Walk me through the DGESVD singular value decomposition", "explain"),
    ("Generate docs for DPOTRF", "docs"),
    ("Map DGELS dependencies", "deps"),
]
