import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone
PINECONE_INDEX_NAME = "legacylens"

# Embedding
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# LLM
LLM_MODEL = "gpt-4o-mini"

# Retrieval
TOP_K = 5
FETCH_MULTIPLIER = 3  # Over-fetch ratio to compensate for filtering

# Generation
LLM_TEMPERATURE = 0.3
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
VALID_MODES = ("ask", "explain", "docs", "patterns")
PATTERNS_TOP_K = 5

# Paths
LAPACK_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "lapack")
