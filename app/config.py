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

# Paths
LAPACK_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "lapack")
