import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Ingestion & Retrieval Configuration
DEFAULT_DATA_DIR = "data"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
DEFAULT_K = 5
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

# LLM Generation Configuration
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
GENERATION_TEMPERATURE = 0.3
GENERATION_MAX_TOKENS = 1024

# Evaluation & Healing Configuration
MIN_PASS_SCORE = 0.75
MIN_RETRIEVAL_SIMILARITY = 0.60
MAX_RETRIES = 3
