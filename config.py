"""
Central configuration for the Conceptual Space Pipeline System
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# Pipeline A paths
A_PIPELINE_DIR = BASE_DIR / "A_concept_pipeline"
A_SCRIPTS_DIR = A_PIPELINE_DIR / "scripts"
A_OUTPUTS_DIR = A_PIPELINE_DIR / "outputs"
A_DATA_DIR = A_PIPELINE_DIR / "data"

# Pipeline B paths
B_PIPELINE_DIR = BASE_DIR / "B_retrieval_pipeline"
B_SCRIPTS_DIR = B_PIPELINE_DIR / "scripts"
B_OUTPUTS_DIR = B_PIPELINE_DIR / "outputs"
B_DATA_DIR = B_PIPELINE_DIR / "data"

# Shared paths
SHARED_DIR = BASE_DIR / "shared"
UTILS_DIR = SHARED_DIR / "utils"
EMBEDDINGS_DIR = SHARED_DIR / "embeddings"

# Data files
DEFAULT_DATA_FILE = "test_5_records.parquet"
DEFAULT_QUESTION_INDEX = 0

# Model configurations
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_TEMPERATURE = 0.7
OPENAI_MAX_TOKENS = 500

# Processing parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MIN_CHUNK_LENGTH = 100
MAX_CHUNK_LENGTH = 1000

# Concept extraction parameters
MIN_KEYWORD_LENGTH = 2
MAX_KEYWORDS_PER_DOC = 50
MIN_CONCEPT_FREQUENCY = 2
CONCEPT_SIMILARITY_THRESHOLD = 0.7

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Pipeline execution settings
TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
BATCH_SIZE = 100

def ensure_directories():
    """Create all necessary directories if they don't exist"""
    directories = [
        DATA_DIR, OUTPUTS_DIR, LOGS_DIR,
        A_OUTPUTS_DIR, A_DATA_DIR,
        B_OUTPUTS_DIR, B_DATA_DIR,
        UTILS_DIR, EMBEDDINGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Create directories on import
ensure_directories()