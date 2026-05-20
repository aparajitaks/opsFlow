import os
from pathlib import Path

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    def __init__(self):
        # Load environment variables from .env file at the root level if present
        env_path = BASE_DIR / ".env"
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'").strip('"')
                        if k:
                            os.environ[k] = v

        # Configurations
        self.GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
        self.GROQ_MODEL_GENERATOR = os.environ.get("GROQ_MODEL_GENERATOR", "llama-3.1-8b-instant")
        self.GROQ_MODEL_AUDITOR = os.environ.get("GROQ_MODEL_AUDITOR", "llama-3.1-8b-instant")
        
        # Directories
        self.DATA_DIR = BASE_DIR / "data"
        self.DOCS_DIR = BASE_DIR / "docs"
        self.LOGS_DIR = BASE_DIR / "logs"
        self.MODEL_ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"
        self.DATABASE_DIR = self.DATA_DIR / "chroma_db"
        
        # Ensure directories exist
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Paths
        self.DATASET_PATH = self.DATA_DIR / "ai4i2020.csv"
        self.LOG_FILE_PATH = self.LOGS_DIR / "retrieved_chunks.log"
        self.APP_LOG_PATH = self.LOGS_DIR / "app.log"
        
        # Security & API limits
        self.RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
        self.MAX_QUERY_LENGTH = int(os.environ.get("MAX_QUERY_LENGTH", "500"))
        
        # MLflow config
        self.MLFLOW_TRACKING_URI = f"file://{os.path.abspath(self.LOGS_DIR / 'mlflow')}"

settings = Settings()
