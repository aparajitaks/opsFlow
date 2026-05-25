import os
from pathlib import Path

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent.parent

def _load_yaml_config(path: Path) -> dict:
    """Loads a YAML config file safely, returns empty dict on failure."""
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _deep_get(d: dict, *keys, default=None):
    """Safe nested dict key access."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, {})
    return d if d != {} else default


class Settings:
    def __init__(self):
        # ── Load .env secrets (API keys etc.) ─────────────────────────────
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

        # ── Load config.yaml for non-secret settings ───────────────────────
        cfg = _load_yaml_config(BASE_DIR / "config.yaml")

        # ── API / Model settings ───────────────────────────────────────────
        self.GROQ_API_KEY            = os.environ.get("GROQ_API_KEY", "")
        self.GROQ_MODEL_GENERATOR    = _deep_get(cfg, "rag", "generator", "groq_model",    default="llama-3.1-8b-instant")
        self.GROQ_MODEL_AUDITOR      = _deep_get(cfg, "rag", "generator", "auditor_model", default="llama-3.1-8b-instant")

        # ── Directory paths ────────────────────────────────────────────────
        self.DATA_DIR             = BASE_DIR / "data"
        self.DOCS_DIR             = BASE_DIR / "docs"
        self.LOGS_DIR             = BASE_DIR / "logs"
        self.MODEL_ARTIFACTS_DIR  = BASE_DIR / "ml" / "artifacts"
        self.DATABASE_DIR         = BASE_DIR / "rag" / "vector_store"

        for d in (self.DATA_DIR, self.DOCS_DIR, self.LOGS_DIR, self.MODEL_ARTIFACTS_DIR):
            d.mkdir(parents=True, exist_ok=True)

        # ── File paths ─────────────────────────────────────────────────────
        self.DATASET_PATH    = self.DATA_DIR / "ai4i2020.csv"
        self.LOG_FILE_PATH   = self.LOGS_DIR / "retrieved_chunks.log"
        self.APP_LOG_PATH    = self.LOGS_DIR / "app.log"
        self.CONFIG_PATH     = BASE_DIR / "config.yaml"

        # ── RAG grounding (assignment-aligned refusal) ─────────────────────
        self.KB_REFUSAL_MESSAGE = "I could not find this in the knowledge base."

        # ── Security ───────────────────────────────────────────────────────
        self.MAX_QUERY_LENGTH      = int(_deep_get(cfg, "security", "max_query_length", default=500))


        # ── ML settings (from config.yaml with env overrides) ─────────────
        ml = cfg.get("ml", {})
        self.RANDOM_STATE = int(ml.get("random_state", 42))
        self.TEST_SIZE    = float(ml.get("test_size", 0.20))
        self.N_CV_SPLITS  = int(ml.get("n_cv_splits", 5))

        feats = ml.get("features", {})
        self.CONTINUOUS_COLS  = feats.get("continuous",    [
            "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
            "Torque [Nm]", "Tool wear [min]", "temp_diff", "power", "wear_torque_ratio"
        ])
        self.CATEGORICAL_COLS = feats.get("categorical",   ["Type"])
        self.FEATURES_ORDER   = feats.get("feature_order", [
            "Type","Air temperature [K]","Process temperature [K]",
            "Rotational speed [rpm]","Torque [Nm]","Tool wear [min]",
            "temp_diff","power","wear_torque_ratio"
        ])
        self.LEAKAGE_COLS     = feats.get("leakage_cols",  ["Machine failure","TWF","HDF","PWF","OSF","RNF"])
        self.DROP_ID_COLS     = feats.get("drop_ids",      ["UDI","Product ID"])
        self.TYPE_MAP         = feats.get("type_map",      {"H": 0, "L": 1, "M": 2})
        self.TARGET_COL       = feats.get("target",        "Machine failure")

        # Root-cause thresholds
        thresh = ml.get("thresholds", {})
        self.HDF_TEMP_DELTA  = float(thresh.get("hdf_temp_delta", 8.5))
        self.PWF_POWER       = float(thresh.get("pwf_power",      9000))
        self.TWF_WEAR        = float(thresh.get("twf_wear",       200))
        self.OSF_TORQUE_HIGH = float(thresh.get("osf_torque_high",65))
        self.OSF_TORQUE_LOW  = float(thresh.get("osf_torque_low", 10))


        # ── RAG settings ───────────────────────────────────────────────────
        rag = cfg.get("rag", {})
        self.EMBEDDING_MODEL   = rag.get("embedding_model", "all-MiniLM-L6-v2")
        self.RERANKER_MODEL    = rag.get("reranker_model",  "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.COLLECTION_NAME   = rag.get("collection_name", "maintenance_kb")

        chunking = rag.get("chunking", {})
        self.CHUNK_SIZE        = int(chunking.get("chunk_size", 300))
        self.CHUNK_OVERLAP     = int(chunking.get("overlap",    50))
        self.USE_SEMANTIC      = bool(chunking.get("use_semantic", False))

        retrieval = rag.get("retrieval", {})
        self.TOP_K             = int(retrieval.get("top_k",           10))
        self.RRF_K             = int(retrieval.get("rrf_k",           60))
        self.TOP_N_RERANK      = int(retrieval.get("top_n_rerank",    3))
        self.CONFIDENCE_THRESH = float(retrieval.get("confidence_threshold", 0.30))

        cache = rag.get("cache", {})
        self.CACHE_SIM_THRESHOLD = float(cache.get("similarity_threshold", 0.98))

        gen = rag.get("generator", {})
        self.GEN_TEMPERATURE   = float(gen.get("temperature",  0.0))
        self.GEN_MAX_RETRIES   = int(gen.get("max_retries",    4))
        self.GEN_RETRY_DELAYS  = list(gen.get("retry_delays",  [3, 6, 15, 30]))

        mem = rag.get("memory", {})
        self.MEMORY_ENABLED       = bool(mem.get("enabled",         True))
        self.MEMORY_MAX_TURNS     = int(mem.get("max_history_turns", 6))

        # ── Logging settings ───────────────────────────────────────────────
        log_cfg = cfg.get("logging", {})
        self.LOG_LEVEL       = log_cfg.get("level",       os.environ.get("LOG_LEVEL", "INFO"))
        self.LOG_FORMAT      = log_cfg.get("format",      "%(asctime)s [%(levelname)s] %(name)s — %(message)s")
        self.LOG_DATEFMT     = log_cfg.get("datefmt",     "%Y-%m-%d %H:%M:%S")
        self.LOG_FILE_NAME   = log_cfg.get("file",        "logs/app.log")
        self.LOG_MAX_BYTES   = int(log_cfg.get("max_bytes",    10_485_760))
        self.LOG_BACKUP_COUNT = int(log_cfg.get("backup_count", 3))


settings = Settings()
