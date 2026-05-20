# opsFlow: Production-Grade Industrial Predictive Maintenance & Grounded RAG Intelligence System

Welcome to **opsFlow**, a production-grade AI system designed for real-time equipment telemetry diagnostics and grounded technical knowledge retrieval in manufacturing and industrial settings.

This codebase has been refactored from a monolithic demonstration script into an enterprise-ready, modular, decoupled architecture consisting of a **FastAPI REST Service**, an interactive **Streamlit Operator Console**, and a robust **Pytest QA automation suite**.

---

## 1. System Architecture

```text
                                    ┌──────────────────────┐
                                    │  Telemetry Sensor    │
                                    │  Stream / CSV Data   │
                                    └──────────┬───────────┘
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │   ML Pipeline &      │
                                    │   Model Server       │
                                    │ (RF / LR Classifier) │
                                    └──────────┬───────────┘
                                               │
                                               ▼
┌──────────────────────┐            ┌──────────────────────┐
│  Technical Manuals   │            │   FastAPI REST API   │
│ & model_summary.json │ ─────────> │   (API Gate Layer)   │
└──────────┬───────────┘            └──────────┬───────────┘
           │                                   ▲
           ▼                                   │ HTTP
┌──────────────────────┐                       │ Requests
│    Hybrid Search     │                       ▼
│ (BM25 + Chroma Vector│            ┌──────────────────────┐
│   RRF Reranker)      │            │ Streamlit Operations │
└──────────────────────┘            │      Dashboard       │
                                    └──────────────────────┘
```

The system splits into three distinct layers:
1. **Core ML Diagnostics Engine**: Performs continuous feature engineering, SMOTE resampling, hyperparameter sweeps, MLflow tracking, and SHAP explainability calculations to classify equipment stability.
2. **Hybrid RAG Knowledge Retrieval**: Ingests technical manuals and model metadata, building sliding window and semantic chunks, indexing them in ChromaDB and BM25, and fusing them via Reciprocal Rank Fusion (RRF) and Cross-Encoder re-ranking. Answer correctness is validated via a double-pass LLM faithfulness claim auditor.
3. **Operations Console**: A client interface communicating exclusively via REST queries to the backend API, featuring real-time diagnostic dials, interactive chatbot panels with visual audit verifications, and ML performance metrics.

---

## 2. Directory Layout

```text
opsFlow/
├── api/                    # FastAPI REST Gateway Layer
│   ├── routes/             # Feature-specific router paths
│   │   ├── ml.py           # Predictions, model status, background retraining
│   │   ├── query.py        # Conversational QA endpoints & cache clears
│   │   └── system.py       # Health check, reindexing, retrieval log audits
│   ├── main.py             # FastAPI entry point & lifespan model warming
│   └── schemas.py          # Pydantic request & response validation contracts
├── core/                   # Core configurations and global definitions
│   ├── config.py           # Pydantic-Settings environmental parsing
│   ├── constants.py        # ML thresholds and retrieval parameter limits
│   └── security.py         # Rate limiters, HTML sanitizers, prompt injection firewalls
├── data/                   # Raw telemetry sensor storage (e.g. ai4i2020.csv)
├── docs/                   # Target technical manuals (.txt / .json) for RAG
├── evaluation/             # RAG validation modules
│   └── faithfulness.py     # Double-pass LLM claim auditor
├── frontend/               # Streamlit Operator Dashboard
│   ├── components/         # Modular layout views
│   │   ├── chat_panel.py   # RAG conversational chatbot & audit logs
│   │   ├── metrics_panel.py# MLOps performance plots & system admin center
│   │   └── telemetry_panel.py # Simulated equipment failure diagnostic sliders
│   ├── app.py              # Main dashboard entrypoint & style injectors
│   └── state.py            # API request clients & state synchronization
├── models/                 # Machinery Diagnostics Classifiers
│   ├── pipeline.py         # Full hyperparameter sweep retraining entry point
│   ├── preprocessing.py    # SMOTE, feature scaling, telemetry calculations
│   └── training.py         # Stratified cross-validation and GridSearchCV tuning
├── retrieval/              # RAG Indexing & Retrieval Pipelines
│   ├── bm25.py             # Sparse keyword indexing (Okapi BM25)
│   ├── cache.py            # Exact & semantic vector query caching
│   ├── chunker.py          # Semantic & sliding window chunk divisions
│   ├── embedder.py         # Dense sentence embedding models
│   ├── hybrid.py           # Reciprocal Rank Fusion (RRF) combiner
│   └── reranker.py         # Cross-Encoder second-pass ranking
├── tests/                  # Pytest QA Test Suite
│   ├── conftest.py         # Reusable mock fixtures and client builders
│   ├── test_api.py         # Integration routes validations
│   ├── test_ml.py          # Preprocessing formulas and fit operations
│   ├── test_retrieval.py   # Sentence splitting, RRF ranks, and cache hits
│   └── test_security.py    # XSS blockages, firewalls, and token rate limits
├── utils/                  # Shared system helpers
│   └── logger.py           # Standard structured logging output
├── docker-compose.yml      # Orchestrates local API and dashboard images
├── Dockerfile.backend      # Multi-stage image build for FastAPI REST API
├── Dockerfile.frontend     # Multi-stage image build for Streamlit Console
├── Makefile                # Automation hooks for operations and QA
└── requirements.txt        # System library declarations
```

---

## 3. Installation & Local Setup

### System Prerequisites
Ensure Python 3.10+ is installed on your local host.

```bash
# Clone the repository
git clone <repository-url>
cd opsFlow

# Create a clean virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Configuration Setup
Copy `.env.example` to `.env` and fill in your Groq API credentials:
```bash
cp .env.example .env
```
*Provide your API key under `GROQ_API_KEY` to enable active LLM conversational QA generation and faithfulness auditing.*

---

## 4. Operational Commands (Makefile)

Use the provided `Makefile` to run system routines:

- **Install Dependencies**:
  ```bash
  make install
  ```
- **Run the Pytest Test Suite**:
  ```bash
  make test
  ```
- **Train and Serialize ML Models**:
  ```bash
  make train
  ```
- **Launch Backend REST API (Port 8000)**:
  ```bash
  make run-backend
  ```
- **Launch Frontend Streamlit Console (Port 8501)**:
  ```bash
  make run-frontend
  ```
- **Clean Bytecode & Cache Directories**:
  ```bash
  make clean
  ```

---

## 5. Deployment with Docker Compose

To build and spin up the backend and frontend in local containers:

```bash
# Build the Docker images
make docker-build

# Launch the compose services
make docker-up
```
- The backend API documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).
- The Operator Console will run at [http://localhost:8501](http://localhost:8501).

---

## 6. QA Test Suite Summary

The system is validated by a thorough automation suite located in `tests/`:

- **`test_ml.py`**: Asserts custom telemetry calculation formulas (e.g. Calculated Power, Wear-Torque Ratio) and splits.
- **`test_retrieval.py`**: Asserts sentence boundary divisions, keyword lookups, RRF ranking prioritization, and cache hit checks.
- **`test_api.py`**: Verifies mock route requests, prediction validations, and model status queries.
- **`test_security.py`**: Verifies that HTML code blocks are escaped and that prompt injection vectors are identified and blocked by the rate limiter.

Run all tests instantly:
```bash
make test
```

---
*Created and maintained for industrial-grade predictive analytics and grounded operations assistance.*
