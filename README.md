# 🔧 opsFlow: Standalone RAG Intelligence & Predictive Maintenance Diagnostic System

Welcome to **opsFlow**, an enterprise-grade, fully self-contained AI system designed for real-time equipment telemetry diagnostics and grounded technical knowledge retrieval in manufacturing and industrial settings.

This application has been engineered to run as a **single, unified, standalone Streamlit application** fully optimized for CPU-only nodes and seamless, robust **Streamlit Cloud deployments** with zero external service or backend dependencies.

---

## 🚀 Key Architectural Innovations

### 1. Direct In-Process Execution
No complex microservice networking or broken localhost APIs. The frontend and backend layers are fused directly inside a single Python process space. The interactive dashboard invokes the RAG pipeline and machine failure classifier via optimized, thread-safe Python interfaces.

### 2. High-Performance Hybrid Retrieval & Reranking
- **Keyword Search (Sparse)**: Exact code search and manual indexing via `rank-bm25` (Okapi BM25).
- **Dense Vector Search**: Semantic context querying via `ChromaDB` and the HuggingFace `all-MiniLM-L6-v2` dense embedder.
- **Fusion & Reranking**: Reciprocal Rank Fusion (RRF) prioritizes multi-channel hits, followed by a second-pass re-ranking using an `ms-marco-MiniLM-L-6-v2` Cross-Encoder.
- **Double-Pass Audit**: Responses are validated via a double-pass LLM claim auditor that flags factual hallucinations, rendering instant visual metrics of answer confidence.

### 3. Machine Learning Equipment Telemetry Diagnostics
- **Real-time Feature Engineering**: Calculates live equipment indices like delta temperature (ΔT), rotational power, and torque-wear ratios directly from input sliders.
- **Inference & Explainability**: Predicts machine breakdown states and likelihoods using tuned, serialized Random Forest and Logistic Regression models.
- **Interactive MLOps Tab**: Allows operators to trigger background retraining sweeps and view evaluation curves (ROC-AUC, Precision-Recall) and local explainability (SHAP).

### 4. Recruiter-Ready UI Aesthetics
- **Typography**: Custom Google Font integration using modern 'Outfit' sans-serif.
- **Design System**: Harmonious HSL colors, premium glassmorphism, and responsive CSS containers.
- **Smooth Micro-Animations**: Native Streamlit elements enhanced with sleek dark-mode custom-themed cards.

---

## 🛠️ Unified System Directory Structure

```text
opsFlow/
├── .streamlit/             # Streamlit configuration settings & styling
│   └── config.toml         # Custom fonts, headless options & dark mode
├── core/                   # Core configurations and global definitions
│   ├── config.py           # Environmental settings parser
│   ├── constants.py        # ML thresholds and RAG parameter bounds
│   └── security.py         # Rate limiters, HTML sanitizers & prompt-injection firewalls
├── data/                   # Raw telemetry sensor storage (ai4i2020.csv)
├── docs/                   # Technical manuals & equipment specs for RAG ingestion
├── evaluation/             # Grounding validation modules
│   └── faithfulness.py     # Double-pass LLM claim auditor (factual checker)
├── frontend/               # Streamlit Modular Interface
│   ├── components/         # Modular layout views
│   │   ├── chat_panel.py   # RAG conversational chatbot & audit cards
│   │   └── telemetry_panel.py # Simulated equipment failure diagnostic sliders
│   ├── app.py              # Main dashboard entrypoint & style injectors
│   └── state.py            # Direct-in-process state controllers & log tailers
├── models/                 # Machinery Diagnostics Classifiers
│   ├── pipeline.py         # Hyperparameter CV sweep and tuning execution
│   ├── preprocessing.py    # Feature engineering formulas & SMOTE balance
│   └── training.py         # Stratified cross-validation model tuning
├── retrieval/              # RAG Search & Cache Pipelines
│   ├── bm25.py             # Sparse keyword indexing (Okapi BM25)
│   ├── cache.py            # Vector-similarity query caching (Exact & Semantic)
│   ├── chunker.py          # Sliding-window & semantic sentence chunking
│   ├── embedder.py         # Dense sentence embedding pipeline
│   ├── hybrid.py           # Reciprocal Rank Fusion (RRF) combiner
│   └── reranker.py         # Cross-Encoder second-pass ranking
├── tests/                  # Pytest QA Test Suite
│   ├── conftest.py         # Reusable mock fixtures
│   ├── test_ml.py          # Telemetry calculations and fit operations
│   ├── test_retrieval.py   # Sentence splitting, RRF ranks, and cache hits
│   └── test_security.py    # XSS blockages, firewalls, and token rate limits
├── utils/                  # Structured logging helpers
│   └── logger.py           # Performance and RAG audit log writer
├── Dockerfile              # Unified single-stage image for production
├── Makefile                # Single-process operational command hooks
├── packages.txt            # System-level graphics packages for Streamlit Cloud
├── requirements.txt        # CPU-only optimized requirements (no CUDA/Triton)
├── run_all.py              # One-command developer pipeline initializer
└── streamlit_app.py        # Root entrypoint compatibility redirect layer
```

---

## ⚡ Setup & Local Operations

### 1. Prerequisites
Ensure Python 3.10 to 3.12 is installed on your local machine.

```bash
# Clone the repository
git clone <repository-url>
cd opsFlow

# Create a clean virtual environment
python3 -m venv venv
source venv/bin/activate

# Install CPU-optimized requirements
make install
```

### 2. Configure Environment
Copy `.env.example` to `.env` and fill in your Groq API credentials:
```bash
cp .env.example .env
```
*Provide your key under `GROQ_API_KEY` to enable active LLM conversational QA generation and claim auditing.*

### 3. Commands & Execution
Use the provided `Makefile` to instantly manage operations:

- **Run unit, integration, and security tests**:
  ```bash
  make test
  ```
- **Train models & export evaluation plots**:
  ```bash
  make train
  ```
- **Run the application locally**:
  ```bash
  make run
  ```
- **Initialize training + run the application in one step**:
  ```bash
  python run_all.py
  ```
- **Clean cached files & bytecode**:
  ```bash
  make clean
  ```

---

## 📈 Quality Assurance (QA) & Robustness
The application enforces strict software engineering standards verified by `pytest`:
- **Accuracy**: Verifies customized mathematical feature calculations and preprocessing formulas.
- **Reliability**: Validates sentence segmentation, hybrid RRF priority rankings, and semantic cache hit checks.
- **Security**: Asserts blockages against HTML script tag cross-site scripting (XSS), prompt-injection bypass instructions, and rate limit bucket capacities.

Run all tests instantly:
```bash
make test
```

---
*Architected for industrial-grade predictive analytics and grounded operations assistance, optimized for instant cloud review.*
