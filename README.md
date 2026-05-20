# ⚙️ opsFlow: Standalone RAG Intelligence & Predictive Maintenance Diagnostic System

Welcome to **opsFlow**, an enterprise-grade, fully modular Python backend system designed for real-time equipment telemetry diagnostics and grounded technical knowledge retrieval in manufacturing and industrial operations.

This project is a 100% technical, assignment-focused python repository, strictly structured around two primary engineering modules:
1. **Task 3 — Equipment Failure Prediction (Traditional ML)**
2. **Task 4 — Retrieval-Based AI Assistant (RAG Basics)**

There are zero frontend UI/Streamlit dashboards or SaaS complexities, maintaining a razor-sharp focus on production-grade Python architecture, full test suites, and clean command-line interfaces.

---

## ⚡ Setup & Local Operations

### 1. Prerequisites
Ensure Python 3.10 to 3.14 is installed on your local macOS or Linux machine.

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
*An active `GROQ_API_KEY` is required to run the real LLM-backed generator and factual claims auditor. If not present, the system automatically falls back to an elegant mock environment for local testing.*

---

## 🚀 CLI Commands & Hook Reference

Use `main.py` or the provided `Makefile` hooks to execute the pipelines:

### ⚙️ Task 3 — Predictive Maintenance ML Pipeline
- **Run python unit and integration tests**:
  ```bash
  make test
  ```
- **Train models, scale, tune hyperparameters, and cache data splits**:
  ```bash
  make train
  # OR: ./venv/bin/python main.py --train
  ```
- **Perform holdout evaluations, generate comparison curves, and export SHAP local explainability**:
  ```bash
  make evaluate
  # OR: ./venv/bin/python main.py --evaluate
  ```
- **Predict equipment failure from a raw JSON telemetry string**:
  ```bash
  ./venv/bin/python main.py --predict '{"Type": "L", "Air temperature [K]": 300.0, "Process temperature [K]": 310.5, "Rotational speed [rpm]": 1500.0, "Torque [Nm]": 40.0, "Tool wear [min]": 50.0}'
  ```

### 🛠️ Task 4 — Retrieval-Augmented Generation (RAG) Assistant
- **Submit a single grounded query to the assistant**:
  ```bash
  ./venv/bin/python main.py --query "What are the symptoms and verification steps for a faulty Pressure Relief Valve PRV-200?"
  ```
- **Start a full interactive terminal RAG session loop**:
  ```bash
  ./venv/bin/python main.py --interactive
  ```
  *Inside the loop, type standard commands or special utilities: `/help`, `/clear` (clear screen), `/history` (show session queries), or `/save` (export transcript).*
- **Purge the semantic query cache database**:
  ```bash
  ./venv/bin/python main.py --clear-cache
  ```

### 🎓 Deep-Dive Architectural Explanations (Recruiter Review)
Fast, zero-heavy-package CLI explanations outlining core architectural decisions:
- **FAISS vs SQLite ChromaDB Vector Database**:
  ```bash
  ./venv/bin/python main.py --explain-chromadb
  ```
- **BM25 Lexical Matching vs Dense Embeddings**:
  ```bash
  ./venv/bin/python main.py --explain-bm25
  ```
- **Reciprocal Rank Fusion (RRF) vs Score Averaging**:
  ```bash
  ./venv/bin/python main.py --explain-rrf
  ```
- **Bi-Encoder Embeddings vs Deep Cross-Encoder Reranking**:
  ```bash
  ./venv/bin/python main.py --explain-rerank
  ```
- **Subject Relevance vs Factual Faithfulness Auditing**:
  ```bash
  ./venv/bin/python main.py --explain-faithfulness
  ```
- **Chunk Source Logging & Safety Compliance**:
  ```bash
  ./venv/bin/python main.py --explain-logging
  ```

---

## 🛠️ Unified System Directory Structure

```text
opsFlow/
├── core/                   # Core configurations and global definitions
│   ├── config.py           # Environmental settings loader & folder generator
│   ├── constants.py        # ML parameter thresholds & feature ordering
│   └── security.py         # Rate limiters, sanitizers & regex firewalls
├── data/                   # Raw telemetry sensor storage (ai4i2020.csv)
├── docs/                   # High-density technical manuals for RAG ingestion
│   ├── model_summary.json  # Synced ML performance metrics (cross-pipeline metadata)
│   ├── cnc_lathe_spindle_systems.txt
│   ├── safety_procedures.txt
│   └── hydraulic_pneumatic_systems.txt  # (Total 10 dense manuals)
├── models/                 # Machinery Diagnostics Classifiers (Task 3)
│   ├── artifacts/          # Serialized models, splits, metrics & plots
│   │   ├── plots/          # Performance curves & SHAP beeswarm/force plots
│   │   ├── scaler.pkl      # Feature scaling checkpoint
│   │   └── random_forest.pkl
│   ├── train.py            # GridSearchCV hyperparameter tuning pipeline
│   ├── evaluate.py         # Balanced vs SMOTE recall comparison & SHAP plots
│   └── predict.py          # In-process real-time telemetry classifier
├── rag/                    # Retrieval-Augmented Generation (Task 4)
│   ├── vector_store/       # Persistent SQLite-backed ChromaDB vectors
│   ├── chunking.py         # Sliding-window and sentence cosine semantic chunker
│   ├── embeddings.py       # Dense text embedding model manager (SentenceTransformers)
│   ├── generator.py        # Groq LLM completion engine & factual claims auditor
│   ├── retriever.py        # Sparse BM25 + Dense RRF retriever and similarity cache
│   └── pipeline.py         # Consolidated modular RAG orchestration pipeline
├── tests/                  # Pytest QA Test Suite
│   ├── conftest.py         # Tiny mock dataframe fixtures
│   ├── test_ml.py          # Preprocessing, tuning, and telemetry calculations
│   ├── test_rag_service.py # API client caching & key validation checks
│   ├── test_retrieval.py   # Token splitting, lexical matches, and semantic cache
│   └── test_security.py    # XSS blockages, firewalls, and token rate limits
├── Makefile                # Unified developer CLI command hooks
├── requirements.txt        # CPU-only optimized requirements (no CUDA/Triton)
└── main.py                 # Root CLI Entrypoint
```

---

## 🔍 Key Architectural Details

### Task 3: Machinery Predictive Diagnostics (Traditional ML)
* **Real-time Feature Engineering**: Engineers customized attributes like Process-Air thermal delta ($\Delta T$), mechanical power ($Torque \times RPM$), and stress-adjusted tool wear.
* **Leakage Prevention**: Systematically drops all sub-failure category leakage columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) during preprocessing and keeps scaling strictly inside cross-validation splits.
* **Recall Strategy Tuning**: Combines Stratified K-Fold CV GridSearch tuning with a comparative SMOTE oversampling study, generating absolute confusion matrices, ROC-AUC, and Precision-Recall comparison plots.
* **Local Explainability**: Computes localized Shapley values (SHAP Beeswarm and Force plots) to give operators deep transparency into exact telemetry failure drivers.

### Task 4: Retrieval-Augmented Generation (RAG Basics)
* **High-Density Corpus**: Loaded with 35 technical chunks covering electrical safety, high-pressure hydraulics, vibrations, gear tribology, and LOTO.
* **Hybrid Retrieval**: Combines high-IDF sparse lexical keyword matches (Okapi BM25) and dense embeddings (`all-MiniLM-L6-v2`) inside a thread-safe Reciprocal Rank Fusion (RRF) system.
* **Deep Cross-Encoder Reranking**: Executes high-fidelity token-to-token attention using a CPU-based `cross-encoder/ms-marco-MiniLM-L-6-v2` over the candidate set, filtering down to the top 3 chunks.
* **Calibrated Sigmoid Confidence**: Maps reranker raw outputs using a customized logistic sigmoid function to generate normalized confidence scores. Gracefully refuses out-of-domain queries when confidence falls below `0.30`.
* **Grounding & Double-Pass Audit**: Prior to output, a secondary pass audits the generated answer using an LLM-based claims evaluator to check every claim against retrieved sources, ensuring factual faithfulness.
* **Rate-Limit & Cache Protection**: Protects backends via an in-memory client-separated TokenBucket rate limiter, rate-limit retries with exponential backoffs, and an exact/semantic similarity query cache.

---

## 📈 Quality Assurance (QA) & Robustness
The application enforces strict software engineering standards verified by `pytest`:
* **XSS Protection**: Asserts blockages against HTML script tag cross-site scripting (XSS).
* **Prompt Injection Firewall**: Blocks adversarial prompt injections and instructions.
* **Rate Limits**: Confirms TokenBucket rate limit parameters function correctly.
* **Algorithm Correctness**: Validates lexical indexing, RRF ranking, cache hits, and ML feature calculations.

Run all tests instantly:
```bash
make test
```
*All 13 unit, integration, and security tests execute cleanly in under 15 seconds.*
