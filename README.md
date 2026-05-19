# opsFlow: AI-Powered Industrial Maintenance Intelligence System

Welcome to **opsFlow**, an integrated enterprise intelligence system designed for predictive mechanical telemetry and automated operations support in manufacturing environments.

## 1. Project Overview

opsFlow is a unified industrial maintenance intelligence platform that combines predictive machinery diagnostics with grounded technical knowledge retrieval. The system employs high-fidelity machine learning to forecast mechanical breakdowns before they happen, while deploying a cognitive retrieval assistant to guide on-site engineers through safety, diagnostics, and repairs.

Task 3 and Task 4 are directly integrated: Task 3 trains production predictive maintenance models and serializes the performance stats, hyperparameter states, and explainability metrics into a centralized database file named `model_summary.json`. Task 4's Retrieval-Augmented Generation (RAG) pipeline ingests this structured dataset as part of its technical knowledge base, allowing operators to query live ML model accuracy, precision, F1-scores, and diagnostic results conversationally.

---

## 2. Architecture Diagram (ASCII)

```text
AI4I 2020 Dataset
      │
      ▼
┌─────────────────────────────────────┐
│         TASK 3: ML PIPELINE         │
│  v1 → v2 → v3 (production)          │
│  Logistic Regression + Random Forest│
│  SMOTE │ GridSearchCV │ SHAP │ MLflow│
└──────────────┬──────────────────────┘
               │ model_summary.json
               ▼
┌─────────────────────────────────────┐
│       TASK 4: RAG ASSISTANT         │
│  v1 → v2 → v3 (production)          │
│  BM25 + Semantic │ Re-ranking       │
│  ChromaDB │ Faithfulness Check      │
└──────────────┬──────────────────────┘
               │
               ▼
        Maintenance Q&A
   grounded in docs + ML results
```

---

## 3. Repository Structure

```text
opsFlow/
├── task3/
│   ├── v1/         # Baseline: LR + RF, basic metrics
│   ├── v2/         # + Feature engineering, SMOTE, CV, modular code
│   ├── v3/         # + MLflow, GridSearchCV, SHAP, serialisation, Docker
│   └── data/
│       └── ai4i2020.csv
├── task4/
│   ├── v1/         # Baseline: FAISS, fixed chunking, Groq generation
│   ├── v2/         # + ChromaDB, overlap chunking, grounding, CLI loop
│   ├── v3/         # + Hybrid search, re-ranking, faithfulness check
│   └── (docs in each version folder)
├── run_all.py      # Single entry point: runs Task 3 v3 → Task 4 v3
├── README.md
└── requirements.txt
```

---

## 4. Versioning Strategy

This project follows an iterative versioning methodology, moving systematically from standard baseline models (v1) through robust feature-engineered codebases (v2) up to enterprise-grade production architectures (v3) for both predictive modeling and information retrieval tasks.

### Task 3: Equipment Failure Prediction
- **v1**: Load dataset, encode features, drop data leakage columns, train LR + RF, evaluate with ROC-AUC and confusion matrix, overfitting analysis.
- **v2**: Engineered features (`temp_diff`, `power`, `wear_torque_ratio`), SMOTE for class imbalance, 5-fold stratified CV, feature importance and PR curve plots, modular `.py` structure.
- **v3**: MLflow experiment tracking, GridSearchCV hyperparameter tuning, SHAP explainability (beeswarm + force plots), joblib model serialisation, `load_and_predict()` demo, Dockerfile.

### Task 4: Retrieval-Based AI Assistant
- **v1**: Fixed-size chunking, sentence-transformers embeddings, FAISS vector store, Groq LLM generation, chunk source logging.
- **v2**: Overlap chunking (300 words, 50 overlap), persistent ChromaDB, strict grounding prompt, rich source logging, multi-query CLI loop.
- **v3**: BM25 keyword index, hybrid retrieval with RRF fusion, cross-encoder re-ranking, faithfulness audit via second Groq call, extended logging.

---

## 5. Setup Instructions

```bash
# Clone and enter project
git clone <repo-url>
cd opsFlow

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt

# Set Groq API key (required for Task 4)
export GROQ_API_KEY='your-key-here'
# Get free key at: https://console.groq.com
```

### Dataset Note
```text
Task 3 uses the AI4I 2020 Predictive Maintenance dataset.
Place it at: task3/data/ai4i2020.csv
Download from: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
```

---

## 6. How to Run

### Option A — Full pipeline (recommended)
Execute both tasks sequentially with dynamic model-to-assistant knowledge transfer using a single command:
```bash
# From opsFlow/ root
python run_all.py
# Runs Task 3 v3 → copies model_summary.json → runs Task 4 v3
```

### Option B — Run each task individually
Step into specific version workspaces to evaluate performance stages:
```bash
# Task 3
cd task3/v1 && python stage2_preprocessing.py   # baseline preprocessing
cd task3/v2 && python main.py                    # feature engineering + CV
cd task3/v3 && python main.py                    # full production pipeline

# Task 4
cd task4/v1 && python rag_pipeline.py            # baseline RAG
cd task4/v2 && python main.py                    # persistent store + CLI
cd task4/v2 && python main.py                    # persistent store + CLI
cd task4/v3 && python main.py                    # hybrid search + faithfulness
```

### Option C — Docker (Task 3 v3 only)
Package and run the production predictive pipeline inside an isolated container:
```bash
cd task3/v3
docker build -t task3-v3 .
docker run -v $(pwd)/../data:/app/data task3-v3
```

---

## 7. Sample Output

### Task 3 v3 Key Outputs
```text
Logistic Regression Best Parameters: {'C': 1, 'solver': 'lbfgs'}
Random Forest Best Parameters: {'max_depth': None, 'min_samples_leaf': 1, 'n_estimators': 100}
Random Forest Best CV F1 Score: 0.8094
Tuned Random Forest AP: 0.8512
MLflow Run ID: 5d29d5924ac54a368eb52a0956af5742
```

### Integration Output
```text
[INTEGRATION] Copied model_summary.json → task4/v2/docs/
[INTEGRATION] Copied model_summary.json → task4/v3/docs/
```

### Task 4 v3 Key Outputs
```text
[Hybrid] RRF scores computed for 10 candidates
[Reranker] Chunk from equipment_manual.txt moved rank 7 → rank 1
Generated Grounded Answer: ...
Faithfulness Check:
  Faithful : Yes
  Score    : 1.00
  Verdict  : All claims are directly supported by the maintenance manual.
```

---

## 8. Key Design Decisions & Assumptions

- **Sub-Failure Column Removal (Data Leakage):** The columns `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` specify exact failure modes which are logical components of the target variable `Machine failure`. Retaining them triggers total target leakage, leading to artificially perfect training metrics that fail completely during live inference. 
- **SMOTE Boundary Restrictions:** Oversampling with SMOTE is strictly constrained to the training splits, leaving test/validation splits completely untouched. Applying SMOTE before splitting or to test sets leaks synthetic coordinates into test bounds, causing severely inflated, unrealistic model evaluations.
- **Precision-Recall Curve over ROC-AUC:** Given the extreme class imbalance in machinery failures (only 3.39% representation), ROC-AUC yields over-optimistic evaluations by using False Positive Rates dominated by the massive negative class. Precision-Recall curves capture exact model performance differences on minority failure instances.
- **BM25 + Semantic Hybrid Search Synergy:** Lexical BM25 search excels at exact-match keyword indexing (such as error codes, technical IDs, or specific model parameters), while semantic vector space search captures abstract conceptual questions. Fusing them with RRF leverages both advantages.
- **Post-Retrieval Cross-Encoder Re-Ranking:** Bi-encoders process queries and documents independently to scale retrieval rapidly. The computationally heavy cross-encoder computes full attention across queries and candidate text sequences, and is placed as a second-pass re-ranker to maximize accuracy without compromising real-time search latencies.
- **Second-Pass LLM Faithfulness Audit:** Implementing a separate LLM pass dedicated strictly to checking generated claims against the retrieved manual chunks separates contextual answer generation from verification logic, catching hallucinations before they reach operators.
- **Groq LPU Selection:** Groq’s custom LPU hardware delivers ultra-fast token generation speeds (exceeding 500 tokens/sec), facilitating real-time interactive Q&A. Its high-availability free tier supports early development without licensing bottlenecks.

---

## 9. Production Improvements

- **Remote MLflow Tracking Server:** Transition from the local filesystem store to a centralized Postgres backend database and an S3 artifact bucket to enable collaboration across multiple engineering teams.
- **Metadata Filtering:** Integrate database-level metadata partitioning in ChromaDB to restrict searches to specific machinery categories, documents, or plant regions prior to vector matching.
- **Streaming UI Outputs:** Refactor the generator pipeline to stream token outputs dynamically to the terminal or web dashboard to minimize user-perceived latencies.
- **Automated Retraining Orchestrator:** Deploy Airflow or Prefect DAGs to trigger retraining and model deployments dynamically as new sensor streams drift from initial training ranges.
- **Assistant Authentication Layer:** Enforce RBAC (Role-Based Access Control) to verify technician credentials before serving high-voltage safety instructions or operating manuals.
- **Faithfulness Threshold Guardrails:** Build an automated system that blocks and re-generates LLM answers scoring below 0.70 in the auditor, ensuring safety-critical correctness in physical plants.

---

## 10. Dependencies

| Library | Purpose |
| :--- | :--- |
| **pandas, numpy** | Data manipulation and telemetry array transformations |
| **scikit-learn** | Standard ML classifiers, cross-validation splits, and parameter tuning |
| **imbalanced-learn** | SMOTE oversampling algorithm implementation for class balancing |
| **matplotlib** | Plotting of diagnostic visual assets (feature importance, PR curve, etc.) |
| **shap** | Unified game-theoretic SHAP explainability calculations and plots |
| **mlflow** | Parameter tracking, metrics logging, and artifact serialization |
| **joblib** | Model and transformer pipeline serialization to local disk |
| **sentence-transformers** | Dense sentence-level embedding matrix and cross-encoder re-ranking calculations |
| **faiss-cpu** | Fast memory-mapped dense vector similarity search |
| **chromadb** | Persistent, SQLite-backed serverless vector database engine |
| **rank-bm25** | Lexical TF-IDF derived keyword indexing algorithm |
| **groq** | High-speed LLM client for answer generation and claim audits |

---

*This project is submitted as an AI Developer Technical Assessment. Created and maintained for industrial-grade predictive analytics.*
