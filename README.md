# Industrial Operations Flow (opsFlow): ML Pipelines & RAG Assistants

Welcome to **opsFlow**, a unified enterprise intelligence system for predictive mechanical engineering, industrial safety orchestration, and technical operations support. This project combines two core AI capabilities:

1. **Task 3 — Equipment Failure Prediction (ML):** A multi-stage Machine Learning pipeline utilizing the **AI4I 2020 Predictive Maintenance Dataset** to preemptively flag structural, thermal, and mechanical breakdowns.
2. **Task 4 — Retrieval-Based AI Assistant (RAG):** A zero-dependency semantic Retrieval-Augmented Generation assistant upgraded to a premium production-grade pipeline (**v3**) featuring **Hybrid Search (BM25 + Semantic)**, **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Re-Ranking**, and **Faithfulness Auditing**.

---

## 1. System Architecture

The opsFlow system integrates predictive telemetry analysis directly with semantic knowledge retrieval. The ML pipeline writes live metrics and model parameters to a shared knowledge layer, which the RAG assistant automatically ingests to answer questions about ML performance:

```text
  +--------------------------------------------------------------------------------+
  |                                   opsFlow System                               |
  +--------------------------------------------------------------------------------+
                                                                                    
     +-----------------------------+               +----------------------------+   
     |  Task 3: Failure telemetry  |               |  Task 4: Technical Manuals  |   
     |  (AI4I 2020 Sensor Stream)  |               |    (LOTO, PPE, FAQ, Guide) |   
     +--------------+--------------+               +--------------+-------------+   
                    |                                             |                 
                    v                                             |                 
     +--------------+--------------+                              |                 
     |  GridSearchCV Model Tuning  |                              |                 
     +--------------+--------------+                              |                 
                    |                                             |                 
                    v (Generates Model Stats)                     v                 
     +--------------+--------------+               +--------------+-------------+   
     |     model_summary.json      | ------------> |    Shared Knowledge Base   |   
     |   (Best Params, Features)   |               |   (Ingested as a Document) |   
     +-----------------------------+               +--------------+-------------+   
                                                                  |                 
                                                                  v                 
                                                   +--------------+-------------+   
                                                   | Hybrid Search (ChromaDB +  |   
                                                   |  rank-bm25 with RRF Fusion)|   
                                                   +--------------+-------------+   
                                                                  |                 
                                                                  v                 
                                                   +--------------+-------------+   
                                                   |  Cross-Encoder Re-Ranking  |   
                                                   | (MiniLM-L-6-v2 Top 10->3)  |   
                                                   +--------------+-------------+   
                                                                  |                 
                                                                  v                 
                                                   +--------------+-------------+   
                                                   |      Grounded Generator     |   
                                                   |   (Groq Llama-3-8B LPU)    |   
                                                   +--------------+-------------+   
                                                                  |                 
                                                                  v                 
                                                   +--------------+-------------+   
                                                   |  Faithfulness Auditor LLM  |   
                                                   | (Claims Verification Pass) |   
                                                   +--------------+-------------+   
                                                                  |                 
                                                                  v                 
                                                   +--------------+-------------+   
                                                   |  Interactive Terminal CLI  |   
                                                   +----------------------------+   
```

---

## 2. Central Local Setup and Environment Guide

opsFlow runs inside a single, unified Python virtual environment.

### Step 1: Create a Virtual Environment
Navigate to the project root and create a sandboxed virtual environment:
```bash
# Navigate to project root
cd /Users/galaxy_grid/Desktop/opsFlow

# Create virtual environment
python3 -m venv venv
```

### Step 2: Activate the Virtual Environment
Activate the environment before executing any commands or scripts:
* **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```
* **Windows:**
  ```cmd
  venv\Scripts\activate
  ```

### Step 3: Install Core Dependencies
Install the required packages listed in the consolidated requirements file:
```bash
pip install -r requirements.txt
```

### Step 4: Export Groq API Key
Set your active API credentials in the terminal shell (do not write them to file):
```bash
export GROQ_API_KEY="your-groq-api-key"
```

---

## 3. How to Run: Unified Entry Point

To execute both tasks in sequence as a single integrated ecosystem, run `run_all.py` at the root of the workspace:

```bash
python run_all.py
```

### Unified Runner Logic (`run_all.py`):
1. **Task 3 execution:** Runs the production ML tuning pipeline `task3/v3/main.py` inside a subprocess, creating local SHAP plots, logging parameter metrics to MLflow, serializing models, and dumping `model_summary.json`.
2. **Knowledge Transfer:** Copies the model summary JSON directly into the Task 4 RAG knowledge folder (`task4/v3/docs/model_summary.json`, `task4/v2/docs/model_summary.json` and `task4/v1/docs/model_summary.json`).
3. **Task 4 execution:** Launches the new hybrid `task4/v3/main.py` inside a subprocess, which loads embeddings and cross-encoders, constructs the local BM25 index, rebuilds ChromaDB dynamically upon count mismatches, runs the full hybrid RRF search suite, audits output claims for faithfulness, and enters the terminal console loop.

---

## 4. Task 3 — Equipment Failure ML Pipeline

Task 3 predicts machinery failures. It is divided into three sequential versions:

### version v1: Sequential ML Stages
Runs individual scripts to demonstrate model pipeline stages:
* **stage2_preprocessing.py:** Prepares data and drops index/ID features.
* **stage4_model_training.py:** Fits standard Logistic Regression & Random Forest classifiers.
* **stage5_model_evaluation.py:** Calculates metrics, confusion matrices, and ROC-AUC curves.
* **stage6_overfitting_analysis.py:** Identifies optimization gaps (train vs. test scores).

```bash
# Run v1 steps sequentially
python task3/v1/explore_predictive_maintenance.py
python task3/v1/stage2_preprocessing.py
python task3/v1/stage4_model_training.py
python task3/v1/stage5_model_evaluation.py
python task3/v1/stage6_overfitting_analysis.py
```

### version v2: Modular Class Balancing & Feature Engineering
Adds engineered telemetry features (`temp_diff`, `power`, `wear_torque_ratio`) and balances minority classes using SMOTE.
```bash
# Run modular v2 pipeline
python task3/v2/main.py
```

### version v3: Production-Grade ML, MLflow, and SHAP Explainability
Utilizes `GridSearchCV` for hyperparameter optimization, logs parameters in MLflow, generates SHAP feature explanations, and exports pickled estimators for inference.
```bash
# Run v3 locally
cd task3/v3
python main.py

# Launch MLflow Dashboard
mlflow ui
```

---

## 5. Task 4 — Retrieval-Based AI Assistant (RAG)

Task 4 builds a cognitive assistant to answer maintenance questions.

### version v1: FAISS & Groq API Pipeline
Uses word-based chunking (300 words, no overlap), `sentence-transformers` embeddings, and a FAISS index to retrieval-ground the **Groq API** completions.
```bash
cd task4/v1
python rag_pipeline.py
```

### version v2: Persistent ChromaDB & Groq Assistant
An advanced assistant utilizing sliding-window overlap chunking (300/50 split), a persistent SQLite-backed **ChromaDB** store, deterministic Groq completions, dynamic decommissioning fallback, and an interactive keyboard console.
```bash
cd task4/v2
python main.py
```

### version v3: Hybrid Search, Re-Ranking, and Faithfulness Auditing
A production-grade, premium RAG pipeline introducing key architectural layers:
1. **Hybrid Retrieval (BM25 + Semantic):** Searches dense embeddings (ChromaDB) and whitespace-tokenized keyword indices (rank-bm25) in parallel.
2. **Reciprocal Rank Fusion (RRF):** Fuses ordinal rankings using dampening constant $k=60$ to resolve uncalibrated dense/lexical score ranges.
3. **Cross-Encoder Re-Ranking:** Scores top 10 fused candidates down to top 3 using a local `ms-marco-MiniLM-L-6-v2` transformer for deep token cross-attention.
4. **Faithfulness Auditor:** Executes a second-pass Groq claim auditor verifying generated answers strictly against context, flagging out-of-scope hallucinations.

```bash
cd task4/v3
python main.py
```

---

## 6. Sample Terminal Outputs

### Task 3 ML Pipeline Execution:
```text
=================================================================
      TASK 3 — EQUIPMENT FAILURE PREDICTION: V3 PIPELINE        
=================================================================
MLflow Active Run ID: 4a2b978d38e24c5598fb87a98ce112bc
Saved StandardScaler to: outputs/models/scaler_v3.pkl
Training SMOTE Random Forest with Best Hyperparameters...
Saved best tuned models to outputs/models/:
 - outputs/models/logistic_regression_v3.pkl
 - outputs/models/random_forest_v3.pkl
Saved model summary JSON to: outputs/model_summary.json
Logging parameters, metrics, and tags to MLflow...

[Random Forest Prediction Result]
Predicted Class:            0 (No Failure)
Failure State Probability:  0.0125
```

### Task 4 v3 RAG Assistant Execution:
```text
=================================================================
      TASK 4 — RETRIEVAL-BASED AI ASSISTANT (RAG): V3            
=================================================================
[Step 1] Overlap Chunking Complete. Total Chunks Created: 11
[Step 2] Loading local SentenceTransformer embedder...
[ChromaDB] Collection 'maintenance_kb' successfully loaded from disk.
[Step 3] Building local BM25 Keyword Index...
[Step 4] Loading local Cross-Encoder re-ranker...

=================================================================
      RUNNING MANDATORY RAG V3 DEMO QUERIES                      
=================================================================

==============================================================
QUERY: What error code is logged when winding temperature exceeds 125 degrees?
==============================================================
[Hybrid Retrieval Fusion Breakdown]
 - Semantic Search Only retrieved: 'safety_procedures.txt' (Chunk 8)
 - BM25 Keyword Search Only retrieved: 'preventive_maintenance.txt' (Chunk 3)
 - Retrieved by BOTH (Intersection):  'maintenance_guide.txt' (Chunk 0)

[Cross-Encoder Re-Ranking Position Shifts]
 • 'maintenance_guide.txt' (Chunk 0) stayed at rank 1 (RE-RANKED TOP 3)
 • 'preventive_maintenance.txt' (Chunk 3) moved from rank 4 → rank 2 (RE-RANKED TOP 3)

Generated Grounded Answer:
When winding temperature exceeds 125 degrees Celsius, the system logs error code ERR-101.

Faithfulness Check:
  Faithful : Yes
  Score    : 1.00
  Verdict  : The claim regarding winding temperature and error code ERR-101 is directly supported by the maintenance guide.
```

---

## 7. Assumptions, Trade-offs, & Production Improvements

### Assumptions:
1. **Model Evaluation Metrics:** F1-score is prioritized over absolute Accuracy due to the severe (3.39%) class imbalance.
2. **Grounding Fallback Threshold:** Factual verification is checked via second-pass LLM claim analysis rather than raw distance boundaries, catching out-of-scope answers.

### Trade-offs:
1. **Local Embeddings vs. API:** Local SentenceTransformers and Cross-Encoders run without network latencies, but consume CPU/memory resources.
2. **FAISS vs. ChromaDB:** FAISS is highly optimized for flat vector operations in memory, but ChromaDB provides comprehensive database persistence and self-healing schemas.

### Production Improvements:
1. **Streaming Ingestion:** Implement Kafka or RabbitMQ event channels to stream live equipment telemetry and dynamically recalculate features.
2. **Multi-Modal Document Layout Ingestion:** Incorporate layout-aware parsers (like Unstructured or PyMuPDF) to ingest engineering blueprint schematics and wiring diagram graphics.
3. **Multilingual and Localized Term Mapping:** Standardize mechanical abbreviations and legacy technician slang onto clean canonical tokens during the tokenization stage.
