# Industrial Operations Flow (opsFlow): ML Pipelines & RAG Assistants

Welcome to **opsFlow**, a unified enterprise intelligence system for predictive mechanical engineering, industrial safety orchestration, and technical operations support. This project combines two core AI capabilities:

1. **Task 3 — Equipment Failure Prediction (ML):** A multi-stage Machine Learning pipeline utilizing the **AI4I 2020 Predictive Maintenance Dataset** to preemptively flag structural, thermal, and mechanical breakdowns.
2. **Task 4 — Retrieval-Based AI Assistant (RAG):** A zero-dependency semantic Retrieval-Augmented Generation assistant that searches plant operation manuals, lock-out/tag-out (LOTO) protocols, and mechanical FAQs to output grounded maintenance procedures.

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
                                                   |    Local Dense Embedder    |   
                                                   |    (all-MiniLM-L6-v2)      |   
                                                   +--------------+-------------+   
                                                                  |                 
                                                                  v                 
                                                   +--------------+-------------+   
                                                   |  Persistent vector Store   |   
                                                   |      (ChromaDB Index)      |   
                                                   +--------------+-------------+   
                                                                  |                 
                                                                  v                 
                                                   +--------------+-------------+   
                                                   |      Grounded Generator     |   
                                                   |   (Groq Llama-3.1-8B LPU)  |   
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
2. **Knowledge Transfer:** Copies the model summary JSON directly into the Task 4 RAG knowledge folder (`task4/v2/docs/model_summary.json` and `task4/v1/docs/model_summary.json`).
3. **Task 4 execution:** Launches `task4/v2/main.py` inside a subprocess, which embeds the new model parameters, loads or rebuilds the ChromaDB vector database, runs verification tests, and initializes the interactive terminal prompt.

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

### Task 4 RAG Assistant Execution:
```text
=================================================================
      TASK 4 — RETRIEVAL-BASED AI ASSISTANT (RAG): V2            
=================================================================
[Step 1] Overlap Chunking Complete. Total Chunks Created: 11
[Step 2 & 3] Loading local SentenceTransformer embedder...
[ChromaDB] Collection count mismatch (10 vs 11 chunks). Rebuilding database for fresh updates...
[ChromaDB] Computing embeddings and populating database for 11 chunks...

-------------------------------------------------------------
QUESTION: What was the best performing model?
-------------------------------------------------------------
Retrieved Sources used for Grounding:
 [1] Doc: model_summary.json | Chunk ID: 0 | Similarity: 0.5422
Generated Grounded Answer:
The best performing model in the training results is the Random Forest. It achieved a best F1 score of 0.8205 and a best ROC-AUC score of 0.9412.
-------------------------------------------------------------
```

---

## 7. Assumptions, Trade-offs, & Production Improvements

### Assumptions:
1. **Model Evaluation Metrics:** F1-score is prioritized over absolute Accuracy due to the severe (3.39%) class imbalance.
2. **Grounding Fallback Threshold:** Similarity scores below a cosine limit default to strict fallback string completions to prevent hallucinations when answers are missing.

### Trade-offs:
1. **Local Embeddings vs. API:** Local SentenceTransformers run without network latencies, but consume CPU/memory resources.
2. **FAISS vs. ChromaDB:** FAISS is highly optimized for flat vector operations in memory, but ChromaDB provides comprehensive database persistence and self-healing schemas.

### Production Improvements:
1. **Streaming Ingestion:** Implement Kafka or RabbitMQ event channels to stream live equipment telemetry and dynamically recalculate features.
2. **Hybrid Retrieval:** Combine ChromaDB dense vector searches with BM25 keyword matching for optimal acronym and error-code lookups.
