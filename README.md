# opsFlow — Industrial AI Maintenance System

An AI Developer Technical Assessment submission implementing two independent pipelines:

- **Task 3**: Supervised ML pipeline for predictive equipment failure classification  
- **Task 4**: Retrieval-Augmented Generation (RAG) assistant for industrial maintenance Q&A

---

## Tasks Implemented

| Task | Description | Status |
|------|-------------|--------|
| Task 3 | Equipment failure prediction (binary classification on AI4I 2020 dataset) | ✅ Complete |
| Task 4 | Grounded RAG assistant over industrial maintenance knowledge base | ✅ Complete |

---

## Architecture

### Task 3 — Predictive Maintenance ML (`ml/`)

```
ml/
├── train.py       # Pipeline: load → engineer → CV → GridSearchCV → persist
├── evaluate.py    # Holdout eval: metrics, SHAP, PR/ROC curves, SMOTE comparison
├── predict.py     # Single-row inference from JSON telemetry input
├── features.py    # Feature engineering (temp_diff, power, wear_torque_ratio)
├── pipeline_builder.py  # sklearn Pipeline: ColumnTransformer + Classifier
├── artifacts.py   # Artifact store (joblib serialization, plots)
└── data.py        # Dataset loading and validation
```

**Key design choices:**
- `sklearn.Pipeline` wraps `ColumnTransformer` + classifier — scaler is fitted per fold, preventing data leakage
- `StratifiedKFold` CV + `GridSearchCV` for hyperparameter tuning
- `SMOTE` comparison at evaluation time (not training time) to test recall trade-off
- `SHAP` TreeExplainer for RF feature importance — mechanically interpretable for failure root-cause

### Task 4 — RAG Assistant (`rag/`)

```
rag/
├── chunking.py    # Fixed-size + semantic sentence chunking; inline text extraction
├── embeddings.py  # SentenceTransformer bi-encoder singleton
├── retriever.py   # BM25 + dense semantic search → RRF fusion → cross-encoder rerank
├── generator.py   # Groq LLM generation + faithfulness auditing + query logging
└── pipeline.py    # Orchestrator: security → cache → retrieve → rerank → generate → audit
```

**Key design choices:**
- **Hybrid retrieval**: BM25 (exact keyword match) + ChromaDB (dense semantic) merged with Reciprocal Rank Fusion — handles both precise code lookups and semantic questions
- **RRF over score averaging**: BM25 and cosine scores are on incompatible scales; rank-based fusion is calibration-independent
- **Cross-encoder reranking**: `ms-marco-MiniLM-L-6-v2` performs joint query+document self-attention for final top-3 selection
- **Faithfulness auditing**: Second Groq call verifies every claim in the answer is grounded in retrieved chunks
- **Prompt injection detection**: Regex firewall on all inputs before retrieval

---

## Setup

```bash
# 1. Create and activate environment
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
cp .env.example .env
# Edit .env and set: GROQ_API_KEY=your_key_here
```

---

## Run Commands

### Task 3

```bash
# Train both classifiers (LR + RF) with GridSearchCV
python main.py --train

# Holdout evaluation with SHAP plots and SMOTE comparison
python main.py --evaluate

# Single-row failure prediction
python main.py --predict '{"Type": "M", "Air temperature [K]": 298.1, "Process temperature [K]": 308.6, "Rotational speed [rpm]": 1551, "Torque [Nm]": 42.8, "Tool wear [min]": 0}'
```

### Task 4

```bash
# Single grounded query
python main.py --query "What causes thermal overload in CNC spindle systems?"

# Interactive session with conversational memory
python main.py --interactive

# Use semantic chunking (slower but boundary-aware)
python main.py --query "..." --use-semantic-chunking
```

---

## Example Output

**`--predict` output:**
```json
{
  "prediction": "FAILURE",
  "failure_probability": 0.82,
  "root_cause": "High tool wear (245 min) above TWF threshold (200 min)",
  "model_used": "random_forest"
}
```

**`--query` output:**
```
Retrieved Chunks (Hybrid RRF + Cross-Encoder):
  [1] vibration_bearing_analysis.txt | chunk 12 | score 4.2891
  [2] equipment_manual.txt          | chunk 7  | score 3.9104

Answer:
Thermal overload in CNC spindle systems is typically caused by...

Faithfulness Audit:
  Faithful : YES
  Score    : 96.0%
  Verdict  : "All claims directly supported by retrieved manual sections."
```

---

## Tradeoffs

| Decision | Chosen | Alternative | Reason |
|----------|--------|-------------|--------|
| Chunking | Fixed-size (default) | Semantic | Deterministic, no model dependency at index time |
| Retrieval | BM25 + ChromaDB + RRF | Dense-only | BM25 catches exact codes/values that dense embeddings smooth over |
| Reranking | Cross-encoder | Bi-encoder | Higher accuracy on small candidate sets (top-10 → top-3) |
| LLM | Groq (llama-3.1-8b-instant) | OpenAI | Fast inference, generous free tier, suitable for assessment |
| ML models | LR + RF | XGBoost, deep nets | Interpretable, SHAP-compatible, suitable for 10k-row tabular data |

---

## Future Improvements

- Add MMR (Maximal Marginal Relevance) to diversify retrieved chunks across documents
- Persist query cache to disk (SQLite) for cross-session reuse  
- Streaming Groq responses for interactive mode latency improvement
- Replace GridSearchCV with Optuna for more efficient hyperparameter search
- Add chunk-level metadata filtering to scope queries by document type
