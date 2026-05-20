# opsFlow — AI Developer Technical Assessment

Two deliverables in one repository:

| Task | Description | Entry point |
|------|-------------|-------------|
| **Task 3** | Equipment failure prediction (Logistic Regression + Random Forest) | `main.py --train` / `--evaluate` / `--predict` |
| **Task 4** | Retrieval-based maintenance assistant (RAG) | `main.py --query` / `--interactive` |

Dataset: [AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/ml/datasets/ai4i+2020+predictive+maintenance) (`data/ai4i2020.csv`).  
Knowledge base: maintenance manuals in `docs/`.

---

## Setup (one time)

```bash
cd opsFlow
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # or: make install
cp .env.example .env              # then set GROQ_API_KEY for live RAG answers
```

> **macOS:** Use `python3`, not `python`. After `source venv/bin/activate`, you can use `python`.

---

## Required CLI commands

Run these from the project root with the virtual environment activated.

### Task 3 — Equipment Failure Prediction

```bash
# 1. Train Logistic Regression + Random Forest (saves models/artifacts/)
python main.py --train

# 2. Evaluate on holdout set (metrics table + plots)
python main.py --evaluate

# 3. Predict failure from telemetry JSON
python main.py --predict '{"Type":"L","Air temperature [K]":300.0,"Process temperature [K]":310.5,"Rotational speed [rpm]":1500.0,"Torque [Nm]":40.0,"Tool wear [min]":50.0}'
```

**Makefile shortcuts:**

```bash
make train
make evaluate
make test
```

**Outputs:**

| Output | Location |
|--------|----------|
| Trained pipelines | `models/artifacts/*_pipeline.pkl` |
| Metrics JSON | `models/artifacts/evaluation_summary.json` |
| Plots (confusion matrix, ROC, PR, SHAP) | `models/artifacts/plots/` |
| Feature engineering notes | `docs/TASK3_FEATURE_ENGINEERING.md` |

---

### Task 4 — Retrieval-Based AI Assistant

```bash
# 1. Single grounded query (prints retrieved chunks + answer)
python main.py --query "What is the LOTO procedure for hydraulic systems?"

# 2. Interactive maintenance chat
python main.py --interactive

# 3. Out-of-domain test (should refuse)
python main.py --query "What is the capital of France?"
```

**Optional:**

```bash
python main.py --clear-cache          # reset query cache
python main.py --use-semantic-chunking  # semantic chunking on re-index
```

**Outputs:**

| Output | Location |
|--------|----------|
| Retrieved chunks (terminal) | printed after each query |
| Chunk audit log | `logs/retrieved_chunks.log` |
| Vector index | `rag/vector_store/` |

**KB refusal (low confidence / out of domain):**

> I could not find this in the knowledge base.

Without `GROQ_API_KEY` in `.env`, retrieval still runs; answers use a mock generator.

---

### Full demo sequence (copy-paste)

```bash
cd /path/to/opsFlow
source venv/bin/activate

# Task 3
python main.py --train
python main.py --evaluate
python main.py --predict '{"Type":"L","Air temperature [K]":300.0,"Process temperature [K]":310.5,"Rotational speed [rpm]":1500.0,"Torque [Nm]":40.0,"Tool wear [min]":50.0}'

# Task 4 (first run downloads embedding models — ~30s)
python main.py --query "What are the safety procedures for high voltage equipment?"
```

---

## Sample metrics (holdout test set)

| Metric | Logistic Regression | Random Forest |
|--------|--------------------:|--------------:|
| Accuracy | 0.86 | 0.99 |
| Recall | 0.87 | 0.85 |
| Precision | 0.18 | 0.85 |
| F1 | 0.30 | 0.85 |
| ROC-AUC | 0.94 | 0.97 |

See `models/artifacts/evaluation_summary.json` for full results after `make evaluate`.

---

## Tests

```bash
make test
# or
ML_N_JOBS=1 pytest tests/ -v
```

---

## Assumptions

- Binary target `Machine failure`; sub-failure columns are leakage and dropped.
- Stratified 80/20 train/test split; scaling inside sklearn `Pipeline` only.
- RAG answers use retrieved maintenance docs only; confidence threshold 0.30 triggers KB refusal.
- Groq API used for generation when `GROQ_API_KEY` is set.

## Trade-offs

- **RF vs LR:** RF wins F1/precision on holdout; LR has higher recall but many false positives.
- **RF overfitting:** train F1 > test F1 — see overfitting table in evaluate output.
- **Hybrid retrieval:** BM25 + dense + rerank improves recall at higher CPU cost.

## Future improvements

- Probability calibration for operator thresholds.
- Incremental vector index updates.
- Stronger RF regularization in `config.yaml`.

---

## Project layout

```text
opsFlow/
├── main.py              # CLI for Tasks 3 & 4
├── config.yaml
├── data/ai4i2020.csv
├── docs/                # RAG knowledge base
├── models/              # train · evaluate · predict · artifacts/
├── rag/                 # chunking · embeddings · vector_store · pipeline
├── core/                # config · logging · security
└── tests/
```

See [ARCHITECTURE.md](ARCHITECTURE.md) and [METRICS.md](METRICS.md) for design notes.
