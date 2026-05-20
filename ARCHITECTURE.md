# opsFlow Architecture (Assessment Scope)

## Layers

```text
main.py (CLI)
    ├── models/     Task 3 — load → features → train → evaluate → predict
    └── rag/        Task 4 — chunk → embed → retrieve → generate
core/               config · logging · security · validation
```

## Task 3 modules

| File | Role |
|------|------|
| `models/data.py` | Load AI4I CSV + validation |
| `models/features.py` | Feature engineering (train/inference parity) |
| `models/pipeline_builder.py` | sklearn Pipeline |
| `models/train.py` | LR + RF training, GridSearch, comparison |
| `models/evaluate.py` | Metrics, plots, overfitting analysis |
| `models/predict.py` | Inference CLI/API |

## Task 4 modules

| File | Role |
|------|------|
| `rag/chunking.py` | Document chunking |
| `rag/embeddings.py` | Sentence-transformer embeddings |
| `rag/vector_store/chroma_store.py` | Chroma persistence |
| `rag/retriever.py` | BM25 + dense + RRF + rerank |
| `rag/generator.py` | Groq grounded generation + faithfulness |
| `rag/pipeline.py` | Orchestration + chunk logging |

## Configuration

- Secrets: `.env` (`GROQ_API_KEY`)
- Parameters: `config.yaml` → `core.config.settings`
