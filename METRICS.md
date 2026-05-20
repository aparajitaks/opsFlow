# opsFlow — Verified Metrics & Golden Behaviors

Last verified: automated test suite + local evaluation pipeline.

## Task 3 — Predictive Maintenance (holdout test set)

| Model | Recall | Precision | F1 | ROC-AUC | AP |
|-------|--------|-----------|-----|---------|-----|
| Logistic Regression (tuned) | 0.8676 | 0.1799 | 0.2980 | 0.9367 | 0.4455 |
| Random Forest (balanced) | 0.7647 | 0.8966 | **0.8254** | **0.9817** | **0.8494** |
| Random Forest (SMOTE†) | 0.8382 | 0.5278 | 0.6477 | 0.9794 | 0.8238 |

† SMOTE comparison trains on **preprocessed** features (same scaler as production pipeline).

### Overfitting watch (Random Forest)

| Split | Train F1 | Test F1 | ΔF1 |
|-------|----------|---------|-----|
| RF | 0.9945 | 0.8254 | +0.1691 |

Production recommendation: prefer **tighter `max_depth` / higher `min_samples_leaf`** (see `config.yaml` grid) or add calibration before thresholding alerts.

**Best model (CV):** Random Forest — F1 **0.8125**, ROC-AUC **0.9722** (`models/artifacts/model_summary.json`).

## Task 4 — RAG golden behaviors (expected)

| Scenario | Expected behavior |
|----------|-------------------|
| In-domain (e.g. LOTO) | Grounded answer citing safety manuals; confidence typically **> 0.5** |
| Out-of-domain (sports, fiction) | Exact refusal phrase; confidence **< 0.05** |
| Prompt injection | **Blocked** by firewall (`blocked: true`) |
| Fake equipment ID (PRV-999) | Refusal or no unsupported confirmation |
| Contradictory physics | Refusal |

## Automated quality gates

```bash
ML_N_JOBS=1 pytest tests/ -q    # 31+ unit/integration tests
make validate                   # dataset integrity on ai4i2020.csv
make evaluate                   # regenerates plots + evaluation_summary.json
```

## Artifacts

- ML plots: `models/artifacts/plots/`
- Evaluation JSON: `models/artifacts/evaluation_summary.json`
- RAG audit log: `logs/retrieved_chunks.log`
