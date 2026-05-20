# Task 3 — Feature Engineering

Dataset: **AI4I 2020 Predictive Maintenance** (`data/ai4i2020.csv`) — public UCI industrial sensor telemetry.

## Target

Binary classification: `Machine failure` (0 = operational, 1 = failure).

## Leakage prevention

Dropped before modeling:

- Sub-failure indicators: `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (direct labels of failure modes)
- IDs: `UDI`, `Product ID`
- Target is never used as a feature

Scaling (`StandardScaler`) is fitted **inside** each CV fold and inside the sklearn `Pipeline`, so test data never influences training statistics.

## Engineered features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `temp_diff` | Process temp − Air temp | Thermal stress on the machine |
| `power` | Torque × RPM | Mechanical load proxy |
| `wear_torque_ratio` | Tool wear / (Torque + 1) | Wear relative to operating stress |
| `Type` | H→0, L→1, M→2 | Product type encoding |

## Models compared

1. **Logistic Regression** — linear baseline, `class_weight=balanced`, GridSearch on `C` and `solver`.
2. **Random Forest** — non-linear ensemble, GridSearch on depth/leaves/estimators.

Holdout metrics, confusion matrices, ROC/PR curves, and train-vs-test overfitting deltas are produced by `make evaluate`.
