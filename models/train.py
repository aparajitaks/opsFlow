"""
models/train.py — Task 3: Equipment Failure Prediction
=======================================================
V3 Production Upgrade:
  • sklearn Pipeline + ColumnTransformer (preprocessing integrated, leak-proof)
  • Config-driven hyperparameters from config.yaml
  • Python logging module (no print() calls)
  • MLflow experiment tracking
  • --model CLI argument for targeted training
  • Full pipeline artifacts persisted as single .pkl for inference
"""
import os
import sys
import json
import shutil
import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

from core.config import settings
from core.logger import get_logger
from models.artifacts import ModelArtifactStore
from models.data import load_dataset
from models.features import engineer_features, prepare_data_pipeline
from models.pipeline_builder import build_pipeline

log = get_logger("models.train")

# Re-export for backward compatibility (tests, external imports)
__all__ = [
    "load_dataset",
    "engineer_features",
    "prepare_data_pipeline",
    "build_pipeline",
    "run_train",
    "perform_cross_validation",
    "tune_pipeline",
]


# ---------------------------------------------------------------------------
# Cross-validation baselines (inside-fold scaling)
# ---------------------------------------------------------------------------

def perform_cross_validation(X: pd.DataFrame, y: pd.Series, n_splits: int = None):
    """
    Executes stratified K-fold CV on baseline (untuned) LR and RF.
    Scaling is fitted inside each fold via Pipeline to prevent data leakage.
    """
    n_splits = n_splits or settings.N_CV_SPLITS
    log.info(f"Running {n_splits}-fold Stratified CV on baseline models...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=settings.RANDOM_STATE)

    scorers = ["f1", "roc_auc", "precision", "recall"]
    lr_cv = {s: [] for s in scorers}
    rf_cv = {s: [] for s in scorers}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # LR pipeline (fit scaler only on fold train data)
        lr_pipe = build_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=settings.RANDOM_STATE)
        )
        lr_pipe.fit(X_tr, y_tr)
        lr_preds = lr_pipe.predict(X_val)
        lr_probs = lr_pipe.predict_proba(X_val)[:, 1]

        # RF pipeline
        rf_pipe = build_pipeline(
            RandomForestClassifier(class_weight="balanced", random_state=settings.RANDOM_STATE)
        )
        rf_pipe.fit(X_tr, y_tr)
        rf_preds = rf_pipe.predict(X_val)
        rf_probs = rf_pipe.predict_proba(X_val)[:, 1]

        lr_cv["f1"].append(f1_score(y_val, lr_preds))
        lr_cv["roc_auc"].append(roc_auc_score(y_val, lr_probs))
        lr_cv["precision"].append(precision_score(y_val, lr_preds, zero_division=0))
        lr_cv["recall"].append(recall_score(y_val, lr_preds))

        rf_cv["f1"].append(f1_score(y_val, rf_preds))
        rf_cv["roc_auc"].append(roc_auc_score(y_val, rf_probs))
        rf_cv["precision"].append(precision_score(y_val, rf_preds, zero_division=0))
        rf_cv["recall"].append(recall_score(y_val, rf_preds))

        log.debug(f"Fold {fold_idx}: LR F1={lr_cv['f1'][-1]:.4f} | RF F1={rf_cv['f1'][-1]:.4f}")

    log.info(f"Baseline LR — CV F1: {np.mean(lr_cv['f1']):.4f} | ROC-AUC: {np.mean(lr_cv['roc_auc']):.4f}")
    log.info(f"Baseline RF — CV F1: {np.mean(rf_cv['f1']):.4f} | ROC-AUC: {np.mean(rf_cv['roc_auc']):.4f}")
    return lr_cv, rf_cv


# ---------------------------------------------------------------------------
# STEP 6 — GridSearchCV tuning
# ---------------------------------------------------------------------------

def tune_pipeline(X_train: pd.DataFrame, y_train: pd.Series, model_name: str = "both"):
    """
    Runs GridSearchCV over Pipeline with nested param keys (classifier__param).
    Returns best estimator pipelines for selected models.
    """
    skf = StratifiedKFold(n_splits=settings.N_CV_SPLITS, shuffle=True, random_state=settings.RANDOM_STATE)
    results = {}

    if model_name in ("logistic_regression", "both"):
        log.info("GridSearchCV — Tuning Logistic Regression pipeline...")
        import yaml
        with open(settings.CONFIG_PATH, "r") as f:
            raw_cfg = yaml.safe_load(f)
        lr_cfg_grid = raw_cfg.get("ml", {}).get("logistic_regression", {}).get("param_grid", {})
        lr_param_grid = {f"classifier__{k}": v for k, v in lr_cfg_grid.items()}

        lr_base = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=settings.RANDOM_STATE
        )
        lr_pipe = build_pipeline(lr_base)
        n_jobs = int(os.environ.get("ML_N_JOBS", "-1"))
        lr_search = GridSearchCV(lr_pipe, lr_param_grid, cv=skf, scoring="f1", n_jobs=n_jobs)
        lr_search.fit(X_train, y_train)
        log.info(f"Best LR params: {lr_search.best_params_} | CV F1: {lr_search.best_score_:.4f}")
        results["logistic_regression"] = {
            "pipeline": lr_search.best_estimator_,
            "best_params": lr_search.best_params_,
            "best_score": lr_search.best_score_,
        }

    if model_name in ("random_forest", "both"):
        log.info("GridSearchCV — Tuning Random Forest pipeline...")
        import yaml
        with open(settings.CONFIG_PATH, "r") as f:
            raw_cfg = yaml.safe_load(f)
        rf_cfg_grid = raw_cfg.get("ml", {}).get("random_forest", {}).get("param_grid", {})
        rf_param_grid = {f"classifier__{k}": v for k, v in rf_cfg_grid.items()}

        rf_base = RandomForestClassifier(
            class_weight="balanced", random_state=settings.RANDOM_STATE
        )
        rf_pipe = build_pipeline(rf_base)
        rf_search = GridSearchCV(rf_pipe, rf_param_grid, cv=skf, scoring="f1", n_jobs=n_jobs)
        rf_search.fit(X_train, y_train)
        log.info(f"Best RF params: {rf_search.best_params_} | CV F1: {rf_search.best_score_:.4f}")
        results["random_forest"] = {
            "pipeline": rf_search.best_estimator_,
            "best_params": rf_search.best_params_,
            "best_score": rf_search.best_score_,
        }

    return results


# ---------------------------------------------------------------------------
# STEP 7 — MLflow logging helper
# ---------------------------------------------------------------------------

def _log_to_mlflow(model_name: str, params: dict, cv_score: float,
                   lr_cv: dict, rf_cv: dict, artifacts_dir):
    """Logs a training run to MLflow if enabled in config."""
    if not settings.MLFLOW_ENABLED:
        return
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        run_name = f"{settings.MLFLOW_RUN_NAME_PREFIX}_{model_name}_{datetime.datetime.now().strftime('%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters
            clean_params = {k.replace("classifier__", ""): v for k, v in params.items()
                            if isinstance(v, (str, int, float, bool)) or v is None}
            mlflow.log_params(clean_params)
            mlflow.log_param("random_state", settings.RANDOM_STATE)
            mlflow.log_param("test_size", settings.TEST_SIZE)
            mlflow.log_param("n_cv_splits", settings.N_CV_SPLITS)

            # Log CV metrics
            mlflow.log_metric("cv_f1_mean", cv_score)
            if lr_cv:
                mlflow.log_metric("lr_cv_f1",     float(np.mean(lr_cv["f1"])))
                mlflow.log_metric("lr_cv_roc_auc", float(np.mean(lr_cv["roc_auc"])))
            if rf_cv:
                mlflow.log_metric("rf_cv_f1",     float(np.mean(rf_cv["f1"])))
                mlflow.log_metric("rf_cv_roc_auc", float(np.mean(rf_cv["roc_auc"])))

            # Log model artifact
            model_file = artifacts_dir / f"{model_name.replace(' ', '_').lower()}_pipeline.pkl"
            if model_file.exists():
                mlflow.log_artifact(str(model_file))

            log.info(f"MLflow run '{run_name}' logged under experiment '{settings.MLFLOW_EXPERIMENT_NAME}'")
    except Exception as e:
        log.warning(f"MLflow logging failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# MAIN RUN FUNCTION
# ---------------------------------------------------------------------------

def run_train(model_name: str = "both"):
    """
    Complete V3 training pipeline:
      1. Load & engineer features
      2. Split (stratified)
      3. 5-fold CV baselines
      4. GridSearchCV pipeline tuning (ColumnTransformer + Classifier)
      5. Persist full pipeline artifacts (no separate scaler needed)
      6. Sync model_summary.json to RAG docs corpus
      7. MLflow experiment tracking
    """
    log.info("=" * 52)
    log.info("  TASK 3 — PREDICTIVE ML PIPELINE: TRAINING (V3)  ")
    log.info("=" * 52)

    # 1. Load
    raw_df = load_dataset()
    failure_rate = float(raw_df[settings.TARGET_COL].mean())
    log.info(f"Dataset failure rate: {failure_rate*100:.2f}%")

    # 2. Engineer & split
    df = engineer_features(raw_df)
    X_train, X_test, y_train, y_test = prepare_data_pipeline(df)

    # 3. CV baselines
    lr_cv, rf_cv = perform_cross_validation(X_train, y_train)

    # 4. GridSearchCV tuning
    tuned = tune_pipeline(X_train, y_train, model_name=model_name)

    # 5. Persist artifacts
    artifacts_dir = settings.MODEL_ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Determine best model
    best_model_name = "Random Forest"
    best_cv_score = 0.0
    top_features = []

    store = ModelArtifactStore(artifacts_dir)

    if "random_forest" in tuned:
        rf_result = tuned["random_forest"]
        rf_pipeline: Pipeline = rf_result["pipeline"]
        store.save_pipeline(rf_pipeline, "random_forest")

        # Extract feature importances from RF step
        try:
            rf_clf = rf_pipeline.named_steps["classifier"]
            feature_names = X_train.columns.tolist()
            importances = rf_clf.feature_importances_
            top_indices = np.argsort(importances)[::-1][:3]
            top_features = [feature_names[i] for i in top_indices]
        except Exception:
            top_features = []

        if rf_result["best_score"] >= best_cv_score:
            best_cv_score = rf_result["best_score"]
            best_model_name = "Random Forest"

    if "logistic_regression" in tuned:
        lr_result = tuned["logistic_regression"]
        lr_pipeline: Pipeline = lr_result["pipeline"]
        store.save_pipeline(lr_pipeline, "logistic_regression")

        if lr_result["best_score"] > best_cv_score:
            best_cv_score = lr_result["best_score"]
            best_model_name = "Logistic Regression"

    # Legacy scaler.pkl — create a dummy so predict.py doesn't crash on older artifact loads
    # In V3, scaling is embedded inside the Pipeline — no standalone scaler needed
    dummy_scaler_path = artifacts_dir / "scaler.pkl"
    if not dummy_scaler_path.exists():
        from sklearn.preprocessing import StandardScaler
        dummy = StandardScaler()
        dummy.fit([[0]*len(settings.CONTINUOUS_COLS)])
        joblib.dump(dummy, dummy_scaler_path)

    # Cache splits for evaluate.py
    joblib.dump((X_train, X_test, y_train, y_test),
                artifacts_dir / "data_splits.pkl")
    log.info("Data splits cached for evaluation script.")

    # 6. Build and save training summary
    best_params_serializable = {}
    if "random_forest" in tuned:
        for k, v in tuned["random_forest"]["best_params"].items():
            clean_k = k.replace("classifier__", "")
            best_params_serializable[clean_k] = (
                int(v) if isinstance(v, (int, np.integer)) else v
            )

    summary = {
        "run_timestamp": datetime.datetime.now().isoformat(),
        "best_model": best_model_name,
        "best_f1": round(float(best_cv_score), 4),
        "best_roc_auc": round(float(np.mean(rf_cv["roc_auc"])) if rf_cv else 0.0, 4),
        "best_params": best_params_serializable,
        "top_features": top_features,
        "failure_rate_in_dataset": round(failure_rate, 4),
        "sklearn_pipeline": True,
        "v3_upgrade": True
    }
    summary_path = artifacts_dir / "model_summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)
    log.info(f"Training summary saved → {summary_path}")

    # Sync to RAG docs
    docs_dir = settings.DOCS_DIR
    shutil.copy(summary_path, docs_dir / "model_summary.json")
    log.info(f"Model summary synced to RAG corpus: {docs_dir / 'model_summary.json'}")

    # 7. MLflow logging
    for name, result in tuned.items():
        _log_to_mlflow(
            model_name=name,
            params=result["best_params"],
            cv_score=result["best_score"],
            lr_cv=lr_cv if name == "logistic_regression" else None,
            rf_cv=rf_cv if name == "random_forest" else None,
            artifacts_dir=artifacts_dir
        )

    log.info("Training pipeline completed successfully. ✓")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train opsFlow ML classifiers.")
    parser.add_argument("--model", choices=["random_forest", "logistic_regression", "both"],
                        default="both", help="Which model(s) to train.")
    args = parser.parse_args()
    run_train(model_name=args.model)
