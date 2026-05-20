"""
models/evaluate.py — Task 3: Model Evaluation (V3)
===================================================
V3 additions:
  • Correlation heatmap (feature_correlation_heatmap.png)
  • Failure distribution bar chart (failure_distribution.png)
  • Train vs test overfitting delta table (printed + saved in JSON)
  • Python logging module
  • Reads pipeline artifacts (no separate scaler)
"""
import os
import sys
import json
import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import settings
from core.logger import get_logger

log = get_logger("models.evaluate")

try:
    import shap
except ImportError:
    shap = None


# ---------------------------------------------------------------------------
# Helper: compute metrics dict
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy":      round(float(accuracy_score(y_true, y_pred)),                    4),
        "recall":        round(float(recall_score(y_true, y_pred)),                       4),
        "precision":     round(float(precision_score(y_true, y_pred, zero_division=0)),   4),
        "f1_score":      round(float(f1_score(y_true, y_pred)),                          4),
        "roc_auc":       round(float(roc_auc_score(y_true, y_prob)),                     4),
        "avg_precision": round(float(average_precision_score(y_true, y_prob)),           4),
    }


# ---------------------------------------------------------------------------
# MAIN EVALUATION FUNCTION
# ---------------------------------------------------------------------------

def run_evaluation():
    log.info("=" * 52)
    log.info("  TASK 3 — PREDICTIVE ML PIPELINE: EVALUATION (V3)")
    log.info("=" * 52)

    artifacts_dir = settings.MODEL_ARTIFACTS_DIR
    splits_path   = artifacts_dir / "data_splits.pkl"
    plots_dir     = artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not splits_path.exists():
        raise FileNotFoundError(
            f"Cached splits not found at {splits_path}. Run `--train` first."
        )

    # ── 1. Load splits and pipelines ────────────────────────────────────────
    log.info("Loading cached splits and pipeline artifacts...")
    splits = joblib.load(splits_path)
    if len(splits) == 4:
        X_train, X_test, y_train, y_test = splits
    elif len(splits) == 6:
        X_train, X_test, y_train, y_test, _, _ = splits
    else:
        raise ValueError(
            f"Unexpected data_splits.pkl format: {len(splits)} items (expected 4 or 6)."
        )

    # Prefer V3 pipeline artifacts
    rf_path  = artifacts_dir / "random_forest_pipeline.pkl"
    lr_path  = artifacts_dir / "logistic_regression_pipeline.pkl"
    if not rf_path.exists():
        rf_path = artifacts_dir / "random_forest.pkl"
    if not lr_path.exists():
        lr_path = artifacts_dir / "logistic_regression.pkl"

    rf_model = joblib.load(rf_path)
    lr_model = joblib.load(lr_path)
    log.info(f"Loaded RF from:  {rf_path.name}")
    log.info(f"Loaded LR from:  {lr_path.name}")

    # ── 2. TRAIN metrics (overfitting check) ────────────────────────────────
    log.info("Computing train-set metrics for overfitting analysis...")
    lr_train_preds = lr_model.predict(X_train)
    lr_train_probs = lr_model.predict_proba(X_train)[:, 1]
    rf_train_preds = rf_model.predict(X_train)
    rf_train_probs = rf_model.predict_proba(X_train)[:, 1]

    lr_train_metrics = _metrics(y_train, lr_train_preds, lr_train_probs)
    rf_train_metrics = _metrics(y_train, rf_train_preds, rf_train_probs)

    # ── 3. TEST metrics ─────────────────────────────────────────────────────
    log.info("Evaluating Tuned Logistic Regression on holdout test set...")
    lr_preds = lr_model.predict(X_test)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    lr_metrics = _metrics(y_test, lr_preds, lr_probs)

    log.info("Evaluating Tuned Random Forest (Balanced) on holdout test set...")
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_metrics = _metrics(y_test, rf_preds, rf_probs)

    # ── 4. SMOTE comparison (must use same preprocessing as tuned RF pipeline) ─
    log.info("Training SMOTE-oversampled RF for comparison (on preprocessed features)...")
    rf_smote_metrics = None
    rf_smote_probs = None
    rf_smote_preds = None
    try:
        if hasattr(rf_model, "named_steps") and "preprocessor" in rf_model.named_steps:
            pre = clone(rf_model.named_steps["preprocessor"])
            X_train_t = pre.fit_transform(X_train, y_train)
            X_test_t = pre.transform(X_test)
            smote = SMOTE(random_state=settings.RANDOM_STATE)
            X_train_sm, y_train_sm = smote.fit_resample(X_train_t, y_train)

            rf_smote = RandomForestClassifier(random_state=settings.RANDOM_STATE)
            try:
                with open(artifacts_dir / "model_summary.json", "r") as f:
                    summary = json.load(f)
                    best_params = {
                        k: v for k, v in summary.get("best_params", {}).items()
                        if k != "class_weight"
                    }
                    rf_smote.set_params(**best_params)
            except Exception:
                pass
            rf_smote.fit(X_train_sm, y_train_sm)
            rf_smote_preds = rf_smote.predict(X_test_t)
            rf_smote_probs = rf_smote.predict_proba(X_test_t)[:, 1]
            rf_smote_metrics = _metrics(y_test, rf_smote_preds, rf_smote_probs)
        else:
            log.warning("RF artifact is not a sklearn Pipeline; skipping SMOTE comparison.")
    except Exception as e:
        log.warning(f"SMOTE comparison skipped due to error: {e}")

    # ── 5. PLOTS ────────────────────────────────────────────────────────────
    log.info(f"Generating performance plots → {plots_dir}")

    # A. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    PrecisionRecallDisplay.from_predictions(
        y_test, lr_probs,
        name=f"Logistic Regression (AP={lr_metrics['avg_precision']})", color="salmon")
    PrecisionRecallDisplay.from_predictions(
        y_test, rf_probs,
        name=f"RF-Balanced (AP={rf_metrics['avg_precision']})", color="steelblue", ax=plt.gca())
    if rf_smote_probs is not None and rf_smote_metrics is not None:
        PrecisionRecallDisplay.from_predictions(
            y_test, rf_smote_probs,
            name=f"RF-SMOTE (AP={rf_smote_metrics['avg_precision']})", color="forestgreen", ax=plt.gca())
    plt.title("Precision-Recall Curves Comparison")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "precision_recall_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Saved: precision_recall_comparison.png")

    # B. ROC Curves
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(
        y_test, lr_probs,
        name=f"Logistic Regression (AUC={lr_metrics['roc_auc']})", color="salmon")
    RocCurveDisplay.from_predictions(
        y_test, rf_probs,
        name=f"RF-Balanced (AUC={rf_metrics['roc_auc']})", color="steelblue", ax=plt.gca())
    if rf_smote_probs is not None and rf_smote_metrics is not None:
        RocCurveDisplay.from_predictions(
            y_test, rf_smote_probs,
            name=f"RF-SMOTE (AUC={rf_smote_metrics['roc_auc']})", color="forestgreen", ax=plt.gca())
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.title("ROC Curves Comparison")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "roc_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Saved: roc_comparison.png")

    # C. Confusion Matrices
    lr_cm = confusion_matrix(y_test, lr_preds)
    rf_cm = confusion_matrix(y_test, rf_preds)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(lr_cm, display_labels=["No Failure", "Failure"]).plot(
        cmap="Reds", ax=axes[0])
    axes[0].set_title("Logistic Regression")
    ConfusionMatrixDisplay(rf_cm, display_labels=["No Failure", "Failure"]).plot(
        cmap="Blues", ax=axes[1])
    axes[1].set_title("Random Forest (Tuned)")
    plt.suptitle("Holdout Confusion Matrices", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrices_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Saved: confusion_matrices_comparison.png")

    # D. Feature Correlation Heatmap (V3 NEW)
    log.info("Generating feature correlation heatmap...")
    try:
        corr_cols = settings.FEATURES_ORDER + [settings.TARGET_COL]
        corr_df   = X_train.copy()
        corr_df[settings.TARGET_COL] = y_train.values
        # Only keep numeric columns that exist
        corr_df = corr_df[[c for c in corr_cols if c in corr_df.columns]]
        corr_matrix = corr_df.corr()

        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            vmin=-1, vmax=1,
            annot_kws={"size": 8}
        )
        plt.title("Feature Correlation Heatmap", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_correlation_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()
        log.info("Saved: feature_correlation_heatmap.png")
    except Exception as e:
        log.warning(f"Could not generate correlation heatmap: {e}")

    # E. Failure Distribution Bar Chart (V3 NEW)
    log.info("Generating failure distribution chart...")
    try:
        full_splits = joblib.load(splits_path)
        X_full_train = full_splits[0]
        y_full_train = full_splits[2]
        all_y = pd.concat([y_full_train, full_splits[3]])

        counts = all_y.value_counts().sort_index()
        labels = ["No Failure (0)", "Failure (1)"]
        colors = ["#4C9BE8", "#E84C4C"]

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", width=0.45)
        ax.set_title("Dataset Failure Class Distribution", fontsize=14, pad=12)
        ax.set_ylabel("Sample Count")
        ax.set_xlabel("Class Label")
        for bar, count in zip(bars, counts.values):
            pct = count / counts.sum() * 100
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 30,
                    f"{count:,}\n({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylim(0, counts.max() * 1.2)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(plots_dir / "failure_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()
        log.info("Saved: failure_distribution.png")
    except Exception as e:
        log.warning(f"Could not generate failure distribution chart: {e}")

    # F. SHAP Explainability
    if shap is not None:
        log.info("Generating SHAP explainability plots (50-sample background)...")
        try:
            # Get the classifier step from pipeline
            rf_clf = rf_model
            if hasattr(rf_model, "named_steps"):
                rf_clf = rf_model.named_steps["classifier"]
                # Transform test data through preprocessor for SHAP
                preprocessor = rf_model.named_steps["preprocessor"]
                X_test_transformed = preprocessor.transform(X_test.iloc[:50])
                feature_names = (
                    list(X_test.columns[:X_test_transformed.shape[1]])
                    if X_test_transformed.shape[1] == len(X_test.columns)
                    else [f"feature_{i}" for i in range(X_test_transformed.shape[1])]
                )
                X_test_sample = X_test_transformed
            else:
                X_test_sample = X_test.iloc[:50]
                feature_names = X_test.columns.tolist()

            explainer   = shap.TreeExplainer(rf_clf)
            shap_values = explainer.shap_values(X_test_sample)

            if isinstance(shap_values, list):
                sv = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                sv = shap_values[:, :, 1]
            else:
                sv = shap_values

            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1:
                base_val = base_val[1]

            # Beeswarm
            plt.figure(figsize=(10, 6))
            shap.summary_plot(sv, X_test_sample, feature_names=feature_names, show=False)
            plt.title("SHAP Beeswarm — Failure Drivers", fontsize=14, pad=15)
            plt.savefig(plots_dir / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
            plt.close()
            log.info("Saved: shap_beeswarm.png")

            # Force plot for first correctly predicted failure
            rf_sample_preds = rf_clf.predict(X_test_sample)
            y_sample        = y_test.iloc[:50].values
            correct_fails   = np.where((y_sample == 1) & (rf_sample_preds == 1))[0]
            if len(correct_fails) > 0:
                idx = correct_fails[0]
                plt.figure(figsize=(12, 4))
                shap.force_plot(base_val, sv[idx], X_test_sample[idx]
                                if isinstance(X_test_sample, np.ndarray)
                                else X_test_sample.iloc[idx],
                                matplotlib=True, show=False)
                plt.title(f"SHAP Force Plot — Instance {idx}", fontsize=12, pad=20)
                plt.savefig(plots_dir / "shap_force_plot.png", dpi=300, bbox_inches="tight")
                plt.close()
                log.info("Saved: shap_force_plot.png")
        except Exception as e:
            log.warning(f"SHAP plot generation failed: {e}")
    else:
        log.warning("SHAP not installed; skipping SHAP plots.")

    # ── 6. Overfitting Analysis Table ────────────────────────────────────────
    log.info("Computing overfitting delta (train vs test)...")
    overfit = {
        "lr": {
            "train_f1":  lr_train_metrics["f1_score"],
            "test_f1":   lr_metrics["f1_score"],
            "delta_f1":  round(lr_train_metrics["f1_score"] - lr_metrics["f1_score"], 4),
            "train_auc": lr_train_metrics["roc_auc"],
            "test_auc":  lr_metrics["roc_auc"],
            "delta_auc": round(lr_train_metrics["roc_auc"] - lr_metrics["roc_auc"], 4),
        },
        "rf": {
            "train_f1":  rf_train_metrics["f1_score"],
            "test_f1":   rf_metrics["f1_score"],
            "delta_f1":  round(rf_train_metrics["f1_score"] - rf_metrics["f1_score"], 4),
            "train_auc": rf_train_metrics["roc_auc"],
            "test_auc":  rf_metrics["roc_auc"],
            "delta_auc": round(rf_train_metrics["roc_auc"] - rf_metrics["roc_auc"], 4),
        },
    }

    # ── 7. Print results ─────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("            EVALUATION RESULTS (HOLDOUT TEST SET)               ")
    log.info("=" * 70)
    print("\n" + "=" * 70)
    print("         EVALUATION RESULTS COMPARISON — V3                        ")
    print("=" * 70)
    if rf_smote_metrics is not None:
        print(f"{'Metric':<26} | {'Tuned LR':>9} | {'RF (Balanced)':>14} | {'RF (SMOTE)':>11}")
        print(f"{'-'*26}-+-{'-'*9}-+-{'-'*14}-+-{'-'*11}")
        for metric, label in [
            ("accuracy",      "Accuracy"),
            ("recall",        "Recall"),
            ("precision",     "Precision"),
            ("f1_score",      "F1-Score"),
            ("roc_auc",       "ROC-AUC"),
            ("avg_precision", "Avg Precision (AP)"),
        ]:
            print(
                f"{label:<26} | {lr_metrics[metric]:>9.4f} | {rf_metrics[metric]:>14.4f} | "
                f"{rf_smote_metrics[metric]:>11.4f}"
            )
    else:
        print(f"{'Metric':<26} | {'Tuned LR':>9} | {'RF (Balanced)':>14} | {'RF (SMOTE)':>11}")
        print(f"{'-'*26}-+-{'-'*9}-+-{'-'*14}-+-{'-'*11}")
        for metric, label in [
            ("accuracy",      "Accuracy"),
            ("recall",        "Recall"),
            ("precision",     "Precision"),
            ("f1_score",      "F1-Score"),
            ("roc_auc",       "ROC-AUC"),
            ("avg_precision", "Avg Precision (AP)"),
        ]:
            print(f"{label:<26} | {lr_metrics[metric]:>9.4f} | {rf_metrics[metric]:>14.4f} | {'N/A':>11}")
    print("=" * 70)

    print("\n── OVERFITTING ANALYSIS (Train Δ Test) ──")
    print(f"{'Model':<22} | Train F1 | Test F1  | ΔF1    | Train AUC | Test AUC  | ΔAUC")
    print(f"{'-'*22}-+---------+----------+--------+-----------+-----------+------")
    for model_key, label in [("lr", "Logistic Regression"), ("rf", "Random Forest")]:
        o = overfit[model_key]
        flag = "⚠️" if abs(o["delta_f1"]) > 0.15 else "✅"
        print(f"{label:<22} | {o['train_f1']:.4f}   | {o['test_f1']:.4f}   | {o['delta_f1']:+.4f} | {o['train_auc']:.4f}    | {o['test_auc']:.4f}    | {o['delta_auc']:+.4f} {flag}")
    print()

    print("── INTERPRETATION ──")
    print(
        f"• Logistic Regression: recall {lr_metrics['recall']:.2f}, precision {lr_metrics['precision']:.2f} — "
        "many false alarms; use when missing failures is unacceptable."
    )
    print(
        f"• Random Forest (balanced): F1 {rf_metrics['f1_score']:.2f}, precision {rf_metrics['precision']:.2f} — "
        "recommended default when false alarms are costly."
    )
    if rf_smote_metrics:
        print(
            f"• RF + SMOTE: recall +{rf_smote_metrics['recall'] - rf_metrics['recall']:.4f} vs balanced "
            f"at cost of lower precision — use when recall must dominate."
        )
    rf_delta = overfit["rf"]["delta_f1"]
    print(
        f"• Overfitting: RF train-test ΔF1 = {rf_delta:+.4f}; "
        "if |Δ| > 0.15, consider stronger regularization in config.yaml."
    )
    print()

    # ── 8. Save evaluation summary JSON ─────────────────────────────────────
    eval_summary = {
        "evaluation_timestamp":  datetime.datetime.now().isoformat(),
        "logistic_regression":   lr_metrics,
        "random_forest_balanced": rf_metrics,
        "random_forest_smote":   rf_smote_metrics,
        "overfitting_analysis":  overfit,
        "recall_comparison": (
            {
                "balanced_strategy_recall": rf_metrics["recall"],
                "smote_strategy_recall":    rf_smote_metrics["recall"],
                "delta_recall":             round(rf_smote_metrics["recall"] - rf_metrics["recall"], 4),
            }
            if rf_smote_metrics is not None
            else {
                "balanced_strategy_recall": rf_metrics["recall"],
                "smote_strategy_recall":    None,
                "delta_recall":             None,
                "note": "SMOTE branch skipped (no pipeline preprocessor or error).",
            }
        ),
        "plots_generated": [
            "precision_recall_comparison.png",
            "roc_comparison.png",
            "confusion_matrices_comparison.png",
            "feature_correlation_heatmap.png",
            "failure_distribution.png",
            "shap_beeswarm.png",
            "shap_force_plot.png",
        ]
    }
    eval_path = artifacts_dir / "evaluation_summary.json"
    with open(eval_path, "w", encoding="utf-8") as ef:
        json.dump(eval_summary, ef, indent=2)
    log.info(f"Evaluation summary saved → {eval_path}")
    log.info("Evaluation run completed successfully. ✓")


if __name__ == "__main__":
    run_evaluation()
