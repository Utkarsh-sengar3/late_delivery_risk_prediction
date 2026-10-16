"""Phase 4: compare Random Forest and Histogram Gradient Boosting.

Random Forest uses the sparse one-hot encoded matrix directly.
Histogram Gradient Boosting uses a train-fitted TruncatedSVD projection because
the encoded matrix is sparse/high-dimensional and HistGradientBoosting requires
dense input. Both models use the same train/validation rows and metrics, but
their representations differ; interpret the comparison accordingly.
"""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay,
    PrecisionRecallDisplay,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "phase2"
PHASE2_ARTIFACTS = ROOT / "artifacts" / "phase2"
ARTIFACT_DIR = ROOT / "artifacts" / "phase4"
REPORT_DIR = ROOT / "reports" / "phase4"
TARGET = "Late_delivery_risk"

def load_split(name):
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run Phase 2 first.")
    frame = pd.read_csv(path)
    if TARGET not in frame:
        raise ValueError(f"Target {TARGET!r} missing in {path}.")
    X = frame.drop(columns=[TARGET])
    y = pd.to_numeric(frame[TARGET], errors="raise").astype("int8")
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"Non-binary labels found in {path}.")
    return X, y

def score_model(name, model, X_val, y_val, model_input):
    probabilities = model.predict_proba(model_input)[:, 1]
    predictions = (probabilities >= 0.5).astype("int8")
    return {
        "model": name,
        "threshold": 0.5,
        "validation_accuracy": float(accuracy_score(y_val, predictions)),
        "validation_precision": float(precision_score(y_val, predictions, zero_division=0)),
        "validation_recall": float(recall_score(y_val, predictions, zero_division=0)),
        "validation_f1": float(f1_score(y_val, predictions, zero_division=0)),
        "validation_roc_auc": float(roc_auc_score(y_val, probabilities)),
        "validation_average_precision": float(average_precision_score(y_val, probabilities)),
        "predicted_positive_rate": float(predictions.mean()),
        "actual_positive_rate": float(y_val.mean()),
    }, probabilities, predictions

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor_path = PHASE2_ARTIFACTS / "preprocessor.joblib"
    if not preprocessor_path.exists():
        raise FileNotFoundError("Missing Phase 2 preprocessor. Run Phase 2 first.")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")
    preprocessor = joblib.load(preprocessor_path)

    print("Loading Phase 2 preprocessor and transforming train/validation...")
    X_train_sparse = preprocessor.transform(X_train)
    X_val_sparse = preprocessor.transform(X_val)
    print(f"Encoded train shape: {X_train_sparse.shape}")
    print(f"Encoded validation shape: {X_val_sparse.shape}")

    results = []
    predictions_by_model = {}

    # Model A: sparse-compatible Random Forest.
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=22,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    print("\nTraining Random Forest (may take several minutes)...")
    with warnings.catch_warnings(record=True) as caught_rf:
        warnings.simplefilter("always")
        rf.fit(X_train_sparse, y_train)
    rf_metrics, rf_prob, rf_pred = score_model(
        "RandomForest", rf, X_val, y_val, X_val_sparse
    )
    rf_metrics["warnings"] = [str(w.message) for w in caught_rf]
    rf_metrics["representation"] = "Phase 2 sparse one-hot encoded features"
    results.append(rf_metrics)
    predictions_by_model["RandomForest"] = (rf_prob, rf_pred)
    joblib.dump(rf, ARTIFACT_DIR / "random_forest.joblib")

    # Model B: SVD + HistGradientBoosting. Fit SVD on training data only.
    n_components = min(80, X_train_sparse.shape[1] - 1, X_train_sparse.shape[0] - 1)
    if n_components < 2:
        raise ValueError("Not enough encoded dimensions for TruncatedSVD.")
    print(f"\nFitting TruncatedSVD with {n_components} components (train data only)...")
    svd = TruncatedSVD(n_components=n_components, n_iter=5, random_state=42)
    X_train_svd = svd.fit_transform(X_train_sparse)
    X_val_svd = svd.transform(X_val_sparse)
    print(f"SVD train shape: {X_train_svd.shape}; validation shape: {X_val_svd.shape}")

    hgb = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
    )
    print("Training Histogram Gradient Boosting on reduced features...")
    with warnings.catch_warnings(record=True) as caught_hgb:
        warnings.simplefilter("always")
        hgb.fit(X_train_svd, y_train)
    hgb_metrics, hgb_prob, hgb_pred = score_model(
        "HistGradientBoosting_SVD", hgb, X_val, y_val, X_val_svd
    )
    hgb_metrics["warnings"] = [str(w.message) for w in caught_hgb]
    hgb_metrics["representation"] = f"TruncatedSVD projection ({n_components} components); SVD fitted on train only"
    hgb_metrics["svd_explained_variance_ratio_sum"] = float(svd.explained_variance_ratio_.sum())
    results.append(hgb_metrics)
    predictions_by_model["HistGradientBoosting_SVD"] = (hgb_prob, hgb_pred)
    joblib.dump(svd, ARTIFACT_DIR / "truncated_svd.joblib")
    joblib.dump(hgb, ARTIFACT_DIR / "hist_gradient_boosting_svd.joblib")
    joblib.dump(preprocessor, ARTIFACT_DIR / "preprocessor.joblib")

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    (REPORT_DIR / "model_comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    prediction_frame = pd.DataFrame({
        "y_true": y_val.to_numpy(),
        "random_forest_probability": rf_prob,
        "random_forest_prediction_0_5": rf_pred,
        "hist_gradient_boosting_probability": hgb_prob,
        "hist_gradient_boosting_prediction_0_5": hgb_pred,
    })
    prediction_frame.to_csv(REPORT_DIR / "validation_predictions.csv", index=False)

    # Comparison bar plot
    metric_cols = [
        "validation_accuracy", "validation_precision", "validation_recall",
        "validation_f1", "validation_roc_auc", "validation_average_precision"
    ]
    plot_df = metrics_df.set_index("model")[metric_cols].T
    ax = plot_df.plot(kind="bar", figsize=(12, 6))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Phase 4 Model Comparison — Validation Set")
    ax.set_xlabel("Metric")
    ax.legend(title="Model", loc="lower right")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_comparison.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_val, rf_prob, name="Random Forest", ax=ax)
    RocCurveDisplay.from_predictions(y_val, hgb_prob, name="HistGradientBoosting + SVD", ax=ax)
    ax.set_title("Validation ROC Curves")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "roc_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_val, rf_prob, name="Random Forest", ax=ax)
    PrecisionRecallDisplay.from_predictions(y_val, hgb_prob, name="HistGradientBoosting + SVD", ax=ax)
    ax.set_title("Validation Precision–Recall Curves")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "precision_recall_curves.png", dpi=160)
    plt.close(fig)

    best = max(results, key=lambda item: item["validation_f1"])
    lines = [
        "# Phase 4 — Model Family Comparison",
        "",
        "Models were compared on the same validation split. The held-out test split was not loaded or evaluated.",
        "",
        "## Validation results",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['validation_accuracy']:.4f} | "
            f"{r['validation_precision']:.4f} | {r['validation_recall']:.4f} | "
            f"{r['validation_f1']:.4f} | {r['validation_roc_auc']:.4f} | "
            f"{r['validation_average_precision']:.4f} |"
        )
    lines += [
        "",
        f"## Best validation F1 at threshold 0.50: `{best['model']}`",
        "",
        "This is a provisional selection based only on validation F1. Operational threshold selection belongs in Phase 5.",
        "",
        "## Method notes",
        "",
        "- Random Forest uses the sparse one-hot encoded Phase 2 features.",
        f"- Histogram Gradient Boosting uses {n_components} TruncatedSVD components because the estimator requires dense input.",
        f"- The SVD was fitted on training data only. Explained variance ratio sum: {svd.explained_variance_ratio_.sum():.4f}.",
        "- Because the two models use different representations, this is a practical model-family comparison rather than a perfectly controlled feature-representation experiment.",
        "- Class weights are enabled for both models.",
        "- Threshold 0.50 is used for consistent baseline metrics; it is not claimed to be optimal.",
        "- Random stratified splitting does not establish future-time generalization.",
        "- Confirm candidate features are available at the actual prediction time before deployment.",
    ]
    (REPORT_DIR / "PHASE_4_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nPHASE 4 COMPLETE")
    for r in results:
        print(
            f"{r['model']}: F1={r['validation_f1']:.4f} | "
            f"ROC-AUC={r['validation_roc_auc']:.4f} | "
            f"Recall={r['validation_recall']:.4f} | "
            f"Precision={r['validation_precision']:.4f}"
        )
    print(f"Best validation F1 at threshold 0.50: {best['model']}")
    print(f"Comparison report: {REPORT_DIR / 'PHASE_4_REPORT.md'}")
    print("Note: test set was not evaluated.")

if __name__ == "__main__":
    main()
