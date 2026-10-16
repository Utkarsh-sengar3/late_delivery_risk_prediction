"""Phase 3: train and evaluate a Logistic Regression baseline.

Uses Phase 2 split CSVs and the Phase 2 preprocessor. The validation set is
used for baseline reporting; the held-out test set is deliberately untouched.
"""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    RocCurveDisplay, PrecisionRecallDisplay,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "phase2"
ARTIFACT_DIR = ROOT / "artifacts" / "phase3"
REPORT_DIR = ROOT / "reports" / "phase3"
TARGET = "Late_delivery_risk"

def load_split(name):
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/phase2_preprocess.py` first."
        )
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"{path} does not contain target column {TARGET!r}.")
    X = frame.drop(columns=[TARGET])
    y = pd.to_numeric(frame[TARGET], errors="coerce")
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"{path} contains missing or non-binary target labels.")
    return X, y.astype("int8")

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    preprocessor_path = ROOT / "artifacts" / "phase2" / "preprocessor.joblib"
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Missing {preprocessor_path}. Run `python scripts/phase2_preprocess.py` first."
        )

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")
    # Deliberately do not load or score test.csv in Phase 3.
    preprocessor = joblib.load(preprocessor_path)

    print("Transforming training and validation data...")
    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    print("Training Logistic Regression baseline...")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train_t, y_train)
    warning_messages = [str(w.message) for w in caught]

    probabilities = model.predict_proba(X_val_t)[:, 1]
    threshold = 0.5
    predictions = (probabilities >= threshold).astype("int8")

    metrics = {
        "model": "LogisticRegression",
        "class_weight": "balanced",
        "threshold": threshold,
        "evaluation_split": "validation",
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)),
        "accuracy": float(accuracy_score(y_val, predictions)),
        "precision": float(precision_score(y_val, predictions, zero_division=0)),
        "recall": float(recall_score(y_val, predictions, zero_division=0)),
        "f1": float(f1_score(y_val, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, probabilities)),
        "average_precision_pr_auc": float(average_precision_score(y_val, probabilities)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_validation": float(y_val.mean()),
        "predicted_positive_rate_validation": float(predictions.mean()),
        "warnings": warning_messages,
        "test_set_evaluated": False,
    }

    joblib.dump(model, ARTIFACT_DIR / "logistic_regression_baseline.joblib")
    joblib.dump(preprocessor, ARTIFACT_DIR / "preprocessor.joblib")
    with (REPORT_DIR / "validation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(
        y_val, predictions, labels=[0, 1],
        target_names=["On-time / lower risk (0)", "Late risk (1)"],
        zero_division=0,
    )
    (REPORT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    pd.DataFrame({
        "y_true": y_val.to_numpy(),
        "probability_late_risk": probabilities,
        "prediction_at_0_5": predictions,
    }).to_csv(REPORT_DIR / "validation_predictions.csv", index=False)

    cm = confusion_matrix(y_val, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm)
    ax.set_title("Logistic Regression — Validation Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])
    for (i, j), value in __import__("numpy").ndenumerate(cm):
        ax.text(j, i, f"{value:,}", ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_val, probabilities, ax=ax)
    ax.set_title("Logistic Regression — Validation ROC Curve")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_val, probabilities, ax=ax)
    ax.set_title("Logistic Regression — Validation Precision–Recall Curve")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "precision_recall_curve.png", dpi=160)
    plt.close(fig)

    md = f"""# Phase 3 — Logistic Regression Baseline

## Validation metrics

| Metric | Value |
|---|---:|
| Accuracy | {metrics['accuracy']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1-score | {metrics['f1']:.4f} |
| ROC-AUC | {metrics['roc_auc']:.4f} |
| Average precision (PR-AUC summary) | {metrics['average_precision_pr_auc']:.4f} |
| Threshold | {threshold:.2f} |

## Setup
- Training rows: {len(y_train):,}
- Validation rows: {len(y_val):,}
- Class weighting: `balanced`
- Preprocessor: loaded from Phase 2 and applied consistently
- Evaluation split: validation only
- Held-out test set evaluated: **No**

## Interpretation notes
- Class weighting is intended to reduce the tendency to ignore the less frequent class; it does not guarantee better operational performance.
- At threshold 0.50, compare precision and recall together. False negatives can represent late orders missed by the system; false positives can create unnecessary interventions.
- The threshold is a baseline default, not an operationally optimized decision threshold.
- Phase 4 will compare stronger model families using validation results.
- Because Phase 2 used a random stratified split, these results do not prove performance on future time periods.
- Confirm every feature is available at the actual prediction moment before operational deployment.

## Classification report

```text
{report}
```
"""
    (REPORT_DIR / "PHASE_3_REPORT.md").write_text(md, encoding="utf-8")

    print("\nPHASE 3 COMPLETE")
    print("Model: Logistic Regression (class_weight='balanced')")
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation Precision: {metrics['precision']:.4f}")
    print(f"Validation Recall: {metrics['recall']:.4f}")
    print(f"Validation F1: {metrics['f1']:.4f}")
    print(f"Validation ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Validation PR-AUC summary: {metrics['average_precision_pr_auc']:.4f}")
    print(f"Saved model: {ARTIFACT_DIR / 'logistic_regression_baseline.joblib'}")
    print(f"Saved report: {REPORT_DIR / 'PHASE_3_REPORT.md'}")
    print("Note: test set was not evaluated.")

if __name__ == "__main__":
    main()
