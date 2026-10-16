"""Phase 5: choose a decision threshold on validation data, then evaluate once on test.

The selected model is chosen using Phase 4 validation F1 at threshold 0.50:
Random Forest. The threshold is then selected on validation predictions by
maximizing F1, with a minimum-recall constraint configurable below. Test data
is evaluated only after model/threshold selection is complete.
"""
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "phase2"
ARTIFACT_DIR = ROOT / "artifacts" / "phase5"
REPORT_DIR = ROOT / "reports" / "phase5"
PHASE2_PREPROCESSOR = ROOT / "artifacts" / "phase2" / "preprocessor.joblib"
PHASE3_MODEL = ROOT / "artifacts" / "phase3" / "logistic_regression_baseline.joblib"
PHASE4_RF = ROOT / "artifacts" / "phase4" / "random_forest.joblib"
PHASE4_SVD = ROOT / "artifacts" / "phase4" / "truncated_svd.joblib"
PHASE4_HGB = ROOT / "artifacts" / "phase4" / "hist_gradient_boosting_svd.joblib"
TARGET = "Late_delivery_risk"

# Set to 0.60 if the use case requires at least 60% recall. None means maximize F1.
MIN_RECALL = None

def load_split(name):
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run Phase 2 first.")
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"Target column {TARGET!r} missing from {path}.")
    X = frame.drop(columns=[TARGET])
    y = pd.to_numeric(frame[TARGET], errors="raise").astype("int8")
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"Non-binary target labels in {path}.")
    return X, y

def evaluate(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype("int8")
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "predicted_positive_rate": float(pred.mean()),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }

def select_threshold(y_true, probabilities, min_recall=None):
    candidates = np.unique(np.concatenate([
        np.linspace(0.05, 0.95, 181),
        np.asarray([0.5]),
        np.asarray(probabilities, dtype=float),
    ]))
    rows = []
    for threshold in candidates:
        pred = (probabilities >= threshold).astype("int8")
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "predicted_positive_rate": float(pred.mean()),
        })
    curve = pd.DataFrame(rows).drop_duplicates(subset=["threshold"]).sort_values("threshold")
    eligible = curve if min_recall is None else curve[curve["recall"] >= min_recall]
    if eligible.empty:
        raise ValueError(
            f"No threshold met minimum recall {min_recall}. Set MIN_RECALL=None or lower it."
        )
    # Maximize F1; tie-break in favor of higher recall, then higher threshold.
    eligible = eligible.sort_values(["f1", "recall", "threshold"], ascending=[False, False, False])
    return float(eligible.iloc[0]["threshold"]), curve

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for p in [PHASE2_PREPROCESSOR, PHASE4_RF]:
        if not p.exists():
            raise FileNotFoundError(f"Required artifact missing: {p}. Run prior phases first.")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")
    X_test, y_test = load_split("test")

    preprocessor = joblib.load(PHASE2_PREPROCESSOR)
    rf = joblib.load(PHASE4_RF)

    print("Transforming validation and test data...")
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    # Validation is used to choose the threshold. Test predictions are held until
    # after the selected threshold is locked.
    print("Selecting threshold using validation predictions...")
    val_prob = rf.predict_proba(X_val_t)[:, 1]
    selected_threshold, threshold_curve = select_threshold(y_val, val_prob, MIN_RECALL)
    val_metrics_default = evaluate(y_val, val_prob, 0.5)
    val_metrics_selected = evaluate(y_val, val_prob, selected_threshold)

    # Final test evaluation exactly once, after model and threshold selection.
    print("Evaluating selected model and locked threshold on held-out test set...")
    test_prob = rf.predict_proba(X_test_t)[:, 1]
    test_metrics = evaluate(y_test, test_prob, selected_threshold)
    test_pred = (test_prob >= selected_threshold).astype("int8")
    val_pred = (val_prob >= selected_threshold).astype("int8")

    # Persist the chosen model and preprocessing artifact together.
    joblib.dump({
        "model_name": "RandomForest",
        "model": rf,
        "preprocessor": preprocessor,
        "threshold": selected_threshold,
        "selection_metric": "validation F1",
        "minimum_recall_constraint": MIN_RECALL,
    }, ARTIFACT_DIR / "selected_model_bundle.joblib")
    threshold_curve.to_csv(REPORT_DIR / "threshold_curve.csv", index=False)

    all_metrics = {
        "selected_model": "RandomForest",
        "model_selection_basis": "Phase 4 validation F1 at threshold 0.50",
        "threshold_selection_metric": "validation F1",
        "minimum_recall_constraint": MIN_RECALL,
        "selected_threshold": selected_threshold,
        "validation_at_default_threshold_0_5": val_metrics_default,
        "validation_at_selected_threshold": val_metrics_selected,
        "held_out_test_at_selected_threshold": test_metrics,
        "test_used_for_model_or_threshold_selection": False,
        "split_method": "random stratified split",
        "limitations": [
            "Random stratified split does not establish future-time generalization.",
            "Threshold optimizes F1 on validation data and may not reflect business costs.",
            "Candidate feature availability at prediction time must be verified.",
        ],
    }
    (REPORT_DIR / "phase5_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    pd.DataFrame({
        "y_true": y_val.to_numpy(),
        "probability_late_risk": val_prob,
        "prediction_at_selected_threshold": val_pred,
    }).to_csv(REPORT_DIR / "validation_predictions_selected_threshold.csv", index=False)
    pd.DataFrame({
        "y_true": y_test.to_numpy(),
        "probability_late_risk": test_prob,
        "prediction_at_selected_threshold": test_pred,
    }).to_csv(REPORT_DIR / "test_predictions_final.csv", index=False)

    report = classification_report(
        y_test, test_pred, labels=[0, 1],
        target_names=["Lower risk (0)", "Late risk (1)"],
        zero_division=0,
    )
    (REPORT_DIR / "test_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Lower risk (0)", "Late risk (1)"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format=",")
    ax.set_title(f"Final Test Confusion Matrix (threshold={selected_threshold:.3f})")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "test_confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(threshold_curve["threshold"], threshold_curve["precision"], label="Precision")
    ax.plot(threshold_curve["threshold"], threshold_curve["recall"], label="Recall")
    ax.plot(threshold_curve["threshold"], threshold_curve["f1"], label="F1")
    ax.axvline(selected_threshold, linestyle="--", label=f"Selected={selected_threshold:.3f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Validation Threshold Selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "threshold_selection.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, test_prob, ax=ax)
    ax.set_title("Final Test ROC Curve — Selected Random Forest")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "test_roc_curve.png", dpi=160)
    plt.close(fig)

    md = f"""# Phase 5 — Threshold Selection and Final Evaluation

## Selected model
**Random Forest**, chosen because it had the best validation F1 at threshold 0.50 in Phase 4.

## Threshold
- Default threshold: 0.500
- Selected threshold: **{selected_threshold:.4f}**
- Selection rule: maximize validation F1{" subject to recall >= " + str(MIN_RECALL) if MIN_RECALL is not None else ""}
- Validation F1 at 0.50: {val_metrics_default['f1']:.4f}
- Validation F1 at selected threshold: {val_metrics_selected['f1']:.4f}
- Validation recall at selected threshold: {val_metrics_selected['recall']:.4f}
- Validation precision at selected threshold: {val_metrics_selected['precision']:.4f}

## Held-out test results at the locked threshold

| Metric | Score |
|---|---:|
| Accuracy | {test_metrics['accuracy']:.4f} |
| Precision | {test_metrics['precision']:.4f} |
| Recall | {test_metrics['recall']:.4f} |
| F1-score | {test_metrics['f1']:.4f} |
| ROC-AUC | {test_metrics['roc_auc']:.4f} |
| Average precision | {test_metrics['average_precision']:.4f} |

## Test classification report

```text
{report}
```

## Evaluation discipline
- The threshold was selected on validation predictions.
- The held-out test set was evaluated only after locking the selected model and threshold.
- Do not tune the model further based on this test result; if changes are made, the reported test result becomes exploratory and a new independent test set would be needed for an unbiased final estimate.

## Limitations
- Random stratified split does not establish future-time generalization.
- Maximizing F1 assumes precision and recall are equally important; real operations may need a cost-based threshold.
- Confirm that all features are available at the actual prediction moment.
"""
    (REPORT_DIR / "PHASE_5_REPORT.md").write_text(md, encoding="utf-8")

    print("\nPHASE 5 COMPLETE")
    print("Selected model: Random Forest")
    print(f"Selected threshold: {selected_threshold:.4f}")
    print(f"Validation F1 at 0.50: {val_metrics_default['f1']:.4f}")
    print(f"Validation F1 at selected threshold: {val_metrics_selected['f1']:.4f}")
    print("\nFINAL HELD-OUT TEST METRICS")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Average Precision: {test_metrics['average_precision']:.4f}")
    print(f"Saved report: {REPORT_DIR / 'PHASE_5_REPORT.md'}")

if __name__ == "__main__":
    main()
