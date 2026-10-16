# Phase 3 — Logistic Regression Baseline

## Prerequisite
Phase 2 must have completed successfully, with:
- `data/processed/phase2/train.csv`
- `data/processed/phase2/validation.csv`
- `data/processed/phase2/test.csv`
- `artifacts/phase2/preprocessor.joblib`

## Run from the project root
```powershell
python scripts/phase3_logistic_regression.py
```

## What it does
- Loads Phase 2 train and validation splits.
- Applies the Phase 2 preprocessor.
- Trains a Logistic Regression classifier with `class_weight="balanced"`.
- Reports Accuracy, Precision, Recall, F1, ROC-AUC and average precision.
- Writes a classification report, validation predictions and plots.
- Saves the classifier and preprocessing artifact.

## Outputs
- `artifacts/phase3/logistic_regression_baseline.joblib`
- `artifacts/phase3/preprocessor.joblib`
- `reports/phase3/validation_metrics.json`
- `reports/phase3/classification_report.txt`
- `reports/phase3/validation_predictions.csv`
- `reports/phase3/confusion_matrix.png`
- `reports/phase3/roc_curve.png`
- `reports/phase3/precision_recall_curve.png`
- `reports/phase3/PHASE_3_REPORT.md`

## Evaluation discipline
The validation set is used for baseline reporting. The held-out test split is not evaluated in this phase. Threshold 0.50 is a baseline default, not an optimized operational threshold. Use Phase 5 for threshold selection and final test evaluation only after model selection is complete.
