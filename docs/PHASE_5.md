# Phase 5 — Threshold Selection and Final Evaluation

## Prerequisite
Run Phases 2–4 successfully. This script requires:
- Phase 2 split CSVs and preprocessor
- `artifacts/phase4/random_forest.joblib`

## Run
```powershell
python scripts/phase5_threshold_final_evaluation.py
```

## What it does
1. Uses Phase 4's selected Random Forest model.
2. Searches thresholds on the validation set to maximize F1.
3. Locks the selected threshold.
4. Evaluates the locked model/threshold on the held-out test set.
5. Saves a model bundle, metrics, threshold curve, predictions, and plots.

`MIN_RECALL = None` maximizes F1 without a recall constraint. If the business requires a minimum recall (for example 0.60), set `MIN_RECALL = 0.60` in the script before running; this choice should be made based on operational requirements, not the test result.

## Outputs
- `artifacts/phase5/selected_model_bundle.joblib`
- `reports/phase5/phase5_metrics.json`
- `reports/phase5/threshold_curve.csv`
- `reports/phase5/validation_predictions_selected_threshold.csv`
- `reports/phase5/test_predictions_final.csv`
- `reports/phase5/test_classification_report.txt`
- `reports/phase5/test_confusion_matrix.png`
- `reports/phase5/threshold_selection.png`
- `reports/phase5/test_roc_curve.png`
- `reports/phase5/PHASE_5_REPORT.md`

## Caution
Once the test result is reported, do not use it to choose models or thresholds. Repeated test-driven changes turn the test set into a validation set. Random stratified splitting also does not prove future-time generalization.
