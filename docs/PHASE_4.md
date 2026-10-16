# Phase 4 — Model Family Comparison

## Prerequisite
Run Phase 2 and Phase 3 first. This script requires the Phase 2 split files and `artifacts/phase2/preprocessor.joblib`.

## Run
```powershell
python scripts/phase4_model_comparison.py
```

## Models
1. **Random Forest** — uses sparse one-hot encoded features directly, with balanced subsample weights.
2. **Histogram Gradient Boosting + TruncatedSVD** — reduces the high-dimensional sparse matrix to 80 dense components because the histogram boosting estimator requires dense data. SVD is fitted on training data only.

Both models are evaluated on the same validation rows. Because the representations differ, the comparison is useful operationally but not a perfectly controlled feature-representation experiment.

## Outputs
- `artifacts/phase4/random_forest.joblib`
- `artifacts/phase4/truncated_svd.joblib`
- `artifacts/phase4/hist_gradient_boosting_svd.joblib`
- `artifacts/phase4/preprocessor.joblib`
- `reports/phase4/model_comparison.csv`
- `reports/phase4/model_comparison.json`
- `reports/phase4/validation_predictions.csv`
- `reports/phase4/model_comparison.png`
- `reports/phase4/roc_curves.png`
- `reports/phase4/precision_recall_curves.png`
- `reports/phase4/PHASE_4_REPORT.md`

## Notes
- Training may take several minutes depending on your computer.
- The script does not load or evaluate the held-out test set.
- The default threshold is 0.50. Phase 5 will handle threshold selection.
- These random-split results do not prove performance on future orders.
