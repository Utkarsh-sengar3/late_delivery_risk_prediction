# Phase 4 — Model Family Comparison

Models were compared on the same validation split. The held-out test split was not loaded or evaluated.

## Validation results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| RandomForest | 0.6954 | 0.8176 | 0.5722 | 0.6732 | 0.7603 | 0.8255 |
| HistGradientBoosting_SVD | 0.6969 | 0.8427 | 0.5497 | 0.6654 | 0.7672 | 0.8304 |

## Best validation F1 at threshold 0.50: `RandomForest`

This is a provisional selection based only on validation F1. Operational threshold selection belongs in Phase 5.

## Method notes

- Random Forest uses the sparse one-hot encoded Phase 2 features.
- Histogram Gradient Boosting uses 80 TruncatedSVD components because the estimator requires dense input.
- The SVD was fitted on training data only. Explained variance ratio sum: 0.9109.
- Because the two models use different representations, this is a practical model-family comparison rather than a perfectly controlled feature-representation experiment.
- Class weights are enabled for both models.
- Threshold 0.50 is used for consistent baseline metrics; it is not claimed to be optimal.
- Random stratified splitting does not establish future-time generalization.
- Confirm candidate features are available at the actual prediction time before deployment.
