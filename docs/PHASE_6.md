# Phase 6 — Model Explainability (Compatibility Fix 2)

## Apply
Extract this ZIP into the project root and merge folders. Replace the existing `scripts/phase6_explainability.py` with this version.

## Run
```powershell
python scripts/phase6_explainability.py
```

## Why this version is different
Both previous versions depended on `sklearn.inspection.permutation_importance`, which rejected the sparse matrix and then the adapter object under the installed scikit-learn API. This version implements permutation importance directly: it shuffles one original feature, applies the saved preprocessor, runs the trained Random Forest, and calculates the F1 drop. It does not need the adapter to satisfy scikit-learn's estimator interface.

To keep runtime manageable, it uses 2,000 validation rows and 3 repeats per feature. This is an approximation for exploratory explainability, not a definitive ranking.

## Outputs
- `reports/phase6/feature_importance.csv`
- `reports/phase6/global_feature_importance.png`
- `reports/phase6/local_case_examples.csv`
- `reports/phase6/local_case_explanations.json`
- `reports/phase6/phase6_explainability_summary.json`
- `reports/phase6/PHASE_6_REPORT.md`

## Interpretation
Importance measures model reliance on the sampled validation data; it does not prove causation. The held-out test set is not used.
