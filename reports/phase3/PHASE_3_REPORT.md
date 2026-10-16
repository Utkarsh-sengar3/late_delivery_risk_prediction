# Phase 3 — Logistic Regression Baseline

## Validation metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.7030 |
| Precision | 0.8151 |
| Recall | 0.5928 |
| F1-score | 0.6864 |
| ROC-AUC | 0.7695 |
| Average precision (PR-AUC summary) | 0.8255 |
| Threshold | 0.50 |

## Setup
- Training rows: 126,363
- Validation rows: 27,078
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
                          precision    recall  f1-score   support

On-time / lower risk (0)       0.63      0.84      0.72     12231
           Late risk (1)       0.82      0.59      0.69     14847

                accuracy                           0.70     27078
               macro avg       0.72      0.71      0.70     27078
            weighted avg       0.73      0.70      0.70     27078

```
