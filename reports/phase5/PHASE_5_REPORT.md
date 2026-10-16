# Phase 5 — Threshold Selection and Final Evaluation

## Selected model
**Random Forest**, chosen because it had the best validation F1 at threshold 0.50 in Phase 4.

## Threshold
- Default threshold: 0.500
- Selected threshold: **0.3790**
- Selection rule: maximize validation F1
- Validation F1 at 0.50: 0.6732
- Validation F1 at selected threshold: 0.7212
- Validation recall at selected threshold: 0.8549
- Validation precision at selected threshold: 0.6237

## Held-out test results at the locked threshold

| Metric | Score |
|---|---:|
| Accuracy | 0.6365 |
| Precision | 0.6233 |
| Recall | 0.8520 |
| F1-score | 0.7199 |
| ROC-AUC | 0.7622 |
| Average precision | 0.8276 |

## Test classification report

```text
                precision    recall  f1-score   support

Lower risk (0)       0.68      0.37      0.48     12231
 Late risk (1)       0.62      0.85      0.72     14847

      accuracy                           0.64     27078
     macro avg       0.65      0.61      0.60     27078
  weighted avg       0.65      0.64      0.61     27078

```

## Evaluation discipline
- The threshold was selected on validation predictions.
- The held-out test set was evaluated only after locking the selected model and threshold.
- Do not tune the model further based on this test result; if changes are made, the reported test result becomes exploratory and a new independent test set would be needed for an unbiased final estimate.

## Limitations
- Random stratified split does not establish future-time generalization.
- Maximizing F1 assumes precision and recall are equally important; real operations may need a cost-based threshold.
- Confirm that all features are available at the actual prediction moment.
