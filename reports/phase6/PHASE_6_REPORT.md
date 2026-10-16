# Phase 6 — Model Explainability

## Method
- Model: Random Forest selected in Phase 5
- Locked threshold: 0.3790
- Validation sample size: 2,000
- Baseline F1 on this sample: 0.7172
- Permutation repeats per feature: 3
- Permutation importance is calculated manually on original input features, applying the saved preprocessor before inference.
- Six local examples are included: up to three high-risk and three low-risk predictions.

## Top 15 features

| Feature | Mean F1 decrease | Standard deviation |
|---|---:|---:|
| `Type` | 0.017156 | 0.002257 |
| `Shipping Mode` | 0.009429 | 0.005520 |
| `Order City` | 0.008666 | 0.001560 |
| `Days for shipment (scheduled)` | 0.007531 | 0.003577 |
| `is_express_shipping` | 0.005818 | 0.004679 |
| `Order State` | 0.004716 | 0.003356 |
| `shipping_pressure_index` | 0.003907 | 0.004124 |
| `Customer State` | 0.003884 | 0.000535 |
| `Latitude` | 0.002754 | 0.002248 |
| `Category Id` | 0.002554 | 0.000337 |
| `Customer Segment` | 0.002368 | 0.001631 |
| `Order Region` | 0.002148 | 0.001369 |
| `Customer Country` | 0.001981 | 0.000409 |
| `discount_to_item_price_ratio` | 0.001788 | 0.001777 |
| `log_order_quantity` | 0.001393 | 0.001914 |

## Interpretation cautions

- Importance is predictive association, not causation.
- Correlated features can share or mask importance.
- Permutation importance uses a reproducible sample of up to 2,000 validation rows to keep runtime manageable.
- Local explanations replace one input at a time with a typical sample value; these are sensitivity analyses, not causal or counterfactual guarantees.
- Do not expose customer-identifying fields in the UI or reports.
- Evaluation used a random stratified split, not a time-based holdout.
