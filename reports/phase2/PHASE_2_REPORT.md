# Phase 2 — Preprocessing and Feature Engineering

- Rows with valid binary target: **180,519**
- Rows dropped for invalid/missing target: **0**
- Input features after engineering: **27**
- Transformed feature count: **3,505**
- Preprocessor fitted on training data only: **Yes**

## Split summary

- Train: 126,363 rows; class 0=57,080; class 1=69,283; late rate=54.83%
- Validation: 27,078 rows; class 0=12,231; class 1=14,847; late rate=54.83%
- Test: 27,078 rows; class 0=12,231; class 1=14,847; late rate=54.83%

## Engineered features

- `shipping_pressure_index`
- `is_express_shipping`
- `discount_to_item_price_ratio`
- `log_order_quantity`

## Important limitations

- Random stratified splitting is used because a trustworthy order timestamp was not identified.
- Feature availability at the actual pre-shipment decision point must be confirmed.
- Derived features are proxies and should not be interpreted as causal drivers.
- The held-out test set must not be used for model selection or threshold tuning.
