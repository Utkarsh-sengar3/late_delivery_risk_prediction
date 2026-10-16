# Phase 2 — Data Preprocessing and Feature Engineering

## Apply this ZIP
Extract the ZIP into your existing project root and merge folders. It updates `src/features/feature_policy.py` and adds the Phase 2 files. Keep `data/raw/APL_Logistics.csv` in place.

## Run
```powershell
python scripts/phase2_preprocess.py
```

## Outputs
- `data/processed/phase2/train.csv`
- `data/processed/phase2/validation.csv`
- `data/processed/phase2/test.csv`
- `artifacts/phase2/preprocessor.joblib`
- `reports/phase2/phase2_summary.json`
- `reports/phase2/PHASE_2_REPORT.md`

## Included engineering
- Candidate features are selected by an explicit allowlist.
- Direct personal identifiers and known outcome fields are excluded.
- Target rows with missing or non-binary labels are removed and counted.
- Numeric values use median imputation and standard scaling.
- Categorical values use most-frequent imputation and one-hot encoding.
- Encoder learns categories on training data only; unknown categories are handled safely.
- Train/validation/test split is stratified and reproducible (approximately 70/15/15).
- Derived features: quantity per scheduled shipping day (`shipping_pressure_index`), express shipping indicator, discount-to-item-price ratio, and log quantity.

## Important caveats
1. Confirm all candidate fields are available at the real prediction moment.
2. This dataset lacks an identified reliable order timestamp in the documented schema, so this phase uses a random stratified split. It does not demonstrate future-time generalization.
3. `shipping_pressure_index` and `log_order_quantity` are transparent proxies, not proven causal indicators.
4. Do not publish the processed CSVs or model artifacts if they contain sensitive business/customer information.
5. Keep the test split untouched for final evaluation.
