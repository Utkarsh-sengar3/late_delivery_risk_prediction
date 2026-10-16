# Phase 7 — Streamlit Dashboard

Incremental package for the existing `late_delivery_risk_prediction` project. It does not replace the project or retrain the model.

## Install
1. Extract this ZIP.
2. Copy `dashboard_app.py` into the project root:
   `C:\Users\utkar\OneDrive\Desktop\Late_Delivery_Risk\late_delivery_risk_prediction`
3. Keep existing `artifacts`, `data`, `reports`, `scripts`, and other folders.
4. In PowerShell, change to the project root and install dependencies:
   ```powershell
   python -m pip install -r requirements_phase7.txt
   ```
   Copy `requirements_phase7.txt` into the project root too, or install directly:
   ```powershell
   python -m pip install streamlit pandas numpy joblib scikit-learn
   ```

## Run
From the project root:
```powershell
python -m streamlit run dashboard_app.py
```
Open the Local URL printed by Streamlit if the browser does not open automatically.

## Required project artifacts
- `artifacts/phase5/selected_model_bundle.joblib`
- `data/processed/phase2/validation.csv`
- `data/processed/phase2/train.csv` (optional fallback to validation)
- `reports/phase5/PHASE_5_REPORT.md`
- `reports/phase6/feature_importance.csv`
- `reports/phase6/global_feature_importance.png`
- `reports/phase6/PHASE_6_REPORT.md`

## Verify
1. Overview shows row count and threshold around `0.3790`.
2. Explainability shows the Phase 6 chart/table.
3. Risk analysis lets you group by Shipping Mode or Type.
4. Predict a shipment returns a probability and risk flag.
5. Prediction history lists the result; it is session-only.
6. Model information shows Phase 5 test metrics.

Phase 5 test metrics recorded: accuracy 0.6365, precision 0.6233, recall 0.8520, F1 0.7199, ROC-AUC 0.7622, average precision 0.8276.

The dashboard loads the saved Phase 5 model and does not retrain it. The split was random, so future-time generalization has not been established. Verify every input is available at the time a prediction is made and remove post-outcome fields before real operational use.
