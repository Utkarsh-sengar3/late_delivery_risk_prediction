# Late Delivery Risk Prediction — Phase 7 Dashboard Upgrade

This ZIP is an incremental update for the existing project. It adds a more complete Streamlit interface while keeping existing model artifacts and reports.

## New pages
- Dashboard
- Model Prediction (single shipment + CSV batch prediction)
- Prediction Analytics (probability charts, segment charts, exploratory numeric clustering)
- Model Performance (accuracy, precision, recall, F1, ROC-AUC, Average Precision, error rate, confusion matrix, FP/FN)
- Risk Analysis
- Explainability
- Prediction History
- About This Website
- About Developer
- Settings (light/dark mode and accent color palette)

## Install
1. Extract the ZIP.
2. Copy `dashboard_app.py` and `requirements_phase7.txt` into the root of the existing project:
   `C:\Users\utkar\OneDrive\Desktop\Late_Delivery_Risk\late_delivery_risk_prediction`
3. Do not delete or replace `artifacts`, `data`, `reports`, `scripts`, or other project files.
4. Open PowerShell in the project root and run:
   ```powershell
   python -m pip install -r requirements_phase7.txt
   ```
5. Start the dashboard:
   ```powershell
   python -m streamlit run dashboard_app.py
   ```

## Required artifacts
- `artifacts/phase5/selected_model_bundle.joblib`
- `data/processed/phase2/validation.csv` (or train/test split as fallback)
- `data/processed/phase2/train.csv` recommended for form value ranges
- `data/processed/phase2/test.csv` for dynamically calculated error metrics
- `reports/phase5/PHASE_5_REPORT.md`
- `reports/phase6/feature_importance.csv`
- `reports/phase6/global_feature_importance.png`
- `reports/phase6/PHASE_6_REPORT.md`

## Notes
- Model performance page calculates metrics from the saved model on the held-out test split when available. It does not fit or retrain the model.
- The threshold slider on the performance page is for analysis only. Do not use the held-out test split to select the production threshold.
- Batch prediction requires all model input columns with the raw feature names expected by the Phase 2 preprocessor.
- Prediction history is session-only and not written to disk.
- Cluster graph uses KMeans and PCA on selected numeric input features as an exploratory segmentation; cluster IDs are not delivery-risk labels.
- Theme settings apply to the active Streamlit session. Streamlit charts and third-party components may not all perfectly match the selected background color.
- The random train/validation/test split does not establish future-time generalization.
- Confirm feature availability at prediction time before operational use.
