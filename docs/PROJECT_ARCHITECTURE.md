# Project Architecture

## Data flow
Raw CSV → schema/data audit → leakage-aware feature pipeline → train/validation/test split → model training → threshold selection → model artifact → Streamlit dashboard → operations action queue.

## Stack
Python, Pandas, NumPy, scikit-learn, Streamlit, Plotly, Joblib and Pytest. A FastAPI scoring service and PostgreSQL can be added after the core model is validated.

## Dashboard modules
1. Delay Risk Overview: predicted risk distribution, high-risk count and performance summary.
2. Order-Level Risk Prediction: probability, risk band and contributing factors.
3. Region & Mode Risk Analysis: market/region and shipping-mode comparisons.
4. Operations Action Panel: ranked high-risk orders, filters and CSV export.
5. Model Performance: ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix and calibration.
6. Data Quality & Model Card: schema, limitations, training date and model version.

## Initial risk bands
Low < 0.35; Medium 0.35–0.69; High ≥ 0.70. These are placeholders. Phase 5 must select thresholds using validation data and business costs.

## Evaluation safeguards
Split before fitting preprocessing. Prefer time-based validation if a trustworthy order timestamp is available. Keep resampling inside training folds. Do not use outcome-derived fields. Feature importance is not proof of causality.
