# Late Delivery Risk Prediction: Jupyter Notebook Guide

This directory contains the complete, educational, and reproducible Machine Learning Jupyter Notebook for the **Late Delivery Risk Prediction** project.

---

## 📓 Notebook File

- **`late_delivery_risk_prediction_ML.ipynb`**: End-to-end Machine Learning pipeline from raw logistics data ingestion to production model inference.

---

## 🎯 Purpose and Demonstration Value

This notebook is designed for technical presentations, project demonstrations, and academic defense. It provides a complete, code-level explanation of the machine learning pipeline powering the deployed [DeliveryIQ Streamlit Application](https://ldr-utkarsh-sengar.streamlit.app/).

Every section bridges statistical theory, domain logic, and scikit-learn code:
1. **What the technique is.**
2. **Why it was selected for this project.**
3. **How it works mathematically.**
4. **How it is implemented in this codebase.**

---

## 📋 Table of Contents & Pipeline Structure

| Section | Title | Description |
|---|---|---|
| **1** | **Project Introduction & Architecture** | Business motivation, ML formulation as supervised binary classification, and end-to-end architecture diagram. |
| **2** | **Dataset Understanding** | Inspection of the DataCo logistics dataset (`180,519` rows, `40` columns), schema, data types, missing values, and summary statistics. |
| **3** | **Exploratory Data Analysis (EDA)** | Deep-dive visualizations of target distribution, shipping mode failure rates, scheduled transit constraints, geographic markets, and correlations. |
| **4** | **Data Preprocessing** | Implementation of `ColumnTransformer`: median imputation and scaling for numerical features, most frequent imputation and one-hot encoding for categoricals. |
| **5** | **Data Leakage Prevention** | Audit of pre-shipment boundaries. Detailed breakdown of why `Days for shipping (real)`, `Delivery Status`, and post-order fields were excluded. |
| **6** | **Feature Engineering** | Mathematical derivation and implementation of the 4 engineered features: `shipping_pressure_index`, `is_express_shipping`, `discount_to_item_price_ratio`, `log_order_quantity`. |
| **7** | **Model Training & Comparison** | Systematic comparison of Logistic Regression (baseline), HistGradientBoosting + TruncatedSVD, and Random Forest on identical validation rows. |
| **8** | **Random Forest Classifier Deep Dive** | Theory of ensemble bagging, decorrelated tree splits, hyperparameter specification (`120` trees, `max_depth=22`, `min_samples_leaf=3`), and inspection of the saved model. |
| **9** | **Decision Threshold Optimization** | Cost-asymmetry in supply chain operations, precision-recall trade-off curve, and validation-based tuning leading to optimal threshold $\tau^* = 0.3790$. |
| **10** | **Final Model Evaluation on Test Set** | Single evaluation on held-out test data (`27,078` records). Verifying test accuracy (`63.65%`), recall (`85.20%`), F1 (`71.99%`), ROC-AUC (`0.7622`), and confusion matrix. |
| **11** | **Model Explainability** | Permutation feature importance analysis from Phase 6, measuring validation F1 drop across features (`Type`, `Shipping Mode`, `Order City`, scheduled days). |
| **12** | **Model Saving and Deployment** | Artifact serialization via `joblib.dump()` and architecture of the `Phase5ModelAdapter` connecting the notebook pipeline to the Streamlit UI. |
| **13** | **Live Prediction Demonstration** | Interactive execution of single-order high-risk vs. low-risk shipments, outputting calibrated probabilities, risk tiers, and local sensitivity drivers. |
| **14** | **Conclusion & Operational Impact** | Summary of the 7-phase methodology and actionable recommendations for supply-chain logistics dispatchers. |

---

## 🛠️ Prerequisites & Setup

### 1. Environment Installation
Ensure you have Python 3.10+ installed. Install project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Recommended additional visualization packages (already in `requirements_phase7.txt`):
```bash
pip install jupyter seaborn matplotlib
```

### 2. Required Dataset & Artifacts Location
The notebook relies on project-relative paths and runs seamlessly from the repository root or the `notebooks/` folder:

- **Raw Dataset:** `data/raw/APL_Logistics.csv`
- **Processed Splits:** `data/processed/phase2/{train.csv, validation.csv, test.csv}`
- **Saved Model Bundle:** `artifacts/phase5/selected_model_bundle.joblib`
- **Evaluation Reports:** `reports/phase4/model_comparison.json`, `reports/phase5/phase5_metrics.json`, `reports/phase5/threshold_curve.csv`, `reports/phase6/feature_importance.csv`

All artifacts and split files are already included in this repository.

---

## 🚀 How to Run the Notebook

1. Open your terminal in the repository root directory:
   ```bash
   jupyter notebook notebooks/late_delivery_risk_prediction_ML.ipynb
   ```
2. In the Jupyter interface, select the Python 3 kernel.
3. Run cells sequentially (**Cell -> Run All** or step through with **Shift + Enter**).
4. All cells execute in under 30 seconds because pre-trained artifacts and pre-computed evaluation matrices are leveraged for maximum interactive responsiveness during live demonstrations.
