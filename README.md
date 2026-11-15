# 🚚 Late Delivery Risk Prediction

An end-to-end Machine Learning project designed to predict late-delivery risk using logistics and order-related data. The project combines data preprocessing, model evaluation, threshold optimization, and explainability through an interactive Streamlit web application.

---

### 👤 Developer & Author
**Utkarsh Sengar**  
- **GitHub:** [https://github.com/Utkarsh-sengar3](https://github.com/Utkarsh-sengar3)  
- **Project:** [https://github.com/Utkarsh-sengar3/late_delivery_risk_prediction](https://github.com/Utkarsh-sengar3/late_delivery_risk_prediction)  
- **Email:** [utkarshsengar3@gmail.com](mailto:utkarshsengar3@gmail.com)

---

## 🌐 Live Website

🚀 **Try the application here:**

### 👉 https://ldr-utkarsh-sengar.streamlit.app/

---

## 📓 Machine Learning Notebook

A complete, educational, and reproducible Jupyter Notebook detailing the entire Machine Learning implementation is available in the [`notebooks/`](notebooks/) directory:

👉 **[`notebooks/late_delivery_risk_prediction_ML.ipynb`](notebooks/late_delivery_risk_prediction_ML.ipynb)**  
📖 **[Notebook Guide & Documentation](notebooks/README.md)**

The notebook provides an in-depth walkthrough:
- **Dataset Understanding & EDA:** Target distribution, shipping mode failure rates, and transit window constraints across 180,519 records.
- **Data Leakage Safeguards:** Audit of post-shipment columns (`Days for shipping (real)`, `Delivery Status`, `Order Status`) and formulation of the 23 raw pre-shipment feature schema.
- **Feature Engineering:** Implementation of `shipping_pressure_index`, `is_express_shipping`, `discount_to_item_price_ratio`, and `log_order_quantity`.
- **Model Comparison:** Benchmark comparison of Logistic Regression (baseline), HistGradientBoosting + TruncatedSVD, and Random Forest.
- **Decision Threshold Optimization:** Cost-asymmetric tuning on validation data ($\tau^* = 0.3790$) increasing delay recall from 57.2% to 85.5%.
- **Held-Out Test Results:** Single locked evaluation on 27,078 test instances (Accuracy: 63.65%, Recall: 85.20%, F1: 71.99%, ROC-AUC: 0.7622).
- **Model Explainability:** Permutation feature importance ranking predictive reliance across features (`Type`, `Shipping Mode`, `Order City`, scheduled days).
- **Production Inference:** Direct integration with `Phase5ModelAdapter` for live single-order and batch predictions with local sensitivity explanations.

---

## 📄 Academic Research Paper

A publication-grade IEEE two-column research paper detailing the theoretical foundations, leakage-free feature engineering, empirical benchmark comparisons, threshold optimization mathematics, and deployment architecture is available in the [`research_paper/`](research_paper/) directory:

📄 **[Download Compiled Research Paper (PDF)](research_paper/compiled/Late_Delivery_Risk_Prediction_Research_Paper.pdf)**  
📝 **[LaTeX Source Manuscript (`main.tex`)](research_paper/main.tex)**  
📚 **[BibTeX Citations (`references.bib`)](research_paper/references.bib)**  
📖 **[Research Paper Documentation & Guide](research_paper/README.md)**

- **Title:** *Late Delivery Risk Prediction in Global Supply Chain Operations Using Machine Learning: An End-to-End Predictive Decision-Support Framework*
- **Author:** Utkarsh Sengar
- **GitHub:** [https://github.com/Utkarsh-sengar3](https://github.com/Utkarsh-sengar3)
- **Project:** [https://github.com/Utkarsh-sengar3/late_delivery_risk_prediction](https://github.com/Utkarsh-sengar3/late_delivery_risk_prediction)
- **Email:** [utkarshsengar3@gmail.com](mailto:utkarshsengar3@gmail.com)
- **Format:** IEEE Two-Column Conference / Journal Format (11 Pages, 20 Figures & Screenshots, 7 Comparative Tables, 25 Authentic Citations)
- **Status:** Complete & Publication-Ready

### Key Empirical Benchmarks (Held-Out Test Set,  = 27,078$)

| Metric | Baseline ($\\tau = 0.5000$) | Tuned Threshold ($\\tau^* = 0.3790$) | Net Delta ($\\Delta$) | Operational Consequence |
| :--- | :---: | :---: | :---: | :--- |
| **Recall (Sensitivity)** | **57.20%** | **85.20%** | **+28.00%** | Rescues 4,157 late shipments from zero intervention |
| **Precision** | **70.62%** | **62.33%** | **-8.29%** | Moderate false-alarm trade-off, highly tolerable |
| **F1-Score** | **63.21%** | **71.99%** | **+8.78%** | Superior harmonic balance for operational planning |
| **Accuracy** | **64.71%** | **63.65%** | **-1.06%** | High overall fidelity preserved |
| **ROC-AUC** | **76.22%** | **76.22%** | **--** | Invariant rank discrimination capacity |
| **Average Precision** | **82.76%** | **82.76%** | **--** | Robust performance over 54.83% naive baseline |

---

## 📌 Project Overview

Late deliveries can negatively impact customer satisfaction, supply-chain efficiency, and logistics operations.

This project explores how machine learning can identify orders at risk of late delivery using historical logistics data. It follows a structured ML workflow, from data auditing and preprocessing to model selection, evaluation, and feature importance analysis.

The project uses the DataCo logistics dataset (`APL_Logistics.csv`).

## ✨ Website Features

### 📊 1. Dashboard
- Overview of the project and dataset.
- Selected machine-learning model.
- Model evaluation metrics.
- Application status and navigation.

### 🔮 2. Model Prediction
- Interface for entering shipment and order attributes.
- Designed for assessing late-delivery risk using the trained model.
- Requires the saved model pipeline and the correct input schema to be connected for live predictions.

### 📈 3. Prediction Analytics
- Designed to display recorded predictions and delivery-risk patterns.
- Supports analysis when genuine prediction records are available.

### 🎯 4. Model Performance
Displays important classification metrics, including:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision

### ⚠️ 5. Risk Analysis
- Explore delivery-related patterns in the dataset.
- Examine the target distribution when the dataset is available.
- Support data-driven investigation of potential delivery risks.

### 🧠 6. Explainability
- Feature-importance visualisation.
- Helps understand which input features influence model performance.
- Uses permutation-importance results from the explainability workflow.

### 🕒 7. Prediction History
- Provides a dedicated interface for reviewing predictions recorded during the current application session.

### ⚙️ 8. Settings
- Customise the interface appearance.
- Choose between supported themes and accent colours.

---

## 🔬 Machine Learning Workflow

The project follows these stages:

1. **Data Auditing** — Inspect dataset dimensions, missing values, duplicate records, and feature availability.
2. **Data Preprocessing** — Prepare features and create training, validation, and test splits.
3. **Model Training and Comparison** — Evaluate candidate classification models.
4. **Threshold Optimisation** — Select a classification threshold using validation data.
5. **Final Evaluation** — Evaluate the selected model on held-out test data.
6. **Explainability** — Use permutation importance to investigate feature influence.

---

## 🤖 Selected Model

**Model:** Random Forest Classifier  
**Selected Decision Threshold:** 0.379

### 📊 Recorded Test Results

| Metric | Score |
|---|---:|
| Accuracy | 0.6365 |
| Precision | 0.6233 |
| Recall | 0.8520 |
| F1-score | 0.7199 |
| ROC-AUC | 0.7622 |
| Average Precision | 0.8276 |

These values represent the project's recorded Phase 5 evaluation results on the held-out test set. They are not guarantees of future prediction performance.

The decision threshold was selected using validation data, while the test set was reserved for final evaluation.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and evaluation |
| Joblib | Model and artifact persistence |
| Matplotlib / Visualisation Tools | Data visualisation and analysis |
| Git and GitHub | Version control and project hosting |

---

## 💻 Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Utkarsh-sengar3/late_delivery_risk_prediction.git
cd late_delivery_risk_prediction
```

### 2. Set Up Python Environment

Recommended Python version: **Python 3.10 or 3.11**.

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Artifacts & Cross-Version Compatibility

The production model bundle is tracked in Git and ready for inference:
- **`artifacts/phase5/selected_model_bundle.joblib`**: Full pipeline bundle containing:
  - `model`: Tuned Random Forest Classifier (120 estimators, max depth 22, balanced subsampling).
  - `preprocessor`: 2-branch `ColumnTransformer` (median + standard scaler for numeric features; most-frequent + one-hot encoding for categorical features).
  - `threshold`: Cost-asymmetric decision threshold (`0.3790`).
  - Cross-version compatibility: Preprocessor imputers are dual-compatible with scikit-learn 1.4–1.6 (`_fill_dtype`) and scikit-learn 1.7+ (`_fit_dtype`).

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 5. Prediction Workflow

1. **Single Shipment Assessment**:
   - Navigate to **Model Prediction** in the sidebar.
   - Enter pre-dispatch shipment details (Scheduled Days, Shipping Mode, Item Price, Quantity, Market, Destination).
   - Click **Predict Late Delivery Risk** to receive the calibrated probability, risk badge, and sensitivity-based feature explanations.
2. **Batch CSV Scoring**:
   - Switch to the **Batch Prediction (CSV)** tab.
   - Upload any logistics CSV manifest. The system automatically filters out leakage columns, scores all rows via vectorized batch inference, and provides a downloadable scored CSV report.

### 6. Run Automated Tests

The repository includes a comprehensive pytest suite covering model loading, leakage enforcement, feature engineering, prediction contracts, imputer cross-version auto-healing, and Streamlit page flows:

```bash
pytest
```

---

## ☁️ Deployment

The application is deployed on **Streamlit Community Cloud**:
- URL: **https://ldr-utkarsh-sengar.streamlit.app/**
- Production branch: `main`
- Configuration: Streamlit automatically installs dependencies from `requirements.txt`.

---

## 👤 Author & Contact

**Utkarsh Sengar**  
- **GitHub:** [https://github.com/Utkarsh-sengar3](https://github.com/Utkarsh-sengar3)  
- **Project Repository:** [https://github.com/Utkarsh-sengar3/late_delivery_risk_prediction](https://github.com/Utkarsh-sengar3/late_delivery_risk_prediction)  
- **Email:** [utkarshsengar3@gmail.com](mailto:utkarshsengar3@gmail.com)

