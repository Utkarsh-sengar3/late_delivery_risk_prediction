# # 🚚 Late Delivery Risk Prediction

An end-to-end Machine Learning project designed to predict late-delivery risk using logistics and order-related data. The project combines data preprocessing, model evaluation, threshold optimization, and explainability through an interactive Streamlit web application.

## 🌐 Live Website

🚀 **Try the application here:**

### 👉 https://ldr-utkarsh-sengar.streamlit.app/

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
