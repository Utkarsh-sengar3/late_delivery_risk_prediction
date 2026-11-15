# Research Paper: Late Delivery Risk Prediction in Global Supply Chain Operations

**Full Title:** Late Delivery Risk Prediction in Global Supply Chain Operations Using Machine Learning: An End-to-End Predictive Decision-Support Framework  
**Author:** Utkarsh Sengar  
**GitHub:** [https://github.com/Utkarsh-sengar3](https://github.com/Utkarsh-sengar3)  
**Project Repository:** [https://github.com/Utkarsh-sengar3/late_delivery_risk_prediction](https://github.com/Utkarsh-sengar3/late_delivery_risk_prediction)  
**Email:** [utkarshsengar3@gmail.com](mailto:utkarshsengar3@gmail.com)  
**Department:** Artificial Intelligence and Machine Learning Engineering  
**Institution:** Lovely Professional University, Phagwara, Punjab, India  
**Project:** DeliveryIQ  
**Live Application:** [https://ldr-utkarsh-sengar.streamlit.app/](https://ldr-utkarsh-sengar.streamlit.app/)  
**Repository:** [https://github.com/Utkarsh-sengar3/late_delivery_risk_prediction](https://github.com/Utkarsh-sengar3/late_delivery_risk_prediction)

---

## Abstract Summary

Global supply chain networks operate under tight operational constraints where delivery schedule adherence directly influences customer satisfaction, retention, contractual penalties, and enterprise brand equity. While conventional enterprise resource planning (ERP) platforms visualize delayed shipments post-facto, predictive decision-support mechanisms capable of forecasting delivery delays prior to transit dispatch remain underdeveloped. 

This research presents **DeliveryIQ**, a rigorous end-to-end machine learning framework for proactive late delivery risk classification using the DataCo Global Supply Chain dataset (180,519 shipments). To eliminate target leakage, all post-order transit variables were purged, retaining 27 strictly pre-shipment attributes alongside four domain-engineered signals (*shipping pressure index*, *express shipping indicator*, *discount-to-price ratio*, and *log-order quantity*). A balanced Random Forest classifier (`n_estimators=120`, `max_depth=22`, `min_samples_leaf=3`, `class_weight='balanced_subsample'`) was trained and optimized. 

To bridge the gap between academic F1 optimization and industry operational requirements where false negatives (unflagged late deliveries) carry steep financial and operational costs, the decision threshold was mathematically tuned from $\tau = 0.50$ to $\tau^* = 0.3790$ on a held-out validation set. On the unseen test partition ($N = 27,078$), this threshold shift surged sensitivity from **57.20% to 85.20%**, achieving an **F1-score of 71.99%**, **Accuracy of 63.65%**, **ROC-AUC of 76.22%**, and **PR-AUC of 82.76%**. The predictive core is containerized and deployed into an interactive 9-module Streamlit decision-support application with model interpretability, batch inference, threshold manipulation, and live audit logging.

---

## Directory Structure

```text
research_paper/
├── main.tex                                     # Master IEEE two-column LaTeX manuscript
├── references.bib                               # Authentic BibTeX citations (25 entries)
├── README.md                                    # This documentation guide
├── compiled/
│   ├── Late_Delivery_Risk_Prediction_Research_Paper.pdf  # Publication-ready compiled PDF (11 pages)
│   └── main.pdf                                 # Primary compiler output
└── figures/                                     # High-resolution figures & live UI screenshots
    ├── system_architecture.png                  # Publication-grade end-to-end architecture diagram
    ├── target_distribution.png                  # Binary target balance visualization (54.83% late)
    ├── numeric_correlation.png                  # Pearson correlation matrix heatmap
    ├── model_comparison.png                     # Multi-metric baseline model benchmark chart
    ├── threshold_selection.png                  # Validation F1 vs. threshold curve (optimal at 0.3790)
    ├── threshold_analysis.png                   # Metric trade-off analysis across threshold range
    ├── confusion_matrix.png                     # Test confusion matrix (TN=4,585, FP=7,646, FN=2,198, TP=12,649)
    ├── roc_curve.png                            # Test ROC curve (AUC = 0.7622)
    ├── precision_recall_curve.png               # Test Precision-Recall curve (AP = 0.8276)
    ├── feature_importance.png                   # Permutation feature importance ranking
    ├── website_dashboard.png                    # Live UI: Operational KPI Dashboard
    ├── single_prediction.png                    # Live UI: Single Shipment Dispatch Form
    ├── prediction_result.png                    # Live UI: Real-Time Risk Assessment & Probability Gauge
    ├── batch_prediction.png                     # Live UI: Multi-Shipment Batch CSV Ingestion
    ├── prediction_analytics.png                 # Live UI: Longitudinal Risk Analytics & Visualizations
    ├── model_performance.png                    # Live UI: Live Evaluation Metrics & Threshold Diagnostics
    ├── risk_analysis.png                        # Live UI: Cohort Risk Drilldowns & Geographic Filters
    ├── explainability.png                       # Live UI: Model Explainability & Permutation Importance
    ├── prediction_history.png                   # Live UI: Audit Trail, Search & CSV History Export
    └── settings_page.png                        # Live UI: Runtime Configuration & System Diagnostics
```

---

## Key Experimental Results

| Metric | Baseline Default ($\tau = 0.5000$) | Optimized Threshold ($\tau^* = 0.3790$) | Net Delta ($\Delta$) | Operational Consequence |
| :--- | :---: | :---: | :---: | :--- |
| **Recall (Sensitivity)** | **57.20%** | **85.20%** | **+28.00%** | Rescues 4,157 late shipments from zero intervention |
| **Precision** | **70.62%** | **62.33%** | **-8.29%** | Moderate increase in false alarms, highly tolerable |
| **F1-Score** | **63.21%** | **71.99%** | **+8.78%** | Superior harmonic balance for operational planning |
| **Accuracy** | **64.71%** | **63.65%** | **-1.06%** | Insignificant overall degradation |
| **ROC-AUC** | **76.22%** | **76.22%** | **--** | Invariant rank discrimination capacity |
| **Average Precision (PR-AUC)** | **82.76%** | **82.76%** | **--** | Strong minority/majority precision retention |

---

## Compilation Instructions

The document is written in standard IEEE Transactions format (`IEEEtran.cls`). It can be compiled using any modern TeX distribution or cloud service.

### Method 1: Using Tectonic (Zero Configuration, Recommended)
```bash
tectonic research_paper/main.tex --outdir research_paper/compiled
```

### Method 2: Using standard pdfLaTeX & BibTeX
```bash
cd research_paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Method 3: Using Overleaf
1. Upload the contents of the `research_paper/` folder to a new Overleaf project.
2. Set the main document to `main.tex`.
3. Select compiler: `pdfLaTeX` or `XeLaTeX`.
4. Click **Recompile**.
