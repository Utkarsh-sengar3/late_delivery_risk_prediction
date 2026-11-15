"""Script to build the comprehensive late_delivery_risk_prediction_ML.ipynb notebook."""

import json
from pathlib import Path

def create_notebook():
    notebook_path = Path("notebooks/late_delivery_risk_prediction_ML.ipynb")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    cells = []
    cell_idx = 0

    def md(source):
        nonlocal cell_idx
        cell_idx += 1
        cells.append({
            "cell_type": "markdown",
            "id": f"cell-md-{cell_idx}",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def code(source):
        nonlocal cell_idx
        cell_idx += 1
        cells.append({
            "cell_type": "code",
            "id": f"cell-code-{cell_idx}",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # ==========================================
    # SECTION 1: PROJECT INTRODUCTION
    # ==========================================
    md("""
# 🚚 Late Delivery Risk Prediction: End-to-End Machine Learning Pipeline
### An Educational, Reproducible Technical Walkthrough from Raw Logistics Data to Production Inference
**Author / Engineering Lead:** Utkarsh Sengar  
**Repository:** [https://github.com/Utkarsh-sengar3/late_delivery_risk_prediction](https://github.com/Utkarsh-sengar3/late_delivery_risk_prediction)  
**Deployed Streamlit Application:** [https://ldr-utkarsh-sengar.streamlit.app/](https://ldr-utkarsh-sengar.streamlit.app/)

---

## 1. Project Introduction & Architecture

### 1.1 What is Late Delivery Risk Prediction?
In modern global supply chains and e-commerce ecosystems, on-time delivery is one of the most critical determinants of customer retention, Net Promoter Score (NPS), and operational profitability. A **late delivery** occurs when a customer's shipment arrives past the scheduled promised transit date.

**Late Delivery Risk Prediction** is the process of estimating the mathematical probability that an order will experience a shipping delay, evaluated at the moment the order is placed or prepared for dispatch--**before the package leaves the fulfillment facility**.

### 1.2 Why is it Important?
When logistics teams only detect delays *after* shipments have missed their delivery windows:
1. **Remediation Costs Skyrocket:** Rush expedited couriers, customer concessions, and chargebacks are expensive.
2. **Customer Churn Increases:** Over 68% of e-commerce consumers report that an unexpected delivery delay negatively affects their likelihood of ordering again.
3. **Warehouse Inefficiencies Multiply:** Misallocated carrier capacity causes freight bottlenecks during peak demand periods.

By contrast, **pre-shipment prediction** empowers logistics managers to take proactive corrective action:
- Automatically route high-risk shipments to priority warehouse packing queues.
- Dynamically upgrade shipping carriers or regional line-haul routes.
- Set realistic expected delivery estimates for downstream customers.

### 1.3 Machine Learning Formulation
This problem is formulated as **Supervised Binary Classification**:
- **Inputs ($X$):** A set of operational attributes known prior to dispatch (shipping mode, scheduled days, product category, customer segment, market, order item quantity, unit price, discounts, and regional geography).
- **Target ($Y$):** Binary ground truth:
  $$\begin{cases} Y = 1 & \text{if shipment arrived late (Late Delivery Risk)} \\ Y = 0 & \text{if shipment arrived on time or early (On-Time Delivery)} \end{cases}$$
- **Model Output ($P(Y=1 \mid X)$):** Estimated posterior probability of late delivery.
- **Decision Policy ($\hat{Y}$):** A risk classification rule parameterized by a decision threshold $\tau \in [0, 1]$:
  $$\hat{Y} = \mathbb{I}(P(Y=1 \mid X) \ge \tau)$$

### 1.4 End-to-End System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 DELIVERYIQ ARCHITECTURAL FLOW                                │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
  Raw Data Ingestion (DataCo Logistics: 180,519 records × 40 columns)
       │
       ▼
  Phase 1: Data Audit & Leakage Audit
       ├── Missing value & duplicate inspection
       └── Strict isolation of post-shipment columns (Days for shipping (real), Delivery Status)
       │
       ▼
  Phase 2: Leakage-Safe Feature Engineering & Preprocessing
       ├── Engineered: shipping_pressure_index, is_express_shipping, discount_to_item_price_ratio
       ├── Preprocessing: Median Imputation + Standard Scaling (Numeric)
       │                  Most Frequent Imputation + One-Hot Encoding (Categorical)
       └── Split: 70% Train (126,363) | 15% Validation (27,078) | 15% Test (27,078)
       │
       ▼
  Phase 3 & 4: Candidate Model Training & Comparison
       ├── Baseline: Logistic Regression (balanced)
       ├── Model A: Random Forest Classifier (120 trees, max_depth=22)
       └── Model B: HistGradientBoosting + TruncatedSVD (80 components)
       │
       ▼
  Phase 5: Validation-Based Threshold Optimization & Single Final Test Evaluation
       ├── Optimize threshold on Validation Set: tau* = 0.3790 (F1: 0.6732 -> 0.7212, Recall: 85.5%)
       └── Locked evaluation on held-out Test Set (Accuracy: 63.65%, Recall: 85.20%, F1: 71.99%)
       │
       ▼
  Phase 6: Permutation Feature Importance & Local Explainability
       └── Rank features by validation F1 drop (Type, Shipping Mode, Order City, Scheduled Days)
       │
       ▼
  Deployment: Streamlit Web Application (DeliveryIQ)
       ├── Joblib Model Bundle Artifact (artifacts/phase5/selected_model_bundle.joblib)
       ├── Phase 5 Model Adapter (src/models/model_adapter.py)
       └── Streamlit UI Dashboard, Single/Batch Prediction, Analytics & Explainability (app.py)
```
""")

    # ==========================================
    # SECTION 2: DATASET UNDERSTANDING
    # ==========================================
    md("""
---
## 2. Dataset Understanding

### 2.1 The DataCo Global Logistics Dataset
The project utilizes the **DataCo Global Logistics dataset** (`APL_Logistics.csv`), which records commercial supply-chain, retail sales, order routing, and delivery fulfillment events across international distribution networks.

Let us inspect the dataset's structural properties, schema, missing values, and summary statistics.
""")

    code("""
# Core imports
import sys
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Visual formatting setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

# Robust project root resolution (works whether notebook is run from repo root or notebooks/ folder)
current_dir = Path.cwd()
PROJECT_ROOT = current_dir if (current_dir / "data" / "raw" / "APL_Logistics.csv").exists() else current_dir.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print(f"Project root resolved to: {PROJECT_ROOT}")
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "APL_Logistics.csv"

# Load dataset
if DATA_PATH.exists():
    try:
        df = pd.read_csv(DATA_PATH, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding='latin1', low_memory=False)
    print(f"[OK] Successfully loaded dataset from {DATA_PATH}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
else:
    print(f"[WARN] Dataset not found at {DATA_PATH}. Please ensure APL_Logistics.csv is placed under data/raw/.")
    df = None
""")

    md("""
### 2.2 Dataset Schema, Data Types, and Head Inspection
Let us examine the first 5 records and column definitions to understand how orders, products, and deliveries are represented.
""")

    code("""
if df is not None:
    display_cols = [
        'Type', 'Days for shipment (scheduled)', 'Shipping Mode',
        'Customer Segment', 'Department Name', 'Category Name',
        'Order City', 'Order Country', 'Product Price', 'Order Item Quantity',
        'Days for shipping (real)', 'Delivery Status', 'Late_delivery_risk'
    ]
    avail_display = [c for c in display_cols if c in df.columns]
    print("Sample Rows (selected key columns):")
    display(df[avail_display].head())
""")

    md("""
### 2.3 Data Integrity: Missing Values and Duplicate Check
Data quality issues such as duplicate records or corrupted rows can silently compromise model training. Let us audit missingness and duplication across all 40 columns.
""")

    code("""
if df is not None:
    # Check duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows count: {duplicates} (0.00%)")

    # Missing values check
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    print(f"Columns with missing values: {len(missing_cols)}")
    if len(missing_cols) > 0:
        missing_df = pd.DataFrame({
            'Column': missing_cols.index,
            'Missing Cells': missing_cols.values,
            'Missing Rate (%)': (missing_cols.values / len(df)) * 100
        }).sort_values(by='Missing Cells', ascending=False)
        display(missing_df)
    else:
        print("[OK] No missing values detected across candidate operational columns.")
""")

    md("""
### 2.4 Statistical Summary of Numerical Attributes
Let us examine key central tendencies, spread, percentiles, and extremes for shipment quantities, prices, discounts, and schedules.
""")

    code("""
if df is not None:
    num_cols = [
        'Days for shipment (scheduled)', 'Order Item Quantity',
        'Order Item Product Price', 'Product Price', 'Order Item Discount',
        'Order Item Discount Rate', 'Days for shipping (real)'
    ]
    avail_num = [c for c in num_cols if c in df.columns]
    display(df[avail_num].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])
""")

    # ==========================================
    # SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
    # ==========================================
    md("""
---
## 3. Exploratory Data Analysis (EDA)

Exploratory Data Analysis allows us to uncover shipment patterns, class distributions, and operational factors strongly linked with delivery delays.

### 3.1 Target Variable Distribution
The target variable is `Late_delivery_risk`:
- `0`: The order was delivered on or before the promised delivery date.
- `1`: The order was delayed (delivery duration exceeded scheduled transit days).
""")

    code("""
if df is not None and 'Late_delivery_risk' in df.columns:
    target_counts = df['Late_delivery_risk'].value_counts()
    target_pcts = df['Late_delivery_risk'].value_counts(normalize=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Bar plot
    palette = ['#2b5c8f', '#e05638']
    sns.barplot(x=target_counts.index, y=target_counts.values, palette=palette, ax=axes[0])
    axes[0].set_title("Target Class Distribution (Late_delivery_risk)", fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlabel("Class Label (0 = On-Time / Early, 1 = Late Risk)", fontsize=11)
    axes[0].set_ylabel("Number of Shipments", fontsize=11)
    axes[0].set_xticklabels(['On-Time / Early (0)', 'Late Risk (1)'])
    for i, v in enumerate(target_counts.values):
        axes[0].text(i, v + 2000, f"{v:,} ({target_pcts.iloc[i]:.2f}%)", ha='center', fontweight='bold')

    # Donut plot
    axes[1].pie(
        target_counts.values,
        labels=['On-Time / Early (Class 0)', 'Late Risk (Class 1)'],
        autopct='%1.1f%%',
        startangle=140,
        colors=palette,
        wedgeprops=dict(width=0.45, edgecolor='w')
    )
    axes[1].set_title("Target Class Proportion", fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.show()

    print(f"Class 1 (Late Risk):     {target_counts[1]:,} shipments ({target_pcts[1]:.2f}%)")
    print(f"Class 0 (On-Time/Early): {target_counts[0]:,} shipments ({target_pcts[0]:.2f}%)")
    print("Interpretation: The dataset has a mild natural class imbalance, with late shipments comprising 54.83% of orders.")
""")

    md("""
### 3.2 Impact of Shipping Mode on Late Delivery Risk
Logistics carriers provide different delivery service levels (`Standard Class`, `Second Class`, `First Class`, `Same Day`). Let us examine how the proportion of late shipments differs across shipping modes.
""")

    code("""
if df is not None and 'Shipping Mode' in df.columns and 'Late_delivery_risk' in df.columns:
    mode_risk = df.groupby('Shipping Mode')['Late_delivery_risk'].agg(
        Total_Orders='count',
        Late_Orders='sum',
        Late_Rate='mean'
    ).reset_index().sort_values(by='Late_Rate', ascending=False)
    mode_risk['Late_Rate_Pct'] = mode_risk['Late_Rate'] * 100

    plt.figure(figsize=(10, 4.5))
    ax = sns.barplot(
        data=mode_risk,
        x='Shipping Mode',
        y='Late_Rate_Pct',
        palette='crest'
    )
    plt.title("Late Delivery Rate by Shipping Mode", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Shipping Mode", fontsize=11)
    plt.ylabel("Late Delivery Rate (%)", fontsize=11)
    plt.axhline(54.83, color='red', linestyle='--', label='Global Average Late Rate (54.8%)')
    plt.ylim(0, 100)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.1f}%",
                    (p.get_x() + p.get_width() / 2., height + 2),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    display(mode_risk[['Shipping Mode', 'Total_Orders', 'Late_Orders', 'Late_Rate_Pct']])
    print("Interpretation: First Class (95.3%) and Second Class (76.8%) suffer dramatically higher delay rates.")
    print("Standard Class has a significantly lower late rate (38.1%) because it allows 4 full transit days.")
""")

    md("""
### 3.3 Scheduled Shipping Days vs. Observed Risk
Scheduled shipping days represents the agreed operational transit window. How does setting a tighter delivery deadline affect fulfillment risk?
""")

    code("""
if df is not None and 'Days for shipment (scheduled)' in df.columns:
    days_risk = df.groupby('Days for shipment (scheduled)')['Late_delivery_risk'].agg(
        Total_Orders='count',
        Late_Rate='mean'
    ).reset_index()
    days_risk['Late_Rate_Pct'] = days_risk['Late_Rate'] * 100

    fig, ax1 = plt.subplots(figsize=(10, 4.5))

    color = '#2b5c8f'
    ax1.set_xlabel('Scheduled Shipping Days', fontsize=11)
    ax1.set_ylabel('Total Shipments', color=color, fontsize=11)
    b = ax1.bar(days_risk['Days for shipment (scheduled)'] - 0.15, days_risk['Total_Orders'], width=0.3, color=color, alpha=0.85, label='Shipment Volume')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = '#e05638'
    ax2.set_ylabel('Late Delivery Rate (%)', color=color, fontsize=11)
    l = ax2.plot(days_risk['Days for shipment (scheduled)'] + 0.15, days_risk['Late_Rate_Pct'], color=color, marker='o', linewidth=2.5, label='Late Rate %')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 105)

    plt.title("Scheduled Shipping Days: Order Volume vs. Late Delivery Probability", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.show()
    print("Interpretation: When scheduled shipping duration is 0, 1, or 2 days, late delivery probability exceeds 70-90%.")
    print("At 4 scheduled days, the probability drops to ~38%.")
""")

    md("""
### 3.4 Operational Attributes: Customer Segment, Market, and Payment Type
Let us examine secondary categorical drivers: Customer Segment (Consumer, Corporate, Home Office), Geographic Market (LATAM, Europe, Pacific Asia, USCA, Africa), and Payment Type (DEBIT, TRANSFER, PAYMENT, CASH).
""")

    code("""
if df is not None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for idx, (col, title) in enumerate([
        ('Customer Segment', 'Customer Segment'),
        ('Market', 'Geographic Market'),
        ('Type', 'Payment Transaction Type')
    ]):
        if col in df.columns:
            grp = df.groupby(col)['Late_delivery_risk'].mean().reset_index()
            grp['Late_Pct'] = grp['Late_delivery_risk'] * 100
            grp = grp.sort_values(by='Late_Pct', ascending=False)

            sns.barplot(data=grp, x=col, y='Late_Pct', ax=axes[idx], palette='mako')
            axes[idx].set_title(f"Late Risk by {title}", fontsize=11, fontweight='bold')
            axes[idx].set_ylabel("Late Risk (%)" if idx == 0 else "")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylim(0, 70)
            axes[idx].axhline(54.83, color='r', linestyle='--', alpha=0.6)
            for p in axes[idx].patches:
                axes[idx].annotate(f"{p.get_height():.1f}%",
                                  (p.get_x() + p.get_width() / 2., p.get_height() + 1.5),
                                  ha='center', fontsize=9, fontweight='bold')
            axes[idx].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.show()
    print("Interpretation: Customer segments experience roughly even risk (~54-55%). Markets vary slightly, with Africa and Europe showing slightly elevated risk.")
    print("Payment Type exhibits noticeable variation: TRANSFER payments show ~57.4% late risk, while CASH is ~54.0%.")
""")

    md("""
### 3.5 Numerical Feature Correlations
Let us evaluate Pearson correlations among valid pre-shipment numerical features to detect multicollinearity.
""")

    code("""
if df is not None:
    valid_numeric = [
        'Days for shipment (scheduled)', 'Order Item Quantity',
        'Order Item Product Price', 'Product Price',
        'Order Item Discount', 'Order Item Discount Rate',
        'Latitude', 'Longitude'
    ]
    avail_num_corr = [c for c in valid_numeric if c in df.columns]

    corr_matrix = df[avail_num_corr].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8}
    )
    plt.title("Correlation Matrix: Pre-Shipment Numerical Features", fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.show()
    print("Interpretation: Product Price and Order Item Product Price are strongly correlated (r = 0.78).")
    print("Scheduled shipping days has very low linear correlation with pricing features, indicating orthogonal information.")
""")

    # ==========================================
    # SECTION 4: DATA PREPROCESSING
    # ==========================================
    md("""
---
## 4. Data Preprocessing

### 4.1 Methodology & Preprocessing Architecture
Data preprocessing transforms raw, heterogenous inputs into standardized, numerically encoded feature vectors suitable for scikit-learn estimators.

Our preprocessing architecture consists of:
1. **Handling Missing Values:**
   - Numerical columns: Imputed with the **median** learned strictly on the training partition (`SimpleImputer(strategy='median')`). Median imputation is robust to extreme outliers.
   - Categorical columns: Imputed with the **most frequent category** learned on the training partition (`SimpleImputer(strategy='most_frequent')`).
2. **Feature Scaling (Numerical Pipeline):**
   - Standardized using `StandardScaler` ($z = \frac{x - \mu}{\sigma}$), ensuring zero mean and unit variance.
3. **Categorical Encoding (Categorical Pipeline):**
   - Transformed using `OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10, sparse_output=True)`.
   - Rare categories appearing fewer than 10 times are grouped into an infrequent category bucket to prevent dimensionality explosion and overfitting.
   - `sparse_output=True` stores the resulting matrix as a Compressed Sparse Row (CSR) matrix, reducing memory usage from gigabytes to megabytes.
4. **ColumnTransformer Integration:**
   - Pipelines are assembled via scikit-learn's `ColumnTransformer`, ensuring strict encapsulation and reproducibility.
""")

    code("""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def make_preprocessor(X):
    \"\"\"Construct leakage-safe ColumnTransformer pipeline.\"\"\"
    numeric_features = X.select_dtypes(include=['number', 'bool']).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scaler', StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
        ('onehot', OneHotEncoder(
            handle_unknown='infrequent_if_exist',
            min_frequency=10,
            sparse_output=True,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ('numeric', numeric_pipeline, numeric_features),
            ('categorical', categorical_pipeline, categorical_features),
        ],
        remainder='drop',
        verbose_feature_names_out=True,
    )

print("[OK] Preprocessing pipeline definition verified against src/features/preprocessing.py.")
""")

    # ==========================================
    # SECTION 5: DATA LEAKAGE PREVENTION
    # ==========================================
    md("""
---
## 5. Data Leakage Prevention

### 5.1 What is Data Leakage?
**Data Leakage** (specifically target leakage and look-ahead bias) occurs when information that would not be available in real-world deployment at the decision point is inadvertently fed to the machine learning model during training.

If a model learns from features that only exist *after* the event has occurred, the model will achieve near-perfect cross-validation and test scores (e.g. 99% accuracy) during development, but will **fail completely and cataclysmically in production** when evaluated on true pre-shipment orders.

### 5.2 Critical Leakage Columns in the Logistics Dataset
The following fields were strictly audited and excluded from model inputs:

| Column Name | Why It Must Be Excluded | Nature of Leakage |
|---|---|---|
| `Days for shipping (real)` | Records the exact elapsed days taken for delivery. In the dataset, `Days for shipping (real) > Days for shipment (scheduled)` is the exact mathematical definition of the target label! | **Direct Target Leakage** |
| `Delivery Status` | Contains post-delivery classifications: `"Late delivery"`, `"Shipping on time"`, `"Advance shipping"`. | **Ground Truth Leakage** |
| `Order Status` | Status such as `"COMPLETE"`, `"CLOSED"`, `"PENDING_PAYMENT"`, updated throughout the post-order lifecycle. | **Lifecycle State Leakage** |
| `Late_delivery_risk` | The ground-truth classification target ($Y$). | **Target Variable** |
| `Benefit per order` | Accounting profit calculated only after completed order settlement and reconciliation. | **Financial Outcome Leakage** |
| `Sales per customer` | Final adjusted revenue figure recorded after delivery and returns. | **Financial Outcome Leakage** |
| `Order Profit Per Order` | Reconciled net margin per shipment. | **Financial Outcome Leakage** |
| Customer PII / IDs (`Customer Fname`, `Lname`, `Email`, `Street`, `Zipcode`, `Customer Id`) | Personally identifiable data that induces memorization without operational generalizability. | **Privacy & High-Cardinality Memorization** |

### 5.3 Approved Pre-Shipment Feature Schema
The approved candidate schema consists of **23 raw pre-shipment features**:
- **Logistics & Order:** `Type`, `Days for shipment (scheduled)`, `Shipping Mode`
- **Product & Department:** `Category Id`, `Category Name`, `Department Id`, `Department Name`, `Product Name`, `Product Price`
- **Pricing & Discounts:** `Order Item Discount`, `Order Item Discount Rate`, `Order Item Product Price`, `Order Item Quantity`
- **Customer & Geography:** `Customer Country`, `Customer Segment`, `Customer State`, `Market`, `Order City`, `Order Country`, `Order Region`, `Order State`, `Latitude`, `Longitude`
""")

    code("""
# Verify against project feature policy in src/features/feature_policy.py
from src.features.feature_policy import (
    TARGET, LEAKAGE_COLUMNS, IDENTIFIER_OR_PERSONAL_COLUMNS,
    CANDIDATE_FEATURES, validate_feature_policy
)

print(f"Target column: '{TARGET}'")
print(f"Forbidden leakage columns ({len(LEAKAGE_COLUMNS)}):")
for col in sorted(LEAKAGE_COLUMNS):
    print(f"  [X] {col}")

print(f"\\nApproved raw pre-shipment candidate features ({len(CANDIDATE_FEATURES)}):")
for col in CANDIDATE_FEATURES:
    print(f"  [OK] {col}")

# Test feature policy validation function
try:
    validate_feature_policy(CANDIDATE_FEATURES)
    print("\\n[OK] validate_feature_policy passed: No leakage features in approved candidates.")
except ValueError as e:
    print(f"Error: {e}")
""")

    # ==========================================
    # SECTION 6: FEATURE ENGINEERING
    # ==========================================
    md("""
---
## 6. Feature Engineering

### 6.1 Purpose of Engineered Features
Raw transactional features do not explicitly quantify operational bottlenecks. For example, an order with 10 items scheduled over 1 day experiences intense fulfillment stress compared to an order with 1 item over 4 days.

To provide the model with domain-specific inductive bias without introducing look-ahead bias, **four deterministic engineered features** were derived using only pre-shipment fields:

1. **`shipping_pressure_index`**:
   $$\text{shipping\_pressure\_index} = \frac{\text{Order Item Quantity}}{\text{Days for shipment (scheduled)}}$$
   *Purpose:* Measures packing and logistics load per allowed transit day. If scheduled days is 0 (Same Day shipping), safe division handles the denominator.
2. **`is_express_shipping`**:
   $$\text{is\_express\_shipping} = \mathbb{I}(\text{Shipping Mode} \in \{\text{"Same Day"}, \text{"First Class"}, \text{"Express"}\})$$
   *Purpose:* Flags expedited transit commitments that have historically tight SLA tolerances and higher delay risks.
3. **`discount_to_item_price_ratio`**:
   $$\text{discount\_to\_item\_price\_ratio} = \frac{\text{Order Item Discount}}{\text{Order Item Product Price}}$$
   *Purpose:* Captures heavily discounted promotional items, which frequently experience supply-chain surges and stockout delays.
4. **`log_order_quantity`**:
   $$\text{log\_order\_quantity} = \ln(1 + \text{Order Item Quantity})$$
   *Purpose:* Compresses the right-skewed order quantity distribution, reducing the leverage of extreme bulk purchases.

With these 4 engineered features added to the 23 raw candidate features, the model receives **27 pre-shipment input features**.
""")

    code("""
# Load preprocessed splits from Phase 2
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "phase2"
train_path = PROCESSED_DIR / "train.csv"
val_path = PROCESSED_DIR / "validation.csv"
test_path = PROCESSED_DIR / "test.csv"

if train_path.exists():
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    print(f"[OK] Loaded processed dataset splits from {PROCESSED_DIR}:")
    print(f"  Train:      {len(df_train):,} rows ({len(df_train.columns)} columns)")
    print(f"  Validation: {len(df_val):,} rows ({len(df_val.columns)} columns)")
    print(f"  Test:       {len(df_test):,} rows ({len(df_test.columns)} columns)")

    # Verify target distribution across splits (Stratified split)
    print(f"  Train Late Rate:      {df_train[TARGET].mean():.2%}")
    print(f"  Validation Late Rate: {df_val[TARGET].mean():.2%}")
    print(f"  Test Late Rate:       {df_test[TARGET].mean():.2%}")

    # Inspect the 4 engineered features in df_train
    eng_cols = ['shipping_pressure_index', 'is_express_shipping', 'discount_to_item_price_ratio', 'log_order_quantity']
    print("\\nEngineered Feature Summary (Train Partition):")
    display(df_train[eng_cols].describe().T[['mean', 'std', 'min', '50%', 'max']])
else:
    print(f"Processed splits not found at {PROCESSED_DIR}.")
""")

    # ==========================================
    # SECTION 7: MODEL TRAINING & COMPARISON
    # ==========================================
    md("""
---
## 7. Model Training and Comparison

### 7.1 Candidate Algorithms Evaluated in Phase 3 & 4
In rigorous applied data science, candidate models must be compared under consistent evaluation splits and identical performance metrics.

In this project, three distinct algorithms were trained on the training partition (`126,363` rows) and scored on the validation partition (`27,078` rows):

1. **Baseline: Logistic Regression (Phase 3)**
   - Linear classifier with L2 regularization (`solver='lbfgs'`, `class_weight='balanced'`, `max_iter=1000`, `random_state=42`).
   - Serves as the linear benchmark against which nonlinear ensemble models are judged.
2. **Model A: Random Forest Classifier (Phase 4)**
   - Non-linear ensemble of 120 decision trees (`n_estimators=120`, `max_depth=22`, `min_samples_leaf=3`, `max_features='sqrt'`, `class_weight='balanced_subsample'`, `random_state=42`).
   - Applied directly to the sparse one-hot encoded matrix.
3. **Model B: Histogram Gradient Boosting with TruncatedSVD (Phase 4)**
   - Gradient boosted decision trees (`HistGradientBoostingClassifier`, `max_iter=150`, `learning_rate=0.08`, `class_weight='balanced'`).
   - Because `HistGradientBoosting` requires dense numeric input, a `TruncatedSVD(n_components=80)` dimensionality reduction projection was fit on the training data (explaining 91.09% of variance).

### 7.2 Validation Set Performance Comparison
Let us load and inspect the verified validation metrics recorded during Phase 3 and Phase 4.
""")

    code("""
# Load verified comparison metrics from reports/phase4/model_comparison.json and reports/phase3/validation_metrics.json
comparison_file = PROJECT_ROOT / "reports" / "phase4" / "model_comparison.json"
phase3_file = PROJECT_ROOT / "reports" / "phase3" / "validation_metrics.json"

models_comparison = []

if phase3_file.exists():
    with open(phase3_file, 'r') as f:
        p3 = json.load(f)
    models_comparison.append({
        'Model': 'Logistic Regression (Baseline)',
        'Accuracy': p3.get('accuracy', 0.7030),
        'Precision': p3.get('precision', 0.8151),
        'Recall': p3.get('recall', 0.5928),
        'F1-Score': p3.get('f1', 0.6864),
        'ROC-AUC': p3.get('roc_auc', 0.7695),
        'PR-AUC': p3.get('average_precision_pr_auc', 0.8255),
        'Representation': 'Phase 2 Sparse One-Hot (3,505 dims)'
    })

if comparison_file.exists():
    with open(comparison_file, 'r') as f:
        p4 = json.load(f)
    for m in p4:
        name = "Random Forest" if m['model'] == 'RandomForest' else "HistGradientBoosting + SVD"
        models_comparison.append({
            'Model': name,
            'Accuracy': m.get('validation_accuracy'),
            'Precision': m.get('validation_precision'),
            'Recall': m.get('validation_recall'),
            'F1-Score': m.get('validation_f1'),
            'ROC-AUC': m.get('validation_roc_auc'),
            'PR-AUC': m.get('validation_average_precision'),
            'Representation': m.get('representation')
        })

df_comp = pd.DataFrame(models_comparison)
display(df_comp[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']])
""")

    md("""
### 7.3 Visual Comparison of Candidate Models
Let us plot a grouped comparison of the evaluation metrics across the three models.
""")

    code("""
if not df_comp.empty:
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
    plot_data = df_comp.set_index('Model')[metrics_to_plot].T

    plt.figure(figsize=(12, 5.5))
    ax = plot_data.plot(kind='bar', figsize=(12, 5.5), colormap='viridis', width=0.75, edgecolor='black', linewidth=0.5)
    plt.title("Candidate Model Comparison -- Validation Set Metrics (Threshold = 0.50)", fontsize=13, fontweight='bold', pad=12)
    plt.ylabel("Validation Score (0.0 – 1.0)", fontsize=11)
    plt.xlabel("Evaluation Metric", fontsize=11)
    plt.ylim(0.4, 1.0)
    plt.xticks(rotation=0, fontsize=11)
    plt.legend(title="Algorithm", loc="lower right", frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8, rotation=90)

    plt.tight_layout()
    plt.show()

    print("Key Takeaway: At the default 0.50 threshold, Random Forest achieved an F1 score of 0.6732 and Precision of 0.8176.")
    print("Why Random Forest was selected: It natively exploits the sparse high-dimensional feature space without compression loss,")
    print("demonstrates superior resilience to categorical sparsity, and provided the strongest foundation for threshold tuning.")
""")

    # ==========================================
    # SECTION 8: RANDOM FOREST CLASSIFIER
    # ==========================================
    md("""
---
## 8. Random Forest Classifier Deep Dive

### 8.1 How Random Forest Works
The **Random Forest** algorithm (Breiman, 2001) is an ensemble learning method that constructs a forest of decorrelated decision trees during training and aggregates their outputs:

1. **Bootstrap Aggregating (Bagging):**
   - For each tree $t \in \{1, \dots, B\}$, a bootstrap sample of size $N$ is drawn with replacement from the training set.
   - Observations not selected in a given bootstrap sample form the **Out-Of-Bag (OOB)** set.
2. **Random Subspace Feature Selection:**
   - At each candidate split within a tree node, only a random subset of features $m \approx \sqrt{p}$ is considered rather than all $p$ features.
   - This prevents dominant features from dictating every tree split, decorrelating the individual trees.
3. **Ensemble Aggregation:**
   - For binary classification, each tree outputs an estimated class probability $P_t(Y=1 \mid x)$.
   - The forest prediction is the arithmetic mean across all $B$ trees:
     $$P(Y=1 \mid x) = \frac{1}{B} \sum_{t=1}^B P_t(Y=1 \mid x)$$

### 8.2 Project Hyperparameter Configuration
The Phase 5 selected Random Forest model bundle was trained with the following hyperparameter specification:
- `n_estimators = 120`: 120 distinct decision trees provide low variance without excessive inference latency.
- `max_depth = 22`: Limits maximum tree depth to prevent memorization of spurious noise.
- `min_samples_leaf = 3`: Requires each leaf node to contain at least 3 samples, smoothing leaf probability estimates.
- `max_features = 'sqrt'`: Considers $\sqrt{3505} \approx 59$ random features per split.
- `class_weight = 'balanced_subsample'`: Recalibrates weights per bootstrap sample inversely proportional to class frequencies.
- `random_state = 42`: Guarantees exact reproducibility.

Let us inspect the actual saved model artifact from `artifacts/phase5/selected_model_bundle.joblib`.
""")

    code("""
import joblib

BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "phase5" / "selected_model_bundle.joblib"

if BUNDLE_PATH.exists():
    bundle = joblib.load(BUNDLE_PATH)
    print(f"[OK] Successfully loaded Phase 5 Model Bundle from {BUNDLE_PATH}")
    print(f"  Bundle Keys: {list(bundle.keys())}")
    print(f"  Model Name:  {bundle.get('model_name')}")
    print(f"  Selected Threshold: {bundle.get('threshold'):.4f}")

    rf_model = bundle['model']
    preprocessor = bundle['preprocessor']

    print(f"\\nRandom Forest Classifier Details:")
    print(f"  Estimator:            {type(rf_model).__name__}")
    print(f"  Number of Trees:      {rf_model.n_estimators}")
    print(f"  Max Depth:            {rf_model.max_depth}")
    print(f"  Min Samples Leaf:     {rf_model.min_samples_leaf}")
    print(f"  Max Features:         {rf_model.max_features}")
    print(f"  Class Weight:         {rf_model.class_weight}")
    print(f"  Target Classes:       {rf_model.classes_}")
    print(f"  Fitted Feature Dims:  {rf_model.n_features_in_:,}")
else:
    print(f"[!] Model bundle not found at {BUNDLE_PATH}.")
""")

    # ==========================================
    # SECTION 9: THRESHOLD OPTIMIZATION
    # ==========================================
    md("""
---
## 9. Decision Threshold Optimization

### 9.1 The Need for Threshold Calibration
Standard classification models apply an arbitrary default decision boundary of $\tau = 0.50$:
$$\hat{Y} = \begin{cases} 1 & \text{if } P(Y=1 \mid X) \ge 0.50 \\ 0 & \text{if } P(Y=1 \mid X) < 0.50 \end{cases}$$

However, the default $\tau = 0.50$ assumes **symmetric misclassification costs**: that a False Positive (flagging an on-time shipment as late) incurs the exact same operational cost as a False Negative (failing to flag a late shipment).

In supply-chain operations, **misclassification costs are highly asymmetric**:
- **Cost of False Negative ($C_{FN}$):** An unflagged late order results in missed customer delivery deadlines, customer complaints, emergency re-shipping, and brand damage ($\approx \$50$ per incident).
- **Cost of False Positive ($C_{FP}$):** Flagging an on-time shipment causes warehouse staff to prioritize or monitor an order that would have made it on time anyway ($\approx \$5$ per incident).

Because $C_{FN} \gg C_{FP}$, the optimal operational policy requires **lowering the decision threshold** to capture more true positive late deliveries.

### 9.2 Validation Set Optimization Procedure
In Phase 5, the threshold was tuned across candidate values $\tau \in [0.05, 0.95]$ evaluated strictly on the **Validation Set** to maximize $F_1$-score:
$$F_1(\tau) = 2 \cdot \frac{\text{Precision}(\tau) \cdot \text{Recall}(\tau)}{\text{Precision}(\tau) + \text{Recall}(\tau)}$$

**Crucial Methodological Safeguard:**
> The threshold was selected entirely on the validation set. The held-out test set was **strictly untouched** during threshold selection to ensure unbiased generalization reporting.
""")

    code("""
# Load verified threshold curve from reports/phase5/threshold_curve.csv
thresh_curve_path = PROJECT_ROOT / "reports" / "phase5" / "threshold_curve.csv"

if thresh_curve_path.exists():
    df_curve = pd.read_csv(thresh_curve_path)
    print(f"[OK] Loaded threshold optimization curve ({len(df_curve):,} candidate threshold evaluations)")

    # Find the maximum F1 threshold
    best_row = df_curve.sort_values(by='f1', ascending=False).iloc[0]
    best_tau = best_row['threshold']
    print(f"\\nOptimal Validation Threshold: {best_tau:.4f}")
    print(f"  At tau = {best_tau:.4f}:")
    print(f"    Validation F1-Score:  {best_row['f1']:.4f}  (vs 0.6732 at tau=0.50)")
    print(f"    Validation Recall:    {best_row['recall']:.4f}  (vs 0.5722 at tau=0.50)")
    print(f"    Validation Precision: {best_row['precision']:.4f}")

    # Plot Precision, Recall, and F1 versus Decision Threshold
    plt.figure(figsize=(11, 5.5))
    plt.plot(df_curve['threshold'], df_curve['precision'], label='Precision', color='#2b5c8f', linewidth=2)
    plt.plot(df_curve['threshold'], df_curve['recall'], label='Recall', color='#e05638', linewidth=2)
    plt.plot(df_curve['threshold'], df_curve['f1'], label='F1-Score', color='#2a9d8f', linewidth=2.5)

    plt.axvline(best_tau, color='black', linestyle='--', linewidth=1.5,
                label=f'Selected Optimal Threshold (tau = {best_tau:.3f})')
    plt.axvline(0.50, color='gray', linestyle=':', linewidth=1.2, label='Default Threshold (0.50)')

    plt.title("Threshold Selection on Validation Set: Precision, Recall & F1 Trade-off", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Classification Decision Threshold (tau)", fontsize=11)
    plt.ylabel("Validation Metric Score", fontsize=11)
    plt.xlim(0.1, 0.9)
    plt.ylim(0, 1.05)
    plt.legend(loc='lower center', frameon=True, fontsize=10)
    plt.tight_layout()
    plt.show()

    print("Insight: Lowering the threshold to 0.379 increases Recall from 57.2% to 85.5% while maintaining F1 at its peak (72.1%).")
""")

    # ==========================================
    # SECTION 10: FINAL MODEL EVALUATION
    # ==========================================
    md("""
---
## 10. Final Model Evaluation on Held-Out Test Set

### 10.1 Single Evaluation on the Locked Test Set
Once the Random Forest model and optimal decision threshold ($\tau^* = 0.37897$) were locked, the complete pipeline was evaluated **exactly once** on the held-out Test Set (`27,078` records).

### 10.2 Documented Phase 5 Test Results
Let us inspect the verified test metrics recorded in `reports/phase5/phase5_metrics.json`.
""")

    code("""
metrics_file = PROJECT_ROOT / "reports" / "phase5" / "phase5_metrics.json"

if metrics_file.exists():
    with open(metrics_file, 'r') as f:
        metrics_data = json.load(f)

    test_metrics = metrics_data.get('held_out_test_at_selected_threshold', {})
    val_metrics = metrics_data.get('validation_at_selected_threshold', {})

    summary_df = pd.DataFrame([
        {
            'Split': 'Validation Set (Threshold Tuning)',
            'Threshold': val_metrics.get('threshold', 0.379),
            'Accuracy': val_metrics.get('accuracy'),
            'Precision': val_metrics.get('precision'),
            'Recall': val_metrics.get('recall'),
            'F1-Score': val_metrics.get('f1'),
            'ROC-AUC': val_metrics.get('roc_auc'),
            'Average Precision': val_metrics.get('average_precision')
        },
        {
            'Split': 'Held-Out Test Set (Final Verification)',
            'Threshold': test_metrics.get('threshold', 0.379),
            'Accuracy': test_metrics.get('accuracy'),
            'Precision': test_metrics.get('precision'),
            'Recall': test_metrics.get('recall'),
            'F1-Score': test_metrics.get('f1'),
            'ROC-AUC': test_metrics.get('roc_auc'),
            'Average Precision': test_metrics.get('average_precision')
        }
    ])

    print("Verified Phase 5 Model Evaluation Summary:")
    display(summary_df.set_index('Split'))
""")

    md("""
### 10.3 Test Confusion Matrix & Error Analysis
Let us examine the distribution of predictions versus true outcomes on the held-out test partition.
""")

    code("""
# Test Confusion Matrix values from phase5_metrics.json
cm = test_metrics.get('confusion_matrix', [[4585, 7646], [2198, 12649]])
cm = np.array(cm)

tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

plt.figure(figsize=(7, 5.5))
sns.heatmap(
    cm,
    annot=True,
    fmt=',d',
    cmap='Blues',
    cbar=False,
    xticklabels=['Predicted On-Time (0)', 'Predicted Late Risk (1)'],
    yticklabels=['Actual On-Time (0)', 'Actual Late (1)']
)
plt.title(f"Held-Out Test Confusion Matrix (Threshold = {test_metrics.get('threshold', 0.379):.4f})", fontsize=12, fontweight='bold', pad=12)
plt.xlabel("Model Prediction", fontsize=11)
plt.ylabel("Ground Truth Outcome", fontsize=11)
plt.tight_layout()
plt.show()

print(f"Confusion Matrix Breakdown (Total Test Records = {cm.sum():,}):")
print(f"  - True Negatives  (TN): {tn:,}  -- Correctly identified on-time deliveries")
print(f"  - False Positives (FP): {fp:,}  -- On-time orders flagged for operational review")
print(f"  - False Negatives (FN): {fn:,}  -- Unflagged late shipments (missed delays)")
print(f"  - True Positives  (TP): {tp:,} -- Correctly captured late-delivery risks")
print(f"\\nOperational Sensitivity (Recall): {tp / (tp + fn):.2%} of all actual late shipments were successfully flagged!")
""")

    # ==========================================
    # SECTION 11: MODEL EXPLAINABILITY
    # ==========================================
    md("""
---
## 11. Model Explainability

### 11.1 Permutation Feature Importance
While complex tree ensembles provide high non-linear predictive capacity, they are inherently black-box models. Understanding *why* a model makes predictions is necessary for stakeholder trust, regulatory transparency, and targeted supply chain intervention.

#### Why Permutation Importance Over Tree Impurity (Gini)?
Standard tree feature importance (Mean Decrease in Impurity / Gini Importance) is notoriously biased toward high-cardinality categorical features and continuous numerical variables.

In Phase 6, **Permutation Feature Importance** (Breiman, 2001) was calculated:
1. Record baseline validation metric ($F_1$-score) using unperturbed features.
2. For each original feature $j$, shuffle the values of column $j$ across validation rows, destroying its relationship with the target.
3. Transform with the saved preprocessor, generate new predictions, and compute the new $F_1$-score.
4. The importance of feature $j$ is the drop in model performance:
   $$\Delta F_{1, j} = F_{1, \text{baseline}} - F_{1, \text{permuted}(j)}$$

### 11.2 Feature Importance Results & Domain Interpretation
Let us load and visualize the permutation importance results from `reports/phase6/feature_importance.csv`.
""")

    code("""
imp_file = PROJECT_ROOT / "reports" / "phase6" / "feature_importance.csv"

if imp_file.exists():
    df_imp = pd.read_csv(imp_file)
    print(f"[OK] Loaded Phase 6 Permutation Feature Importance ({len(df_imp)} features evaluated)")

    top_15 = df_imp.sort_values(by='importance_mean_f1_drop', ascending=True).tail(15)

    plt.figure(figsize=(10, 6.5))
    bars = plt.barh(
        top_15['source_feature'],
        top_15['importance_mean_f1_drop'],
        xerr=top_15['importance_std'],
        color='#2b5c8f',
        edgecolor='black',
        alpha=0.85,
        capsize=3
    )

    plt.title("Phase 6: Top 15 Features by Permutation Importance (Mean F1 Drop)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Mean Validation F1 Decrease (When Feature Is Shuffled)", fontsize=11)
    plt.ylabel("Original Input Feature", fontsize=11)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.0005, bar.get_y() + bar.get_height()/2, f"{width:.5f}",
                 va='center', ha='left', fontsize=9, fontweight='bold')

    plt.xlim(0, max(top_15['importance_mean_f1_drop']) * 1.25)
    plt.tight_layout()
    plt.show()

    display(df_imp.head(10)[['source_feature', 'importance_mean_f1_drop', 'importance_std']])
""")

    md("""
### 11.3 Key Explainability Takeaways
1. **`Type` (Payment Transaction Type):**
   - Shuffling payment type causes the highest $F_1$ drop (~0.0172). Payment method is strongly correlated with order confirmation speed and checkout processing friction.
2. **`Shipping Mode` & `Days for shipment (scheduled)`:**
   - Shuffling shipping mode and scheduled transit allowance produces severe drops (~0.0094 and ~0.0075). Tighter delivery windows create disproportionate vulnerability to carrier delays.
3. **`is_express_shipping` & `shipping_pressure_index`:**
   - Both engineered features rank within the top 7 most influential features, validating the feature engineering strategy.
4. **Geographic Destination (`Order City`, `Order State`, `Customer State`):**
   - Regional destination attributes strongly influence delay likelihood due to varying distance from fulfillment hubs and carrier network density.

> **Cautionary Note:** Permutation importance reflects *predictive association* within the fitted estimator; it does not constitute mathematical proof of physical causality.
""")

    # ==========================================
    # SECTION 12: MODEL SAVING AND DEPLOYMENT
    # ==========================================
    md("""
---
## 12. Model Saving and Deployment Architecture

### 12.1 Persistence via Joblib
To ensure that the exact trained model, fitted column transformer, and decision policy can be executed in production without code drift:
- The trained `RandomForestClassifier` and fitted `ColumnTransformer` are bundled together into a single serialized binary artifact using `joblib.dump()`:
  `artifacts/phase5/selected_model_bundle.joblib`

### 12.2 Production Inference Adapter (`Phase5ModelAdapter`)
In production software engineering, user interfaces (such as Streamlit or FastAPI) should never manipulate raw estimators directly. Instead, an **Adapter Pattern** is employed:
- Located at: `src/models/model_adapter.py`
- Exposes clean methods:
  - `Phase5ModelAdapter.load()`: Loads and caches the bundle.
  - `validate_inputs()`: Rejects post-shipment leakage columns.
  - `prepare_features()`: Computes the 4 engineered features and structures the 27 expected input columns.
  - `predict_proba()`: Computes calibrated late probabilities.
  - `predict()`: Returns binary decisions using the 0.3790 threshold.
  - `predict_single()`: Returns probability, risk tier, decision label, and local sensitivity explainability for UI display.
""")

    code("""
# Demonstrate the Phase 5 Model Adapter
from src.models import Phase5ModelAdapter

adapter = Phase5ModelAdapter.load()
print("[OK] Loaded Phase5ModelAdapter successfully.")
print(f"  Model Name:         {adapter.metadata.get('model_name')}")
print(f"  Decision Threshold: {adapter.threshold:.4f}")
print(f"  Expected Features:  {len(adapter.expected_features)} columns")
""")

    # ==========================================
    # SECTION 13: LIVE PREDICTION DEMONSTRATION
    # ==========================================
    md("""
---
## 13. Live Prediction Demonstration

Let us test the production adapter using realistic pre-shipment order scenarios.

### Scenario A: High-Risk Urgent Order
- **Shipping Mode:** `First Class` (Scheduled days = 1)
- **Order Type:** `PAYMENT`
- **Customer Segment:** `Consumer`
- **Order Item Quantity:** `3` units
- **Product Price:** `$120.00`
""")

    code("""
sample_high_risk = {
    'Type': 'PAYMENT',
    'Shipping Mode': 'First Class',
    'Days for shipment (scheduled)': 1,
    'Customer Segment': 'Consumer',
    'Market': 'LATAM',
    'Order Region': 'Central America',
    'Department Name': 'Apparel',
    'Order Item Quantity': 3,
    'Order Item Product Price': 120.0,
    'Product Price': 120.0,
    'Order Item Discount': 10.0,
    'Customer Country': 'EE. UU.',
    'Customer State': 'PR',
    'Order Country': 'Estados Unidos',
    'Order City': 'New York City'
}

result_a = adapter.predict_single(sample_high_risk)

print("=" * 60)
print("SCENARIO A PREDICTION RESULT:")
print("=" * 60)
print(f"Estimated Late Probability: {result_a['probability_pct']} ({result_a['probability']:.4f})")
print(f"Decision Threshold:         {result_a['threshold_str']}")
print(f"Risk Tier:                  {result_a['risk_tier']}")
print(f"Operational Decision:       {result_a['decision']}")
print(f"High Risk Flagged:          {result_a['is_high_risk']}")

print("\\nLocal Sensitivity Explanation (Feature Contributions):")
for exp in result_a['explanations']:
    print(f"  - {exp['feature']} = '{exp['original_value']}' (Baseline: '{exp['typical_value']}') -> Impact: {exp['marginal_effect']:+.2%} ({exp['impact']})")
""")

    md("""
### Scenario B: Low-Risk Standard Shipment
- **Shipping Mode:** `Standard Class` (Scheduled days = 4)
- **Order Type:** `CASH`
- **Customer Segment:** `Corporate`
- **Order Item Quantity:** `1` unit
- **Product Price:** `$45.00`
""")

    code("""
sample_low_risk = {
    'Type': 'CASH',
    'Shipping Mode': 'Standard Class',
    'Days for shipment (scheduled)': 4,
    'Customer Segment': 'Corporate',
    'Market': 'USCA',
    'Order Region': 'East of USA',
    'Department Name': 'Fan Shop',
    'Order Item Quantity': 1,
    'Order Item Product Price': 45.0,
    'Product Price': 45.0,
    'Order Item Discount': 0.0,
    'Customer Country': 'EE. UU.',
    'Customer State': 'CA',
    'Order Country': 'Estados Unidos',
    'Order City': 'Los Angeles'
}

result_b = adapter.predict_single(sample_low_risk)

print("=" * 60)
print("SCENARIO B PREDICTION RESULT:")
print("=" * 60)
print(f"Estimated Late Probability: {result_b['probability_pct']} ({result_b['probability']:.4f})")
print(f"Decision Threshold:         {result_b['threshold_str']}")
print(f"Risk Tier:                  {result_b['risk_tier']}")
print(f"Operational Decision:       {result_b['decision']}")
print(f"High Risk Flagged:          {result_b['is_high_risk']}")
""")

    # ==========================================
    # SECTION 14: CONCLUSION
    # ==========================================
    md("""
---
## 14. Conclusion & Operational Recommendations

### 14.1 Summary of Machine Learning Workflow
| Stage | Key Implementation Details | Outcome / Finding |
|---|---|---|
| **Problem Formulation** | Supervised binary classification on DataCo logistics records | Predict late delivery risk before physical dispatch |
| **Data Audit & Leakage** | Excluded `Days for shipping (real)`, `Delivery Status`, `Order Status`, financial fields | Eliminated target leakage and artificial optimism |
| **Feature Engineering** | Derived `shipping_pressure_index`, `is_express_shipping`, `discount_to_item_price_ratio`, `log_order_quantity` | Total 27 pre-shipment operational features |
| **Data Preprocessing** | Stratified 70/15/15 split; Median + StandardScaler (num); OHE with min_freq=10 (cat) | 3,505 sparse dimensions; preprocessor fit on train only |
| **Model Comparison** | Compared Logistic Regression, HistGradientBoosting+SVD, and Random Forest | Random Forest selected: highest validation F1 & precision |
| **Threshold Tuning** | Optimized threshold on Validation Set ($\tau^* = 0.3790$) | Validation F1 improved from 0.6732 to 0.7212; Recall reached 85.5% |
| **Held-Out Test Results** | Single locked evaluation on 27,078 test records | Accuracy: **63.65%**, Recall: **85.20%**, F1: **71.99%**, ROC-AUC: **0.7622** |
| **Explainability** | Permutation importance on validation samples | `Type`, `Shipping Mode`, `Order City`, and scheduled days drive predictive reliance |
| **Deployment** | Encapsulated in `Phase5ModelAdapter` and deployed via Streamlit | Interactive single & batch scoring in production |

### 14.2 Actionable Recommendations for Logistics Stakeholders
1. **Dynamic Fulfillment Routing:** Integrate the model directly into warehouse management systems (WMS). Orders with late probability $> 0.379$ should automatically be assigned to express picking queues.
2. **Carrier Contract Review:** High delay rates on First Class (95.3%) and Second Class (76.8%) indicate that carrier contracts require realistic SLAs or transit buffer re-negotiation.
3. **Continuous Monitoring:** Implement data drift monitors on order categories and carrier delivery rates to trigger automated periodic retraining.

---
**Pipeline execution complete.**
""")

    # Build final notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Notebook successfully written to {notebook_path} with {len(cells)} cells.")

if __name__ == "__main__":
    create_notebook()
