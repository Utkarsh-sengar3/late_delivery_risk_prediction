import pandas as pd
import pytest
from src.features.build_features import build_feature_frame
from src.features.preprocessing import make_preprocessor
from src.data.split import split_dataset

def sample_data(n=40):
    return pd.DataFrame({
        "Late_delivery_risk": [i % 2 for i in range(n)],
        "Type": ["Debit" if i % 2 else "Credit" for i in range(n)],
        "Days for shipment (scheduled)": [2 + (i % 3) for i in range(n)],
        "Order Item Quantity": [1 + (i % 4) for i in range(n)],
        "Shipping Mode": ["Standard Class" if i % 2 else "First Class" for i in range(n)],
        "Days for shipping (real)": [99] * n,
        "Delivery Status": ["Late delivery"] * n,
        "Customer Fname": ["Private"] * n,
    })

def test_feature_engineering_excludes_leakage_and_personal_fields():
    X, y, meta = build_feature_frame(sample_data())
    assert "Days for shipping (real)" not in X.columns
    assert "Delivery Status" not in X.columns
    assert "Customer Fname" not in X.columns
    assert "shipping_pressure_index" in X.columns
    assert len(X) == len(y) == 40

def test_split_sizes_and_target_stratification():
    X, y, _ = build_feature_frame(sample_data(100))
    Xtr, Xv, Xte, ytr, yv, yte = split_dataset(X, y)
    assert len(Xtr) + len(Xv) + len(Xte) == 100
    assert abs(ytr.mean() - 0.5) < 0.03
    assert abs(yv.mean() - 0.5) < 0.05
    assert abs(yte.mean() - 0.5) < 0.05

def test_preprocessor_fits_and_transforms():
    X, y, _ = build_feature_frame(sample_data())
    pre = make_preprocessor(X.iloc[:30])
    train = pre.fit_transform(X.iloc[:30])
    test = pre.transform(X.iloc[30:])
    assert train.shape[1] == test.shape[1]
    assert train.shape[0] == 30
