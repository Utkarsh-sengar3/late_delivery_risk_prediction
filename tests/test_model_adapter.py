"""Tests for Phase 5 Model Adapter and Prediction Pipeline."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.models import (
    Phase5ModelAdapter,
    DEFAULT_THRESHOLD,
    LEAKAGE_COLUMNS,
    TYPICAL_VALUES,
)


@pytest.fixture(scope="module")
def adapter():
    return Phase5ModelAdapter.load()


def test_adapter_loading(adapter):
    """Verify that Phase 5 model bundle loads successfully with expected components."""
    assert adapter is not None
    assert adapter.model is not None
    assert adapter.preprocessor is not None
    assert 0.35 <= adapter.threshold <= 0.40
    assert len(adapter.expected_features) == 27


def test_feature_preparation_and_engineering(adapter):
    """Verify deterministic feature engineering matches Phase 2 pipeline."""
    sample = {
        "Type": "DEBIT",
        "Shipping Mode": "First Class",
        "Days for shipment (scheduled)": 1,
        "Order Item Quantity": 2,
        "Order Item Product Price": 100.0,
        "Order Item Discount": 10.0,
    }
    prepared = adapter.prepare_features(sample)
    assert isinstance(prepared, pd.DataFrame)
    assert list(prepared.columns) == adapter.expected_features
    assert prepared["shipping_pressure_index"].iloc[0] == pytest.approx(2.0 / 1.0)
    assert prepared["is_express_shipping"].iloc[0] == 1.0
    assert prepared["discount_to_item_price_ratio"].iloc[0] == pytest.approx(10.0 / 100.0)
    assert prepared["log_order_quantity"].iloc[0] == pytest.approx(np.log1p(2.0))


def test_data_leakage_prevention(adapter):
    """Ensure that post-shipment or outcome-dependent features are rejected."""
    for forbidden_col in ["Days for shipping (real)", "Delivery Status", "Late_delivery_risk", "Order Status"]:
        with pytest.raises(ValueError, match="Data leakage detected"):
            adapter.validate_inputs({forbidden_col: "some_value", "Type": "DEBIT"})


def test_prediction_output_properties(adapter):
    """Ensure model probabilities are strictly between 0 and 1, and threshold logic holds."""
    sample = {
        "Type": "PAYMENT",
        "Shipping Mode": "Standard Class",
        "Days for shipment (scheduled)": 4,
        "Order Item Quantity": 1,
        "Order Item Product Price": 50.0,
    }
    probs = adapter.predict_proba(sample)
    assert len(probs) == 1
    assert 0.0 <= probs[0] <= 1.0

    preds = adapter.predict(sample, threshold=0.3790)
    assert preds[0] in [0, 1]
    assert preds[0] == int(probs[0] >= 0.3790)


def test_predict_single_and_explanation(adapter):
    """Ensure predict_single produces full explainable risk contract."""
    sample = {
        "Type": "TRANSFER",
        "Shipping Mode": "First Class",
        "Days for shipment (scheduled)": 1,
        "Order Item Quantity": 3,
        "Order Item Product Price": 150.0,
    }
    res = adapter.predict_single(sample)
    assert "probability" in res
    assert "threshold" in res
    assert "risk_tier" in res
    assert "decision" in res
    assert "explanations" in res
    assert isinstance(res["explanations"], list)
    for exp in res["explanations"]:
        assert "feature" in exp
        assert "marginal_effect" in exp
        assert "impact" in exp


def test_batch_prediction(adapter):
    """Ensure batch scoring processes data frames and appends correct prediction columns."""
    df = pd.DataFrame([
        {"Type": "DEBIT", "Shipping Mode": "Standard Class", "Days for shipment (scheduled)": 4},
        {"Type": "PAYMENT", "Shipping Mode": "First Class", "Days for shipment (scheduled)": 1},
    ])
    scored = adapter.batch_predict(df)
    assert len(scored) == 2
    assert "predicted_late_probability" in scored.columns
    assert "predicted_late_risk" in scored.columns
    assert "risk_decision" in scored.columns


def test_simple_imputer_cross_version_compatibility(adapter):
    """Verify that all SimpleImputer instances have both _fill_dtype and _fit_dtype populated."""
    imputers_found = 0
    for _, trans, _ in adapter.preprocessor.transformers_:
        if hasattr(trans, "steps"):
            for _, step in trans.steps:
                if type(step).__name__ == "SimpleImputer":
                    imputers_found += 1
                    assert hasattr(step, "_fill_dtype"), "SimpleImputer must have _fill_dtype for sklearn 1.4-1.6"
                    assert hasattr(step, "_fit_dtype"), "SimpleImputer must have _fit_dtype for sklearn 1.7+"
                    assert step._fill_dtype is not None
                    assert step._fit_dtype is not None
    assert imputers_found >= 2, f"Expected at least 2 SimpleImputers, found {imputers_found}"


def test_imputer_auto_healing(adapter):
    """Simulate missing _fill_dtype or _fit_dtype and verify auto-healing mechanism."""
    from src.models import ensure_imputer_compatibility

    # Temporarily remove attribute to test auto-healing
    for _, trans, _ in adapter.preprocessor.transformers_:
        if hasattr(trans, "steps"):
            for _, step in trans.steps:
                if type(step).__name__ == "SimpleImputer":
                    if hasattr(step, "_fill_dtype"):
                        delattr(step, "_fill_dtype")
                    assert not hasattr(step, "_fill_dtype")

    # Call healing function
    ensure_imputer_compatibility(adapter.preprocessor)

    # Verify both attributes are re-attached
    for _, trans, _ in adapter.preprocessor.transformers_:
        if hasattr(trans, "steps"):
            for _, step in trans.steps:
                if type(step).__name__ == "SimpleImputer":
                    assert hasattr(step, "_fill_dtype")
                    assert hasattr(step, "_fit_dtype")

    # Verify transform succeeds without exception
    sample = {"Shipping Mode": "Standard Class", "Days for shipment (scheduled)": 4}
    res = adapter.predict_single(sample)
    assert "probability" in res


def test_data_validation_robustness(adapter):
    """Test handling of missing, unknown categorical, and unexpected extra columns."""
    # 1. Missing values and unexpected columns
    noisy_sample = {
        "Shipping Mode": "Standard Class",
        "Days for shipment (scheduled)": None,
        "Order Item Quantity": np.nan,
        "Unexpected_Column_XYZ": "Ignore me",
        "Market": "UNKNOWN_NEW_MARKET_2026",
        "Order Country": "NON_EXISTENT_COUNTRY_99",
    }
    res = adapter.predict_single(noisy_sample)
    assert 0.0 <= res["probability"] <= 1.0
    assert res["risk_tier"] in ["Low Risk", "Moderate / Monitored", "High Risk", "Critical Risk"]

    # 2. Batch with diverse edge cases
    batch_data = pd.DataFrame([
        {"Shipping Mode": "First Class", "Days for shipment (scheduled)": 1, "Order Item Quantity": 1},
        {"Shipping Mode": "Standard Class", "Days for shipment (scheduled)": 4, "Order Item Quantity": 5},
        {"Shipping Mode": "UNKNOWN", "Days for shipment (scheduled)": 2, "Order Item Quantity": 2},
    ])
    scored = adapter.batch_predict(batch_data)
    assert len(scored) == 3
    assert scored["predicted_late_probability"].between(0.0, 1.0).all()
    assert set(scored["predicted_late_risk"].unique()).issubset({0, 1})
