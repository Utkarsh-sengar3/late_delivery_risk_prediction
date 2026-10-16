import pandas as pd
from src.data.loader import inspect_dataset
from src.features.feature_policy import get_candidate_features, LEAKAGE_COLUMNS

def test_inspect_dataset():
    result = inspect_dataset(pd.DataFrame({"Late_delivery_risk": [0, 1, 1], "x": [1, None, 3]}))
    assert result["rows"] == 3
    assert result["columns"] == 2
    assert result["missing_cells"] == 1
    assert result["target_rate"] == 2 / 3

def test_candidate_features():
    assert get_candidate_features(["Type", "Shipping Mode", "Late_delivery_risk"]) == ["Type", "Shipping Mode"]
    assert "Late_delivery_risk" in LEAKAGE_COLUMNS
