"""Phase 5 model adapter for pre-shipment late delivery risk prediction.

Loads the selected Random Forest pipeline and preprocessing artifacts,
validates pre-shipment input schema, prevents data leakage,
computes deterministic engineered features, generates calibrated probabilities,
and provides local sensitivity-based explanations.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

# Root resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "phase5" / "selected_model_bundle.joblib"
DEFAULT_THRESHOLD = 0.3789747461496299  # Phase 5 locked validation F1 threshold

# Strict leakage columns that MUST NOT be used for pre-shipment prediction
LEAKAGE_COLUMNS = {
    "Late_delivery_risk",
    "Days for shipping (real)",
    "Delivery Status",
    "Order Status",
    "Benefit per order",
    "Sales per customer",
    "Order Profit Per Order",
}

# Personal / tracking identifiers that do not carry operational predictive signal
IDENTIFIER_COLUMNS = {
    "Customer Fname",
    "Customer Lname",
    "Customer Email",
    "Customer Password",
    "Customer Street",
    "Customer Zipcode",
    "Customer Id",
    "Order Customer Id",
    "Order Id",
    "Order Item Id",
    "Product Description",
}

# Baseline typical values computed from training data (medians for numerics, modes for categoricals)
TYPICAL_VALUES: Dict[str, Any] = {
    "Type": "DEBIT",
    "Days for shipment (scheduled)": 4.0,
    "Category Id": 29.0,
    "Category Name": "Cleats",
    "Customer Country": "EE. UU.",
    "Customer Segment": "Consumer",
    "Customer State": "PR",
    "Department Id": 5.0,
    "Department Name": "Fan Shop",
    "Latitude": 33.13901901,
    "Longitude": -76.81587219,
    "Market": "LATAM",
    "Order City": "New York City",
    "Order Country": "Estados Unidos",
    "Order Item Discount": 14.0,
    "Order Item Discount Rate": 0.1,
    "Order Item Product Price": 59.99,
    "Order Item Quantity": 1.0,
    "Order Region": "Central America",
    "Order State": "Inglaterra",
    "Product Name": "Perfect Fitness Perfect Rip Deck",
    "Product Price": 59.99,
    "Shipping Mode": "Standard Class",
    "shipping_pressure_index": 0.5,
    "is_express_shipping": 0.0,
    "discount_to_item_price_ratio": 0.1500075,
    "log_order_quantity": 0.69314718,
}


class Phase5ModelAdapter:
    """Production inference adapter for the Phase 5 Random Forest model bundle."""

    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        threshold: float = DEFAULT_THRESHOLD,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = float(threshold)
        self.metadata = metadata or {}

        # Cache expected preprocessor feature schema
        if hasattr(preprocessor, "feature_names_in_"):
            self.expected_features: List[str] = list(preprocessor.feature_names_in_)
        else:
            self.expected_features = list(TYPICAL_VALUES.keys())

    @classmethod
    def load(cls, bundle_path: Optional[Union[str, Path]] = None) -> Phase5ModelAdapter:
        """Load and instantiate the adapter from a saved joblib bundle."""
        path = Path(bundle_path) if bundle_path else DEFAULT_BUNDLE_PATH
        if not path.exists():
            # Try alternate fallback locations
            candidates = [
                PROJECT_ROOT / "artifacts" / "phase5" / "selected_model_bundle.joblib",
                Path("artifacts/phase5/selected_model_bundle.joblib"),
                Path("../artifacts/phase5/selected_model_bundle.joblib"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"Phase 5 model bundle not found at {path}. Ensure artifacts/phase5/selected_model_bundle.joblib exists."
                )

        bundle = joblib.load(path)
        if not isinstance(bundle, dict):
            raise ValueError(f"Expected dict bundle in {path}, got {type(bundle)}")

        model = bundle.get("model") or bundle.get("estimator") or bundle.get("classifier")
        preprocessor = bundle.get("preprocessor") or bundle.get("transformer")
        threshold = bundle.get("threshold", bundle.get("selected_threshold", DEFAULT_THRESHOLD))

        if model is None or preprocessor is None:
            raise ValueError(
                f"Bundle at {path} is missing 'model' or 'preprocessor'. Available keys: {list(bundle.keys())}"
            )

        metadata = {
            "model_name": bundle.get("model_name", "RandomForest"),
            "selection_metric": bundle.get("selection_metric", "validation F1"),
            "threshold": float(threshold),
            "bundle_path": str(path),
        }
        return cls(model=model, preprocessor=preprocessor, threshold=float(threshold), metadata=metadata)

    def validate_inputs(self, df_or_dict: Union[pd.DataFrame, Dict[str, Any]]) -> None:
        """Enforce strict pre-shipment boundaries and data validation."""
        keys = set(df_or_dict.keys()) if isinstance(df_or_dict, dict) else set(df_or_dict.columns)

        # Check for forbidden outcome-dependent or post-shipment leakage columns
        forbidden_present = keys & LEAKAGE_COLUMNS
        if forbidden_present:
            raise ValueError(
                f"Data leakage detected! Pre-shipment prediction cannot use outcome features: {sorted(forbidden_present)}. "
                "These features are only known after dispatch or delivery."
            )

    def prepare_features(self, input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        """Map raw pre-shipment inputs to the exact 27-feature schema with deterministic feature engineering."""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

        self.validate_inputs(df)

        # Fill any missing required columns with baseline defaults
        for col, default_val in TYPICAL_VALUES.items():
            if col not in df.columns:
                df[col] = default_val

        # Coerce known numeric features
        numeric_cols = [
            "Days for shipment (scheduled)", "Category Id", "Department Id",
            "Latitude", "Longitude", "Order Item Discount",
            "Order Item Discount Rate", "Order Item Product Price",
            "Order Item Quantity", "Product Price",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(TYPICAL_VALUES[col])

        # Feature Engineering: exact match with Phase 2 build_feature_frame
        scheduled = pd.to_numeric(df["Days for shipment (scheduled)"], errors="coerce")
        quantity = pd.to_numeric(df["Order Item Quantity"], errors="coerce").clip(lower=0)
        item_price = pd.to_numeric(df["Order Item Product Price"], errors="coerce")
        discount = pd.to_numeric(df["Order Item Discount"], errors="coerce").fillna(0.0)

        # 1. shipping_pressure_index = Order Item Quantity / Days for shipment (scheduled)
        safe_denom = scheduled.where(scheduled > 0)
        df["shipping_pressure_index"] = (quantity / safe_denom).fillna(0.0)

        # 2. is_express_shipping = 1 if mode in express modes else 0
        mode_str = df["Shipping Mode"].astype(str).str.strip().str.lower()
        df["is_express_shipping"] = mode_str.isin(["same day", "first class", "express"]).astype(float)

        # 3. discount_to_item_price_ratio = discount / item_price
        safe_price = item_price.where(item_price > 0)
        df["discount_to_item_price_ratio"] = (discount / safe_price).fillna(0.0)

        # 4. log_order_quantity = log1p(quantity)
        df["log_order_quantity"] = np.log1p(quantity).fillna(0.0)

        # Ensure all columns are clean and in the exact expected preprocessor order
        df = df.replace([np.inf, -np.inf], np.nan)
        return df[self.expected_features]

    def predict_proba(self, X: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> np.ndarray:
        """Compute the estimated probability of late delivery (class 1)."""
        df_prepared = self.prepare_features(X)
        X_trans = self.preprocessor.transform(df_prepared)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_trans)
            # Find class 1 index safely
            classes = list(getattr(self.model, "classes_", [0, 1]))
            idx = classes.index(1) if 1 in classes else probs.shape[1] - 1
            return probs[:, idx]
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X_trans)
            return 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
        else:
            return np.asarray(self.model.predict(X_trans), dtype=float)

    def predict(
        self,
        X: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Classify binary risk using the specified or Phase 5 locked threshold."""
        thresh = float(threshold) if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def explain_single(
        self,
        df_prepared: pd.DataFrame,
        base_probability: float,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Sensitivity-based local explanation: replaces key features with typical values to measure marginal effect."""
        salient_features = [
            "Shipping Mode",
            "Days for shipment (scheduled)",
            "Type",
            "Order Item Quantity",
            "Customer Segment",
            "Market",
            "Department Name",
            "Order State",
            "Product Price",
            "Order Item Discount",
        ]

        effects = []
        for feat in salient_features:
            if feat not in df_prepared.columns:
                continue

            orig_val = df_prepared.iloc[0][feat]
            typ_val = TYPICAL_VALUES.get(feat)
            if typ_val is None or orig_val == typ_val:
                continue

            perturbed = df_prepared.copy()
            perturbed.iloc[0, perturbed.columns.get_loc(feat)] = typ_val

            # Recompute linked engineered features if primary input changed
            if feat in ("Days for shipment (scheduled)", "Order Item Quantity", "Shipping Mode", "Order Item Product Price", "Order Item Discount"):
                sched = float(perturbed["Days for shipment (scheduled)"].iloc[0])
                qty = float(perturbed["Order Item Quantity"].iloc[0])
                mode = str(perturbed["Shipping Mode"].iloc[0]).strip().lower()
                price = float(perturbed["Order Item Product Price"].iloc[0])
                disc = float(perturbed["Order Item Discount"].iloc[0])

                perturbed["shipping_pressure_index"] = (qty / sched) if sched > 0 else 0.0
                perturbed["is_express_shipping"] = 1.0 if mode in ["same day", "first class", "express"] else 0.0
                perturbed["discount_to_item_price_ratio"] = (disc / price) if price > 0 else 0.0
                perturbed["log_order_quantity"] = float(np.log1p(max(0, qty)))

            p_trans = self.preprocessor.transform(perturbed)
            p_new = float(self.model.predict_proba(p_trans)[0, 1])
            diff = base_probability - p_new  # positive means original value increased risk

            effects.append({
                "feature": feat,
                "original_value": orig_val,
                "typical_value": typ_val,
                "marginal_effect": diff,
                "impact": "Increases late risk" if diff > 0.005 else "Decreases late risk" if diff < -0.005 else "Neutral impact",
            })

        effects.sort(key=lambda x: abs(x["marginal_effect"]), reverse=True)
        return effects[:top_k]

    def predict_single(
        self,
        input_dict: Dict[str, Any],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform full prediction and local explainability for a single shipment assessment."""
        thresh = float(threshold) if threshold is not None else self.threshold
        df_prepared = self.prepare_features(input_dict)
        prob = float(self.predict_proba(df_prepared)[0])
        is_high_risk = bool(prob >= thresh)

        # Risk band policy
        if prob >= 0.65:
            risk_tier = "Critical Risk"
            badge = "danger"
        elif prob >= thresh:
            risk_tier = "High Risk"
            badge = "warning"
        elif prob >= 0.30:
            risk_tier = "Moderate / Monitored"
            badge = "info"
        else:
            risk_tier = "Low Risk"
            badge = "success"

        explanation = self.explain_single(df_prepared, prob, top_k=5)

        return {
            "probability": prob,
            "probability_pct": f"{prob:.1%}",
            "threshold": thresh,
            "threshold_str": f"{thresh:.4f}",
            "is_high_risk": is_high_risk,
            "decision": "Flagged (Late Delivery Likely)" if is_high_risk else "Not Flagged (On-Time Likely)",
            "risk_tier": risk_tier,
            "badge": badge,
            "model_name": self.metadata.get("model_name", "Random Forest"),
            "explanations": explanation,
            "prepared_record": df_prepared.iloc[0].to_dict(),
        }

    def batch_predict(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Batch score a DataFrame, appending predicted probability and risk label."""
        thresh = float(threshold) if threshold is not None else self.threshold
        probs = self.predict_proba(df)
        flags = (probs >= thresh).astype(int)

        out = df.copy()
        out["predicted_late_probability"] = np.round(probs, 4)
        out["predicted_late_risk"] = flags
        out["risk_decision"] = np.where(flags == 1, "High Risk", "Not Flagged")
        return out
