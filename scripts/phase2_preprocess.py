"""Phase 2: feature engineering, split data, and save reusable preprocessing pipeline."""
from pathlib import Path
import json
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataset
from src.data.split import split_dataset
from src.features.build_features import build_feature_frame
from src.features.preprocessing import make_preprocessor

DATA_PATH = ROOT / "data" / "raw" / "APL_Logistics.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "phase2"
ARTIFACT_DIR = ROOT / "artifacts" / "phase2"
REPORT_DIR = ROOT / "reports" / "phase2"

def main():
    df = load_dataset(DATA_PATH)
    if df is None:
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Copy your CSV to data/raw/APL_Logistics.csv first."
        )
    df.columns = [str(col).strip() for col in df.columns]
    X, y, metadata = build_feature_frame(df)

    if y.nunique() != 2:
        raise ValueError(
            f"Expected binary target with both classes; found labels: {sorted(y.unique().tolist())}"
        )
    if y.value_counts().min() < 3:
        raise ValueError("Too few examples in one target class to create stratified splits.")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Store untransformed, selected feature rows for transparent inspection.
    # These files can contain location/category data; keep them out of public Git.
    split_frames = {
        "train": (X_train, y_train),
        "validation": (X_val, y_val),
        "test": (X_test, y_test),
    }
    split_summary = {}
    for name, (features, labels) in split_frames.items():
        out = features.copy()
        out["Late_delivery_risk"] = labels
        out.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        split_summary[name] = {
            "rows": int(len(out)),
            "class_0": int((labels == 0).sum()),
            "class_1": int((labels == 1).sum()),
            "class_1_rate": float(labels.mean()),
        }

    # Fit preprocessing ONLY on training data; do not fit on validation/test.
    preprocessor = make_preprocessor(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)
    joblib.dump(preprocessor, ARTIFACT_DIR / "preprocessor.joblib")

    try:
        transformed_feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        transformed_feature_names = []

    metadata.update({
        "input_features_after_engineering": X.columns.tolist(),
        "numeric_features": X.select_dtypes(include=["number", "bool"]).columns.tolist(),
        "categorical_features": [c for c in X.columns if c not in X.select_dtypes(include=["number", "bool"]).columns],
        "split_fractions_approx": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "split_summary": split_summary,
        "transformed_shapes": {
            "train": list(X_train_transformed.shape),
            "validation": list(X_val_transformed.shape),
            "test": list(X_test_transformed.shape),
        },
        "transformed_feature_count": int(X_train_transformed.shape[1]),
        "preprocessing_fit_on": "train only",
        "random_state": 42,
        "note": "Random stratified split used because no trustworthy order timestamp was identified. Do not claim temporal generalization.",
    })
    with (REPORT_DIR / "phase2_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    (REPORT_DIR / "PHASE_2_REPORT.md").write_text(
        "# Phase 2 — Preprocessing and Feature Engineering\n\n"
        f"- Rows with valid binary target: **{metadata['rows_after_target_filter']:,}**\n"
        f"- Rows dropped for invalid/missing target: **{metadata['rows_dropped_for_invalid_target']:,}**\n"
        f"- Input features after engineering: **{len(metadata['input_features_after_engineering'])}**\n"
        f"- Transformed feature count: **{metadata['transformed_feature_count']:,}**\n"
        f"- Preprocessor fitted on training data only: **Yes**\n\n"
        "## Split summary\n\n"
        + "\n".join(
            f"- {name.title()}: {info['rows']:,} rows; class 0={info['class_0']:,}; "
            f"class 1={info['class_1']:,}; late rate={info['class_1_rate']:.2%}"
            for name, info in split_summary.items()
        )
        + "\n\n## Engineered features\n\n"
        + "\n".join(f"- `{col}`" for col in metadata["engineered_features"])
        + "\n\n## Important limitations\n\n"
        "- Random stratified splitting is used because a trustworthy order timestamp was not identified.\n"
        "- Feature availability at the actual pre-shipment decision point must be confirmed.\n"
        "- Derived features are proxies and should not be interpreted as causal drivers.\n"
        "- The held-out test set must not be used for model selection or threshold tuning.\n",
        encoding="utf-8",
    )
    print("PHASE 2 COMPLETE")
    print(f"Valid labeled rows: {len(X):,}")
    print(f"Train / validation / test: {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
    print(f"Features before encoding: {X.shape[1]}")
    print(f"Features after encoding: {X_train_transformed.shape[1]:,}")
    print(f"Saved outputs: {OUTPUT_DIR}")
    print(f"Saved preprocessor: {ARTIFACT_DIR / 'preprocessor.joblib'}")
    print(f"Saved report: {REPORT_DIR / 'PHASE_2_REPORT.md'}")

if __name__ == "__main__":
    main()
