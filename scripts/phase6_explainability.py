"""Phase 6: explain selected Random Forest using manual permutation importance.

This version intentionally avoids sklearn.inspection.permutation_importance,
which requires estimator.fit and has had compatibility issues with the adapter.
It computes the baseline F1, permutes one raw feature at a time, transforms with
the saved preprocessor, and records the drop in F1. The test set is not used.
"""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "phase2"
BUNDLE_PATH = ROOT / "artifacts" / "phase5" / "selected_model_bundle.joblib"
REPORT_DIR = ROOT / "reports" / "phase6"
TARGET = "Late_delivery_risk"
RANDOM_STATE = 42
SAMPLE_SIZE = 2000
N_REPEATS = 3

def load_split(name):
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run Phase 2 first.")
    frame = pd.read_csv(path)
    if TARGET not in frame:
        raise ValueError(f"Target {TARGET!r} missing in {path}.")
    X = frame.drop(columns=[TARGET])
    y = pd.to_numeric(frame[TARGET], errors="raise").astype("int8")
    return X, y

def predict_probabilities(model, preprocessor, X):
    return model.predict_proba(preprocessor.transform(X))[:, 1]

def score_at_threshold(model, preprocessor, X, y, threshold):
    probabilities = predict_probabilities(model, preprocessor, X)
    predictions = (probabilities >= threshold).astype("int8")
    return float(f1_score(y, predictions, zero_division=0)), probabilities, predictions

def json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)

def main():
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing selected model bundle: {BUNDLE_PATH}. Run Phase 5 first."
        )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(BUNDLE_PATH)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    threshold = float(bundle["threshold"])

    X_val, y_val = load_split("validation")
    rng = np.random.default_rng(RANDOM_STATE)
    n = min(SAMPLE_SIZE, len(X_val))
    sample_indices = np.sort(rng.choice(len(X_val), size=n, replace=False))
    X_sample = X_val.iloc[sample_indices].reset_index(drop=True)
    y_sample = y_val.iloc[sample_indices].reset_index(drop=True)

    print(f"Explainability sample: {n:,} validation rows")
    print("Calculating baseline validation F1...")
    baseline_f1, base_prob, base_pred = score_at_threshold(
        model, preprocessor, X_sample, y_sample, threshold
    )
    print(f"Baseline F1 at locked threshold {threshold:.4f}: {baseline_f1:.4f}")
    print("Computing permutation importance on raw features (this can take several minutes)...")

    records = []
    # Sequential by design: avoids copying a large model across worker processes.
    for idx, col in enumerate(X_sample.columns, start=1):
        drops = []
        for repeat in range(N_REPEATS):
            permuted = X_sample.copy()
            permuted[col] = rng.permutation(permuted[col].to_numpy())
            score, _, _ = score_at_threshold(
                model, preprocessor, permuted, y_sample, threshold
            )
            drops.append(baseline_f1 - score)
        records.append({
            "source_feature": col,
            "importance_mean_f1_drop": float(np.mean(drops)),
            "importance_std": float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0,
            "permutation_repeats": N_REPEATS,
        })
        if idx % 5 == 0 or idx == len(X_sample.columns):
            print(f"  Processed {idx}/{len(X_sample.columns)} raw features")

    importance = pd.DataFrame(records).sort_values(
        "importance_mean_f1_drop", ascending=False
    )
    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    top = importance.head(20).sort_values("importance_mean_f1_drop")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["source_feature"], top["importance_mean_f1_drop"], xerr=top["importance_std"])
    ax.set_xlabel("Mean decrease in validation F1 when feature is permuted")
    ax.set_title("Global Feature Importance (Permutation, Validation Sample)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "global_feature_importance.png", dpi=160)
    plt.close(fig)

    print("Building local case explanations...")
    probabilities = base_prob
    predictions = base_pred
    sample_results = X_sample.copy()
    sample_results["actual_label"] = y_sample
    sample_results["predicted_probability"] = probabilities
    sample_results["predicted_label"] = predictions
    high_idx = np.argsort(probabilities)[-3:][::-1]
    low_idx = np.argsort(probabilities)[:3]
    selected_rows = list(dict.fromkeys(list(high_idx) + list(low_idx)))
    local_records = []

    typical_values = {}
    for col in X_sample.columns:
        if pd.api.types.is_numeric_dtype(X_sample[col]):
            typical_values[col] = X_sample[col].median()
        else:
            modes = X_sample[col].mode(dropna=True)
            typical_values[col] = modes.iloc[0] if not modes.empty else None

    for row_idx in selected_rows:
        original = X_sample.iloc[[row_idx]].copy()
        base_probability = float(probabilities[row_idx])
        changes = []
        for col in X_sample.columns:
            perturbed = original.copy()
            perturbed[col] = typical_values[col]
            p = float(predict_probabilities(model, preprocessor, perturbed)[0])
            changes.append({
                "feature": col,
                "original_value": json_value(original.iloc[0][col]),
                "typical_value": json_value(typical_values[col]),
                "probability_after_replacing_with_typical": p,
                "probability_change_from_original": p - base_probability,
            })
        changes.sort(key=lambda item: abs(item["probability_change_from_original"]), reverse=True)
        local_records.append({
            "sample_row": int(row_idx),
            "actual_label": int(y_sample.iloc[row_idx]),
            "predicted_probability": base_probability,
            "predicted_label": int(predictions[row_idx]),
            "threshold": threshold,
            "top_feature_sensitivity": changes[:8],
        })

    selected_cases = sample_results.iloc[selected_rows].copy()
    selected_cases.insert(0, "sample_row", selected_rows)
    selected_cases.to_csv(REPORT_DIR / "local_case_examples.csv", index=False)
    (REPORT_DIR / "local_case_explanations.json").write_text(
        json.dumps(local_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    top_rows = importance.head(15)
    summary = {
        "model": bundle.get("model_name", "RandomForest"),
        "threshold": threshold,
        "validation_sample_size": n,
        "baseline_f1": baseline_f1,
        "permutation_repeats": N_REPEATS,
        "scoring_metric": "F1 at locked Phase 5 threshold",
        "top_features": [
            {
                "feature": str(row["source_feature"]),
                "mean_f1_drop": float(row["importance_mean_f1_drop"]),
                "std": float(row["importance_std"]),
            }
            for _, row in top_rows.iterrows()
        ],
        "interpretation_note": "Permutation importance measures predictive reliance under this validation sample; it does not establish causation.",
    }
    (REPORT_DIR / "phase6_explainability_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# Phase 6 — Model Explainability",
        "",
        "## Method",
        "- Model: Random Forest selected in Phase 5",
        f"- Locked threshold: {threshold:.4f}",
        f"- Validation sample size: {n:,}",
        f"- Baseline F1 on this sample: {baseline_f1:.4f}",
        f"- Permutation repeats per feature: {N_REPEATS}",
        "- Permutation importance is calculated manually on original input features, applying the saved preprocessor before inference.",
        "- Six local examples are included: up to three high-risk and three low-risk predictions.",
        "",
        "## Top 15 features",
        "",
        "| Feature | Mean F1 decrease | Standard deviation |",
        "|---|---:|---:|",
    ]
    for _, row in top_rows.iterrows():
        lines.append(
            f"| `{row['source_feature']}` | {row['importance_mean_f1_drop']:.6f} | {row['importance_std']:.6f} |"
        )
    lines += [
        "",
        "## Interpretation cautions",
        "",
        "- Importance is predictive association, not causation.",
        "- Correlated features can share or mask importance.",
        "- Permutation importance uses a reproducible sample of up to 2,000 validation rows to keep runtime manageable.",
        "- Local explanations replace one input at a time with a typical sample value; these are sensitivity analyses, not causal or counterfactual guarantees.",
        "- Do not expose customer-identifying fields in the UI or reports.",
        "- Evaluation used a random stratified split, not a time-based holdout.",
    ]
    (REPORT_DIR / "PHASE_6_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nPHASE 6 COMPLETE")
    print(f"Validation sample: {n:,}")
    print("\nTop 10 features by permutation importance:")
    for _, row in importance.head(10).iterrows():
        print(f"- {row['source_feature']}: {row['importance_mean_f1_drop']:.6f}")
    print(f"Saved report: {REPORT_DIR / 'PHASE_6_REPORT.md'}")

if __name__ == "__main__":
    main()
