"""Phase 1: reproducible dataset audit and exploratory data analysis."""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataset
from src.features.feature_policy import TARGET, LEAKAGE_COLUMNS, IDENTIFIER_OR_PERSONAL_COLUMNS

DATA_PATH = ROOT / "data" / "raw" / "APL_Logistics.csv"
REPORT_DIR = ROOT / "reports" / "phase1"
PLOT_DIR = REPORT_DIR / "plots"

def json_safe(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(DATA_PATH)
    if df is None:
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Copy APL_Logistics(1).csv to data/raw/APL_Logistics.csv and rerun."
        )

    df.columns = [str(c).strip() for c in df.columns]
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        df[col] = df[col].replace({"": pd.NA, "NULL": pd.NA, "null": pd.NA, "N/A": pd.NA, "NA": pd.NA})

    missing = df.isna().sum().sort_values(ascending=False)
    duplicate_rows = int(df.duplicated().sum())
    target_present = TARGET in df.columns
    summary = {
        "dataset_file": DATA_PATH.name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": duplicate_rows,
        "duplicate_row_percent": round(100 * duplicate_rows / max(len(df), 1), 4),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_cell_percent": round(100 * df.isna().sum().sum() / max(df.size, 1), 4),
        "target_column": TARGET,
        "target_present": target_present,
        "column_names": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing_by_column": {k: int(v) for k, v in missing.items()},
        "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
        "high_cardinality_columns": [
            c for c in df.select_dtypes(include=["object", "string"]).columns
            if df[c].nunique(dropna=True) > min(1000, max(100, int(0.25 * len(df))))
        ],
        "leakage_columns_present": [c for c in LEAKAGE_COLUMNS if c in df.columns],
        "identifier_or_personal_columns_present": [c for c in IDENTIFIER_OR_PERSONAL_COLUMNS if c in df.columns],
    }

    if target_present:
        target = df[TARGET]
        summary["target_missing"] = int(target.isna().sum())
        summary["target_unique_values"] = [json_safe(x) for x in target.dropna().unique().tolist()[:50]]
        counts = target.value_counts(dropna=False)
        summary["target_distribution"] = {str(json_safe(k)): int(v) for k, v in counts.items()}
        numeric_target = pd.to_numeric(target, errors="coerce")
        valid = numeric_target.dropna()
        summary["target_values_outside_0_1"] = int((~valid.isin([0, 1])).sum())
        summary["target_late_rate_if_binary"] = (
            float(valid.mean()) if len(valid) and valid.isin([0, 1]).all() else None
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        # FIX: plot numeric frequencies. Converting the counts to strings caused
        # pandas/matplotlib to raise "TypeError: no numeric data to plot".
        counts.plot(kind="bar", ax=ax)
        ax.set_title("Target distribution (including missing labels)")
        ax.set_xlabel(TARGET)
        ax.set_ylabel("Rows")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "target_distribution.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    positive_missing = missing[missing > 0].head(25).sort_values()
    if not positive_missing.empty:
        positive_missing.plot(kind="barh", ax=ax)
    else:
        ax.text(0.5, 0.5, "No missing values found", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Missing values by column (top 25)")
    ax.set_xlabel("Missing cells")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "missing_values.png", dpi=150)
    plt.close(fig)

    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in IDENTIFIER_OR_PERSONAL_COLUMNS]
    if numeric_cols:
        df[numeric_cols].describe().T.to_csv(REPORT_DIR / "numeric_summary.csv", index=True)
        corr = df[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(max(9, len(numeric_cols) * 0.55), max(7, len(numeric_cols) * 0.45)))
        image = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=7)
        ax.set_title("Numeric feature correlation (descriptive; not causal)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "numeric_correlation.png", dpi=150)
        plt.close(fig)

    with (REPORT_DIR / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    missing.rename_axis("column").reset_index(name="missing_cells").to_csv(
        REPORT_DIR / "missing_values.csv", index=False
    )

    md = [
        "# Phase 1 — Data Audit and EDA", "",
        "## Dataset overview",
        f"- Rows: **{summary['rows']:,}**",
        f"- Columns: **{summary['columns']:,}**",
        f"- Duplicate rows: **{summary['duplicate_rows']:,}** ({summary['duplicate_row_percent']}%)",
        f"- Missing cells: **{summary['missing_cells']:,}** ({summary['missing_cell_percent']}%)",
        f"- Target present: **{summary['target_present']}**", "", "## Target",
    ]
    if target_present:
        md += [
            f"- Target missing labels: **{summary['target_missing']:,}**",
            f"- Target distribution: `{summary['target_distribution']}`",
            f"- Values outside binary 0/1: **{summary['target_values_outside_0_1']:,}**",
        ]
    md += [
        "", "## Data quality findings",
        f"- Constant columns: `{summary['constant_columns']}`",
        f"- High-cardinality text columns: `{summary['high_cardinality_columns']}`",
        f"- Outcome/leakage-sensitive columns present: `{summary['leakage_columns_present']}`",
        f"- Personal/identifier columns present (column names only): `{summary['identifier_or_personal_columns_present']}`",
        "", "## Interpretation guardrails",
        "- Correlations do not establish causality.",
        "- Review duplicates before deleting them.",
        "- Exclude actual shipping duration, delivery status, target, and post-outcome fields from pre-shipment features.",
        "- The schema does not show a clear order-created timestamp; valid temporal evaluation may require additional data.",
        "- Review feature availability with an operations/domain owner before model training.",
        "", "## Outputs",
        "- `audit_summary.json`, `missing_values.csv`, `numeric_summary.csv` (when numeric columns exist), and `plots/`",
    ]
    (REPORT_DIR / "EDA_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print("PHASE 1 COMPLETE")
    print(f"Rows: {summary['rows']:,} | Columns: {summary['columns']:,}")
    print(f"Duplicate rows: {duplicate_rows:,} | Missing cells: {summary['missing_cells']:,}")
    print(f"Report directory: {REPORT_DIR}")

if __name__ == "__main__":
    main()
