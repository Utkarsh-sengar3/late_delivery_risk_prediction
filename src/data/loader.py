from pathlib import Path
from typing import Optional
import pandas as pd

TARGET_COLUMN = "Late_delivery_risk"

def load_dataset(path: str | Path, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8", nrows=nrows, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", nrows=nrows, low_memory=False)

def inspect_dataset(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "target_present": TARGET_COLUMN in df.columns,
        "target_rate": float(pd.to_numeric(df[TARGET_COLUMN], errors="coerce").mean())
            if TARGET_COLUMN in df.columns else None,
    }
