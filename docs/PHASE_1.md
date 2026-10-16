# Phase 1 — Data Audit and EDA (corrected)

Run from the project root:
```powershell
python scripts/phase1_data_audit.py
```
Dataset path: `data/raw/APL_Logistics.csv`

Outputs: `reports/phase1/EDA_REPORT.md`, `audit_summary.json`, `missing_values.csv`, numeric summary (if available), and plots.

This version fixes the target-distribution plotting error by plotting numeric value counts instead of converting the counts to strings. It also handles datasets with no missing values without failing.
