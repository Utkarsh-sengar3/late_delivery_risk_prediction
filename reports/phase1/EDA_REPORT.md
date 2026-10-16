# Phase 1 — Data Audit and EDA

## Dataset overview
- Rows: **180,519**
- Columns: **40**
- Duplicate rows: **0** (0.0%)
- Missing cells: **11** (0.0002%)
- Target present: **True**

## Target
- Target missing labels: **0**
- Target distribution: `{'1': 98977, '0': 81542}`
- Values outside binary 0/1: **0**

## Data quality findings
- Constant columns: `[]`
- High-cardinality text columns: `['Customer Lname', 'Customer Street', 'Order City', 'Order State']`
- Outcome/leakage-sensitive columns present: `['Days for shipping (real)', 'Delivery Status', 'Order Status', 'Late_delivery_risk', 'Benefit per order', 'Sales per customer', 'Order Profit Per Order']`
- Personal/identifier columns present (column names only): `['Customer Fname', 'Customer Lname', 'Customer Street', 'Customer Zipcode', 'Customer Id', 'Order Customer Id']`

## Interpretation guardrails
- Correlations do not establish causality.
- Review duplicates before deleting them.
- Exclude actual shipping duration, delivery status, target, and post-outcome fields from pre-shipment features.
- The schema does not show a clear order-created timestamp; valid temporal evaluation may require additional data.
- Review feature availability with an operations/domain owner before model training.

## Outputs
- `audit_summary.json`, `missing_values.csv`, `numeric_summary.csv` (when numeric columns exist), and `plots/`