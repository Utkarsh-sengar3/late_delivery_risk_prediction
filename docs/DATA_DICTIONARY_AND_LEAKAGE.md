# Data and leakage notes

Target: `Late_delivery_risk` (binary).

Exclude from pre-shipment model inputs:
- `Days for shipping (real)`: actual duration, only known after shipment.
- `Delivery Status`: describes the delivery outcome.
- `Order Status`: may reveal fulfillment state/outcome.
- `Late_delivery_risk`: target, never a feature.
- `Benefit per order`, `Sales per customer`, `Order Profit Per Order`: likely post-outcome or outcome-related; exclude until timing is confirmed.

Exclude direct identifiers/personal fields by default: customer names, street, ZIP/postal code and IDs. Never expose these in public dashboards or logs.

The supplied schema does not show a clear order-created timestamp, so a trustworthy temporal holdout may not be possible. Verify data provenance, target semantics, field availability at prediction time and dataset license before publication. Treat results as a retrospective prototype until operationally validated.
