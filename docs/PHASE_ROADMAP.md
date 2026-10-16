# Phase-by-phase delivery roadmap

| Phase | Deliverable | Acceptance criteria |
|---|---|---|
| 0 | Architecture starter (this ZIP) | App opens, loader works, privacy and leakage guardrails documented |
| 1 | Data audit + EDA | Reproducible profile, target balance, missingness, duplicates, plots and leakage review |
| 2 | Preprocessing + feature engineering | Reproducible splits and sklearn ColumnTransformer pipeline |
| 3 | Logistic Regression baseline | Saved pipeline, probabilities and metrics |
| 4 | Random Forest + boosting | Comparable experiment table, no test-set tuning |
| 5 | Evaluation + threshold selection | ROC-AUC, PR-AUC, precision/recall/F1, confusion matrix and chosen threshold |
| 6 | Explainability | Global and per-order explanations with caveats |
| 7 | Full Streamlit dashboard | Filters, risk scoring, region/mode analysis, action queue and export |
| 8 | Testing + hardening | Unit/integration tests, schema and artifact validation |
| 9 | Submission deliverables | Research paper, executive summary, model card and deployment guide |

Apply one phase ZIP at a time. Back up the project or create a Git commit before overlaying files.
