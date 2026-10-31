from pathlib import Path
import json
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from ui import (
    inject_css, glass_card, metric_card, section_header, status_badge,
    load_project_data, load_model_metrics, load_feature_importance,
    load_model_adapter
)

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Late Delivery Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

PAGES = [
    ("Overview", "Dashboard", "◈"),
    ("Operations", "Model Prediction", "⌁"),
    ("Operations", "Prediction Analytics", "▥"),
    ("Model", "Model Performance", "⌂"),
    ("Model", "Risk Analysis", "◉"),
    ("Model", "Explainability", "✳"),
    ("Operations", "Prediction History", "↺"),
    ("Workspace", "About Website", "ⓘ"),
    ("Workspace", "About Developer", "♙"),
    ("Workspace", "Settings", "⚙"),
]

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"
if "accent" not in st.session_state:
    st.session_state.accent = "#7C5CFC"
if "density" not in st.session_state:
    st.session_state.density = "Comfortable"
if "history" not in st.session_state:
    st.session_state.history = []
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

inject_css(st.session_state.theme_mode, st.session_state.accent, st.session_state.density)

with st.sidebar:
    st.markdown('<div class="brand-lockup"><div class="brand-mark">LD</div><div><div class="brand-title">DeliveryIQ</div><div class="brand-subtitle">RISK INTELLIGENCE</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">WORKSPACE</div>', unsafe_allow_html=True)
    for group in ["Overview", "Operations", "Model", "Workspace"]:
        rows = [(label, icon) for category, label, icon in PAGES if category == group]
        if not rows:
            continue
        st.markdown(f'<div class="nav-group">{group}</div>', unsafe_allow_html=True)
        for label, icon in rows:
            selected = st.session_state.page == label
            if st.button(f"{icon}   {label}", key=f"nav_{label}", use_container_width=True, type="primary" if selected else "secondary"):
                st.session_state.page = label
                st.rerun()
    st.markdown('<div class="sidebar-bottom"><div class="connection-dot"></div><span>Local workspace</span><span class="sidebar-version">v1.0</span></div>', unsafe_allow_html=True)

page = st.session_state.page
df = load_project_data()
metrics = load_model_metrics()
importance = load_feature_importance()
adapter = load_model_adapter()

# Header
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown(f'<div class="eyebrow">LATE DELIVERY RISK PREDICTION / {page.upper()}</div>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="page-title">{page}</h1>', unsafe_allow_html=True)
    subtitle = {
        "Dashboard": "A clear view of delivery-risk intelligence, model readiness, and the data behind it.",
        "Model Prediction": "Assess pre-shipment delivery risk from order attributes using your trained Phase 5 model pipeline.",
        "Prediction Analytics": "Explore recorded predictions and delivery-risk patterns without inventing missing data.",
        "Model Performance": "Review verified model evaluation results, ROC curve, confusion matrix, and threshold behaviour.",
        "Risk Analysis": "Understand the distribution and operational drivers of delivery risk.",
        "Explainability": "Inspect which input features influence the model most in the saved explainability report.",
        "Prediction History": "Review predictions saved locally during this session.",
        "About Website": "Project purpose, methodology, and system boundaries.",
        "About Developer": "Project ownership and development information.",
        "Settings": "Personalise the interface for your workspace."
    }.get(page, "")
    st.markdown(f'<p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)
with col_status:
    st.markdown('<div class="top-status"><span class="live-dot"></span> LOCAL MODE</div>', unsafe_allow_html=True)

# ----------------- Dashboard -----------------
if page == "Dashboard":
    section_header("Workspace overview", "Project signals at a glance")
    rows = len(df) if df is not None else 180519
    cols = len(df.columns) if df is not None else 40
    m1, m2, m3, m4 = st.columns(4)
    metric_card(m1, "Dataset records", f"{rows:,}", "DataCo logistics dataset" if df is not None else "Expected project dataset")
    metric_card(m2, "Input columns", str(cols), "Raw data schema" if df is not None else "Dataset not loaded from this folder")
    metric_card(m3, "Selected model", "Random Forest", "Phase 5 selected pipeline")
    metric_card(m4, "Decision threshold", "0.379", "Selected on validation data")
    st.write("")
    left, right = st.columns([1.45, 1])
    with left:
        with st.container(border=True):
            section_header("Model snapshot", "Phase 5 · held-out test results")
            if metrics:
                cols_top = st.columns(3)
                for c, key, label in zip(cols_top, ["accuracy", "precision", "recall"], ["Accuracy", "Precision", "Recall"]):
                    val = metrics.get(key)
                    metric_card(c, label, f"{val:.1%}" if isinstance(val, (int, float)) else "—", "Verified report value")
                cols_bot = st.columns(3)
                for c, key, label in zip(cols_bot, ["f1", "roc_auc", "average_precision"], ["F1 score", "ROC-AUC", "Average precision"]):
                    val = metrics.get(key)
                    metric_card(c, label, f"{val:.3f}" if isinstance(val, (int, float)) else "—", "Verified report value")
            else:
                st.info("Model metrics file was not found in the expected reports folder. Keep this area empty rather than displaying invented scores.")
    with right:
        with st.container(border=True):
            section_header("System readiness", "Local project checks")
            status_badge("Dataset", "Available" if df is not None else "Not found", "success" if df is not None else "warning")
            status_badge("Phase 5 model bundle", "Loaded & Active (Random Forest)" if adapter is not None else "Not found", "success" if adapter is not None else "warning")
            status_badge("Inference Adapter", "Connected & Validated" if adapter is not None else "Disconnected", "success" if adapter is not None else "warning")
            st.markdown('<div class="small-note">The Streamlit interface runs with the trained Phase 5 Random Forest model bundle, leakage-safe pre-shipment feature schema, and locked 0.3790 decision threshold.</div>', unsafe_allow_html=True)
    st.write("")
    with st.container(border=True):
        section_header("Explore the workspace", "Navigate using the left sidebar")
        cards = st.columns(3)
        for c, title, desc in zip(
            cards,
            ["Predict a delivery", "Inspect model quality", "Understand risk drivers"],
            [
                "Enter order details and assess a shipment with real-time probability and sensitivity explainability.",
                "Compare verified test metrics, confusion matrix, and threshold trade-offs.",
                "Review feature importance generated by Phase 6."
            ]
        ):
            with c:
                st.markdown(f'<div class="feature-tile"><div class="tile-kicker">DELIVERYIQ / MODULE</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

# ----------------- Model Prediction -----------------
elif page == "Model Prediction":
    section_header("Shipment risk assessment", "Operational pre-shipment risk inference using Phase 5 trained model")

    if adapter is None:
        st.error("⚠️ Phase 5 model adapter could not be loaded. Please ensure `artifacts/phase5/selected_model_bundle.joblib` is present.")
    else:
        tab_single, tab_batch = st.tabs(["Single shipment assessment", "Batch CSV assessment"])

        with tab_single:
            st.markdown('<div class="small-note" style="margin-bottom:12px;">Pre-shipment prediction policy: Outcome-derived fields (<i>Days for shipping (real)</i>, <i>Delivery Status</i>, <i>Order Status</i>) are strictly excluded to prevent data leakage.</div>', unsafe_allow_html=True)

            with st.form("single_prediction_form", border=True):
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.markdown("##### Transport & Logistics")
                    shipping_mode = st.selectbox(
                        "Shipping mode",
                        ["Standard Class", "Second Class", "First Class", "Same Day"],
                        index=0,
                        help="Carriers and delivery speeds for the shipment"
                    )
                    default_days = 4 if shipping_mode == "Standard Class" else 2 if shipping_mode == "Second Class" else 1 if shipping_mode == "First Class" else 0
                    scheduled_days = st.number_input(
                        "Scheduled shipping days",
                        min_value=0,
                        max_value=14,
                        value=default_days,
                        help="Standard scheduled transit days agreed with customer"
                    )
                    order_type = st.selectbox(
                        "Payment type",
                        ["DEBIT", "TRANSFER", "PAYMENT", "CASH"],
                        index=0,
                        help="Transaction settlement method"
                    )

                with col_b:
                    st.markdown("##### Customer & Geography")
                    customer_segment = st.selectbox(
                        "Customer segment",
                        ["Consumer", "Corporate", "Home Office"],
                        index=0
                    )
                    market = st.selectbox(
                        "Market",
                        ["LATAM", "Europe", "Pacific Asia", "USCA", "Africa"],
                        index=0
                    )
                    order_region = st.selectbox(
                        "Order region",
                        [
                            "Central America", "Western Europe", "South America", "Oceania",
                            "Southeast Asia", "Eastern Asia", "West of USA", "East of USA",
                            "Southern Europe", "South Asia", "North Africa", "Central Asia",
                            "Eastern Europe", "Canada", "Caribbean", "East Africa"
                        ],
                        index=0
                    )

                with col_c:
                    st.markdown("##### Order Items & Pricing")
                    department = st.selectbox(
                        "Department",
                        ["Fan Shop", "Apparel", "Golf", "Footwear", "Outdoors", "Fitness", "Discs Shop", "Technology", "Book Shop"],
                        index=0
                    )
                    order_quantity = st.number_input(
                        "Order item quantity",
                        min_value=1,
                        max_value=20,
                        value=1,
                        step=1,
                        help="Number of units in this order line"
                    )
                    product_price = st.number_input(
                        "Product unit price ($)",
                        min_value=1.0,
                        max_value=2500.0,
                        value=59.99,
                        step=5.0
                    )
                    order_discount = st.number_input(
                        "Item discount ($)",
                        min_value=0.0,
                        max_value=500.0,
                        value=0.0,
                        step=5.0
                    )

                with st.expander("Additional geographic attributes (optional)", expanded=False):
                    g1, g2, g3, g4 = st.columns(4)
                    with g1:
                        cust_country = st.text_input("Customer country", value="EE. UU.")
                    with g2:
                        cust_state = st.text_input("Customer state", value="PR")
                    with g3:
                        order_country = st.text_input("Order destination country", value="Estados Unidos")
                    with g4:
                        order_city = st.text_input("Order destination city", value="New York City")

                submitted = st.form_submit_button("Assess delivery risk", use_container_width=True, type="primary")

            if submitted:
                payload = {
                    "Type": order_type,
                    "Shipping Mode": shipping_mode,
                    "Days for shipment (scheduled)": scheduled_days,
                    "Customer Segment": customer_segment,
                    "Market": market,
                    "Order Region": order_region,
                    "Department Name": department,
                    "Order Item Quantity": order_quantity,
                    "Order Item Product Price": product_price,
                    "Product Price": product_price,
                    "Order Item Discount": order_discount,
                    "Order Item Discount Rate": (order_discount / product_price) if product_price > 0 else 0.0,
                    "Customer Country": cust_country,
                    "Customer State": cust_state,
                    "Order Country": order_country,
                    "Order City": order_city,
                }

                try:
                    res = adapter.predict_single(payload)
                    prob = res["probability"]
                    thresh = res["threshold"]

                    st.write("")
                    with st.container(border=True):
                        if res["is_high_risk"]:
                            st.error(
                                f"🚨 **HIGHER LATE-DELIVERY RISK** — Model probability of **{res['probability_pct']}** "
                                f"exceeds the locked decision threshold (**{res['threshold_str']}**). Proactive operational intervention recommended."
                            )
                        else:
                            st.success(
                                f"✅ **LOWER LATE-DELIVERY RISK** — Model probability of **{res['probability_pct']}** "
                                f"is below the locked decision threshold (**{res['threshold_str']}**). Normal transit timeline expected."
                            )

                        r1, r2, r3, r4 = st.columns(4)
                        metric_card(r1, "Estimated late probability", res["probability_pct"], "Model predicted probability")
                        metric_card(r2, "Locked threshold", res["threshold_str"], "Phase 5 validation F1 optimum")
                        metric_card(r3, "Risk tier", res["risk_tier"], f"Status badge: {res['badge']}")
                        metric_card(r4, "Decision", "Flagged for delay" if res["is_high_risk"] else "Not flagged", res["decision"])

                        st.progress(float(np.clip(prob, 0.0, 1.0)), text=f"Calculated late probability: {res['probability_pct']} (Decision threshold: {res['threshold_str']})")

                    # Local Explainability
                    if res.get("explanations"):
                        with st.container(border=True):
                            section_header("Local sensitivity explanation", "Marginal impact of shipment features relative to typical baseline")
                            st.caption("How much the predicted late probability would change if each feature were replaced by its baseline reference value:")
                            exp_df = pd.DataFrame([
                                {
                                    "Feature": e["feature"],
                                    "Original Value": str(e["original_value"]),
                                    "Typical Baseline": str(e["typical_value"]),
                                    "Marginal Impact": f"{e['marginal_effect']:+.2%}",
                                    "Influence Direction": e["impact"],
                                }
                                for e in res["explanations"]
                            ])
                            st.dataframe(exp_df, use_container_width=True, hide_index=True)

                    # Save record to session history
                    history_entry = {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Source": "Single Shipment",
                        "Shipping Mode": shipping_mode,
                        "Scheduled Days": scheduled_days,
                        "Order Type": order_type,
                        "Segment": customer_segment,
                        "Market": market,
                        "Quantity": order_quantity,
                        "Price": f"${product_price:.2f}",
                        "Probability": f"{prob:.4f}",
                        "Threshold": res["threshold_str"],
                        "Risk Tier": res["risk_tier"],
                        "Decision": "High Risk" if res["is_high_risk"] else "Low Risk",
                    }
                    st.session_state.history.insert(0, history_entry)

                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        with tab_batch:
            section_header("Batch CSV assessment", "Upload and score shipments in bulk")
            st.markdown('<div class="small-note">Upload a CSV file containing pre-shipment orders. Outcome columns (<i>Late_delivery_risk</i>, <i>Days for shipping (real)</i>, <i>Delivery Status</i>) will be ignored if present to prevent data leakage.</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Choose shipments CSV file", type=["csv"], key="batch_uploader")

            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file, low_memory=False)
                    st.write(f"Loaded **{len(batch_df):,}** rows with **{len(batch_df.columns)}** columns.")

                    # Filter out forbidden leakage columns if present
                    safe_batch = batch_df.copy()
                    for col in ["Late_delivery_risk", "Days for shipping (real)", "Delivery Status", "Order Status"]:
                        if col in safe_batch.columns:
                            safe_batch = safe_batch.drop(columns=[col])

                    if st.button("Score batch shipments", type="primary"):
                        scored_df = adapter.batch_predict(safe_batch)
                        st.session_state.batch_results = scored_df

                        n_total = len(scored_df)
                        n_flagged = int((scored_df["predicted_late_risk"] == 1).sum())
                        flag_rate = n_flagged / n_total if n_total > 0 else 0.0
                        avg_prob = float(scored_df["predicted_late_probability"].mean())

                        # Record batch summary in history
                        st.session_state.history.insert(0, {
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Source": f"Batch CSV ({n_total:,} rows)",
                            "Shipping Mode": "Batch Multi-mode",
                            "Scheduled Days": "—",
                            "Order Type": "—",
                            "Segment": "—",
                            "Market": "—",
                            "Quantity": n_total,
                            "Price": "—",
                            "Probability": f"{avg_prob:.4f}",
                            "Threshold": f"{adapter.threshold:.4f}",
                            "Risk Tier": f"{flag_rate:.1%} High Risk",
                            "Decision": f"{n_flagged:,} flagged / {n_total:,}",
                        })

                    if st.session_state.batch_results is not None:
                        res_df = st.session_state.batch_results
                        with st.container(border=True):
                            b1, b2, b3, b4 = st.columns(4)
                            n_total = len(res_df)
                            n_flagged = int((res_df["predicted_late_risk"] == 1).sum())
                            metric_card(b1, "Total shipments", f"{n_total:,}", "Scored batch records")
                            metric_card(b2, "Flagged high risk", f"{n_flagged:,}", "Exceeds decision threshold")
                            metric_card(b3, "High risk share", f"{(n_flagged / n_total):.1%}" if n_total > 0 else "0%", "Proportion requiring action")
                            metric_card(b4, "Mean predicted prob", f"{res_df['predicted_late_probability'].mean():.1%}", "Across all uploaded rows")

                            st.write("##### Scored Shipment Preview")
                            preview_cols = [c for c in ["Shipping Mode", "Days for shipment (scheduled)", "Type", "predicted_late_probability", "predicted_late_risk", "risk_decision"] if c in res_df.columns]
                            st.dataframe(res_df[preview_cols].head(50), use_container_width=True, hide_index=True)

                            st.download_button(
                                label="Download scored shipments CSV",
                                data=res_df.to_csv(index=False).encode("utf-8"),
                                file_name="late_delivery_predictions.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                except Exception as exc:
                    st.error(f"Error processing CSV: {exc}")

# ----------------- Prediction Analytics -----------------
elif page == "Prediction Analytics":
    section_header("Prediction analytics", "Recorded outcomes from local session inference")
    if "history" in st.session_state and st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)

        with st.container(border=True):
            section_header("Session prediction signals", "Overview of predictions recorded during this session")
            p1, p2, p3, p4 = st.columns(4)
            total_assessments = len(hist_df)
            metric_card(p1, "Session predictions", f"{total_assessments:,}", "Single & batch assessments")

            high_risk_count = sum(1 for d in hist_df["Decision"] if "High Risk" in str(d) or "flagged" in str(d).lower())
            metric_card(p2, "Flagged risk shipments", f"{high_risk_count:,}", "Requiring proactive review")

            # Extract numeric probabilities
            probs = []
            for p in hist_df["Probability"]:
                try:
                    probs.append(float(p))
                except Exception:
                    pass

            if probs:
                metric_card(p3, "Average late risk", f"{np.mean(probs):.1%}", "Mean across session records")
                metric_card(p4, "Max assessed risk", f"{np.max(probs):.1%}", "Highest risk order assessed")
            else:
                metric_card(p3, "Average late risk", "—", "Awaiting numeric probabilities")
                metric_card(p4, "Max assessed risk", "—", "Awaiting numeric probabilities")

        st.write("")
        with st.container(border=True):
            section_header("Recorded prediction stream", "Detailed log of session predictions")
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            '<div class="empty-state"><div class="empty-icon">▥</div><h3>No prediction records yet</h3>'
            '<p>Once real predictions are generated on the Model Prediction page, charts, risk flag counts, and operational patterns will appear here. No synthetic records are shown as if they were real.</p></div>',
            unsafe_allow_html=True
        )

    if df is not None:
        st.write("")
        with st.container(border=True):
            section_header("Dataset profile", "Descriptive statistics from the loaded source")
            numeric = df.select_dtypes(include=np.number)
            if not numeric.empty:
                st.bar_chart(numeric.nunique().sort_values(ascending=False).head(12), height=250)
                st.caption("Columns with the most distinct numeric values. This is a dataset profile, not a prediction chart.")

# ----------------- Model Performance -----------------
elif page == "Model Performance":
    section_header("Evaluation report", "Phase 5 selected Random Forest on held-out test data")
    if metrics:
        labels = [
            ("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
            ("F1 score", "f1"), ("ROC-AUC", "roc_auc"), ("Average precision", "average_precision")
        ]
        cols = st.columns(3)
        for i, (label, key) in enumerate(labels):
            val = metrics.get(key)
            metric_card(cols[i % 3], label, f"{val:.4f}" if isinstance(val, (int, float)) else "—", "Held-out test report")
        st.caption("Values are verified Phase 5 evaluation results on the held-out test set (27,078 records). No scores are fabricated.")
    else:
        st.warning("No recognised Phase 5 metrics JSON/CSV was found. Add your existing evaluation report under reports/ or update the loader in ui.py.")

    st.write("")
    with st.container(border=True):
        section_header("Threshold policy & optimization", "Validation F1 trade-off")
        st.markdown(
            "Phase 5 selected decision threshold: `0.3790`. By optimizing the decision threshold on validation data, "
            "the validation F1 improved from `0.6732` (at default 0.50) to `0.7212` (at 0.3790). "
            "On the held-out test set, the final F1 score achieved was `0.7199` with a recall of `0.8520`."
        )

    # Show verified visual curves if available
    cm_path = ROOT / "reports" / "phase5" / "test_confusion_matrix.png"
    roc_path = ROOT / "reports" / "phase5" / "test_roc_curve.png"
    if cm_path.exists() and roc_path.exists():
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                section_header("Confusion matrix", "Held-out test evaluation")
                st.image(str(cm_path), use_container_width=True)
        with c2:
            with st.container(border=True):
                section_header("ROC curve", "Held-out test evaluation (AUC = 0.762)")
                st.image(str(roc_path), use_container_width=True)

# ----------------- Risk Analysis -----------------
elif page == "Risk Analysis":
    section_header("Risk analysis", "Separate data patterns from model predictions")
    if df is None:
        st.warning("Dataset not found. Expected `data/raw/APL_Logistics.csv` or `APL_Logistics.csv` near this app.")
    else:
        target_candidates = [c for c in df.columns if any(x in c.lower() for x in ["late_delivery", "late delivery", "late_delivery_risk"])]
        if target_candidates:
            target = target_candidates[0]
            counts = df[target].value_counts(dropna=False)
            l, r = st.columns([1, 1])
            with l:
                with st.container(border=True):
                    section_header("Target distribution", target)
                    st.bar_chart(counts)
            with r:
                with st.container(border=True):
                    section_header("Class share", "Dataset labels")
                    st.dataframe(counts.rename("records").to_frame().assign(share=lambda x: x["records"] / x["records"].sum()).style.format({"share": "{:.1%}"}), use_container_width=True)
        else:
            st.info("Could not identify the target label automatically. Choose the target column after confirming the dataset schema.")

# ----------------- Explainability -----------------
elif page == "Explainability":
    section_header("Feature influence", "Phase 6 permutation importance")
    if importance is not None and not importance.empty:
        st.markdown("Importance is loaded from the Phase 6 permutation importance report; it describes model sensitivity under the evaluated validation sample, not causal impact.")
        st.bar_chart(importance.set_index("feature")["importance"].head(15), height=380)
        st.dataframe(importance.head(25), use_container_width=True, hide_index=True)
    else:
        st.warning("Feature-importance report not found. Run Phase 6 and place its CSV under reports/phase6/.")

# ----------------- Prediction History -----------------
elif page == "Prediction History":
    section_header("Prediction history", "Local browser session")
    if st.session_state.get("history"):
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        btn_col1, btn_col2 = st.columns([1, 4])
        with btn_col1:
            if st.button("Clear session history", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        with btn_col2:
            st.download_button(
                label="Download prediction history (CSV)",
                data=hist_df.to_csv(index=False).encode("utf-8"),
                file_name="deliveryiq_prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.markdown('<div class="empty-state"><div class="empty-icon">↺</div><h3>Your history is empty</h3><p>Genuine predictions will be stored here during this session after assessing shipments.</p></div>', unsafe_allow_html=True)

# ----------------- About Website -----------------
elif page == "About Website":
    section_header("About DeliveryIQ", "Late Delivery Risk Prediction")
    st.markdown("""
    <div class="prose-card">
      <h3>Operational risk, made interpretable.</h3>
      <p>DeliveryIQ is a machine-learning project interface for reviewing late-delivery risk, model evaluation, and feature-level explanations using the DataCo logistics dataset.</p>
      <h3>Methodology</h3>
      <p>The project follows a rigorous 7-phase data science methodology: data auditing, leakage prevention, pre-shipment feature engineering, candidate model comparison, validation-based threshold optimization (locked at 0.3790), held-out test evaluation, and permutation-based explainability.</p>
      <h3>Data integrity</h3>
      <p>Evaluation metrics and predictions are generated only from the verified Phase 5 model bundle. The interface avoids synthetic predictions or fabricated performance metrics.</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------- About Developer -----------------
elif page == "About Developer":
    section_header("Developer profile", "Project ownership")
    st.markdown("""
    <div class="developer-card">
      <div class="developer-avatar">US</div>
      <div><div class="eyebrow">MACHINE LEARNING / DATA APPLICATIONS</div><h2>Utkarsh Sengar</h2>
      <p>AI/ML Engineering student building applied machine-learning projects with a focus on practical evaluation, explainability, and usable interfaces.</p></div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- Settings -----------------
elif page == "Settings":
    section_header("Workspace settings", "Make the interface feel like yours")
    with st.container(border=True):
        st.markdown("#### Appearance")
        theme = st.radio("Color mode", ["Dark", "Light"], horizontal=True, index=0 if st.session_state.theme_mode == "Dark" else 1)
        accent = st.selectbox("Accent color", ["Violet", "Ocean", "Emerald", "Amber"], index=["#7C5CFC", "#2F80ED", "#10B981", "#F59E0B"].index(st.session_state.accent) if st.session_state.accent in ["#7C5CFC", "#2F80ED", "#10B981", "#F59E0B"] else 0)
        accent_map = {"Violet": "#7C5CFC", "Ocean": "#2F80ED", "Emerald": "#10B981", "Amber": "#F59E0B"}
        density = st.selectbox("Layout density", ["Comfortable", "Compact"], index=0 if st.session_state.density == "Comfortable" else 1)
        if st.button("Apply appearance", type="primary"):
            st.session_state.theme_mode = theme
            st.session_state.accent = accent_map[accent]
            st.session_state.density = density
            st.rerun()
    st.caption("Appearance preferences are stored for the current Streamlit session.")

st.markdown('<div class="footer-line"><span>DELIVERYIQ</span><span>Late Delivery Risk Prediction · Streamlit edition</span><span>Built for clarity, not noise.</span></div>', unsafe_allow_html=True)
