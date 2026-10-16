from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
from ui import inject_css, glass_card, metric_card, section_header, status_badge, load_project_data, load_model_metrics, load_feature_importance

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Late Delivery Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
inject_css()

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

# Header
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown(f'<div class="eyebrow">LATE DELIVERY RISK PREDICTION / {page.upper()}</div>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="page-title">{page}</h1>', unsafe_allow_html=True)
    subtitle = {
        "Dashboard":"A clear view of delivery-risk intelligence, model readiness, and the data behind it.",
        "Model Prediction":"Assess delivery risk from order attributes using your saved model pipeline.",
        "Prediction Analytics":"Explore recorded predictions and delivery-risk patterns without inventing missing data.",
        "Model Performance":"Review verified model evaluation results and threshold behaviour.",
        "Risk Analysis":"Understand the distribution and operational drivers of delivery risk.",
        "Explainability":"Inspect which input features influence the model most in the saved explainability report.",
        "Prediction History":"Review predictions saved locally during this session.",
        "About Website":"Project purpose, methodology, and system boundaries.",
        "About Developer":"Project ownership and development information.",
        "Settings":"Personalise the interface for your workspace."
    }.get(page, "")
    st.markdown(f'<p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)
with col_status:
    st.markdown('<div class="top-status"><span class="live-dot"></span> LOCAL MODE</div>', unsafe_allow_html=True)

if page == "Dashboard":
    section_header("Workspace overview", "Project signals at a glance")
    rows = len(df) if df is not None else 180519
    cols = len(df.columns) if df is not None else 40
    m1,m2,m3,m4 = st.columns(4)
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
                cols = st.columns(3)
                for c, key, label in zip(cols, ["accuracy","precision","recall"], ["Accuracy","Precision","Recall"]):
                    val = metrics.get(key)
                    metric_card(c, label, f"{val:.1%}" if isinstance(val,(int,float)) else "—", "Verified report value")
                cols2 = st.columns(3)
                for c, key, label in zip(cols2, ["f1","roc_auc","average_precision"], ["F1 score","ROC-AUC","Average precision"]):
                    val = metrics.get(key)
                    metric_card(c, label, f"{val:.3f}" if isinstance(val,(int,float)) else "—", "Verified report value")
            else:
                st.info("Model metrics file was not found in the expected reports folder. Keep this area empty rather than displaying invented scores.")
    with right:
        with st.container(border=True):
            section_header("System readiness", "Local project checks")
            status_badge("Dataset", "Available" if df is not None else "Not found", "success" if df is not None else "warning")
            status_badge("Phase 5 model bundle", "Check project artifacts folder", "info")
            status_badge("API service", "Not configured", "warning")
            st.markdown('<div class="small-note">The Streamlit interface can run independently. Prediction requires the project’s saved model bundle and its exact feature schema.</div>', unsafe_allow_html=True)
    st.write("")
    with st.container(border=True):
        section_header("Explore the workspace", "Navigate using the left sidebar")
        cards = st.columns(3)
        for c, title, desc in zip(cards, ["Predict a delivery","Inspect model quality","Understand risk drivers"], ["Enter order details and assess a shipment once the model adapter is connected.","Compare verified test metrics and threshold trade-offs.","Review feature importance generated by Phase 6."]):
            with c:
                st.markdown(f'<div class="feature-tile"><div class="tile-kicker">DELIVERYIQ / MODULE</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

elif page == "Model Prediction":
    section_header("Single shipment assessment", "Provide operational features for a delivery")
    st.info("The UI is ready, but the model bundle's input schema has not yet been wired into this page. This form does not fabricate a risk score.")
    with st.form("prediction_form", border=True):
        a,b,c = st.columns(3)
        with a:
            order_type = st.selectbox("Order type", ["Choose…","Type A","Type B","Type C"])
            shipping = st.selectbox("Shipping mode", ["Choose…","Standard Class","Second Class","First Class","Same Day"])
        with b:
            scheduled_days = st.number_input("Scheduled shipping days", min_value=0, max_value=30, value=4)
            customer_segment = st.selectbox("Customer segment", ["Choose…","Consumer","Corporate","Home Office"])
        with c:
            category = st.text_input("Product category", placeholder="Enter category from dataset")
            order_value = st.number_input("Order value (optional)", min_value=0.0, value=0.0, step=10.0)
        submitted = st.form_submit_button("Assess delivery risk", use_container_width=True, type="primary")
        if submitted:
            st.warning("Prediction was not run: connect the Phase 5 model adapter and confirm which features are available at prediction time before enabling this action.")

elif page == "Prediction Analytics":
    section_header("Prediction analytics", "Recorded outcomes only")
    if "history" in st.session_state and st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist, use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="empty-state"><div class="empty-icon">▥</div><h3>No prediction records yet</h3><p>Once real predictions are connected and recorded, charts and segment comparisons will appear here. No sample records are shown as if they were real.</p></div>', unsafe_allow_html=True)
    if df is not None:
        with st.container(border=True):
            section_header("Dataset profile", "Descriptive statistics from the loaded source")
            numeric = df.select_dtypes(include=np.number)
            if not numeric.empty:
                st.bar_chart(numeric.nunique().sort_values(ascending=False).head(12), height=250)
                st.caption("Columns with the most distinct numeric values. This is a dataset profile, not a prediction chart.")

elif page == "Model Performance":
    section_header("Evaluation report", "Phase 5 selected Random Forest")
    if metrics:
        labels = [("Accuracy","accuracy"),("Precision","precision"),("Recall","recall"),("F1 score","f1"),("ROC-AUC","roc_auc"),("Average precision","average_precision")]
        cols = st.columns(3)
        for i,(label,key) in enumerate(labels):
            val = metrics.get(key)
            metric_card(cols[i%3], label, f"{val:.4f}" if isinstance(val,(int,float)) else "—", "Held-out test report")
        st.caption("Values are loaded from a project report file. Check the report path and metric definitions before publishing externally.")
    else:
        st.warning("No recognised Phase 5 metrics JSON/CSV was found. Add your existing evaluation report under reports/ or update the loader in ui.py.")
    st.markdown("**Threshold note:** Phase 5 selected threshold is `0.3790`. The validation F1 improved from `0.6732` at threshold `0.5` to `0.7212` at `0.3790`. Held-out test F1 was `0.7199`.")

elif page == "Risk Analysis":
    section_header("Risk analysis", "Separate data patterns from model predictions")
    if df is None:
        st.warning("Dataset not found. Expected `data/raw/APL_Logistics.csv` or `APL_Logistics.csv` near this app.")
    else:
        target_candidates = [c for c in df.columns if any(x in c.lower() for x in ["late_delivery", "late delivery", "late_delivery_risk"])]
        if target_candidates:
            target = target_candidates[0]
            counts = df[target].value_counts(dropna=False)
            l,r = st.columns([1,1])
            with l:
                with st.container(border=True):
                    section_header("Target distribution", target)
                    st.bar_chart(counts)
            with r:
                with st.container(border=True):
                    section_header("Class share", "Dataset labels")
                    st.dataframe(counts.rename("records").to_frame().assign(share=lambda x: x["records"]/x["records"].sum()).style.format({"share":"{:.1%}"}), use_container_width=True)
        else:
            st.info("Could not identify the target label automatically. Choose the target column after confirming the dataset schema.")

elif page == "Explainability":
    section_header("Feature influence", "Phase 6 permutation importance")
    if importance is not None and not importance.empty:
        st.markdown("Importance is loaded from the project report; it describes model sensitivity under the evaluated validation sample, not causal impact.")
        st.bar_chart(importance.set_index("feature")["importance"].head(15), height=420)
        st.dataframe(importance.head(25), use_container_width=True, hide_index=True)
    else:
        st.warning("Feature-importance report not found. Run Phase 6 and place its CSV under reports/phase6/.")

elif page == "Prediction History":
    section_header("Prediction history", "Local browser session")
    if st.session_state.get("history"):
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if st.button("Clear session history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown('<div class="empty-state"><div class="empty-icon">↺</div><h3>Your history is empty</h3><p>Only genuine predictions will be stored here after model integration.</p></div>', unsafe_allow_html=True)

elif page == "About Website":
    section_header("About DeliveryIQ", "Late Delivery Risk Prediction")
    st.markdown("""
    <div class="prose-card">
      <h3>Operational risk, made interpretable.</h3>
      <p>DeliveryIQ is a machine-learning project interface for reviewing late-delivery risk, model evaluation, and feature-level explanations using the project's logistics dataset.</p>
      <h3>Methodology</h3>
      <p>The current project workflow includes data auditing, preprocessing, candidate-model evaluation, threshold selection, and permutation-based explainability.</p>
      <h3>Data integrity</h3>
      <p>Evaluation metrics are shown only when a recognised project report is available. The interface avoids presenting synthetic predictions or fabricated analytics as real operational data.</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "About Developer":
    section_header("Developer profile", "Project ownership")
    st.markdown("""
    <div class="developer-card">
      <div class="developer-avatar">US</div>
      <div><div class="eyebrow">MACHINE LEARNING / DATA APPLICATIONS</div><h2>Utkarsh Sengar</h2>
      <p>AI/ML Engineering student building applied machine-learning projects with a focus on practical evaluation, explainability, and usable interfaces.</p></div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Add verified portfolio, GitHub, and LinkedIn URLs here before publishing.")

elif page == "Settings":
    section_header("Workspace settings", "Make the interface feel like yours")
    with st.container(border=True):
        st.markdown("#### Appearance")
        theme = st.radio("Color mode", ["Dark","Light"], horizontal=True, index=0 if st.session_state.theme_mode=="Dark" else 1)
        accent = st.selectbox("Accent color", ["Violet","Ocean","Emerald","Amber"], index=["#7C5CFC","#2F80ED","#10B981","#F59E0B"].index(st.session_state.accent) if st.session_state.accent in ["#7C5CFC","#2F80ED","#10B981","#F59E0B"] else 0)
        accent_map = {"Violet":"#7C5CFC","Ocean":"#2F80ED","Emerald":"#10B981","Amber":"#F59E0B"}
        density = st.selectbox("Layout density", ["Comfortable","Compact"], index=0 if st.session_state.density=="Comfortable" else 1)
        if st.button("Apply appearance", type="primary"):
            st.session_state.theme_mode = theme
            st.session_state.accent = accent_map[accent]
            st.session_state.density = density
            st.rerun()
    st.caption("Appearance preferences are stored for the current Streamlit session.")

st.markdown('<div class="footer-line"><span>DELIVERYIQ</span><span>Late Delivery Risk Prediction · Streamlit edition</span><span>Built for clarity, not noise.</span></div>', unsafe_allow_html=True)
