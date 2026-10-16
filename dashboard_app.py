from pathlib import Path
from datetime import datetime
import warnings
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "artifacts/phase5/selected_model_bundle.joblib"
TRAIN_PATH = ROOT / "data/processed/phase2/train.csv"
VALIDATION_PATH = ROOT / "data/processed/phase2/validation.csv"
TEST_PATH = ROOT / "data/processed/phase2/test.csv"
PHASE5_REPORT = ROOT / "reports/phase5/PHASE_5_REPORT.md"
PHASE6_DIR = ROOT / "reports/phase6"
TARGET = "Late_delivery_risk"
DEFAULT_THRESHOLD = 0.3790
LEAKAGE_NAMES = {
    TARGET, "Days for shipping (real)", "Delivery Status", "Order Status",
    "Order Id", "Order Item Id", "Order Customer Id", "Customer Id",
    "Customer Email", "Customer Fname", "Customer Lname", "Customer Password",
    "Customer Street", "Customer Zipcode", "Product Description"
}
PALETTES = {
    "Ocean Blue": {"accent": "#2F6FED", "accent2": "#4F8DF7"},
    "Teal": {"accent": "#0F8B8D", "accent2": "#25A99B"},
    "Violet": {"accent": "#7657D6", "accent2": "#9B86EA"},
    "Amber": {"accent": "#B7791F", "accent2": "#D69E2E"},
    "Graphite": {"accent": "#536273", "accent2": "#8190A2"},
}

st.set_page_config(page_title="Late Delivery Risk | Intelligence", page_icon="📦", layout="wide", initial_sidebar_state="expanded")

# Persist visual preferences in this browser session.
if "ui_mode" not in st.session_state:
    st.session_state.ui_mode = "Light"
if "ui_palette" not in st.session_state:
    st.session_state.ui_palette = "Ocean Blue"

palette = PALETTES.get(st.session_state.ui_palette, PALETTES["Ocean Blue"])
dark = st.session_state.ui_mode == "Dark"
bg = "#10151D" if dark else "#F5F7FA"
panel = "#171F2B" if dark else "#FFFFFF"
text = "#E7ECF3" if dark else "#17212F"
muted = "#9AA8B8" if dark else "#687588"
border = "#2A3544" if dark else "#E2E8F0"
st.markdown(f"""
<style>
:root {{ --accent: {palette['accent']}; --accent2: {palette['accent2']}; }}
.stApp {{ background: {bg}; color: {text}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1550px; padding-top: 1.25rem; padding-bottom: 2.5rem; }}
h1, h2, h3 {{ color: {text}; letter-spacing: -0.025em; }}
p, label, .stMarkdown {{ color: {text}; }}
div[data-testid="stMetric"] {{ background: {panel}; border: 1px solid {border}; padding: 15px 17px; border-radius: 10px; }}
div[data-testid="stMetricLabel"] {{ color: {muted}; }}
div[data-testid="stMetricValue"] {{ color: {text}; }}
section[data-testid="stSidebar"] {{ background: {panel}; border-right: 1px solid {border}; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 7px; border: 1px solid {border}; }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{ background: {palette['accent']}; border-color: {palette['accent']}; color: white; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {border}; border-radius: 8px; }}
hr {{ border-color: {border}; }}
small, .muted {{ color: {muted}; }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bundle(path):
    return joblib.load(path)

@st.cache_data
def load_csv(path, modified):
    return pd.read_csv(path, low_memory=False)

def safe_csv(path):
    if not path.exists():
        return None
    try:
        return load_csv(str(path), path.stat().st_mtime)
    except Exception as exc:
        st.warning(f"Could not read `{path.relative_to(ROOT)}`: {exc}")
        return None

def unpack_bundle(bundle):
    if not isinstance(bundle, dict):
        raise ValueError("Expected the Phase 5 model bundle to be a dictionary.")
    model = bundle.get("model")
    if model is None: model = bundle.get("estimator")
    if model is None: model = bundle.get("classifier")
    preprocessor = bundle.get("preprocessor")
    if preprocessor is None: preprocessor = bundle.get("transformer")
    threshold = bundle.get("threshold", bundle.get("selected_threshold", DEFAULT_THRESHOLD))
    if model is None or preprocessor is None:
        nested = bundle.get("selected_model_bundle")
        if isinstance(nested, dict):
            model = model or nested.get("model") or nested.get("estimator")
            preprocessor = preprocessor or nested.get("preprocessor")
            threshold = nested.get("threshold", threshold)
    if model is None or preprocessor is None:
        raise ValueError("Model/preprocessor not found. Bundle keys: " + ", ".join(map(str, bundle.keys())))
    try: threshold = float(threshold)
    except Exception: threshold = DEFAULT_THRESHOLD
    return model, preprocessor, threshold

def positive_probability(model, preprocessor, frame):
    transformed = preprocessor.transform(frame)
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(transformed))
        classes = list(getattr(model, "classes_", [0, 1]))
        index = classes.index(1) if 1 in classes else raw.shape[1] - 1
        return raw[:, index]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(transformed))
        return 1 / (1 + np.exp(-np.clip(scores, -30, 30)))
    return np.asarray(model.predict(transformed), dtype=float)

def expected_features(preprocessor, data):
    names = getattr(preprocessor, "feature_names_in_", None)
    if names is not None:
        return [str(c) for c in names]
    return [c for c in data.columns if c not in LEAKAGE_NAMES]

def markdown_report(path):
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8", errors="replace"))
    else:
        st.info(f"Report not found: `{path.relative_to(ROOT)}`")

def make_risk_label(p, threshold):
    return "High risk" if p >= threshold else "Not flagged"

def grouped_risk(data, group, threshold, model, prep, features):
    frame = data[features].copy()
    probs = positive_probability(model, prep, frame)
    tmp = data[[group]].copy()
    tmp["Predicted probability"] = probs
    tmp["Predicted risk"] = (probs >= threshold).astype(int)
    if TARGET in data.columns:
        tmp["Observed label"] = pd.to_numeric(data[TARGET], errors="coerce").values
    agg = tmp.groupby(group, dropna=False).agg(
        Shipments=("Predicted probability", "size"),
        Mean_predicted_risk=("Predicted probability", "mean"),
        Predicted_high_risk_count=("Predicted risk", "sum")
    ).reset_index()
    if "Observed label" in tmp.columns:
        observed = tmp.groupby(group, dropna=False)["Observed label"].mean().reset_index(name="Observed_late_rate")
        agg = agg.merge(observed, on=group, how="left")
    return agg.sort_values("Mean_predicted_risk", ascending=False)

# Sidebar and navigation
st.sidebar.markdown("## Late Delivery Risk")
st.sidebar.caption("Supply chain risk intelligence")
pages = [
    "Dashboard", "Model Prediction", "Prediction Analytics",
    "Model Performance", "Risk Analysis", "Explainability",
    "Prediction History", "About This Website", "About Developer", "Settings"
]
page = st.sidebar.radio("NAVIGATION", pages, label_visibility="visible")
st.sidebar.divider()
st.sidebar.caption("Saved Phase 5 model · Phase 6 explainability")

# Settings page needs to work even if the model bundle is missing.
if page == "Settings":
    st.title("Settings")
    st.caption("Customize the dashboard appearance. Changes apply to this session.")
    left, right = st.columns(2)
    with left:
        mode = st.radio("Appearance", ["Light", "Dark"], index=0 if st.session_state.ui_mode == "Light" else 1, horizontal=True)
        palette_name = st.selectbox("Accent color", list(PALETTES.keys()), index=list(PALETTES.keys()).index(st.session_state.ui_palette))
        st.session_state.ui_mode = mode
        st.session_state.ui_palette = palette_name
        st.success("Settings updated. If some components do not repaint immediately, navigate to another page and return.")
    with right:
        st.markdown("### Preview")
        p = PALETTES[palette_name]
        st.markdown(f"""
        <div style="padding:24px;border-radius:10px;background:{'#171F2B' if mode == 'Dark' else '#FFFFFF'};border:1px solid {'#2A3544' if mode == 'Dark' else '#E2E8F0'}">
          <div style="font-size:12px;letter-spacing:.12em;color:{'#9AA8B8' if mode == 'Dark' else '#687588'}">PREVIEW CARD</div>
          <h3 style="color:{'#E7ECF3' if mode == 'Dark' else '#17212F'}">Shipment risk overview</h3>
          <div style="font-size:32px;font-weight:650;color:{p['accent']}">72.0%</div>
          <div style="height:7px;border-radius:4px;background:{p['accent2']};width:72%;margin-top:10px"></div>
        </div>""", unsafe_allow_html=True)
    st.info("Theme preferences persist while this Streamlit session is active. For a deployment-wide permanent theme, configure Streamlit's theme settings separately.")
    st.stop()

if not BUNDLE_PATH.exists():
    st.error(f"Missing model bundle: `{BUNDLE_PATH.relative_to(ROOT)}`")
    st.info("Run this app from the existing project root. Preserve the artifacts/, data/, and reports/ folders.")
    st.stop()
try:
    bundle = load_bundle(str(BUNDLE_PATH))
    model, preprocessor, saved_threshold = unpack_bundle(bundle)
except Exception as exc:
    st.error(f"Could not load the Phase 5 model bundle: {exc}")
    st.stop()

train = safe_csv(TRAIN_PATH)
validation = safe_csv(VALIDATION_PATH)
test = safe_csv(TEST_PATH)
data = validation if validation is not None else (train if train is not None else test)
if data is None:
    st.error("Could not load any Phase 2 split CSV.")
    st.stop()

features = expected_features(preprocessor, data)
missing_data_features = [c for c in features if c not in data.columns]
available_features = [c for c in features if c in data.columns]
if missing_data_features:
    st.warning("Some expected features are absent from the loaded split: " + ", ".join(missing_data_features[:12]))
if not available_features:
    st.error("No model input features were found in the loaded CSV.")
    st.stop()

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

# ---------------- Dashboard ----------------
if page == "Dashboard":
    st.title("Supply Chain Risk Dashboard")
    st.caption("A decision-support view for shipment risk, model performance, and operational patterns.")
    late_rate = pd.to_numeric(data[TARGET], errors="coerce").mean() if TARGET in data.columns else np.nan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records in active split", f"{len(data):,}")
    c2.metric("Model input features", f"{len(features):,}")
    c3.metric("Saved decision threshold", f"{saved_threshold:.4f}")
    c4.metric("Observed late-label rate", f"{late_rate:.1%}" if pd.notna(late_rate) else "Not available")
    st.divider()
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Shipment label distribution")
        if TARGET in data.columns:
            counts = data[TARGET].value_counts(dropna=False).sort_index()
            fig = px.bar(x=[str(x) for x in counts.index], y=counts.values,
                         labels={"x":"Target label","y":"Shipments"}, title="Observed labels in selected split")
            fig.update_layout(showlegend=False, margin=dict(l=10,r=10,t=50,b=10), height=330)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Target labels are unavailable in the selected split.")
    with right:
        st.subheader("Project status")
        statuses = pd.DataFrame({
            "Artifact": ["Phase 5 model bundle", "Training split", "Validation split", "Test split", "Phase 6 importance chart"],
            "Status": ["Available" if p.exists() else "Missing" for p in [
                BUNDLE_PATH, TRAIN_PATH, VALIDATION_PATH, TEST_PATH, PHASE6_DIR/"global_feature_importance.png"]]
        })
        st.dataframe(statuses, use_container_width=True, hide_index=True)
        st.markdown("**Current model:** Random Forest · selected threshold from Phase 5")
    st.warning("Validation data and historical labels are descriptive. The original split is random, so future-time generalization has not been established.")
    st.markdown("### Quick actions")
    q1,q2,q3 = st.columns(3)
    with q1: st.button("Open Model Prediction", on_click=lambda: st.session_state.update({"quick_nav": "Model Prediction"}))
    with q2: st.button("Open Prediction Analytics", on_click=lambda: st.session_state.update({"quick_nav": "Prediction Analytics"}))
    with q3: st.button("Open Model Performance", on_click=lambda: st.session_state.update({"quick_nav": "Model Performance"}))
    if "quick_nav" in st.session_state:
        st.info(f"Use the sidebar to open **{st.session_state.quick_nav}**.")

# ---------------- Model Prediction ----------------
elif page == "Model Prediction":
    st.title("Model Prediction")
    st.caption("Single-shipment prediction and batch CSV scoring using the saved Phase 5 model.")
    tab_single, tab_batch = st.tabs(["Single shipment", "Batch CSV prediction"])
    with tab_single:
        st.markdown("### Enter shipment attributes")
        source = train if train is not None else data
        form_features = [c for c in features if c in source.columns]
        vals = {}
        with st.form("single_prediction_form"):
            st.caption("Controls use observed training-data values/ranges. Only enter fields available at prediction time.")
            columns = st.columns(3)
            for i, name in enumerate(form_features):
                series = source[name]
                with columns[i % 3]:
                    if pd.api.types.is_numeric_dtype(series):
                        clean = pd.to_numeric(series, errors="coerce").dropna()
                        default = float(clean.median()) if not clean.empty else 0.0
                        lo = float(clean.quantile(.01)) if not clean.empty else default-1
                        hi = float(clean.quantile(.99)) if not clean.empty else default+1
                        if lo > hi: lo, hi = hi, lo
                        if lo == hi:
                            vals[name] = st.number_input(name, value=default)
                        else:
                            vals[name] = st.number_input(name, min_value=lo, max_value=hi,
                                                         value=min(max(default,lo),hi),
                                                         help="Input range is limited to the training-data 1st–99th percentile.")
                    else:
                        opts = sorted(series.dropna().astype(str).value_counts().head(60).index.tolist())
                        if not opts: opts = ["Unknown"]
                        mode_val = str(series.dropna().mode().iloc[0]) if not series.dropna().mode().empty else opts[0]
                        vals[name] = st.selectbox(name, opts, index=opts.index(mode_val) if mode_val in opts else 0)
            submitted = st.form_submit_button("Calculate shipment risk", type="primary", use_container_width=True)
        if submitted:
            try:
                row = {c: vals.get(c, 0) for c in features}
                X = pd.DataFrame([row], columns=features)
                # Add missing expected columns only if a fallback exists in the project split.
                for col in missing_data_features:
                    X[col] = 0
                X = X[features]
                prob = float(positive_probability(model, preprocessor, X)[0])
                threshold = st.slider("Decision threshold for this view", 0.0, 1.0, float(saved_threshold), 0.01, key="single_threshold")
                flagged = prob >= threshold
                if flagged: st.error("HIGHER LATE-DELIVERY RISK — review recommended")
                else: st.success("LOWER LATE-DELIVERY RISK — not flagged at this threshold")
                a,b,c = st.columns(3)
                a.metric("Estimated probability", f"{prob:.1%}")
                b.metric("Applied threshold", f"{threshold:.2f}")
                c.metric("Decision", "Flagged" if flagged else "Not flagged")
                st.progress(float(np.clip(prob,0,1)), text="Estimated late-delivery probability")
                st.caption("The score is a model estimate, not a guarantee. Changing the display threshold changes the flag, not the model probability.")
                st.session_state.prediction_history.insert(0, {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Probability": prob, "Threshold": threshold,
                    "Prediction": "High risk" if flagged else "Not flagged",
                    "Source": "Single shipment"
                })
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
    with tab_batch:
        st.markdown("### Score a CSV file")
        st.write("Upload a CSV containing the model's required input columns. Extra columns are ignored; target labels and known post-outcome fields are not used as inputs.")
        uploaded = st.file_uploader("Choose shipment CSV", type=["csv"], key="batch_upload")
        batch_threshold = st.slider("Batch decision threshold", 0.0, 1.0, float(saved_threshold), 0.01, key="batch_threshold")
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded, low_memory=False)
                st.write(f"Uploaded rows: **{len(raw):,}** · Columns: **{len(raw.columns):,}**")
                required = list(features)
                missing = [c for c in required if c not in raw.columns]
                if missing:
                    st.error("Missing required model input columns: " + ", ".join(missing[:25]))
                    st.caption("Use a CSV with the exact Phase 2 raw feature names.")
                else:
                    if st.button("Run batch predictions", type="primary"):
                        Xbatch = raw[required].copy()
                        probs = positive_probability(model, preprocessor, Xbatch)
                        out = raw.copy()
                        out["predicted_late_probability"] = probs
                        out["predicted_late_risk"] = (probs >= batch_threshold).astype(int)
                        out["risk_label"] = np.where(out["predicted_late_risk"] == 1, "High risk", "Not flagged")
                        st.session_state.batch_results = out
                        st.session_state.prediction_history.insert(0, {
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Probability": float(np.mean(probs)), "Threshold": batch_threshold,
                            "Prediction": f"Batch: {(probs >= batch_threshold).sum()} flagged / {len(probs)}",
                            "Source": "Batch CSV"
                        })
                    if st.session_state.batch_results is not None:
                        result = st.session_state.batch_results
                        st.success(f"Batch scored: {len(result):,} rows")
                        a,b,c = st.columns(3)
                        a.metric("Rows scored", f"{len(result):,}")
                        c1 = int((result["predicted_late_risk"] == 1).sum())
                        b.metric("Flagged high-risk", f"{c1:,}")
                        c.metric("Average predicted probability", f"{result['predicted_late_probability'].mean():.1%}")
                        st.dataframe(result.head(100), use_container_width=True, hide_index=True)
                        st.download_button("Download all predictions CSV", result.to_csv(index=False).encode("utf-8"),
                                           "late_delivery_batch_predictions.csv", "text/csv")
            except Exception as exc:
                st.error(f"Could not process this CSV: {exc}")

# ---------------- Prediction Analytics ----------------
elif page == "Prediction Analytics":
    st.title("Prediction Analytics")
    st.caption("Prediction distributions, group comparisons, and exploratory clustering. Cluster IDs are exploratory segments, not delivery labels.")
    n = min(len(data), 10000)
    sample = data.sample(n=n, random_state=42) if len(data) > n else data.copy()
    try:
        p = positive_probability(model, preprocessor, sample[features])
        view = sample.copy()
        view["Predicted probability"] = p
        view["Predicted risk"] = np.where(p >= saved_threshold, "High risk", "Not flagged")
        a,b,c = st.columns(3)
        a.metric("Rows analyzed", f"{len(view):,}")
        b.metric("Flagged by model", f"{(p >= saved_threshold).sum():,}")
        c.metric("Mean predicted probability", f"{np.mean(p):.1%}")
        left,right = st.columns(2)
        with left:
            fig = px.histogram(view, x="Predicted probability", nbins=30, title="Predicted probability distribution",
                               color_discrete_sequence=[palette["accent"]])
            st.plotly_chart(fig, use_container_width=True)
        with right:
            risk_counts = view["Predicted risk"].value_counts().rename_axis("Prediction").reset_index(name="Shipments")
            fig = px.pie(risk_counts, names="Prediction", values="Shipments", hole=.48, title="Model risk flags")
            st.plotly_chart(fig, use_container_width=True)
        group_candidates = [c for c in ["Shipping Mode","Type","Category Name","Order State","Customer State","Market","is_express_shipping"] if c in view.columns]
        if group_candidates:
            group = st.selectbox("Compare predicted risk by segment", group_candidates)
            agg = view.groupby(group, dropna=False).agg(
                Shipments=("Predicted probability","size"),
                Mean_predicted_probability=("Predicted probability","mean"),
                Flagged=("Predicted risk",lambda x:(x=="High risk").sum())
            ).reset_index().sort_values("Mean_predicted_probability",ascending=False).head(20)
            l,r = st.columns(2)
            with l: st.dataframe(agg,use_container_width=True,hide_index=True)
            with r:
                fig = px.bar(agg, x=group, y="Mean_predicted_probability", title=f"Mean predicted risk by {group}",
                             color_discrete_sequence=[palette["accent"]])
                fig.update_layout(xaxis_tickangle=-35)
                st.plotly_chart(fig,use_container_width=True)
        st.download_button("Download prediction analytics CSV", view.to_csv(index=False).encode("utf-8"),
                           "prediction_analytics.csv","text/csv")
    except Exception as exc:
        st.error(f"Could not calculate prediction analytics: {exc}")

    st.divider()
    st.subheader("Exploratory shipment clusters")
    st.caption("Uses numeric model input features only. It is an unsupervised view to explore similar records; clusters do not mean high/low risk.")
    numeric_candidates = [c for c in features if c in data.columns and pd.api.types.is_numeric_dtype(data[c]) and c not in LEAKAGE_NAMES]
    if len(numeric_candidates) < 2:
        st.info("At least two numeric input features are required to build an exploratory cluster plot.")
    else:
        chosen_numeric = st.multiselect("Numeric features for clustering", numeric_candidates,
                                        default=numeric_candidates[:min(5,len(numeric_candidates))])
        k = st.slider("Number of clusters", 2, 8, 4)
        if len(chosen_numeric) >= 2 and st.button("Build cluster graph"):
            try:
                clust_data = data[chosen_numeric].apply(pd.to_numeric, errors="coerce")
                clust_data = clust_data.replace([np.inf,-np.inf],np.nan)
                medians = clust_data.median()
                clust_data = clust_data.fillna(medians).fillna(0)
                if len(clust_data) > 5000:
                    idx = clust_data.sample(n=5000,random_state=42).index
                    clust_data = clust_data.loc[idx]
                    source_for_plot = data.loc[idx]
                else:
                    source_for_plot = data.loc[clust_data.index]
                scaled = StandardScaler().fit_transform(clust_data)
                cluster_ids = KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(scaled)
                coords = PCA(n_components=2,random_state=42).fit_transform(scaled)
                plot = pd.DataFrame({"PC 1":coords[:,0],"PC 2":coords[:,1],"Cluster":cluster_ids.astype(str)})
                if TARGET in source_for_plot.columns:
                    plot["Observed label"] = source_for_plot[TARGET].astype(str).values
                fig = px.scatter(plot,x="PC 1",y="PC 2",color="Cluster",hover_data=["Observed label"] if "Observed label" in plot else None,
                                 title="PCA projection of exploratory shipment clusters", opacity=.75)
                st.plotly_chart(fig,use_container_width=True)
                st.caption(f"Clustered {len(plot):,} rows using {len(chosen_numeric)} numeric features. PCA axes are projections, not business units.")
                st.download_button("Download cluster assignments", plot.to_csv(index=False).encode("utf-8"),
                                   "exploratory_shipment_clusters.csv","text/csv")
            except Exception as exc:
                st.error(f"Clustering failed: {exc}")

# ---------------- Model Performance ----------------
elif page == "Model Performance":
    st.title("Model Performance & Error Analysis")
    st.caption("Metrics below are calculated from the saved model on the held-out Phase 2 test split when available.")
    if test is None:
        st.error("Test split is unavailable. No test metrics are recalculated.")
        st.markdown("### Recorded Phase 5 metrics")
        rec = {"Accuracy":.6365,"Precision":.6233,"Recall":.8520,"F1-score":.7199,"ROC-AUC":.7622,"Average Precision":.8276}
        cols = st.columns(3)
        for i,(key,val) in enumerate(rec.items()): cols[i%3].metric(key,f"{val:.4f}")
        st.info("These are the recorded Phase 5 metrics; confusion matrix cannot be reconstructed without test predictions.")
    elif TARGET not in test.columns:
        st.error("Test target labels are missing.")
    else:
        try:
            Xtest = test[features].copy()
            ytest = pd.to_numeric(test[TARGET],errors="coerce")
            valid = ytest.notna()
            Xtest, ytest = Xtest.loc[valid], ytest.loc[valid].astype(int)
            probs = positive_probability(model,preprocessor,Xtest)
            threshold = st.slider("Evaluation threshold",0.0,1.0,float(saved_threshold),0.01,key="eval_threshold")
            pred = (probs >= threshold).astype(int)
            auc = roc_auc_score(ytest,probs) if ytest.nunique() == 2 else np.nan
            ap = average_precision_score(ytest,probs) if ytest.nunique() == 2 else np.nan
            metrics = {
                "Accuracy":accuracy_score(ytest,pred),
                "Precision":precision_score(ytest,pred,zero_division=0),
                "Recall":recall_score(ytest,pred,zero_division=0),
                "F1-score":f1_score(ytest,pred,zero_division=0),
                "ROC-AUC":auc,
                "Average Precision":ap,
                "Error rate":1-accuracy_score(ytest,pred),
                "False negative rate":( ((ytest.to_numpy()==1)&(pred==0)).sum() / max((ytest.to_numpy()==1).sum(),1) ),
            }
            cols=st.columns(4)
            for i,(key,val) in enumerate(metrics.items()):
                cols[i%4].metric(key, f"{val:.4f}" if pd.notna(val) else "N/A")
            cm=confusion_matrix(ytest,pred,labels=[0,1])
            left,right=st.columns([1,1])
            with left:
                fig=px.imshow(cm,text_auto=True,x=["Predicted not late","Predicted late"],
                              y=["Actual not late","Actual late"],color_continuous_scale="Blues",
                              title="Confusion matrix")
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig,use_container_width=True)
            with right:
                report=classification_report(ytest,pred,labels=[0,1],target_names=["Not late","Late"],output_dict=True,zero_division=0)
                rep=pd.DataFrame(report).T
                st.markdown("### Classification report")
                st.dataframe(rep.round(4),use_container_width=True)
            st.markdown("### Error breakdown")
            tn,fp,fn,tp=cm.ravel()
            e1,e2,e3,e4=st.columns(4)
            e1.metric("True negatives",f"{tn:,}")
            e2.metric("False positives",f"{fp:,}")
            e3.metric("False negatives",f"{fn:,}")
            e4.metric("True positives",f"{tp:,}")
            st.caption("The threshold slider changes classification metrics only. ROC-AUC and Average Precision use continuous probabilities. Do not use the test split to choose a production threshold; the threshold was selected on validation data.")
            perf_df=pd.DataFrame({"actual":ytest.to_numpy(),"predicted":pred,"probability":probs})
            st.download_button("Download test prediction errors",perf_df.to_csv(index=False).encode("utf-8"),
                               "test_prediction_error_analysis.csv","text/csv")
        except Exception as exc:
            st.error(f"Could not evaluate test performance: {exc}")
            st.info("Ensure the saved model bundle and Phase 2 test.csv come from the same run.")

# ---------------- Risk Analysis ----------------
elif page == "Risk Analysis":
    st.title("Risk Analysis")
    st.caption("Explore predicted risk by operational segments; observed rates are shown separately where labels exist.")
    candidates=[c for c in ["Shipping Mode","Type","Category Name","Order State","Customer State","Market","is_express_shipping"] if c in data.columns]
    if not candidates:
        st.info("No categorical segment columns found.")
    else:
        group=st.selectbox("Segment",candidates)
        max_rows=min(len(data),10000)
        sample=data.sample(n=max_rows,random_state=42) if len(data)>max_rows else data
        try:
            summary=grouped_risk(sample,group,saved_threshold,model,preprocessor,features)
            st.dataframe(summary,use_container_width=True,hide_index=True)
            ycol="Mean_predicted_risk"
            fig=px.bar(summary.head(20),x=group,y=ycol,color_discrete_sequence=[palette["accent"]],
                       title=f"Mean predicted probability by {group}")
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig,use_container_width=True)
            st.download_button("Download segment risk CSV",summary.to_csv(index=False).encode("utf-8"),
                               "segment_risk_analysis.csv","text/csv")
        except Exception as exc:
            st.error(f"Risk analysis failed: {exc}")

# ---------------- Explainability ----------------
elif page == "Explainability":
    st.title("Explainability")
    st.write("Permutation importance reflects predictive reliance, not causation.")
    chart=PHASE6_DIR/"global_feature_importance.png"
    csv=PHASE6_DIR/"feature_importance.csv"
    if chart.exists(): st.image(str(chart),caption="Phase 6 global permutation importance",use_container_width=True)
    else: st.info("Phase 6 chart not found.")
    if csv.exists():
        importance=pd.read_csv(csv)
        st.dataframe(importance.head(30),use_container_width=True,hide_index=True)
        st.download_button("Download feature importance",csv.read_bytes(),"feature_importance.csv","text/csv")
    local=PHASE6_DIR/"local_case_examples.csv"
    if local.exists():
        with st.expander("Local case examples"):
            st.dataframe(pd.read_csv(local),use_container_width=True,hide_index=True)
    with st.expander("Phase 6 report"): markdown_report(PHASE6_DIR/"PHASE_6_REPORT.md")

# ---------------- Prediction History ----------------
elif page == "Prediction History":
    st.title("Prediction History")
    st.caption("Session-only history. It is not persisted to disk or shared with an external service.")
    hist=st.session_state.prediction_history
    if not hist:
        st.info("No predictions in this session yet. Use Model Prediction first.")
    else:
        frame=pd.DataFrame(hist)
        st.dataframe(frame,use_container_width=True,hide_index=True)
        st.download_button("Download history CSV",frame.to_csv(index=False).encode("utf-8"),
                           "prediction_history.csv","text/csv")
        if st.button("Clear session history"):
            st.session_state.prediction_history=[]
            st.rerun()

# ---------------- About website ----------------
elif page == "About This Website":
    st.title("About This Website")
    st.caption("Project overview, methodology, and responsible-use notes.")
    st.markdown("""
    ### Late Delivery Risk Prediction
    This application demonstrates a machine-learning workflow for estimating whether a shipment may be late, based on the features provided to the saved model.

    **What the website provides**
    - Single-shipment and CSV batch predictions.
    - Risk analytics by shipping and geographic segments.
    - Model evaluation with error breakdowns.
    - Phase 6 global permutation-importance results.
    - Session-only prediction history and appearance settings.

    **Methodology**
    - Phase 1: data audit and exploratory analysis.
    - Phase 2: preprocessing and feature engineering.
    - Phase 3: Logistic Regression baseline.
    - Phase 4: candidate-model comparison.
    - Phase 5: threshold selection and held-out test evaluation.
    - Phase 6: global permutation importance and local case examples.
    - Phase 7: Streamlit dashboard.

    **Responsible use**
    This is a portfolio/research decision-support project. A predicted risk is not a delivery guarantee. The original data split was random, so future-time generalization has not been demonstrated. Before operational use, verify that every input is available at the prediction time and remove unavailable or post-outcome fields.
    """)
    with st.expander("Recorded Phase 5 report"): markdown_report(PHASE5_REPORT)

# ---------------- About developer ----------------
elif page == "About Developer":
    st.title("About the Developer")
    st.caption("Project author and implementation details")
    st.markdown("""
    ### Utkarsh Sengar
    **AI/ML Engineering student · Machine Learning & Data Applications**

    This project explores the end-to-end development of a late-delivery risk prediction system: data preparation, model evaluation, explainability, and a practical dashboard.

    **Project work demonstrated**
    - Python-based data processing and machine learning.
    - Scikit-learn model training and evaluation.
    - Feature engineering and model explainability.
    - Streamlit interface development and interactive analytics.
    - Responsible reporting of metrics, limitations, and model assumptions.

    **Project note**
    This section describes the developer and work included in this project. It does not imply external certification, production deployment, or business results that have not been independently verified.
    """)
    st.info("You can personalize this page later with your GitHub, LinkedIn, portfolio URL, and a professional profile photo.")

st.divider()
st.caption("Late Delivery Risk Prediction · Phase 7 dashboard upgrade · Built with Streamlit and Plotly")
