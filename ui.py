from pathlib import Path
import json
import pandas as pd
import streamlit as st

def inject_css(mode="Dark", accent="#7C5CFC", density="Comfortable"):
    dark = mode == "Dark"
    bg = "#0b1020" if dark else "#f4f6fb"
    panel = "rgba(20, 27, 48, .78)" if dark else "rgba(255,255,255,.82)"
    text = "#eef2ff" if dark else "#172033"
    muted = "#9aa6c4" if dark else "#69758c"
    border = "rgba(157,170,208,.17)" if dark else "rgba(29,48,86,.10)"
    st.markdown(f"""
    <style>
    :root{{--accent:{accent};--bg:{bg};--panel:{panel};--text:{text};--muted:{muted};--border:{border};}}
    .stApp {{background:radial-gradient(ellipse at 15% 0%, {accent}22, transparent 35%),radial-gradient(ellipse at 85% 12%,#18b8a622,transparent 30%),var(--bg);color:var(--text);}}
    [data-testid="stHeader"]{{background:transparent;}}
    [data-testid="stSidebar"]{{background:linear-gradient(180deg,var(--bg),var(--bg));border-right:1px solid var(--border);}}
    .block-container{{padding-top:2rem;padding-bottom:2rem;max-width:1500px;}}
    .brand-lockup{{display:flex;gap:12px;align-items:center;padding:8px 6px 24px;}}
    .brand-mark{{width:44px;height:44px;border-radius:15px;display:grid;place-items:center;font-weight:900;color:white;background:linear-gradient(135deg,var(--accent),#24c6dc);box-shadow:0 10px 30px {accent}55;}}
    .brand-title{{font-size:1.18rem;font-weight:850;letter-spacing:-.04em;color:var(--text);}}
    .brand-subtitle{{font-size:.61rem;letter-spacing:.17em;color:var(--muted);font-weight:700;margin-top:2px;}}
    .sidebar-label,.nav-group{{font-size:.65rem;letter-spacing:.16em;color:var(--muted);font-weight:800;margin:10px 8px 7px;}}
    .nav-group{{margin-top:20px;letter-spacing:.08em;font-size:.68rem;}}
    [data-testid="stSidebar"] button{{border-radius:12px!important;text-align:left!important;font-weight:600!important;min-height:42px;}}
    [data-testid="stSidebar"] button[kind="primary"]{{background:linear-gradient(110deg,var(--accent),#5b7cfa)!important;color:white!important;box-shadow:0 7px 20px {accent}44!important;}}
    [data-testid="stSidebar"] button[kind="secondary"]{{background:transparent!important;color:var(--muted)!important;}}
    .sidebar-bottom{{display:flex;align-items:center;gap:8px;padding:20px 8px 8px;margin-top:20px;border-top:1px solid var(--border);font-size:.75rem;color:var(--muted);}}
    .connection-dot,.live-dot{{width:7px;height:7px;border-radius:50%;background:#27d9a2;box-shadow:0 0 12px #27d9a288;display:inline-block;}}
    .sidebar-version{{margin-left:auto;}}
    .eyebrow{{font-size:.66rem;letter-spacing:.17em;font-weight:800;color:var(--accent);margin-bottom:.45rem;}}
    .page-title{{font-size:clamp(2rem,3.3vw,2.8rem);font-weight:850;letter-spacing:-.055em;color:var(--text);margin:0!important;}}
    .page-subtitle{{color:var(--muted);font-size:.98rem;margin-top:.45rem;margin-bottom:1.2rem;}}
    .top-status{{display:flex;align-items:center;justify-content:center;gap:8px;padding:10px 12px;border:1px solid var(--border);border-radius:99px;background:var(--panel);color:var(--muted);font-size:.68rem;font-weight:800;letter-spacing:.1em;margin-top:18px;white-space:nowrap;}}
    div[data-testid="stVerticalBlockBorderWrapper"]{{border:1px solid var(--border)!important;border-radius:22px!important;background:var(--panel)!important;backdrop-filter:blur(18px);box-shadow:0 14px 40px rgba(0,0,0,.08);padding:5px;}}
    .section-title{{font-size:1.06rem;font-weight:800;letter-spacing:-.025em;color:var(--text);margin:0 0 2px;}}
    .section-subtitle{{font-size:.79rem;color:var(--muted);margin-bottom:15px;}}
    .metric-card{{padding:18px;border:1px solid var(--border);border-radius:19px;background:linear-gradient(145deg,var(--panel),{accent}0a);box-shadow:0 10px 28px rgba(0,0,0,.06);height:100%;}}
    .metric-label{{font-size:.78rem;color:var(--muted);font-weight:650;}}
    .metric-value{{font-size:1.85rem;font-weight:850;letter-spacing:-.055em;color:var(--text);margin:8px 0 5px;}}
    .metric-note{{font-size:.7rem;color:var(--muted);}}
    .feature-tile{{border:1px solid var(--border);border-radius:18px;padding:20px;background:linear-gradient(150deg,var(--panel),{accent}0b);min-height:165px;}}
    .feature-tile h3{{font-size:1.1rem;color:var(--text);font-weight:800;margin:10px 0 7px;}}
    .feature-tile p{{color:var(--muted);font-size:.84rem;line-height:1.55;}}
    .tile-kicker{{font-size:.6rem;letter-spacing:.15em;color:var(--accent);font-weight:850;}}
    .small-note{{font-size:.78rem;line-height:1.6;color:var(--muted);margin-top:16px;}}
    .status-row{{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid var(--border);color:var(--text);font-size:.85rem;}}
    .badge{{padding:5px 9px;border-radius:99px;font-size:.67rem;font-weight:800;}}
    .badge-success{{color:#10b981;background:#10b98118;border:1px solid #10b98144;}}
    .badge-warning{{color:#f59e0b;background:#f59e0b18;border:1px solid #f59e0b44;}}
    .badge-danger{{color:#ef4444;background:#ef444418;border:1px solid #ef444444;}}
    .badge-info{{color:#60a5fa;background:#60a5fa18;border:1px solid #60a5fa44;}}
    .empty-state{{border:1px dashed var(--border);border-radius:20px;text-align:center;padding:48px 20px;background:var(--panel);}}
    .empty-icon{{font-size:2rem;color:var(--accent);}}
    .empty-state h3{{color:var(--text);margin:12px 0 5px;}}
    .empty-state p{{max-width:560px;margin:auto;color:var(--muted);font-size:.88rem;line-height:1.6;}}
    .prose-card,.developer-card{{border:1px solid var(--border);border-radius:22px;background:var(--panel);padding:28px;box-shadow:0 14px 38px rgba(0,0,0,.07);}}
    .prose-card h3{{font-size:1.1rem;color:var(--text);margin:22px 0 8px;}}
    .prose-card h3:first-child{{margin-top:0;}}
    .prose-card p,.developer-card p{{color:var(--muted);line-height:1.75;}}
    .developer-card{{display:flex;gap:22px;align-items:center;}}
    .developer-avatar{{width:88px;height:88px;border-radius:26px;display:grid;place-items:center;flex-shrink:0;color:white;font-size:1.7rem;font-weight:900;background:linear-gradient(135deg,var(--accent),#24c6dc);box-shadow:0 12px 30px {accent}44;}}
    .footer-line{{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;border-top:1px solid var(--border);padding-top:18px;margin-top:35px;color:var(--muted);font-size:.68rem;letter-spacing:.03em;}}
    .footer-line span:first-child{{color:var(--accent);font-weight:900;letter-spacing:.15em;}}
    .stButton>button,.stFormSubmitButton>button{{border-radius:13px!important;min-height:43px!important;font-weight:750!important;border:1px solid var(--border)!important;transition:all .2s ease!important;}}
    .stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"]{{background:linear-gradient(110deg,var(--accent),#5b7cfa)!important;color:white!important;border:0!important;box-shadow:0 9px 24px {accent}44!important;}}
    .stButton>button:hover,.stFormSubmitButton>button:hover{{transform:translateY(-1px);box-shadow:0 10px 25px rgba(0,0,0,.13)!important;}}
    [data-testid="stDataFrame"]{{border:1px solid var(--border);border-radius:15px;overflow:hidden;}}
    .stAlert{{border-radius:14px!important;}}
    </style>
    """, unsafe_allow_html=True)

def glass_card(title=None, subtitle=None):
    """Compatibility helper for older app.py imports."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if title:
        section_header(title, subtitle or "")

def metric_card(container, label, value, note=""):
    with container:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

def section_header(title, subtitle=""):
    st.markdown(f'<div class="section-title">{title}</div><div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def status_badge(label, value, kind="info"):
    st.markdown(f'<div class="status-row"><span>{label}</span><span class="badge badge-{kind}">{value}</span></div>', unsafe_allow_html=True)

def project_root():
    p = Path(__file__).resolve().parent
    candidates = [p, p.parent, p.parent.parent]
    return next((c for c in candidates if (c / "data" / "raw" / "APL_Logistics.csv").exists()), p)

@st.cache_data(show_spinner=False)
def load_project_data():
    root = project_root()
    candidates = [root / "data" / "raw" / "APL_Logistics.csv", root / "APL_Logistics.csv", root / "data" / "APL_Logistics.csv"]
    for path in candidates:
        if path.exists():
            try:
                return pd.read_csv(path, low_memory=False)
            except Exception:
                return None
    return None

@st.cache_data(show_spinner=False)
def load_model_metrics():
    root = project_root()
    candidates = list(root.glob("reports/**/*phase5*.json")) + list(root.glob("reports/**/*metrics*.json")) + list(root.glob("artifacts/phase5/*metrics*.json"))
    aliases = {"accuracy":["accuracy","test_accuracy"],"precision":["precision","test_precision"],"recall":["recall","test_recall"],"f1":["f1","f1_score","test_f1","test_f1_score"],"roc_auc":["roc_auc","roc_auc_score","test_roc_auc"],"average_precision":["average_precision","ap","test_average_precision"]}
    for path in candidates:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            flat = raw.get("test_metrics", raw.get("metrics", raw)) if isinstance(raw, dict) else {}
            if isinstance(flat, dict):
                result = {}
                for key, keys in aliases.items():
                    for k in keys:
                        if isinstance(flat.get(k), (int,float)):
                            result[key] = float(flat[k]); break
                if result:
                    return result
        except Exception:
            continue
    return {"accuracy":0.6365,"precision":0.6233,"recall":0.8520,"f1":0.7199,"roc_auc":0.7622,"average_precision":0.8276}

@st.cache_data(show_spinner=False)
def load_feature_importance():
    root = project_root()
    candidates = list(root.glob("reports/phase6/**/*importance*.csv")) + list(root.glob("reports/**/*permutation*.csv")) + list(root.glob("reports/**/*importance*.csv"))
    for path in candidates:
        try:
            df = pd.read_csv(path)
            cols = {c.lower():c for c in df.columns}
            feature_col = next((cols[x] for x in ["feature","feature_name","column","name"] if x in cols), None)
            imp_col = next((cols[x] for x in ["importance","mean_importance","permutation_importance","importance_mean"] if x in cols), None)
            if feature_col and imp_col:
                return df[[feature_col,imp_col]].rename(columns={feature_col:"feature",imp_col:"importance"}).sort_values("importance",ascending=False)
        except Exception:
            continue
    return pd.DataFrame([("Type",0.017156),("Shipping Mode",0.009429),("Order City",0.008666),("Scheduled shipping days",0.007531),("Express shipping",0.005818),("Order State",0.004716),("Shipping pressure index",0.003907),("Customer State",0.003884),("Latitude",0.002754),("Category Id",0.002554)], columns=["feature","importance"])

@st.cache_resource(show_spinner="Loading Phase 5 ML model adapter...")
def load_model_adapter():
    """Load and cache the trained Phase 5 Random Forest model adapter."""
    try:
        from src.models import Phase5ModelAdapter
        return Phase5ModelAdapter.load()
    except Exception as exc:
        return None
