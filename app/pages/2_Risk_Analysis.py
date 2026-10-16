from pathlib import Path
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data.loader import load_dataset

st.set_page_config(page_title="Risk Analysis", layout="wide")
st.title("Risk Analysis")
df = load_dataset(ROOT / "data" / "raw" / "APL_Logistics.csv")
if df is None:
    st.warning("Place the CSV at data/raw/APL_Logistics.csv.")
    st.stop()
if "Late_delivery_risk" not in df:
    st.error("Target column Late_delivery_risk not found.")
    st.stop()
st.metric("Historical late-delivery rate", f"{df['Late_delivery_risk'].mean():.1%}")
st.info("This is descriptive history, not an individual order prediction. A validated scoring model arrives in later phases.")
