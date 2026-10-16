from pathlib import Path
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data.loader import load_dataset, inspect_dataset

st.set_page_config(page_title="Late Delivery Risk Intelligence", page_icon="📦", layout="wide")
st.title("Late Delivery Risk Intelligence")
st.caption("Machine Learning–based Late Delivery Risk Prediction in Global Supply Chain Operations")
st.info("Phase 0 architecture starter. Later phases add leakage-aware training, model evaluation, explanations and operational prioritization.")

df = load_dataset(ROOT / "data" / "raw" / "APL_Logistics.csv")
if df is None:
    st.warning("Dataset not found. Copy your CSV to data/raw/APL_Logistics.csv and refresh.")
    st.code("data/raw/APL_Logistics.csv")
    st.markdown("See README.md for setup and phase roadmap.")
    st.stop()

p = inspect_dataset(df)
a, b, c, d = st.columns(4)
a.metric("Rows", f"{p['rows']:,}")
b.metric("Columns", f"{p['columns']:,}")
c.metric("Duplicate rows", f"{p['duplicate_rows']:,}")
d.metric("Missing cells", f"{p['missing_cells']:,}")
st.subheader("Safe preview")
private = ["Customer Fname", "Customer Lname", "Customer Street", "Customer Zipcode", "Customer Id", "Order Customer Id"]
st.dataframe(df.drop(columns=private, errors="ignore").head(20), use_container_width=True)
st.subheader("Modeling guardrails")
st.markdown("- Target: `Late_delivery_risk`\n- Exclude actual shipping duration and delivery outcome fields.\n- Only use features known at the intended prediction time.")
