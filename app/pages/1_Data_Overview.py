from pathlib import Path
import sys
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data.loader import load_dataset

st.set_page_config(page_title="Data Overview", layout="wide")
st.title("Data Overview")
df = load_dataset(ROOT / "data" / "raw" / "APL_Logistics.csv")
if df is None:
    st.warning("Place the CSV at data/raw/APL_Logistics.csv.")
    st.stop()
private = ["Customer Fname", "Customer Lname", "Customer Street", "Customer Zipcode", "Customer Id", "Order Customer Id"]
st.dataframe(df.drop(columns=private, errors="ignore").head(100), use_container_width=True)
if "Late_delivery_risk" in df:
    counts = df["Late_delivery_risk"].astype(str).value_counts().rename_axis("Target").reset_index(name="Orders")
    st.plotly_chart(px.bar(counts, x="Target", y="Orders", title="Historical target distribution"), use_container_width=True)
missing = df.isna().sum().sort_values(ascending=False).head(20).reset_index()
missing.columns = ["Column", "Missing cells"]
st.plotly_chart(px.bar(missing, x="Column", y="Missing cells", title="Missing values by column"), use_container_width=True)
