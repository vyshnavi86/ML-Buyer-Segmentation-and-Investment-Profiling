import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Buyer Segmentation", layout="wide")
st.title("🏠 ML Buyer Segmentation & Investment Profiling")
st.write("Machine learning based Buyer Segmentation for Real Estate Market Intelligence")

@st.cache_data
def load_data():
    df = pd.read_csv("segmented_clients.csv")
    return df

try:
    df = load_data()
except:
    st.error("segmented_clients.csv not found!")
    st.stop()

# Find cluster column
cluster_col = None
for c in df.columns:
    if c.lower() in ["cluster", "segment", "segment_label", "profile", "buyer_segment"]:
        cluster_col = c
        break
if cluster_col is None:
    cluster_col = df.columns[-1]

st.sidebar.header("Filter")
segments = sorted(df[cluster_col].astype(str).unique().tolist())
selected = st.sidebar.multiselect("Select Segment", segments, default=segments)
filtered = df[df[cluster_col].astype(str).isin(selected)]

col1, col2, col3 = st.columns(3)
col1.metric("Total Clients", len(df))
col2.metric("Filtered Clients", len(filtered))
col3.metric("Total Segments", df[cluster_col].nunique())

st.subheader("Segment Distribution")
fig = px.pie(df, names=cluster_col, title="Buyer Segments")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Filtered Data")
st.dataframe(filtered, use_container_width=True)

st.subheader("Segment-wise Average")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if numeric_cols:
    st.dataframe(df.groupby(cluster_col)[numeric_cols].mean().round(2))

st.download_button("Download Filtered Data", filtered.to_csv(index=False), "filtered_clients.csv")
