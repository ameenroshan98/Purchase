import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Purchase Dashboard", layout="wide")

SHEET_ID = "1R7o4xKMeYWcYWwAOorMyYDtjg0-74FqDK0xAFKN6Iuo"
RANGE = "Sheet1!A1:I1000"  # A..I = Code..Bonus %
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=0"

def load_data():
    # try API key first (if present)
    api_key = st.secrets.get("gcp", {}).get("api_key")
    if api_key:
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}/values/{RANGE}?key={api_key}"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        if "values" not in data or not data["values"]:
            return pd.DataFrame()
        values = data["values"]
        return pd.DataFrame(values[1:], columns=values[0])
    # fallback to public CSV
    return pd.read_csv(CSV_URL)

st.title("📊 Purchase Dashboard")

try:
    df = load_data()
    if df.empty:
        st.warning("No data found.")
    else:
        # numeric columns—coerce types
        numeric_cols = [
            "Qty Purchased","Bonus Received","No. of Times Purchased",
            "No. of Times Bonus Received","Avg. Purchase Qty","Avg. Bonus Qty","Bonus %"
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        st.dataframe(df, use_container_width=True)

        # quick KPIs
        c1,c2,c3 = st.columns(3)
        if "Qty Purchased" in df: c1.metric("Total Purchased", f"{int(df['Qty Purchased'].sum()):,}")
        if "Bonus Received" in df: c2.metric("Total Bonus", f"{int(df['Bonus Received'].sum()):,}")
        c3.metric("Products", f"{len(df):,}")
except Exception as e:
    st.error(f"⚠️ {e}")
