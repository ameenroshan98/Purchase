import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Purchase Dashboard", layout="wide")

SHEET_ID = "1R7o4xKMeYWcYWwAOorMyYDtjg0-74FqDK0xAFKN6Iuo"
RANGE = "Sheet1!A1:I1000"  # A..I covers: Code..Bonus %
API_KEY = st.secrets["gcp"]["api_key"]  # put your key in Streamlit Secrets

URL = f"https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}/values/{RANGE}?key={API_KEY}"

st.title("Google Sheets Purchase Dashboard")

try:
    r = requests.get(URL, timeout=20)
    r.raise_for_status()
    data = r.json()

    if "values" not in data or not data["values"]:
        st.warning("No data found in sheet.")
    else:
        values = data["values"]
        df = pd.DataFrame(values[1:], columns=values[0])

        # coerce numeric columns
        numeric_cols = [
            "Qty Purchased", "Bonus Received", "No. of Times Purchased",
            "No. of Times Bonus Received", "Avg. Purchase Qty",
            "Avg. Bonus Qty", "Bonus %"
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        st.dataframe(df, use_container_width=True)

        # quick KPIs
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Purchased", f"{int(df['Qty Purchased'].sum()):,}")
        with col2: st.metric("Total Bonus", f"{int(df['Bonus Received'].sum()):,}")
        with col3: st.metric("Products", f"{len(df):,}")

except Exception as e:
    st.error(f"⚠️ {e}")
