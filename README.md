# app.py
import os
import json
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Purchase Dashboard", layout="wide")

# ── Config ─────────────────────────────────────────────────────────────────────
SHEET_ID = "1R7o4xKMeYWcYWwAOorMyYDtjg0-74FqDK0xAFKN6Iuo"
RANGE = "Sheet1!A1:I1000"  # A..I covers: Code..Bonus %

API_KEY = None
try:
    # Safe access: handle missing [gcp] or api_key without crashing
    API_KEY = st.secrets.get("gcp", {}).get("api_key")
except Exception:
    API_KEY = None

API_URL = (
    f"https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}/values/{RANGE}"
    + (f"?key={API_KEY}" if API_KEY else "")
)

CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

NUMERIC_COLS = [
    "Qty Purchased", "Bonus Received", "No. of Times Purchased",
    "No. of Times Bonus Received", "Avg. Purchase Qty", "Avg. Bonus Qty", "Bonus %"
]

# ── Data loaders with caching ──────────────────────────────────────────────────
@st.cache_data(ttl=600)
def fetch_via_api(url: str):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    if "values" not in data or not data["values"]:
        raise ValueError("Google Sheets API returned no 'values'.")
    values = data["values"]
    return pd.DataFrame(values[1:], columns=values[0])

@st.cache_data(ttl=600)
def fetch_via_csv(url: str):
    return pd.read_csv(url)

def load_data():
    # Prefer API if we have a key; otherwise try CSV export
    if API_KEY:
        try:
            df = fetch_via_api(API_URL)
            return df, "api"
        except Exception as e_api:
            # Fall back to CSV if API fails (quota, bad range, etc.)
            try:
                df = fetch_via_csv(CSV_URL)
                return df, f"csv_fallback (API error: {type(e_api).__name__})"
            except Exception as e_csv:
                raise RuntimeError(f"API and CSV both failed: {e_api} | {e_csv}")
    else:
        # No key configured → use CSV directly
        return fetch_via_csv(CSV_URL), "csv_no_key"

# ── UI header ──────────────────────────────────────────────────────────────────
st.title("Google Sheets Purchase Dashboard")

# Optional: show page links only if files exist, preventing crashes
def safe_page_link(rel_path: str, label: str):
    if os.path.exists(rel_path):
        try:
            st.page_link(rel_path, label=label)
        except Exception:
            pass  # Never crash on a link
# Example (uncomment if you actually have this page):
# safe_page_link("pages/Compare_Yearly_Patterns.py", "📅 Compare Ranges")

# ── Load & display data ────────────────────────────────────────────────────────
try:
    df, source = load_data()

    # Coerce numeric columns gracefully
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    with st.expander("Data source & diagnostics", expanded=False):
        st.write({"data_source": source, "rows": len(df), "cols": list(df.columns)})
        if not API_KEY:
            st.info("Running without GCP API key. Using CSV export.")

    st.dataframe(df, use_container_width=True)

    # KPIs (guard against missing columns)
    kpi1 = int(df["Qty Purchased"].sum()) if "Qty Purchased" in df.columns else 0
    kpi2 = int(df["Bonus Received"].sum()) if "Bonus Received" in df.columns else 0
    kpi3 = df["Code"].nunique() if "Code" in df.columns else len(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Purchased (Units)", f"{kpi1:,}")
    col2.metric("Total Bonus (Units)", f"{kpi2:,}")
    col3.metric("Products", f"{kpi3:,}")

    # Simple filter (optional)
    if "Product" in df.columns:
        selected = st.selectbox("Filter by Product", ["All"] + sorted(df["Product"].dropna().unique().tolist()))
        if selected != "All":
            df = df[df["Product"] == selected]
            st.dataframe(df, use_container_width=True)

except Exception as e:
    st.error("The app encountered an error but didn’t crash. Open the expander below for details.")
    with st.expander("Error details", expanded=True):
        st.exception(e)
