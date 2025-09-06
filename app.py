import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Purchase Dashboard", layout="wide")

SHEET_ID = "1R7o4xKMeYWcYWwAOorMyYDtjg0-74FqDK0xAFKN6Iuo"
RANGE    = "Sheet1!A1:I20000"     # A..I = Code..Bonus %
GID      = 0                      # change if your tab is not the first
CSV_URL  = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

NUMERIC_COLS = [
    "Qty Purchased", "Bonus Received", "No. of Times Purchased",
    "No. of Times Bonus Received", "Avg. Purchase Qty", "Avg. Bonus Qty", "Bonus %"
]

# -----------------------------
# Data loader (API -> fallback CSV)
# -----------------------------
@st.cache_data(ttl=600)
def load_data():
    api_key = st.secrets.get("gcp", {}).get("api_key")
    if api_key:
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}/values/{RANGE}?key={api_key}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        js = r.json()
        vals = js.get("values", [])
        if not vals:
            return pd.DataFrame()
        df = pd.DataFrame(vals[1:], columns=vals[0])
    else:
        df = pd.read_csv(CSV_URL)

    # Coerce numerics
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Standardize column names we’ll reference
    rename_map = {
        "Code": "Code",
        "Product": "Product",
        "Qty Purchased": "Qty Purchased",
        "Bonus Received": "Bonus Received",
        "No. of Times Purchased": "Times Purchased",
        "No. of Times Bonus Received": "Times Bonus",
        "Avg. Purchase Qty": "Avg Purchase Qty",
        "Avg. Bonus Qty": "Avg Bonus Qty",
        "Bonus %": "Bonus %"
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    return df

df = load_data()

st.title("📊 Purchase Dashboard")

if df.empty:
    st.warning("No data found. Check your sheet sharing, range, or API key.")
    st.stop()

# -----------------------------
# Sidebar Filters
# -----------------------------
with st.sidebar:
    st.header("Filters")
    q = st.text_input("Search (code or product)", "")
    bonus_filter = st.selectbox("Bonus filter", ["All", "With Bonus", "Without Bonus"])
    min_qty = st.slider("Min Qty Purchased", 0, int(df["Qty Purchased"].max()), 0, step=100)
    top_n = st.slider("Top-N for charts", 5, 30, 10)

# Apply filters
mask = pd.Series(True, index=df.index)
if q.strip():
    q_lower = q.strip().lower()
    mask &= df["Product"].str.lower().str.contains(q_lower, na=False) | df["Code"].astype(str).str.contains(q_lower, na=False)

if bonus_filter == "With Bonus":
    mask &= df["Bonus %"] > 0
elif bonus_filter == "Without Bonus":
    mask &= df["Bonus %"] == 0

mask &= df["Qty Purchased"] >= min_qty

fdf = df[mask].copy()

# -----------------------------
# KPIs
# -----------------------------
total_purchased = int(fdf["Qty Purchased"].sum())
total_bonus = int(fdf["Bonus Received"].sum())
overall_bonus_rate = (fdf["Bonus Received"].sum() / fdf["Qty Purchased"].sum() * 100) if fdf["Qty Purchased"].sum() > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Products (after filters)", f"{len(fdf):,}")
c2.metric("Total Purchased", f"{total_purchased:,}")
c3.metric("Total Bonus", f"{total_bonus:,}")
c4.metric("Overall Bonus %", f"{overall_bonus_rate:.1f}%")

st.divider()

# -----------------------------
# Prepare Top-N (by Qty)
# -----------------------------
top_by_qty = fdf.sort_values("Qty Purchased", ascending=False).head(top_n).copy()
top_by_qty["Code • Product"] = top_by_qty["Code"].astype(str) + " • " + top_by_qty["Product"].astype(str)

# -----------------------------
# Bar: Purchased vs Bonus (Top-N)
# -----------------------------
st.subheader("Purchased vs Bonus — Top Products")
if not top_by_qty.empty:
    bar_df = top_by_qty.melt(
        id_vars=["Code • Product"],
        value_vars=["Qty Purchased", "Bonus Received"],
        var_name="Metric", value_name="Value"
    )
    fig_bar = px.bar(
        bar_df, x="Code • Product", y="Value", color="Metric",
        barmode="group", height=420
    )
    fig_bar.update_layout(xaxis_title="", yaxis_title="", margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("No rows after filters for bar chart.")

# -----------------------------
# Pie: Purchase share (Top-N by Qty)
# -----------------------------
colA, colB = st.columns(2)
with colA:
    st.subheader("Purchase Share — Top Products")
    if not top_by_qty.empty:
        fig_pie = px.pie(
            top_by_qty, names="Code • Product", values="Qty Purchased", height=420
        )
        fig_pie.update_layout(margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No rows after filters for pie chart.")

# -----------------------------
# Line: Bonus % (Top-N by Qty or by Bonus %)
# -----------------------------
with colB:
    st.subheader("Bonus % across Products")
    line_df = fdf.sort_values("Bonus %", ascending=False).head(top_n).copy()
    if not line_df.empty:
        line_df["Code • Product"] = line_df["Code"].astype(str) + " • " + line_df["Product"].astype(str)
        fig_line = px.line(
            line_df, x="Code • Product", y="Bonus %", markers=True, height=420
        )
        fig_line.update_layout(xaxis_title="", yaxis_title="Bonus %", margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No rows after filters for line chart.")

st.divider()

# -----------------------------
# Highlights (like your React view)
# -----------------------------
st.subheader("Highlights")
left, right = st.columns(2)

with left:
    st.markdown("**Top by Volume (Qty Purchased)**")
    t1 = fdf.sort_values("Qty Purchased", ascending=False).head(10)[
        ["Code", "Product", "Qty Purchased", "Bonus Received", "Avg Purchase Qty", "Bonus %"]
    ]
    st.dataframe(t1, use_container_width=True, hide_index=True)

with right:
    st.markdown("**Top by Bonus %**")
    t2 = fdf[fdf["Bonus %"] > 0].sort_values("Bonus %", ascending=False).head(10)[
        ["Code", "Product", "Qty Purchased", "Bonus Received", "Avg Bonus Qty", "Bonus %"]
    ]
    st.dataframe(t2, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# Detailed Table (filtered)
# -----------------------------
st.subheader(f"Detailed Table ({len(fdf):,} products)")
cols = ["Code","Product","Qty Purchased","Bonus Received","Times Purchased","Times Bonus","Avg Purchase Qty","Avg Bonus Qty","Bonus %"]
present_cols = [c for c in cols if c in fdf.columns]
st.dataframe(fdf[present_cols].sort_values("Qty Purchased", ascending=False), use_container_width=True, hide_index=True)
