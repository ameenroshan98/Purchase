import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Purchase Dashboard", layout="wide")

# ---- Google Sheet config (edit these if needed)
SHEET_ID = "1R7o4xKMeYWcYWwAOorMyYDtjg0-74FqDK0xAFKN6Iuo"
GID      = 0  # change if your data tab is not the first
RANGE    = "Sheet1!A1:G100000"   # A..G = Supplier..Bonus %
CSV_URL  = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def short_label(code, name, n=40):
    name = str(name)
    return f"{code} — {name[:n]}…" if len(name) > n else f"{code} — {name}"

# -------------------------------------------------
# Load data (API -> CSV fallback)
# -------------------------------------------------
@st.cache_data(ttl=600)
def load_transactions():
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

    # Normalize expected headers
    rename_map = {
        "Supplier Name": "Supplier Name",
        "Code": "Code",
        "Product": "Product",
        "Date": "Date",
        "Qty Purchased": "Qty Purchased",
        "Bonus": "Bonus",
        "Bonus %": "Bonus %"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Parse types
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

    for c in ["Qty Purchased", "Bonus"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Clean/convert Bonus %
    if "Bonus %" in df.columns:
        df["Bonus %"] = (
            df["Bonus %"].astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["Bonus %"] = pd.to_numeric(df["Bonus %"], errors="coerce").fillna(0.0)

    return df

tx = load_transactions()

st.title("📊 Purchase Dashboard")

if tx.empty:
    st.warning("No data found. Check your sheet sharing, tab gid, or API key.")
    st.stop()

# -------------------------------------------------
# Optional: basic slicers on the transaction level
# (We aggregate AFTER these)
# -------------------------------------------------
with st.sidebar:
    st.header("Filters")

    q = st.text_input("Search (code or product)", "")
    bonus_filter = st.selectbox("Bonus filter", ["All", "With Bonus", "Without Bonus"])
    min_qty = st.slider("Min Qty Purchased (product total)", 0, int(tx["Qty Purchased"].sum()), 0, step=100)

# Filter at transaction-level for search
mask_tx = pd.Series(True, index=tx.index)
if q.strip():
    ql = q.strip().lower()
    mask_tx &= tx["Product"].str.lower().str.contains(ql, na=False) | tx["Code"].astype(str).str.contains(ql, na=False)

tx_f = tx[mask_tx].copy()

# -------------------------------------------------
# Aggregate to PRODUCT level (for charts/KPIs)
# -------------------------------------------------
if tx_f.empty:
    st.info("No transactions match your filters.")
    st.stop()

agg = (
    tx_f.groupby(["Code", "Product"], as_index=False)
        .agg(
            **{
                "Qty Purchased": ("Qty Purchased", "sum"),
                "Bonus Received": ("Bonus", "sum"),
                "Times Purchased": ("Qty Purchased", "count"),
                "Times Bonus": ("Bonus", lambda s: (s > 0).sum()),
                "Avg Purchase Qty": ("Qty Purchased", "mean"),
                "Avg Bonus Qty": ("Bonus", "mean"),
                # Effective product-level bonus % (bonus/qty * 100)
                "Bonus %": ("Qty Purchased", lambda q: 0)  # placeholder; we set below
            }
        )
)
# Effective bonus % calc
agg["Bonus %"] = (agg["Bonus Received"] / agg["Qty Purchased"]).replace([pd.NA, pd.NaT, float("inf")], 0) * 100
agg["Bonus %"] = agg["Bonus %"].fillna(0.0)

# Apply bonus presence & min qty filters on aggregated table
if bonus_filter == "With Bonus":
    agg = agg[agg["Bonus %"] > 0]
elif bonus_filter == "Without Bonus":
    agg = agg[agg["Bonus %"] == 0]

agg = agg[agg["Qty Purchased"] >= min_qty]

if agg.empty:
    st.info("No products after applying filters. Try lowering the Min Qty or changing Bonus filter.")
    st.stop()

# -------------------------------------------------
# KPIs (from aggregated)
# -------------------------------------------------
total_purchased = int(agg["Qty Purchased"].sum())
total_bonus = int(agg["Bonus Received"].sum())
overall_bonus_rate = (total_bonus / total_purchased * 100) if total_purchased > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Products (after filters)", f"{len(agg):,}")
c2.metric("Total Purchased", f"{total_purchased:,}")
c3.metric("Total Bonus", f"{total_bonus:,}")
c4.metric("Overall Bonus %", f"{overall_bonus_rate:.1f}%")

st.divider()

# -------------------------------------------------
# Top-N by Qty for charts
# -------------------------------------------------
top_n = st.slider("Top-N for charts", 5, 40, 15, key="topn_main")
top_by_qty = agg.sort_values("Qty Purchased", ascending=False).head(top_n).copy()
top_by_qty["Label"] = top_by_qty.apply(lambda r: short_label(r["Code"], r["Product"], 42), axis=1)

# -------------------------------------------------
# Chart 1: Horizontal grouped bars (Qty vs Bonus)
# -------------------------------------------------
st.subheader("Purchased vs Bonus — Top Products")
if not top_by_qty.empty:
    bar_df = top_by_qty.melt(
        id_vars=["Label"],
        value_vars=["Qty Purchased", "Bonus Received"],
        var_name="Metric", value_name="Value"
    )
    fig_bar = px.bar(
        bar_df, y="Label", x="Value", color="Metric",
        orientation="h", height=max(420, 30 * len(top_by_qty))
    )
    fig_bar.update_layout(
        yaxis=dict(categoryorder="total ascending", automargin=True),
        legend_title="", bargap=0.15,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    fig_bar.update_traces(hovertemplate="%{y}<br>%{legendgroup}: %{x:,.0f}")
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("No rows for bar chart.")

# -------------------------------------------------
# Chart 2 & 3 side-by-side (Treemap + Bubble)
# -------------------------------------------------
colA, colB = st.columns(2)

with colA:
    st.subheader("Purchase Share — Top Products (Treemap)")
    if not top_by_qty.empty:
        fig_tree = px.treemap(
            top_by_qty,
            path=[px.Constant("Top-N"), "Label"],
            values="Qty Purchased",
            hover_data={"Qty Purchased":":,", "Bonus Received":":,"}
        )
        fig_tree.update_traces(root_color="lightgrey")
        fig_tree.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info("No rows for treemap.")

with colB:
    st.subheader("Bonus % vs Purchased (Bubble)")
    bub = agg.copy()
    bub["Label"] = bub.apply(lambda r: short_label(r["Code"], r["Product"], 30), axis=1)
    if not bub.empty:
        fig_bub = px.scatter(
            bub, x="Qty Purchased", y="Bonus %",
            size="Qty Purchased", color="Bonus %",
            hover_name="Label", height=420
        )
        fig_bub.add_hline(y=10, line_dash="dot", annotation_text="10% ref")
        fig_bub.update_layout(margin=dict(l=10, r=10, t=30, b=10),
                              xaxis_title="Qty Purchased", yaxis_title="Bonus %")
        st.plotly_chart(fig_bub, use_container_width=True)
    else:
        st.info("No rows for bubble chart.")

st.divider()

# -------------------------------------------------
# Highlights tables
# -------------------------------------------------
st.subheader("Highlights")
left, right = st.columns(2)

with left:
    st.markdown("**Top by Volume (Qty Purchased)**")
    t1 = agg.sort_values("Qty Purchased", ascending=False).head(10)[
        ["Code", "Product", "Qty Purchased", "Bonus Received", "Avg Purchase Qty", "Bonus %"]
    ]
    st.dataframe(t1, use_container_width=True, hide_index=True)

with right:
    st.markdown("**Top by Bonus % (min 100 units)**")
    t2 = agg[agg["Qty Purchased"] >= 100].sort_values("Bonus %", ascending=False).head(10)[
        ["Code", "Product", "Qty Purchased", "Bonus Received", "Avg Bonus Qty", "Bonus %"]
    ]
    st.dataframe(t2, use_container_width=True, hide_index=True)

st.divider()

# -------------------------------------------------
# Detailed table (aggregated products)
# -------------------------------------------------
st.subheader(f"Detailed Products ({len(agg):,})")
cols = ["Code","Product","Qty Purchased","Bonus Received","Times Purchased","Times Bonus","Avg Purchase Qty","Avg Bonus Qty","Bonus %"]
present = [c for c in cols if c in agg.columns]
st.dataframe(agg[present].sort_values("Qty Purchased", ascending=False), use_container_width=True, hide_index=True)

# Optional raw transactions in an expander
with st.expander("🔎 View raw transactions (filtered)"):
    show_cols = ["Supplier Name","Code","Product","Date","Qty Purchased","Bonus","Bonus %"]
    show_cols = [c for c in show_cols if c in tx_f.columns]
    st.dataframe(tx_f[show_cols].sort_values(["Product","Date"]), use_container_width=True, hide_index=True)

# Download buttons
csv_agg = agg[present].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download product summary (CSV)", csv_agg, "product_summary.csv", "text/csv")
