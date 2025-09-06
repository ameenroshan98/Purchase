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

# Earliest available data (requested banner + min date)
EARLIEST_AVAILABLE = pd.Timestamp(2024, 8, 1)  # 01.08.2024

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def short_label(code, name, n=40):
    name = str(name)
    return f"{code} — {name[:n]}…" if len(name) > n else f"{code} — {name}"

def avg_gap_days(s: pd.Series) -> float:
    """Average days between purchases using sorted UNIQUE dates."""
    dates = (
        pd.to_datetime(s, errors="coerce", dayfirst=True)
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )
    if len(dates) <= 1:
        return float("nan")
    diffs = dates.diff().dt.days.dropna()
    return float(diffs.mean())

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

    # Normalize headers
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

    # Types
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

    for c in ["Qty Purchased", "Bonus"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

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
st.caption("ℹ️ Data available from **01.08.2024** onward.")

if tx.empty:
    st.warning("No data found. Check your sheet sharing, tab gid, or API key.")
    st.stop()

# -------------------------------------------------
# Sidebar Filters (now includes DATE RANGE)
# -------------------------------------------------
with st.sidebar:
    st.header("Filters")

    # Date range setup from data but not earlier than EARLIEST_AVAILABLE
    data_min = tx["Date"].min()
    data_max = tx["Date"].max()
    min_allowed = max(EARLIEST_AVAILABLE, data_min) if pd.notna(data_min) else EARLIEST_AVAILABLE
    max_allowed = data_max if pd.notna(data_max) else pd.Timestamp.today()

    default_range = (min_allowed.date(), max_allowed.date())
    date_range = st.date_input(
        "Date range",
        value=default_range,
        min_value=min_allowed.date(),
        max_value=max_allowed.date(),
    )
    # Handle both single date and tuple
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = [pd.Timestamp(d) for d in date_range]
    else:
        start_date = pd.Timestamp(date_range)
        end_date = start_date

    q = st.text_input("Search (code or product)", "")
    bonus_filter = st.selectbox("Bonus filter", ["All", "With Bonus", "Without Bonus"])
    min_qty = st.slider("Min Qty Purchased (product total)", 0, int(tx["Qty Purchased"].sum()), 0, step=100)
    top_n = st.slider("Top-N for charts", 5, 40, 15)

st.caption(f"Showing data from **{start_date:%d.%m.%Y}** to **{end_date:%d.%m.%Y}**.")

# Transaction-level mask (date + search)
mask_tx = pd.Series(True, index=tx.index)
mask_tx &= tx["Date"].between(start_date, end_date, inclusive="both")
if q.strip():
    ql = q.strip().lower()
    mask_tx &= tx["Product"].str.lower().str.contains(ql, na=False) | tx["Code"].astype(str).str.contains(ql, na=False)

tx_f = tx[mask_tx].copy()

if tx_f.empty:
    st.info("No transactions match your filters/date range.")
    st.stop()

# -------------------------------------------------
# Aggregate to PRODUCT level (with Times Purchased & Avg Days Between)
# -------------------------------------------------
agg_base = (
    tx_f.groupby(["Code", "Product"], as_index=False)
        .agg(
            Qty_Purchased=("Qty Purchased", "sum"),
            Bonus_Received=("Bonus", "sum"),
            Times_Purchased=("Date", "count"),
            Times_Bonus=("Bonus", lambda s: (s > 0).sum()),
            Avg_Purchase_Qty=("Qty Purchased", "mean"),
            Avg_Bonus_Qty=("Bonus", "mean"),
        )
        .rename(columns={
            "Qty_Purchased": "Qty Purchased",
            "Bonus_Received": "Bonus Received",
            "Times_Purchased": "Times Purchased",
            "Times_Bonus": "Times Bonus",
            "Avg_Purchase_Qty": "Avg Purchase Qty",
            "Avg_Bonus_Qty": "Avg Bonus Qty",
        })
)

gap_series = (
    tx_f.groupby(["Code", "Product"])["Date"]
        .apply(avg_gap_days)
        .reset_index(name="Avg Days Between")
)

agg = agg_base.merge(gap_series, on=["Code", "Product"], how="left")
agg["Bonus %"] = (agg["Bonus Received"] / agg["Qty Purchased"] * 100).replace([pd.NA, pd.NaT, float("inf")], 0).fillna(0)
agg["Avg Days Between"] = agg["Avg Days Between"].round(1)

# Product-level filters
if bonus_filter == "With Bonus":
    agg = agg[agg["Bonus %"] > 0]
elif bonus_filter == "Without Bonus":
    agg = agg[agg["Bonus %"] == 0]
agg = agg[agg["Qty Purchased"] >= min_qty]

if agg.empty:
    st.info("No products after applying filters. Try adjusting date range or Min Qty.")
    st.stop()

# -------------------------------------------------
# KPIs
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
# Charts
# -------------------------------------------------
top_by_qty = agg.sort_values("Qty Purchased", ascending=False).head(top_n).copy()
top_by_qty["Label"] = top_by_qty.apply(lambda r: short_label(r["Code"], r["Product"], 42), axis=1)

st.subheader("Purchased vs Bonus — Top Products")
if not top_by_qty.empty:
    m = top_by_qty.melt(
        id_vars=["Label"],
        value_vars=["Qty Purchased", "Bonus Received"],
        var_name="Metric", value_name="Value"
    )
    fig_bar = px.bar(
        m, y="Label", x="Value", color="Metric",
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
# Highlights
# -------------------------------------------------
st.subheader("Highlights")
left, right = st.columns(2)

with left:
    st.markdown("**Top by Volume (Qty Purchased)**")
    t1 = agg.sort_values("Qty Purchased", ascending=False).head(10)[
        ["Code", "Product", "Qty Purchased", "Bonus Received", "Times Purchased", "Avg Days Between", "Bonus %"]
    ]
    st.dataframe(t1, use_container_width=True, hide_index=True)

with right:
    st.markdown("**Top by Bonus % (min 100 units)**")
    t2 = agg[agg["Qty Purchased"] >= 100].sort_values("Bonus %", ascending=False).head(10)[
        ["Code", "Product", "Qty Purchased", "Bonus Received", "Times Purchased", "Avg Days Between", "Bonus %"]
    ]
    st.dataframe(t2, use_container_width=True, hide_index=True)

st.divider()

# -------------------------------------------------
# Detailed Products table
# -------------------------------------------------
st.subheader(f"Detailed Products ({len(agg):,})")
cols = [
    "Code","Product","Qty Purchased","Bonus Received",
    "Times Purchased","Times Bonus","Avg Purchase Qty","Avg Bonus Qty",
    "Avg Days Between","Bonus %"
]
present = [c for c in cols if c in agg.columns]
st.dataframe(
    agg[present].sort_values("Qty Purchased", ascending=False),
    use_container_width=True, hide_index=True
)

# Raw transactions (filtered)
with st.expander("🔎 View raw transactions (filtered)"):
    show_cols = ["Supplier Name","Code","Product","Date","Qty Purchased","Bonus","Bonus %"]
    show_cols = [c for c in show_cols if c in tx_f.columns]
    st.dataframe(tx_f[show_cols].sort_values(["Product","Date"]), use_container_width=True, hide_index=True)

# Download summary
csv_agg = agg[present].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download product summary (CSV)", csv_agg, "product_summary.csv", "text/csv")
