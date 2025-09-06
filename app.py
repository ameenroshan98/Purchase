import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Purchase Dashboard", layout="wide")

# ---- Google Sheet config
SHEET_ID = "1R7o4xKMeYWcYWwAOorMyYDtjg0-74FqDK0xAFKN6Iuo"
GID      = 0  # change if your data tab is not the first
RANGE    = "Sheet1!A1:G100000"   # A..G = Supplier..Bonus %
CSV_URL  = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

# Earliest available data + display format
EARLIEST_AVAILABLE = pd.Timestamp(2024, 8, 1)   # 01/08/2024
DATE_FMT_DISPLAY   = "%d/%m/%Y"                 # dd/mm/yyyy

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def short_label(code, name, n=40):
    name = str(name)
    return f"{code} — {name[:n]}…" if len(name) > n else f"{code} — {name}"

def avg_gap_days(s: pd.Series) -> float:
    """Average days between purchases for a product (unique dates only)."""
    dates = (
        pd.to_datetime(s, errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )
    if len(dates) <= 1:
        return float("nan")
    diffs = dates.diff().dt.days.dropna()
    return float(diffs.mean())

def parse_dates_ddmmyyyy(col: pd.Series) -> pd.Series:
    """
    Robust dd/mm/yyyy parser:
    - trims, replaces non-breaking spaces
    - tries dayfirst=True
    - retries with '-' swapped to '/' if needed
    - normalizes to midnight
    """
    s = col.astype(str).str.strip().str.replace("\u00A0", " ", regex=False)
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    mask = dt.isna() & s.notna()
    if mask.any():
        s2 = s.str.replace("-", "/", regex=False)
        dt2 = pd.to_datetime(s2, errors="coerce", dayfirst=True)
        dt.loc[mask] = dt2.loc[mask]
    return dt.dt.normalize()

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
        df = pd.read_csv(CSV_URL, dtype=str).fillna("")

    # Clean headers
    df.columns = (
        pd.Index(df.columns)
        .str.replace("\u00A0", " ", regex=False)
        .str.strip()
    )

    # Standardize expected headers
    rename_map = {
        "Supplier Name": "Supplier Name",
        "Code": "Code",
        "Product": "Product",
        "Date": "Date",
        "Qty Purchased": "Qty Purchased",
        "Bonus": "Bonus",
        "Bonus %": "Bonus %"
    }
    # Tolerate variants
    for c in list(df.columns):
        lc = c.lower().strip()
        if lc in ["qty purchased", "quantity purchased"]:
            rename_map[c] = "Qty Purchased"
    df = df.rename(columns=rename_map)

    # Parse Date (strict dd/mm/yyyy behavior)
    if "Date" in df.columns:
        df["Date"] = parse_dates_ddmmyyyy(df["Date"])

    # Numeric columns
    for c in ["Qty Purchased", "Bonus"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(0)

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
st.caption("ℹ️ Data available from **01/08/2024** onward.")

if tx.empty or "Date" not in tx.columns:
    st.warning("No data found or 'Date' column missing. Check sharing, tab gid, or column names.")
    st.stop()

# -------------------------------------------------
# Sidebar Filters (DATE + SUPPLIER dropdown + search)
# -------------------------------------------------
with st.sidebar:
    st.header("Filters")

    # Date bounds (respect earliest available)
    data_min = tx["Date"].min()
    data_max = tx["Date"].max()
    min_allowed = max(EARLIEST_AVAILABLE, data_min) if pd.notna(data_min) else EARLIEST_AVAILABLE
    max_allowed = data_max if pd.notna(data_max) else pd.Timestamp.today().normalize()

    default_range = (min_allowed.date(), max_allowed.date())
    date_range = st.date_input(
        "Date range (dd/mm/yyyy)",
        value=default_range,
        min_value=min_allowed.date(),
        max_value=max_allowed.date(),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = [pd.Timestamp(d) for d in date_range]
    else:
        start_date = pd.Timestamp(date_range)
        end_date = start_date

    # Supplier DROPDOWN (single select)
    sup_series = tx.get("Supplier Name", pd.Series("", index=tx.index)).astype(str).str.strip()
    supplier_options = sorted([s for s in sup_series.unique() if s])
    supplier_dropdown_options = ["All suppliers"] + supplier_options
    selected_supplier = st.selectbox(
        "Supplier",
        options=supplier_dropdown_options,
        index=0,
        help="Choose a single supplier or 'All suppliers'."
    )

    q = st.text_input("Search (code or product)", "")
    bonus_filter = st.selectbox("Bonus filter", ["All", "With Bonus", "Without Bonus"])
    total_qty = pd.to_numeric(tx.get("Qty Purchased", pd.Series([], dtype=float)), errors="coerce").fillna(0).sum()
    min_qty = st.slider("Min Qty Purchased (product total)", 0, int(max(total_qty, 0)), 0, step=100)
    top_n = st.slider("Top-N for charts", 5, 40, 15)

st.caption(
    f"Showing data from **{start_date.strftime(DATE_FMT_DISPLAY)}** "
    f"to **{end_date.strftime(DATE_FMT_DISPLAY)}**."
)

# Transaction-level mask (DATE + SUPPLIER + SEARCH)
start_norm = start_date.normalize()
end_norm   = end_date.normalize()
mask_tx = (tx["Date"] >= start_norm) & (tx["Date"] <= end_norm)

if selected_supplier != "All suppliers":
    mask_tx &= (sup_series == selected_supplier)

if q.strip():
    ql = q.strip().lower()
    mask_tx &= (
        tx.get("Product", pd.Series("", index=tx.index)).astype(str).str.lower().str.contains(ql, na=False)
        | tx.get("Code", pd.Series("", index=tx.index)).astype(str).str.contains(ql, na=False)
    )

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
            Last_Purchase=("Date", "max"),
        )
        .rename(columns={
            "Qty_Purchased": "Qty Purchased",
            "Bonus_Received": "Bonus Received",
            "Times_Purchased": "Times Purchased",
            "Times_Bonus": "Times Bonus",
            "Avg_Purchase_Qty": "Avg Purchase Qty",
            "Avg_Bonus_Qty": "Avg Bonus Qty",
            "Last_Purchase": "Last Purchase Date",
        })
)

gap_series = (
    tx_f.groupby(["Code", "Product"])["Date"]
        .apply(avg_gap_days)
        .reset_index(name="Avg Days Between")
)

# Per-transaction bonus % variability (std dev)
tx_tmp = tx_f.copy()
if "Bonus %" in tx_tmp.columns:
    tx_tmp["bonus_pct_tx"] = pd.to_numeric(tx_tmp["Bonus %"], errors="coerce")
else:
    tx_tmp["bonus_pct_tx"] = (
        (pd.to_numeric(tx_tmp["Bonus"], errors="coerce") /
         pd.to_numeric(tx_tmp["Qty Purchased"], errors="coerce"))
        .replace([float("inf"), -float("inf")], 0) * 100
    )

var_df = (
    tx_tmp.groupby(["Code","Product"])["bonus_pct_tx"]
          .std(ddof=0).fillna(0).reset_index(name="Bonus Variability (pp)")
)

# Merge & compute effective Bonus %
agg = agg_base.merge(gap_series, on=["Code","Product"], how="left").merge(var_df, on=["Code","Product"], how="left")
agg["Bonus %"] = (agg["Bonus Received"] / agg["Qty Purchased"] * 100).replace([pd.NA, pd.NaT, float("inf")], 0).fillna(0).round(1)
agg["Avg Days Between"] = agg["Avg Days Between"].round(1)
agg["Bonus Presence Rate"] = (agg["Times Bonus"] / agg["Times Purchased"]).fillna(0)
agg["Recency Days"] = (end_norm - agg["Last Purchase Date"]).dt.days

# Product-level filters
if bonus_filter == "With Bonus":
    agg = agg[agg["Bonus %"] > 0]
elif bonus_filter == "Without Bonus":
    agg = agg[agg["Bonus %"] == 0]
agg = agg[agg["Qty Purchased"] >= min_qty]

if agg.empty:
    st.info("No products after applying filters. Try adjusting supplier, date range, or Min Qty.")
    st.stop()

# -------------------------------------------------
# 3-status human classification (Core / Promo-timed / Review)
# -------------------------------------------------
VOL_MED = 300          # units
VOL_HIGH = 1000        # units
BONUS_PROMO = 8.0      # %: considered promo-driven if >= this
BPR_PROMO = 0.50       # share of orders with bonus
VAR_HIGH = 5.0         # pp std dev of bonus%
STALE_DAYS_MIN = 90    # days
FACTOR_GAP = 1.5       # recency > 1.5x typical gap -> stale

def classify_simple(r):
    vol = r["Qty Purchased"]
    ebp = r["Bonus %"]
    bpr = r["Bonus Presence Rate"]
    var = r.get("Bonus Variability (pp)", 0) or 0
    rec = r.get("Recency Days", None)
    gap = r.get("Avg Days Between", None)
    # Review: stale or anomalous
    stale_gate = max(STALE_DAYS_MIN, (gap * FACTOR_GAP) if pd.notna(gap) else STALE_DAYS_MIN)
    if (pd.notna(rec) and rec > stale_gate) or ebp > 40 or var > 15:
        return "Review"
    # Promo-timed: meaningfully bonus-driven
    if (ebp >= BONUS_PROMO) or (bpr >= BPR_PROMO):
        return "Promo-timed"
    # Otherwise Core
    return "Core"

agg["StatusKey"] = agg.apply(classify_simple, axis=1)

STATUS_COPY = {
    "Core": ("🟢 Core", "Stable demand; buy steadily and negotiate base price."),
    "Promo-timed": ("🟠 Promo-timed", "Buy during bonus windows; avoid outside promos."),
    "Review": ("🔴 Review", "Dormant/anomalous; verify demand, data, or supplier terms."),
}
agg[["Status", "Status Note"]] = agg["StatusKey"].apply(lambda k: pd.Series(STATUS_COPY[k]))

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

# (Optional) Status filter after classification
with st.sidebar:
    status_choice = st.selectbox("Status", ["(All)", "🟢 Core", "🟠 Promo-timed", "🔴 Review"], index=0)
if status_choice != "(All)":
    agg = agg[agg["Status"] == status_choice]

# -------------------------------------------------
# Charts
# -------------------------------------------------
top_n = min(top_n, max(1, len(agg)))
top_by_qty = agg.sort_values("Qty Purchased", ascending=False).head(top_n).copy()
top_by_qty["Label"] = top_by_qty.apply(lambda r: short_label(r["Code"], r["Product"], 42), axis=1)

# ---------- Purchased vs Bonus — Top Products (labels on Bonus only)
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
        margin=dict(l=10, r=60, t=30, b=10)  # more right padding for labels
    )
    fig_bar.update_traces(hovertemplate="%{y}<br>%{legendgroup}: %{x:,.0f}",
                          cliponaxis=False)

    # ✅ Show value labels ONLY on "Bonus Received" bars
    for tr in fig_bar.data:
        if tr.name == "Bonus Received":
            tr.text = [f"{v:,.0f}" for v in tr.x]  # or :,.1f for one decimal
            tr.textposition = "outside"
            tr.texttemplate = "%{text}"
            tr.textfont = dict(size=12)
        else:
            tr.text = None

    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("No rows for bar chart.")

# Treemap + Bubble
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
            size="Qty Purchased", color="Status",
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
# Highlights (with conditional formatting via column_config)
# -------------------------------------------------
st.subheader("Highlights")
left, right = st.columns(2)

with left:
    st.markdown("**Top by Volume (Qty Purchased)**")
    t1 = agg.sort_values("Qty Purchased", ascending=False).head(10)[
        ["Status", "Code", "Product", "Qty Purchased", "Bonus Received",
         "Times Purchased", "Avg Days Between", "Bonus %", "Status Note"]
    ]
    st.dataframe(
        t1,
        use_container_width=True, hide_index=True,
        column_config={
            "Qty Purchased": st.column_config.NumberColumn(format="%,d"),
            "Bonus Received": st.column_config.NumberColumn(format="%,d"),
            "Avg Days Between": st.column_config.NumberColumn(format="%.1f"),
            "Bonus %": st.column_config.ProgressColumn(
                "Bonus %",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )

with right:
    st.markdown("**Top by Bonus % (min 100 units)**")
    t2 = agg[agg["Qty Purchased"] >= 100].sort_values("Bonus %", ascending=False).head(10)[
        ["Status", "Code", "Product", "Qty Purchased", "Bonus Received",
         "Times Purchased", "Avg Days Between", "Bonus %", "Status Note"]
    ]
    st.dataframe(
        t2,
        use_container_width=True, hide_index=True,
        column_config={
            "Qty Purchased": st.column_config.NumberColumn(format="%,d"),
            "Bonus Received": st.column_config.NumberColumn(format="%,d"),
            "Avg Days Between": st.column_config.NumberColumn(format="%.1f"),
            "Bonus %": st.column_config.ProgressColumn(
                "Bonus %",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )

st.divider()

# -------------------------------------------------
# Detailed Products table (with 1-decimal Bonus % + formatting)
# -------------------------------------------------
st.subheader(f"Detailed Products ({len(agg):,})")
cols = [
    "Status","Code","Product","Qty Purchased","Bonus Received",
    "Times Purchased","Times Bonus","Avg Purchase Qty","Avg Bonus Qty",
    "Avg Days Between","Bonus %","Bonus Presence Rate","Bonus Variability (pp)","Status Note"
]
present = [c for c in cols if c in agg.columns]
st.dataframe(
    agg[present].sort_values("Qty Purchased", ascending=False),
    use_container_width=True, hide_index=True,
    column_config={
        "Qty Purchased": st.column_config.NumberColumn(format="%,d"),
        "Bonus Received": st.column_config.NumberColumn(format="%,d"),
        "Times Purchased": st.column_config.NumberColumn(format="%,d"),
        "Times Bonus": st.column_config.NumberColumn(format="%,d"),
        "Avg Purchase Qty": st.column_config.NumberColumn(format="%.1f"),
        "Avg Bonus Qty": st.column_config.NumberColumn(format="%.1f"),
        "Avg Days Between": st.column_config.NumberColumn(format="%.1f"),
        "Bonus Presence Rate": st.column_config.NumberColumn(format="%.2f"),
        "Bonus Variability (pp)": st.column_config.NumberColumn(format="%.1f"),
        "Bonus %": st.column_config.ProgressColumn(
            "Bonus %",
            format="%.1f%%",
            min_value=0,
            max_value=100,
        ),
    },
)

st.divider()

# -------------------------------------------------
# 🔎 SKU Drill-down: pick a product and see purchase history
# -------------------------------------------------
st.subheader("🔎 SKU Drill-down")

# SKU selector from the aggregated list
agg_sorted = agg.sort_values(["Qty Purchased", "Product"], ascending=[False, True]).copy()
agg_sorted["SKU"] = agg_sorted["Code"].astype(str) + " — " + agg_sorted["Product"].astype(str)
sku_list = agg_sorted["SKU"].tolist()

selected_sku = st.selectbox("Select a product to view its purchase history:", ["(Choose a product)"] + sku_list, index=0)
if selected_sku != "(Choose a product)":
    row = agg_sorted.loc[agg_sorted["SKU"] == selected_sku].iloc[0]
    sel_code, sel_product = row["Code"], row["Product"]

    # Filter transactions for this SKU within current filters
    hist = tx_f[(tx_f["Code"].astype(str) == str(sel_code)) & (tx_f["Product"] == sel_product)].copy()

    # Compute per-transaction bonus %
    if "Bonus %" in hist.columns:
        hist["Bonus % (tx)"] = pd.to_numeric(hist["Bonus %"], errors="coerce").fillna(0).round(1)
    else:
        hist["Bonus % (tx)"] = ((pd.to_numeric(hist["Bonus"], errors="coerce") /
                                 pd.to_numeric(hist["Qty Purchased"], errors="coerce"))
                                .replace([float("inf"), -float("inf")], 0) * 100).fillna(0).round(1)

    # Format date for display
    if "Date" in hist.columns:
        hist["_sort_date"] = hist["Date"]
        hist["Date"] = hist["Date"].dt.strftime(DATE_FMT_DISPLAY)

    # Tiny trend
    st.markdown(f"**{selected_sku}** — history in selected period")
    if len(hist) > 0:
        trend = hist.sort_values("_sort_date" if "_sort_date" in hist else "Date")
        fig_hist = px.line(
            trend,
            x="_sort_date" if "_sort_date" in trend else "Date",
            y="Qty Purchased",
            markers=True,
            title=None,
        )
        fig_hist.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Date",
            yaxis_title="Qty Purchased",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # History table
    cols_hist = ["Date", "Supplier Name", "Qty Purchased", "Bonus", "Bonus % (tx)"]
    cols_hist = [c for c in cols_hist if c in hist.columns]
    st.dataframe(
        hist.sort_values("_sort_date" if "_sort_date" in hist else "Date")[cols_hist],
        use_container_width=True, hide_index=True,
        column_config={
            "Qty Purchased": st.column_config.NumberColumn(format="%,d"),
            "Bonus": st.column_config.NumberColumn(format="%,d"),
            "Bonus % (tx)": st.column_config.ProgressColumn(
                "Bonus % (tx)",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )

# -------------------------------------------------
# Raw transactions (filtered) with DD/MM/YYYY display
# -------------------------------------------------
with st.expander("🧾 View all filtered transactions"):
    show_cols = ["Supplier Name","Code","Product","Date","Qty Purchased","Bonus","Bonus %"]
    show_cols = [c for c in show_cols if c in tx_f.columns]
    tx_show = tx_f[show_cols].copy()
    if "Date" in tx_show.columns:
        tx_show["_sort_date"] = tx_show["Date"]  # for correct sort
        tx_show["Date"] = tx_show["Date"].dt.strftime(DATE_FMT_DISPLAY)
        tx_show = tx_show.sort_values(["Product", "_sort_date"]).drop(columns=["_sort_date"])
    st.dataframe(
        tx_show,
        use_container_width=True, hide_index=True,
        column_config={
            "Qty Purchased": st.column_config.NumberColumn(format="%,d"),
            "Bonus": st.column_config.NumberColumn(format="%,d"),
            "Bonus %": st.column_config.ProgressColumn(
                "Bonus %",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )

# Download summary
csv_cols = [
    "Status","Code","Product","Qty Purchased","Bonus Received",
    "Times Purchased","Times Bonus","Avg Purchase Qty","Avg Bonus Qty",
    "Avg Days Between","Bonus %","Bonus Presence Rate","Bonus Variability (pp)","Status Note"
]
csv_agg = agg[[c for c in csv_cols if c in agg.columns]].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download product summary (CSV)", csv_agg, "product_summary.csv", "text/csv")
