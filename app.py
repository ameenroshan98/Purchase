# app.py
import streamlit as st
st.set_page_config(page_title="Purchase Dashboard", layout="wide")  # must be first

# =============================
# App proper (no authentication)
# =============================
import pandas as pd
import requests
from io import BytesIO
import plotly.express as px
import re
import time

# Quick link to the comparison page
try:
    st.page_link("pages/Compare_Yearly_Patterns.py", label="📅 Compare Ranges (Purchase vs Bonus)")
except Exception:
    pass

# ---- Google Sheet config (entire workbook)
SHEET_ID = "1R7o4xKMeYWcYWwAOorMyYDtjg0-74FqDK0xAFKN6Iuo"
GID      = 0  # used only by CSV fallback
XLSX_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=xlsx"
CSV_URL  = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

DATE_FMT_DISPLAY = "%m/%d/%Y"  # mm/dd/yyyy

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def short_label(code, name, n=42):
    name = str(name)
    s = f"{code} — {name}"
    return s if len(s) <= n + len(str(code)) + 3 else f"{code} — {name[:n]}…"

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

# -------------------------------------------------
# Fast loader (CSV one-tab) or full workbook, with pruning and compact dtypes
# -------------------------------------------------
@st.cache_data(ttl=1800, max_entries=4)
def load_transactions(prefer="mdy", fast_mode=True):
    # Columns used downstream
    usecols = ["Supplier Name", "Code", "Product", "Date", "Qty Purchased", "Bonus", "Bonus %"]

    def _standardize_headers(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = (
            pd.Index(df.columns)
            .astype(str)
            .str.replace("\u00A0", " ", regex=False)
            .str.strip()
        )
        # Tolerate variants
        rename_map = {
            "Supplier Name": "Supplier Name",
            "Code": "Code",
            "Product": "Product",
            "Date": "Date",
            "Qty Purchased": "Qty Purchased",
            "quantity purchased": "Qty Purchased",
            "qty purchased": "Qty Purchased",
            "qty": "Qty Purchased",
            "Bonus": "Bonus",
            "Bonus %": "Bonus %",
            "Bonus%": "Bonus %",
        }
        df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
        return df

    def _fast_dates(series: pd.Series, prefer: str) -> pd.Series:
        # Vectorized date parsing with serial support
        s = series.astype(str).str.replace("\u00A0", " ", regex=False).str.strip()
        s = s.str.replace(".", "/", regex=False).str.replace("-", "/", regex=False)
        # Excel serials
        serial = pd.to_numeric(s, errors="coerce")
        out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        serial_mask = serial.notna()
        if serial_mask.any():
            out.loc[serial_mask] = pd.to_datetime(
                serial.loc[serial_mask], unit="D", origin="1899-12-30", errors="coerce"
            )
        # Vector parse the rest (prefer mdy/dmy on ambiguous)
        rest = ~serial_mask
        if rest.any():
            dayfirst = (prefer.lower() == "dmy")
            out.loc[rest] = pd.to_datetime(
                s.loc[rest], errors="coerce", dayfirst=dayfirst, infer_datetime_format=True
            )
        return out.dt.normalize()

    def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            empty = pd.DataFrame(columns=usecols)
            for c in usecols:
                empty[c] = pd.NA
            return empty

        df = _standardize_headers(df)
        # Ensure needed cols exist
        for c in usecols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[usecols]

        # Dates (fast path)
        df["Date"] = _fast_dates(df["Date"], prefer)

        # Numerics
        for c in ["Qty Purchased", "Bonus", "Bonus %"]:
            df[c] = df[c].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
        df["Qty Purchased"] = pd.to_numeric(df["Qty Purchased"], errors="coerce").fillna(0).astype("int32")
        df["Bonus"]         = pd.to_numeric(df["Bonus"], errors="coerce").fillna(0).astype("int32")
        df["Bonus %"]       = pd.to_numeric(df["Bonus %"], errors="coerce").fillna(0.0).astype("float32")

        # Categories for faster groupby/filter
        for c in ["Supplier Name", "Code", "Product"]:
            df[c] = df[c].astype("category")

        return df

    if fast_mode:
        # FAST PATH: CSV single tab (small payload)
        try:
            df = pd.read_csv(CSV_URL, dtype=str)
            df["__sheet__"] = f"gid_{GID}"
            return _postprocess(df)
        except Exception:
            pass  # fall through to XLSX if CSV fails

    # FULL PATH: Entire workbook
    try:
        for attempt in range(3):
            r = requests.get(XLSX_URL, timeout=60)
            if r.ok:
                with BytesIO(r.content) as bio:
                    book = pd.read_excel(bio, sheet_name=None, dtype=str, engine="openpyxl")
                break
            time.sleep(1.2 * (attempt + 1))
        else:
            raise RuntimeError("XLSX fetch failed")

        frames = []
        for sheet_name, df in (book or {}).items():
            if df is None or df.empty:
                continue
            df = df.dropna(how="all")
            df["__sheet__"] = sheet_name
            frames.append(df)
        big = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return _postprocess(big)
    except Exception:
        # Final fallback to CSV if XLSX fails
        df = pd.read_csv(CSV_URL, dtype=str)
        df["__sheet__"] = f"gid_{GID}"
        return _postprocess(df)

# -------------------------------------------------
# UI & Analytics
# -------------------------------------------------
st.title("📊 Purchase Dashboard")

# Controls: data locale + refresh + performance
with st.sidebar:
    st.markdown("### Data refresh & locale")
    c1, c2 = st.columns([1, 1])
    prefer_order = c1.radio(
        "Ambiguous dates",
        ["mdy", "dmy"],
        index=0,
        horizontal=True,
        help="Used only when both parts ≤ 12 (e.g., 03/04/25)."
    )
    if c2.button("🔄 Force refresh", use_container_width=True):
        load_transactions.clear()
        st.experimental_rerun()

with st.sidebar:
    st.markdown("### Performance")
    FAST_MODE = st.toggle(
        "⚡ Fast mode (CSV, one tab)",
        value=True,
        help="Loads the CSV of the primary tab only. Disable to load the entire workbook (slower)."
    )

tx = load_transactions(prefer=prefer_order, fast_mode=FAST_MODE)

if tx.empty or "Code" not in tx.columns or "Product" not in tx.columns:
    st.warning("No data found or required columns missing (Code/Product).")
    st.stop()

# Diagnostics + global date bounds
min_date = tx["Date"].min(skipna=True) if "Date" in tx.columns else None
max_date = tx["Date"].max(skipna=True) if "Date" in tx.columns else None
cap = []
if pd.notna(min_date): cap.append(f"from **{min_date.strftime(DATE_FMT_DISPLAY)}**")
if pd.notna(max_date): cap.append(f"to **{max_date.strftime(DATE_FMT_DISPLAY)}**")
if cap:
    st.caption("ℹ️ Data range " + " ".join(cap))

with st.expander("📊 Load diagnostics"):
    st.write({
        "Raw rows loaded": len(tx),
        "Rows with parseable Date": int(tx["Date"].notna().sum()) if "Date" in tx else "n/a",
        "Rows with missing/unparseable Date": int(tx["Date"].isna().sum()) if "Date" in tx else "n/a",
        "Distinct suppliers": int(tx["Supplier Name"].nunique() if "Supplier Name" in tx else 0),
        "Sheets combined": int(tx["__sheet__"].nunique() if "__sheet__" in tx else 1),
        "Fast mode": FAST_MODE,
    })

if "Date" in tx.columns and tx["Date"].isna().sum() > 0:
    with st.expander("⚠️ Examples of unparseable dates"):
        bad = tx.loc[tx["Date"].isna(), "Date"].astype(str).value_counts().head(10)
        st.write(bad)

# Sidebar filters (with DATE RANGE)
with st.sidebar:
    st.header("Filters")
    sup_series = tx.get("Supplier Name", pd.Series("", index=tx.index)).astype(str).str.strip()
    supplier_options = sorted([s for s in sup_series.unique() if s])
    selected_supplier = st.selectbox(
        "Supplier",
        options=["All suppliers"] + supplier_options,
        index=0
    )

    q = st.text_input("Search (code or product)", "")

    # Date range filter (if we have valid dates)
    include_missing_dates = False
    if pd.notna(min_date) and pd.notna(max_date):
        start_d, end_d = st.date_input(
            "Date range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
            format="MM/DD/YYYY",
        )
        include_missing_dates = st.checkbox("Include rows with missing Date", value=False)
    else:
        start_d = end_d = None

    bonus_filter = st.selectbox("Bonus filter", ["All", "With Bonus", "Without Bonus"])

    # Min-qty slider max based on supplier/search slice (ignores date for preview simplicity)
    mask_preview = pd.Series(True, index=tx.index)
    if selected_supplier != "All suppliers":
        mask_preview &= (sup_series == selected_supplier)
    if q.strip():
        ql = q.strip().lower()
        mask_preview &= (
            tx.get("Product", pd.Series("", index=tx.index)).astype(str).str.lower().str.contains(ql, na=False)
            | tx.get("Code", pd.Series("", index=tx.index)).astype(str).str.contains(ql, na=False)
        )
    preview_max = 0
    if mask_preview.any() and "Qty Purchased" in tx.columns:
        prev_agg = (
            tx.loc[mask_preview]
              .groupby(["Code", "Product"], as_index=False)["Qty Purchased"]
              .sum()
        )
        if not prev_agg.empty:
            preview_max = int(prev_agg["Qty Purchased"].max())
    min_qty = st.slider("Min Qty Purchased (product total)", 0, max(preview_max, 0), 0, step=100)

    scope = st.radio("Chart scope", ["Top-N", "All"], horizontal=True)
    top_n = st.slider("Top-N for charts", 5, 100, 15) if scope == "Top-N" else None

# Build transaction-level mask (supplier + search + DATE)
mask_tx = pd.Series(True, index=tx.index)
if selected_supplier != "All suppliers":
    mask_tx &= (sup_series == selected_supplier)
if q.strip():
    ql = q.strip().lower()
    mask_tx &= (
        tx.get("Product", pd.Series("", index=tx.index)).astype(str).str.lower().str.contains(ql, na=False)
        | tx.get("Code", pd.Series("", index=tx.index)).astype(str).str.contains(ql, na=False)
    )

# Apply date range (if present)
if "Date" in tx.columns and pd.notna(min_date) and pd.notna(max_date) and start_d and end_d:
    s_ts, e_ts = pd.Timestamp(start_d).normalize(), pd.Timestamp(end_d).normalize()
    date_mask = tx["Date"].between(s_ts, e_ts, inclusive="both")
    if include_missing_dates:
        date_mask = date_mask | tx["Date"].isna()
    mask_tx &= date_mask

tx_f = tx[mask_tx].copy()
if tx_f.empty:
    st.info("No transactions match your filters (including date range).")
    st.stop()

# Aggregate to PRODUCT level (filtered dataset)
agg_base = (
    tx_f.groupby(["Code", "Product"], as_index=False)
        .agg(
            Qty_Purchased=("Qty Purchased", "sum"),
            Bonus_Received=("Bonus", "sum"),
            Times_Purchased=("Product", "count"),
            Times_Bonus=("Bonus", lambda s: (s > 0).sum()),
            Avg_Purchase_Qty=("Qty Purchased", "mean"),
            Avg_Bonus_Qty=("Bonus", "mean"),
            Last_Purchase=("Date", "max") if "Date" in tx_f.columns else ("Product", "count")
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

# Avg Days Between (uses only valid dates)
if "Date" in tx_f.columns:
    gap_series = (
        tx_f.groupby(["Code", "Product"])["Date"]
            .apply(avg_gap_days)
            .reset_index(name="Avg Days Between")
    )
else:
    gap_series = pd.DataFrame({"Code": [], "Product": [], "Avg Days Between": []})

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
agg = (
    agg_base
    .merge(gap_series, on=["Code","Product"], how="left")
    .merge(var_df, on=["Code","Product"], how="left")
)
agg["Bonus %"] = (
    agg["Bonus Received"].replace(0, 0) /
    agg["Qty Purchased"].replace(0, pd.NA) * 100
).fillna(0).astype(float).round(1)
agg["Avg Days Between"] = agg["Avg Days Between"].round(1)

# Recency vs latest date found (of filtered set)
if "Date" in tx_f.columns and pd.notna(tx_f["Date"]).any():
    end_norm = tx_f["Date"].dropna().max().normalize()
    agg["Recency Days"] = (end_norm - agg["Last Purchase Date"]).dt.days
else:
    agg["Recency Days"] = pd.NA

# Bonus presence rate
agg["Bonus Presence Rate"] = (agg["Times Bonus"] / agg["Times Purchased"]).fillna(0)

# Apply Min Qty + Bonus filters at product level
agg = agg[agg["Qty Purchased"] >= min_qty]
if bonus_filter == "With Bonus":
    agg = agg[agg["Bonus %"] > 0]
elif bonus_filter == "Without Bonus":
    agg = agg[agg["Bonus %"] == 0]

if agg.empty:
    st.info("No products after Min Qty / Bonus / Date filters. Adjust filters.")
    st.stop()

# 3-status classification (Core / Promo-timed / Review)
BONUS_PROMO     = 8.0   # %
BPR_PROMO       = 0.50  # share of orders with bonus
STALE_DAYS_MIN  = 90
FACTOR_GAP      = 1.5

def classify_simple(r):
    ebp = r["Bonus %"]
    bpr = r["Bonus Presence Rate"]
    var = r.get("Bonus Variability (pp)", 0) or 0
    rec = r.get("Recency Days", None)
    gap = r.get("Avg Days Between", None)
    stale_gate = max(STALE_DAYS_MIN, (gap * FACTOR_GAP) if pd.notna(gap) else STALE_DAYS_MIN)
    if (pd.notna(rec) and rec > stale_gate) or ebp > 40 or var > 15:
        return "Review"
    if (ebp >= BONUS_PROMO) or (bpr >= BPR_PROMO):
        return "Promo-timed"
    return "Core"

agg["StatusKey"] = agg.apply(classify_simple, axis=1)
STATUS_COPY = {
    "Core": ("🟢 Core", "Stable demand; buy steadily and negotiate base price."),
    "Promo-timed": ("🟠 Promo-timed", "Buy during bonus windows; avoid outside promos."),
    "Review": ("🔴 Review", "Dormant/anomalous; verify demand, data, or supplier terms."),
}
agg[["Status", "Status Note"]] = agg["StatusKey"].apply(lambda k: pd.Series(STATUS_COPY[k]))

# Status filter
with st.sidebar:
    status_choice = st.selectbox("Status", ["(All)", "🟢 Core", "🟠 Promo-timed", "🔴 Review"], index=0)
if status_choice != "(All)":
    agg = agg[agg["Status"] == status_choice]

# KPIs
total_purchased = int(agg["Qty Purchased"].sum())
total_bonus     = int(agg["Bonus Received"].sum())
overall_bonus_rate = (total_bonus / total_purchased * 100) if total_purchased > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Products (after filters)", f"{len(agg):,}")
c2.metric("Total Purchased", f"{total_purchased:,}")
c3.metric("Total Bonus", f"{total_bonus:,}")
c4.metric("Overall Bonus %", f"{overall_bonus_rate:.1f}%")

# ---- Status definitions (compact)
st.markdown(
    """
**Status definitions:**  
- 🟢 **Core** — Stable demand; buy steadily and negotiate base price.  
- 🟠 **Promo-timed** — Buy during bonus windows; avoid outside promos.  
- 🔴 **Review** — Dormant/anomalous; verify demand, data, or supplier terms.
"""
)

st.divider()

# Charts — keep only the bar chart (fast)
agg["Label"] = (agg["Code"].astype(str) + " — " + agg["Product"].astype(str))
chart_df = agg.nlargest(top_n or len(agg), "Qty Purchased")[["Label", "Qty Purchased", "Bonus Received"]].copy()

top_qty   = chart_df["Qty Purchased"].sum()
top_bonus = chart_df["Bonus Received"].sum()
cov_qty   = (top_qty / total_purchased * 100) if total_purchased else 0
cov_bonus = (top_bonus / total_bonus * 100) if total_bonus else 0
st.caption(
    f"Bars cover {top_qty:,.0f} purchased ({cov_qty:.1f}% of total) "
    f"and {top_bonus:,.0f} bonus ({cov_bonus:.1f}% of total)."
)

st.subheader("Purchased vs Bonus — Top Products")
if not chart_df.empty:
    m = chart_df.melt(
        id_vars=["Label"],
        value_vars=["Qty Purchased", "Bonus Received"],
        var_name="Metric", value_name="Value"
    )
    fig_bar = px.bar(
        m, y="Label", x="Value", color="Metric",
        orientation="h", height=max(420, 30 * len(chart_df))
    )
    fig_bar.update_layout(
        yaxis=dict(categoryorder="total ascending", automargin=True),
        legend_title="", bargap=0.15,
        margin=dict(l=10, r=60, t=30, b=10)
    )
    fig_bar.update_traces(hovertemplate="%{y}<br>%{legendgroup}: %{x:,.0f}", cliponaxis=False)
    # Out-of-bar labels for bonus series only (cheaper rendering)
    for tr in fig_bar.data:
        if tr.name == "Bonus Received":
            tr.text = [f"{v:,.0f}" for v in tr.x]
            tr.textposition = "outside"
            tr.texttemplate = "%{text}"
            tr.textfont = dict(size=12)
        else:
            tr.text = None
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("No rows for bar chart.")

st.divider()

# Highlights
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
            "Qty Purchased": st.column_config.NumberColumn(format="%d"),
            "Bonus Received": st.column_config.NumberColumn(format="%d"),
            "Avg Days Between": st.column_config.NumberColumn(format="%.1f"),
            "Bonus %": st.column_config.ProgressColumn("Bonus %", format="%.1f%%", min_value=0, max_value=100),
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
            "Qty Purchased": st.column_config.NumberColumn(format="%d"),
            "Bonus Received": st.column_config.NumberColumn(format="%d"),
            "Avg Days Between": st.column_config.NumberColumn(format="%.1f"),
            "Bonus %": st.column_config.ProgressColumn("Bonus %", format="%.1f%%", min_value=0, max_value=100),
        },
    )

st.divider()

# Detailed Products table (lazy)
show_detailed = st.toggle("Show detailed products table", value=False)
if show_detailed:
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
            "Qty Purchased": st.column_config.NumberColumn(format="%d"),
            "Bonus Received": st.column_config.NumberColumn(format="%d"),
            "Times Purchased": st.column_config.NumberColumn(format="%d"),
            "Times Bonus": st.column_config.NumberColumn(format="%d"),
            "Avg Purchase Qty": st.column_config.NumberColumn(format="%.1f"),
            "Avg Bonus Qty": st.column_config.NumberColumn(format="%.1f"),
            "Avg Days Between": st.column_config.NumberColumn(format="%.1f"),
            "Bonus Presence Rate": st.column_config.NumberColumn(format="%.2f"),
            "Bonus Variability (pp)": st.column_config.NumberColumn(format="%.1f"),
            "Bonus %": st.column_config.ProgressColumn("Bonus %", format="%.1f%%", min_value=0, max_value=100),
        },
    )

st.divider()

# 🔎 SKU Drill-down (respects current filters)
st.subheader("🔎 SKU Drill-down")

agg_sorted = agg.sort_values(["Qty Purchased", "Product"], ascending=[False, True]).copy()
agg_sorted["SKU"] = agg_sorted["Code"].astype(str) + " — " + agg_sorted["Product"].astype(str)
sku_list = agg_sorted["SKU"].tolist()

selected_sku = st.selectbox("Select a product to view its purchase history:", ["(Choose a product)"] + sku_list, index=0)
if selected_sku != "(Choose a product)":
    row = agg_sorted.loc[agg_sorted["SKU"] == selected_sku].iloc[0]
    sel_code, sel_product = row["Code"], row["Product"]

    hist = tx_f[(tx_f["Code"].astype(str) == str(sel_code)) & (tx_f["Product"] == sel_product)].copy()

    if "Bonus %" in hist.columns:
        hist["Bonus % (tx)"] = pd.to_numeric(hist["Bonus %"], errors="coerce").fillna(0).round(1)
    else:
        hist["Bonus % (tx)"] = (
            (pd.to_numeric(hist["Bonus"], errors="coerce") /
             pd.to_numeric(hist["Qty Purchased"], errors="coerce"))
            .replace([float("inf"), -float("inf")], 0) * 100
        ).fillna(0).round(1)

    if "Date" in hist.columns:
        hist["_sort_date"] = hist["Date"]
        hist["Date"] = hist["Date"].dt.strftime(DATE_FMT_DISPLAY)

    st.markdown(f"**{selected_sku}** — history in selected filters")
    if len(hist) > 0 and "_sort_date" in hist:
        trend = hist.sort_values("_sort_date")
        fig_hist = px.line(
            trend,
            x="_sort_date",
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

    cols_hist = ["Date", "Supplier Name", "Qty Purchased", "Bonus", "Bonus % (tx)"]
    cols_hist = [c for c in cols_hist if c in hist.columns]
    st.dataframe(
        (hist.sort_values("_sort_date") if "_sort_date" in hist else hist)[cols_hist],
        use_container_width=True, hide_index=True,
        column_config={
            "Qty Purchased": st.column_config.NumberColumn(format="%d"),
            "Bonus": st.column_config.NumberColumn(format="%d"),
            "Bonus % (tx)": st.column_config.ProgressColumn("Bonus % (tx)", format="%.1f%%", min_value=0, max_value=100),
        },
    )

# Raw transactions (filtered table) — lazy inside expander
with st.expander("🧾 View all filtered transactions (lazy)"):
    show_cols = ["Supplier Name","Code","Product","Date","Qty Purchased","Bonus","Bonus %","__sheet__"]
    show_cols = [c for c in show_cols if c in tx_f.columns or c == "__sheet__"]
    tx_show = tx_f.copy()
    if "__sheet__" not in tx_show.columns:
        tx_show["__sheet__"] = "sheet"
    tx_show = tx_show[show_cols]
    if "Date" in tx_show.columns:
        tx_show["_sort_date"] = tx_show["Date"]
        tx_show["Date"] = tx_show["Date"].dt.strftime(DATE_FMT_DISPLAY)
        tx_show = tx_show.sort_values(["__sheet__", "Product", "_sort_date"]).drop(columns=["_sort_date"])
    st.dataframe(
        tx_show,
        use_container_width=True, hide_index=True,
        column_config={
            "Qty Purchased": st.column_config.NumberColumn(format="%d"),
            "Bonus": st.column_config.NumberColumn(format="%d"),
            "Bonus %": st.column_config.ProgressColumn("Bonus %", format="%.1f%%", min_value=0, max_value=100),
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
