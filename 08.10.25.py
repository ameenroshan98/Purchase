# app.py
import streamlit as st
st.set_page_config(page_title="Purchase Dashboard", layout="wide")  # must be first

import pandas as pd
import requests
from io import BytesIO
import re
import time
import warnings
from typing import Tuple

# Quiet noisy pandas deprecations in UI (optional but tidy)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# ---------- Quick link ----------
try:
    st.page_link("pages/Compare_Yearly_Patterns.py", label="📅 Compare Ranges (Purchase vs Bonus)")
except Exception:
    pass

# ---------- Config ----------
SHEET_ID = "1R7o4xKMeYWcYWwAOorMyYDtjg0-74FqDK0xAFKN6Iuo"
GID      = 0
XLSX_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=xlsx"
CSV_URL  = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
DATE_FMT_DISPLAY = "%m/%d/%Y"

# ---------- Helpers ----------
def avg_gap_days(s: pd.Series) -> float:
    d = pd.to_datetime(s, errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()
    if len(d) <= 1:
        return float("nan")
    return float(d.diff().dt.days.dropna().mean())

def _standardize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (pd.Index(df.columns).astype(str)
                  .str.replace("\u00A0"," ",regex=False).str.strip())
    rename_map = {
        "quantity purchased":"Qty Purchased","qty purchased":"Qty Purchased","qty":"Qty Purchased",
        "Bonus%":"Bonus %",
    }
    return df.rename(columns={c: rename_map.get(c,c) for c in df.columns})

def _fast_dates(series: pd.Series, prefer: str) -> pd.Series:
    s = series.astype(str).str.replace("\u00A0"," ",regex=False).str.strip()
    s = s.str.replace(".","/",regex=False).str.replace("-","/",regex=False)
    serial = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    m = serial.notna()
    if m.any():
        out.loc[m] = pd.to_datetime(serial.loc[m], unit="D", origin="1899-12-30", errors="coerce")
    rest = ~m
    if rest.any():
        dayfirst = (prefer.lower()=="dmy")
        # NOTE: no infer_datetime_format (deprecated in pandas 2.3+)
        out.loc[rest] = pd.to_datetime(s.loc[rest], errors="coerce", dayfirst=dayfirst)
    return out.dt.normalize()

@st.cache_data(ttl=1800, max_entries=4, show_spinner=False)
def load_transactions(prefer="mdy", fast_mode=True) -> pd.DataFrame:
    usecols = ["Supplier Name","Code","Product","Date","Qty Purchased","Bonus","Bonus %"]

    def _post(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            out = pd.DataFrame(columns=usecols)
            for c in usecols: out[c]=pd.NA
            return out
        df = _standardize_headers(df)
        for c in usecols:
            if c not in df.columns: df[c]=pd.NA
        df = df[usecols]
        df["Date"] = _fast_dates(df["Date"], prefer)

        for c in ["Qty Purchased","Bonus","Bonus %"]:
            df[c] = (df[c].astype(str)
                            .str.replace(",","",regex=False)
                            .str.replace("%","",regex=False)
                            .str.strip())
        df["Qty Purchased"] = pd.to_numeric(df["Qty Purchased"], errors="coerce").fillna(0).astype("int32")
        df["Bonus"]         = pd.to_numeric(df["Bonus"], errors="coerce").fillna(0).astype("int32")
        df["Bonus %"]       = pd.to_numeric(df["Bonus %"], errors="coerce").fillna(0.0).astype("float32")

        for c in ["Supplier Name","Code","Product"]:
            df[c] = df[c].astype("category")
        return df

    if fast_mode:
        try:
            df = pd.read_csv(CSV_URL, dtype=str)
            df["__sheet__"] = f"gid_{GID}"
            return _post(df)
        except Exception:
            pass

    # XLSX fallback
    try:
        for attempt in range(2):
            r = requests.get(XLSX_URL, timeout=45)
            if r.ok:
                with BytesIO(r.content) as bio:
                    book = pd.read_excel(bio, sheet_name=None, dtype=str, engine="openpyxl")
                break
            time.sleep(0.8*(attempt+1))
        else:
            raise RuntimeError("XLSX fetch failed")

        frames = []
        for sh, df in (book or {}).items():
            if df is None or df.empty: continue
            df = df.dropna(how="all"); df["__sheet__"]=sh; frames.append(df)
        big = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return _post(big)
    except Exception:
        df = pd.read_csv(CSV_URL, dtype=str)
        df["__sheet__"] = f"gid_{GID}"
        return _post(df)

# ---------- UI ----------
st.title("📊 Purchase Dashboard")

with st.sidebar:
    st.markdown("### Data refresh & locale")
    c1, c2 = st.columns([1,1])
    prefer_order = c1.radio("Ambiguous dates", ["mdy","dmy"], index=0, horizontal=True,
                            help="Used only when both parts ≤ 12 (e.g., 03/04/25).")
    if c2.button("🔄 Force refresh", use_container_width=True):
        load_transactions.clear(); st.experimental_rerun()

with st.sidebar:
    st.markdown("### Performance")
    FAST_MODE = st.toggle("⚡ Fast mode (CSV, one tab)", value=True,
                          help="Loads the CSV of the primary tab only.")
    COMPUTE_ADVANCED = st.toggle("Compute advanced metrics", value=False,
                                 help="Recency, Avg Days Between, Variability.")
    SHOW_CHARTS = st.toggle("Show charts (Plotly)", value=False)
    SHOW_DEBUG = st.toggle("Show debug info", value=False)

tx = load_transactions(prefer=prefer_order, fast_mode=FAST_MODE)

# Hard guard for required columns
for req in ["Code","Product","Qty Purchased","Bonus"]:
    if req not in tx.columns:
        st.error(f"Required column missing: **{req}**. Check sheet headers.")
        st.stop()

# Global bounds (safe)
min_date = tx["Date"].min(skipna=True) if "Date" in tx.columns else pd.NaT
max_date = tx["Date"].max(skipna=True) if "Date" in tx.columns else pd.NaT
if pd.notna(min_date) and pd.notna(max_date):
    st.caption(f"ℹ️ Data range from **{min_date.strftime(DATE_FMT_DISPLAY)}** to **{max_date.strftime(DATE_FMT_DISPLAY)}**")

if SHOW_DEBUG:
    with st.expander("🔧 Debug"):
        st.write({"shape": tx.shape, "cols": list(tx.columns), "sample": tx.head(3)})

# ---------- Filters ----------
with st.sidebar:
    st.header("Filters")
    sup_series = tx.get("Supplier Name", pd.Series("", index=tx.index)).astype(str).str.strip()
    supplier_options = sorted([s for s in sup_series.unique() if s])
    selected_supplier = st.selectbox("Supplier", ["All suppliers"] + supplier_options, index=0)
    q = st.text_input("Search (code or product)", "")

    # Date range — safe unpack
    include_missing_dates = False
    if pd.notna(min_date) and pd.notna(max_date):
        dr_val: Tuple = (min_date.date(), max_date.date())
        dr = st.date_input("Date range", value=dr_val,
                           min_value=min_date.date(), max_value=max_date.date(),
                           format="MM/DD/YYYY")
        if isinstance(dr, (list, tuple)) and len(dr)==2:
            start_d, end_d = dr[0], dr[1]
        else:
            start_d, end_d = min_date.date(), max_date.date()
        include_missing_dates = st.checkbox("Include rows with missing Date", value=False)
    else:
        start_d = end_d = None

    bonus_filter = st.selectbox("Bonus filter", ["All","With Bonus","Without Bonus"])

    # Robust preview_max
    mask_preview = pd.Series(True, index=tx.index)
    if selected_supplier != "All suppliers": mask_preview &= (sup_series == selected_supplier)
    if q.strip():
        ql = q.strip().lower()
        mask_preview &= (
            tx["Product"].astype(str).str.lower().str.contains(ql, na=False) |
            tx["Code"].astype(str).str.contains(ql, na=False)
        )
    try:
        if mask_preview.any():
            prev = tx.loc[mask_preview, ["Code","Product","Qty Purchased"]].copy()
            prev["Qty Purchased"] = pd.to_numeric(prev["Qty Purchased"], errors="coerce").fillna(0)
            preview_max = int(prev.groupby(["Code","Product"], observed=False)["Qty Purchased"].sum().max() or 0)
        else:
            preview_max = 0
    except Exception:
        preview_max = 0

    min_qty = st.slider("Min Qty Purchased (product total)", 0, max(preview_max, 0), 0, step=100)
    scope = st.radio("Chart scope", ["Top-N","All"], horizontal=True)
    top_n = st.slider("Top-N for charts", 5, 100, 15) if scope=="Top-N" else None

# ---------- Apply filters ----------
mask_tx = pd.Series(True, index=tx.index)
if selected_supplier != "All suppliers": mask_tx &= (sup_series == selected_supplier)
if q.strip():
    ql = q.strip().lower()
    mask_tx &= (
        tx["Product"].astype(str).str.lower().str.contains(ql, na=False) |
        tx["Code"].astype(str).str.contains(ql, na=False)
    )
if "Date" in tx.columns and pd.notna(min_date) and pd.notna(max_date) and start_d and end_d:
    s_ts, e_ts = pd.Timestamp(start_d).normalize(), pd.Timestamp(end_d).normalize()
    date_mask = tx["Date"].between(s_ts, e_ts, inclusive="both")
    if include_missing_dates: date_mask = date_mask | tx["Date"].isna()
    mask_tx &= date_mask

tx_f = tx[mask_tx].copy()
if tx_f.empty:
    st.info("No transactions match your filters (including date range).")
    st.stop()

# ---------- Aggregations ----------
agg_base = (
    tx_f.groupby(["Code","Product"], as_index=False, observed=False)
        .agg(
            Qty_Purchased=("Qty Purchased","sum"),
            Bonus_Received=("Bonus","sum"),
            Times_Purchased=("Product","count"),
            Times_Bonus=("Bonus", lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum()),
            Avg_Purchase_Qty=("Qty Purchased","mean"),
            Avg_Bonus_Qty=("Bonus","mean"),
            Last_Purchase=("Date","max") if "Date" in tx_f.columns else ("Product","count"),
        )
        .rename(columns={
            "Qty_Purchased":"Qty Purchased",
            "Bonus_Received":"Bonus Received",
            "Times_Purchased":"Times Purchased",
            "Times_Bonus":"Times Bonus",
            "Avg_Purchase_Qty":"Avg Purchase Qty",
            "Avg_Bonus_Qty":"Avg Bonus Qty",
            "Last_Purchase":"Last Purchase Date",
        })
)

# Advanced metrics (optional)
if COMPUTE_ADVANCED and "Date" in tx_f.columns:
    gap_series = (
        tx_f.groupby(["Code","Product"], observed=False)["Date"]
            .apply(avg_gap_days).reset_index(name="Avg Days Between")
    )
    tx_tmp = tx_f.copy()
    if "Bonus %" in tx_tmp.columns:
        tx_tmp["bonus_pct_tx"] = pd.to_numeric(tx_tmp["Bonus %"], errors="coerce")
    else:
        tx_tmp["bonus_pct_tx"] = (
            (pd.to_numeric(tx_tmp["Bonus"], errors="coerce") /
             pd.to_numeric(tx_tmp["Qty Purchased"], errors="coerce"))
        ).replace([float("inf"), -float("inf")], 0).fillna(0) * 100
    var_df = (
        tx_tmp.groupby(["Code","Product"], observed=False)["bonus_pct_tx"]
              .std(ddof=0).fillna(0).reset_index(name="Bonus Variability (pp)")
    )
else:
    gap_series = pd.DataFrame({"Code": agg_base["Code"], "Product": agg_base["Product"], "Avg Days Between": pd.NA})
    var_df   = pd.DataFrame({"Code": agg_base["Code"], "Product": agg_base["Product"], "Bonus Variability (pp)": 0})

agg = (agg_base
       .merge(gap_series, on=["Code","Product"], how="left")
       .merge(var_df, on=["Code","Product"], how="left"))

agg["Bonus %"] = (
    pd.to_numeric(agg["Bonus Received"], errors="coerce").fillna(0) /
    pd.to_numeric(agg["Qty Purchased"], errors="coerce").replace(0, pd.NA)
) * 100
agg["Bonus %"] = agg["Bonus %"].fillna(0).astype(float).round(1)
if COMPUTE_ADVANCED:
    agg["Avg Days Between"] = pd.to_numeric(agg["Avg Days Between"], errors="coerce").round(1)

# Recency (optional)
if COMPUTE_ADVANCED and "Date" in tx_f.columns and pd.notna(tx_f["Date"]).any():
    end_norm = tx_f["Date"].dropna().max().normalize()
    agg["Recency Days"] = (end_norm - agg["Last Purchase Date"]).dt.days
else:
    agg["Recency Days"] = pd.NA

# Bonus presence rate
agg["Bonus Presence Rate"] = (agg["Times Bonus"] / agg["Times Purchased"]).fillna(0)

# Product-level filters
agg = agg[pd.to_numeric(agg["Qty Purchased"], errors="coerce").fillna(0) >= min_qty]
if bonus_filter == "With Bonus":
    agg = agg[agg["Bonus %"] > 0]
elif bonus_filter == "Without Bonus":
    agg = agg[agg["Bonus %"] == 0]

if agg.empty:
    st.info("No products after Min Qty / Bonus / Date filters. Adjust filters.")
    st.stop()

# ---------- Classification ----------
BONUS_PROMO = 8.0
BPR_PROMO   = 0.50
STALE_MIN   = 90
FACTOR_GAP  = 1.5

def classify_simple(r):
    ebp = r.get("Bonus %", 0) or 0
    bpr = r.get("Bonus Presence Rate", 0) or 0
    var = r.get("Bonus Variability (pp)", 0) or 0
    rec = r.get("Recency Days", None)
    gap = r.get("Avg Days Between", None)
    try:
        stale_gate = max(STALE_MIN, (float(gap) * FACTOR_GAP) if pd.notna(gap) else STALE_MIN)
    except Exception:
        stale_gate = STALE_MIN
    if (pd.notna(rec) and rec is not None and rec > stale_gate) or ebp > 40 or var > 15:
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
agg[["Status","Status Note"]] = agg["StatusKey"].apply(lambda k: pd.Series(STATUS_COPY[k]))

with st.sidebar:
    status_choice = st.selectbox("Status", ["(All)","🟢 Core","🟠 Promo-timed","🔴 Review"], index=0)
if status_choice != "(All)":
    agg = agg[agg["Status"] == status_choice]

# ---------- KPIs ----------
total_purchased = int(pd.to_numeric(agg["Qty Purchased"], errors="coerce").fillna(0).sum())
total_bonus     = int(pd.to_numeric(agg["Bonus Received"], errors="coerce").fillna(0).sum())
overall_bonus_rate = (total_bonus / total_purchased * 100) if total_purchased > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Products (after filters)", f"{len(agg):,}")
c2.metric("Total Purchased", f"{total_purchased:,}")
c3.metric("Total Bonus", f"{total_bonus:,}")
c4.metric("Overall Bonus %", f"{overall_bonus_rate:.1f}%")

st.markdown("""
**Status definitions:**  
- 🟢 **Core** — Stable demand; buy steadily and negotiate base price.  
- 🟠 **Promo-timed** — Buy during bonus windows; avoid outside promos.  
- 🔴 **Review** — Dormant/anomalous; verify demand, data, or supplier terms.
""")
st.divider()

# ---------- Charts (optional) ----------
if SHOW_CHARTS:
    import plotly.express as px
    agg["Label"] = agg["Code"].astype(str) + " — " + agg["Product"].astype(str)
    n = (top_n or len(agg))
    chart_df = agg.nlargest(n, "Qty Purchased")[["Label","Qty Purchased","Bonus Received"]].copy()
    top_qty   = chart_df["Qty Purchased"].sum()
    top_bonus = chart_df["Bonus Received"].sum()
    cov_qty   = (top_qty / total_purchased * 100) if total_purchased else 0
    cov_bonus = (top_bonus / total_bonus * 100) if total_bonus else 0
    st.caption(f"Bars cover {top_qty:,.0f} purchased ({cov_qty:.1f}% of total) and {top_bonus:,.0f} bonus ({cov_bonus:.1f}% of total).")
    st.subheader("Purchased vs Bonus — Top Products")
    if not chart_df.empty:
        m = chart_df.melt(id_vars=["Label"], value_vars=["Qty Purchased","Bonus Received"],
                          var_name="Metric", value_name="Value")
        fig_bar = px.bar(m, y="Label", x="Value", color="Metric",
                         orientation="h", height=max(420, 30*len(chart_df)))
        fig_bar.update_layout(yaxis=dict(categoryorder="total ascending", automargin=True),
                              legend_title="", bargap=0.15,
                              margin=dict(l=10,r=60,t=30,b=10))
        fig_bar.update_traces(hovertemplate="%{y}<br>%{legendgroup}: %{x:,.0f}", cliponaxis=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    st.divider()

# ---------- Highlights ----------
st.subheader("Highlights")
left, right = st.columns(2)

with left:
    st.markdown("**Top by Volume (Qty Purchased)**")
    t1_cols = [c for c in ["Status","Code","Product","Qty Purchased","Bonus Received","Times Purchased","Avg Days Between","Bonus %","Status Note"] if c in agg.columns]
    st.dataframe(agg.sort_values("Qty Purchased", ascending=False).head(10)[t1_cols],
                 use_container_width=True, hide_index=True,
                 column_config={"Qty Purchased": st.column_config.NumberColumn(format="%d"),
                                "Bonus Received": st.column_config.NumberColumn(format="%d"),
                                "Avg Days Between": st.column_config.NumberColumn(format="%.1f"),
                                "Bonus %": st.column_config.ProgressColumn("Bonus %", format="%.1f%%", min_value=0, max_value=100)})

with right:
    st.markdown("**Top by Bonus % (min 100 units)**")
    t2 = agg[pd.to_numeric(agg["Qty Purchased"], errors="coerce").fillna(0) >= 100].sort_values("Bonus %", ascending=False).head(10)
    t2_cols = [c for c in ["Status","Code","Product","Qty Purchased","Bonus Received","Times Purchased","Avg Days Between","Bonus %","Status Note"] if c in t2.columns]
    st.dataframe(t2[t2_cols],
                 use_container_width=True, hide_index=True,
                 column_config={"Qty Purchased": st.column_config.NumberColumn(format="%d"),
                                "Bonus Received": st.column_config.NumberColumn(format="%d"),
                                "Avg Days Between": st.column_config.NumberColumn(format="%.1f"),
                                "Bonus %": st.column_config.ProgressColumn("Bonus %", format="%.1f%%", min_value=0, max_value=100)})

st.divider()

# ---------- Detailed table (lazy) ----------
show_detailed = st.toggle("Show detailed products table", value=False)
if show_detailed:
    cols = ["Status","Code","Product","Qty Purchased","Bonus Received","Times Purchased","Times Bonus","Avg Purchase Qty","Avg Bonus Qty","Avg Days Between","Bonus %","Bonus Presence Rate","Bonus Variability (pp)","Status Note"]
    present = [c for c in cols if c in agg.columns]
    st.subheader(f"Detailed Products ({len(agg):,})")
    st.dataframe(agg[present].sort_values("Qty Purchased", ascending=False),
                 use_container_width=True, hide_index=True,
                 column_config={"Qty Purchased": st.column_config.NumberColumn(format="%d"),
                                "Bonus Received": st.column_config.NumberColumn(format="%d"),
                                "Times Purchased": st.column_config.NumberColumn(format="%d"),
                                "Times Bonus": st.column_config.NumberColumn(format="%d"),
                                "Avg Purchase Qty": st.column_config.NumberColumn(format="%.1f"),
                                "Avg Bonus Qty": st.column_config.NumberColumn(format="%.1f"),
                                "Avg Days Between": st.column_config.NumberColumn(format="%.1f"),
                                "Bonus Presence Rate": st.column_config.NumberColumn(format="%.2f"),
                                "Bonus Variability (pp)": st.column_config.NumberColumn(format="%.1f"),
                                "Bonus %": st.column_config.ProgressColumn("Bonus %", format="%.1f%%", min_value=0, max_value=100)})

st.divider()

# ---------- Drill-down ----------
st.subheader("🔎 SKU Drill-down")
agg_sorted = agg.sort_values(["Qty Purchased","Product"], ascending=[False, True]).copy()
agg_sorted["SKU"] = agg_sorted["Code"].astype(str) + " — " + agg_sorted["Product"].astype(str)
sku = st.selectbox("Select a product to view its purchase history:", ["(Choose a product)"] + agg_sorted["SKU"].tolist(), index=0)

if sku != "(Choose a product)":
    row = agg_sorted.loc[agg_sorted["SKU"] == sku].iloc[0]
    sel_code, sel_product = str(row["Code"]), row["Product"]
    hist = tx_f[(tx_f["Code"].astype(str) == sel_code) & (tx_f["Product"] == sel_product)].copy()

    if "Bonus %" in hist.columns:
        hist["Bonus % (tx)"] = pd.to_numeric(hist["Bonus %"], errors="coerce").fillna(0).round(1)
    else:
        hist["Bonus % (tx)"] = ((pd.to_numeric(hist["Bonus"], errors="coerce") /
                                  pd.to_numeric(hist["Qty Purchased"], errors="coerce"))
                                 .replace([float("inf"), -float("inf")], 0).fillna(0) * 100).round(1)

    if "Date" in hist.columns:
        hist["_sort_date"] = hist["Date"]
        hist["Date"] = hist["Date"].dt.strftime(DATE_FMT_DISPLAY)

    st.markdown(f"**{sku}** — history in selected filters")

    if SHOW_CHARTS and len(hist) > 0 and "_sort_date" in hist:
        import plotly.express as px
        trend = hist.sort_values("_sort_date")
        fig_hist = px.line(trend, x="_sort_date", y="Qty Purchased", markers=True, title=None)
        fig_hist.update_layout(margin=dict(l=10,r=10,t=10,b=10), xaxis_title="Date", yaxis_title="Qty Purchased")
        st.plotly_chart(fig_hist, use_container_width=True)

    cols_hist = [c for c in ["Date","Supplier Name","Qty Purchased","Bonus","Bonus % (tx)"] if c in hist.columns]
    st.dataframe((hist.sort_values("_sort_date") if "_sort_date" in hist else hist)[cols_hist],
                 use_container_width=True, hide_index=True,
                 column_config={"Qty Purchased": st.column_config.NumberColumn(format="%d"),
                                "Bonus": st.column_config.NumberColumn(format="%d"),
                                "Bonus % (tx)": st.column_config.ProgressColumn("Bonus % (tx)", format="%.1f%%", min_value=0, max_value=100)})

# ---------- Download ----------
csv_cols = ["Status","Code","Product","Qty Purchased","Bonus Received","Times Purchased","Times Bonus","Avg Purchase Qty","Avg Bonus Qty","Avg Days Between","Bonus %","Bonus Presence Rate","Bonus Variability (pp)","Status Note"]
csv_agg = agg[[c for c in csv_cols if c in agg.columns]].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download product summary (CSV)", csv_agg, "product_summary.csv", "text/csv")
