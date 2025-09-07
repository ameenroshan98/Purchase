import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from io import BytesIO
import re

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Purchase Dashboard", layout="wide")

# ---- Google Sheet config (entire workbook)
SHEET_ID = "1R7o4xKMeYWcYWwAOorMyYDtjg0-74FqDK0xAFKN6Iuo"
GID      = 0  # used only by CSV fallback
XLSX_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=xlsx"
CSV_URL  = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

DATE_FMT_DISPLAY = "%m/%d/%Y"  # mm/dd/yyyy
EARLIEST_AVAILABLE = pd.Timestamp(2024, 8, 1)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def short_label(code, name, n=42):
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

def parse_dates_smart(raw_col: pd.Series, prefer: str = "mdy"):
    """
    Smart parser with per-row detection + serials + ISO.
    prefer: "mdy" (default) or "dmy" used only when ambiguous (both parts <= 12).
    Returns (parsed_Timestamps, original_strings)
    """
    s_raw = raw_col.astype(str)

    # Clean NBSP, trim, unify separators
    s = (s_raw
         .str.replace("\u00A0", " ", regex=False)
         .str.strip()
         .str.replace(".", "/", regex=False))

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    # 1) Excel/Sheets serial numbers
    serial = pd.to_numeric(s, errors="coerce")
    serial_mask = serial.notna()
    if serial_mask.any():
        out.loc[serial_mask] = pd.to_datetime(
            serial.loc[serial_mask], unit="D", origin="1899-12-30", errors="coerce"
        )

    # 2) Strings with separators
    mask = ~serial_mask
    s2 = s[mask].str.replace("-", "/", regex=False)

    def parse_one(val: str):
        if not val:
            return pd.NaT
        # Try ISO first
        iso_try = pd.to_datetime(val, errors="coerce")
        if pd.notna(iso_try):
            return iso_try.normalize()

        parts = re.split(r"[\/]", val)
        if len(parts) != 3:
            return pd.NaT
        try:
            a, b, c = [int(p) for p in parts]
        except Exception:
            return pd.NaT

        # Fix 2-digit year
        if c < 100:
            c += 2000 if c < 70 else 1900

        # Decide month/day; fall back to preference when ambiguous
        if a > 12 and b <= 12:   # clearly D/M/Y
            m, d, y = b, a, c
        elif b > 12 and a <= 12: # clearly M/D/Y
            m, d, y = a, b, c
        elif a > 12 and b > 12:  # impossible
            return pd.NaT
        else:
            if prefer.lower() == "dmy":
                m, d, y = b, a, c
            else:
                m, d, y = a, b, c

        try:
            return pd.Timestamp(year=y, month=m, day=d).normalize()
        except Exception:
            return pd.NaT

    parsed = s2.apply(parse_one)
    out.loc[mask] = parsed.values
    return out, s_raw

# -------------------------------------------------
# Load ALL data from the Google Sheet (all tabs), fallback to CSV
# -------------------------------------------------
@st.cache_data(ttl=900)
def load_transactions(prefer="mdy"):
    # Try XLSX (entire workbook)
    try:
        r = requests.get(XLSX_URL, timeout=60)
        r.raise_for_status()
        with BytesIO(r.content) as bio:
            book = pd.read_excel(bio, sheet_name=None, dtype=str)
        frames = []
        for sheet_name, df in book.items():
            if df is None or df.empty:
                continue
            df = df.dropna(how="all")
            df.columns = pd.Index(df.columns).astype(str).str.replace("\u00A0", " ", regex=False).str.strip()
            df["__sheet__"] = sheet_name
            frames.append(df)
        df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception:
        # Fallback: CSV of one tab
        df_all = pd.read_csv(CSV_URL, dtype=str).fillna("")
        df_all.columns = pd.Index(df_all.columns).astype(str).str.replace("\u00A0", " ", regex=False).str.strip()
        df_all["__sheet__"] = f"gid_{GID}"

    if df_all.empty:
        return df_all

    # Standardize headers (tolerate variants)
    rename_map = {
        "Supplier Name": "Supplier Name",
        "Code": "Code",
        "Product": "Product",
        "Date": "Date",
        "Qty Purchased": "Qty Purchased",
        "Bonus": "Bonus",
        "Bonus %": "Bonus %"
    }
    for c in list(df_all.columns):
        lc = c.lower().strip()
        if lc in ["qty purchased", "quantity purchased", "qty"]:
            rename_map[c] = "Qty Purchased"
    df_all = df_all.rename(columns=rename_map)

    # Parse Date (smart; prefers MDY)
    if "Date" in df_all.columns:
        parsed, date_raw = parse_dates_smart(df_all["Date"], prefer=prefer)
        df_all["_Date_raw"] = date_raw
        df_all["Date"] = parsed

    # Numerics
    for c in ["Qty Purchased", "Bonus"]:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c].astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(0)

    if "Bonus %" in df_all.columns:
        df_all["Bonus %"] = (
            df_all["Bonus %"].astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df_all["Bonus %"] = pd.to_numeric(df_all["Bonus %"], errors="coerce").fillna(0.0)

    return df_all

# -------------------------------------------------
# UI & Filters (WITH Date range)
# -------------------------------------------------
tx = load_transactions(prefer="mdy")

st.title("📊 Purchase Dashboard")

if tx.empty or "Code" not in tx.columns or "Product" not in tx.columns:
    st.warning("No data found or required columns missing (Code/Product).")
    st.stop()

# Show data range
min_date = tx["Date"].min(skipna=True) if "Date" in tx.columns else None
max_date = tx["Date"].max(skipna=True) if "Date" in tx.columns else None
cap = []
if pd.notna(min_date): cap.append(f"from **{min_date.strftime(DATE_FMT_DISPLAY)}**")
if pd.notna(max_date): cap.append(f"to **{max_date.strftime(DATE_FMT_DISPLAY)}**")
if cap:
    st.caption("ℹ️ Data range " + " ".join(cap))

# Sidebar filters
with st.sidebar:
    st.header("Filters")

    # Date range (mm/dd/yyyy)
    if "Date" in tx.columns and pd.notna(tx["Date"]).any():
        data_min = tx["Date"].min(skipna=True)
        data_max = tx["Date"].max(skipna=True)
        min_allowed = max(EARLIEST_AVAILABLE, data_min) if pd.notna(data_min) else EARLIEST_AVAILABLE
        max_allowed = data_max if pd.notna(data_max) else pd.Timestamp.today().normalize()

        default_range = (min_allowed.date(), max_allowed.date())
        date_range = st.date_input(
            "Date range (mm/dd/yyyy)",
            value=default_range,
            min_value=min_allowed.date(),
            max_value=max_allowed.date(),
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = [pd.Timestamp(d) for d in date_range]
        else:
            start_date = pd.Timestamp(date_range)
            end_date = start_date
    else:
        # If no valid dates in data, treat as unbounded
        start_date = pd.Timestamp(EARLIEST_AVAILABLE)
        end_date = pd.Timestamp.today().normalize()

    sup_series = tx.get("Supplier Name", pd.Series("", index=tx.index)).astype(str).str.strip()
    supplier_options = sorted([s for s in sup_series.unique() if s])
    selected_supplier = st.selectbox(
        "Supplier",
        options=["All suppliers"] + supplier_options,
        index=0
    )

    q = st.text_input("Search (code or product)", "")
    bonus_filter = st.selectbox("Bonus filter", ["All", "With Bonus", "Without Bonus"])

    # Min-qty slider max based on preview slice (date+supplier+search)
    start_norm_preview = start_date.normalize()
    end_norm_preview   = end_date.normalize()
    mask_preview = pd.Series(True, index=tx.index)
    if "Date" in tx.columns:
        mask_preview &= tx["Date"].notna() & (tx["Date"] >= start_norm_preview) & (tx["Date"] <= end_norm_preview)
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
        prev_agg = (tx.loc[mask_preview]
                      .groupby(["Code","Product"], as_index=False)["Qty Purchased"]
                      .sum())
        if not prev_agg.empty:
            preview_max = int(prev_agg["Qty Purchased"].max())
    min_qty = st.slider("Min Qty Purchased (product total)", 0, max(preview_max, 0), 0, step=100)

    scope = st.radio("Chart scope", ["Top-N", "All"], horizontal=True)
    if scope == "Top-N":
        top_n = st.slider("Top-N for charts", 5, 100, 15)
    else:
        top_n = None

st.caption(
    f"Showing data from **{start_date.strftime(DATE_FMT_DISPLAY)}** "
    f"to **{end_date.strftime(DATE_FMT_DISPLAY)}**."
)

# -------------------------------------------------
# Build transaction-level mask (DATE + supplier + search)
# -------------------------------------------------
mask_tx = pd.Series(True, index=tx.index)
if "Date" in tx.columns:
    start_norm = start_date.normalize()
    end_norm   = end_date.normalize()
    mask_tx &= tx["Date"].notna() & (tx["Date"] >= start_norm) & (tx["Date"] <= end_norm)

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
# Aggregate to PRODUCT level
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

# Recency vs selected end date
end_norm = end_date.normalize()
agg["Recency Days"] = (end_norm - agg["Last Purchase Date"]).dt.days

# Apply Min Qty + Bonus filters at product level
agg = agg[agg["Qty Purchased"] >= min_qty]
if bonus_filter == "With Bonus":
    agg = agg[agg["Bonus %"] > 0]
elif bonus_filter == "Without Bonus":
    agg = agg[agg["Bonus %"] == 0]

if agg.empty:
    st.info("No products after Min Qty / Bonus filter. Adjust filters.")
    st.stop()

# -------------------------------------------------
# 3-status classification (Core / Promo-timed / Review)
# -------------------------------------------------
BONUS_PROMO = 8.0      # %: considered promo-driven if >= this
BPR_PROMO   = 0.50     # share of orders with bonus
STALE_DAYS_MIN = 90    # days
FACTOR_GAP     = 1.5   # recency > 1.5x typical gap -> stale

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

# ---- Status filter (optional)
with st.sidebar:
    status_choice = st.selectbox("Status", ["(All)", "🟢 Core", "🟠 Promo-timed", "🔴 Review"], index=0)

agg_view = agg.copy()
if status_choice != "(All)":
    agg_view = agg_view[agg_view["Status"] == status_choice]

# -------------------------------------------------
# Legend / Explanations
# -------------------------------------------------
with st.expander("ℹ️ Status definitions & rules", expanded=False):
    st.markdown(
        f"""
- **🟢 Core** — Stable demand; **buy steadily** and negotiate base price.  
  Criteria: Bonus % < {BONUS_PROMO:.0f}% **and** bonus on < {int(BPR_PROMO*100)}% of orders, no stale recency/anomaly.
- **🟠 Promo-timed** — Demand responds to promos; **buy during bonus windows**.  
  Criteria: Bonus % ≥ {BONUS_PROMO:.0f}% **or** bonus on ≥ {int(BPR_PROMO*100)}% of orders.
- **🔴 Review** — Dormant/anomalous; **verify demand, data, or terms**.  
  Criteria: Recency > max({STALE_DAYS_MIN}d, 1.5×Avg Gap) **or** Bonus % > 40% **or** variability > 15 pp.
        """
    )

# -------------------------------------------------
# KPIs (from the current view after status filter)
# -------------------------------------------------
total_purchased = int(agg_view["Qty Purchased"].sum())
total_bonus = int(agg_view["Bonus Received"].sum())
overall_bonus_rate = (total_bonus / total_purchased * 100) if total_purchased > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Products (after filters)", f"{len(agg_view):,}")
c2.metric("Total Purchased", f"{total_purchased:,}")
c3.metric("Total Bonus", f"{total_bonus:,}")
c4.metric("Overall Bonus %", f"{overall_bonus_rate:.1f}%")

st.divider()

# -------------------------------------------------
# Charts (use current view)
# -------------------------------------------------
chart_df = agg_view.sort_values("Qty Purchased", ascending=False)
if top_n := (None if scope == "All" else top_n):
    chart_df = chart_df.head(top_n)
chart_df["Label"] = chart_df.apply(lambda r: short_label(r["Code"], r["Product"]), axis=1)

# Coverage vs totals
top_qty   = chart_df["Qty Purchased"].sum()
top_bonus = chart_df["Bonus Received"].sum()
cov_qty   = (top_qty / total_purchased * 100) if total_purchased else 0
cov_bonus = (top_bonus / total_bonus * 100) if total_bonus else 0
st.caption(f"Bars cover {top_qty:,.0f} purchased ({cov_qty:.1f}% of total) "
           f"and {top_bonus:,.0f} bonus ({cov_bonus:.1f}% of total).")

# Purchased vs Bonus — Top Products (labels on Bonus only)
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
    fig_bar.update_traces(hovertemplate="%{y}<br>%{legendgroup}: %{x:,.0f}",
                          cliponaxis=False)
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

# Treemap + Bubble
colA, colB = st.columns(2)
with colA:
    st.subheader("Purchase Share — Top Products (Treemap)")
    if not chart_df.empty:
        fig_tree = px.treemap(
            chart_df,
            path=[px.Constant("Top-N" if scope == "Top-N" else "All"), "Label"],
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
    bub = agg_view.copy()
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
# Highlights
# -------------------------------------------------
st.subheader("Highlights")
left, right = st.columns(
