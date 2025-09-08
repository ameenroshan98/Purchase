# pages/Compare_Yearly_Patterns.py
import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import plotly.express as px
import re

st.set_page_config(page_title="Range Compare — Purchase vs Bonus", layout="wide")

# --- Config
SHEET_ID = "1R7o4xKMeYWcYWwAOorMyYDtjg0-74FqDK0xAFKN6Iuo"
XLSX_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=xlsx"
CSV_URL  = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=0"
DATE_FMT_DISPLAY = "%m/%d/%Y"

# Fixed month order for x-axes
MONTHS_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------------- Helpers ----------------
def parse_dates_smart(raw_col: pd.Series, prefer: str = "mdy"):
    s_raw = raw_col.astype(str)
    s = (s_raw
         .str.replace("\u00A0", " ", regex=False)
         .str.strip()
         .str.replace(".", "/", regex=False))
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    serial = pd.to_numeric(s, errors="coerce")
    serial_mask = serial.notna()
    if serial_mask.any():
        out.loc[serial_mask] = pd.to_datetime(
            serial.loc[serial_mask], unit="D", origin="1899-12-30", errors="coerce"
        )

    mask = ~serial_mask
    s2 = s[mask].str.replace("-", "/", regex=False)

    def parse_one(val: str):
        if not val:
            return pd.NaT
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
        if c < 100:
            c += 2000 if c < 70 else 1900
        if a > 12 and b <= 12:   # D/M/Y
            m, d, y = b, a, c
        elif b > 12 and a <= 12: # M/D/Y
            m, d, y = a, b, c
        elif a > 12 and b > 12:
            return pd.NaT
        else:
            m, d, y = (a, b, c) if prefer.lower() != "dmy" else (b, a, c)
        try:
            return pd.Timestamp(year=y, month=m, day=d).normalize()
        except Exception:
            return pd.NaT

    parsed = s2.apply(parse_one)
    out.loc[mask] = parsed.values
    return out, s_raw

@st.cache_data(ttl=900)
def load_all(prefer="mdy"):
    """Load entire workbook (xlsx) or fall back to first tab CSV if openpyxl isn't available."""
    try:
        r = requests.get(XLSX_URL, timeout=60)
        r.raise_for_status()
        with BytesIO(r.content) as bio:
            book = pd.read_excel(bio, sheet_name=None, dtype=str)  # needs openpyxl
        frames = []
        for sheet_name, df in book.items():
            if df is None or df.empty:
                continue
            df = df.dropna(how="all")
            df.columns = pd.Index(df.columns).astype(str).str.replace("\u00A0", " ", regex=False).str.strip()
            df["__sheet__"] = sheet_name
            frames.append(df)
        df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        fallback = False
    except Exception:
        df_all = pd.read_csv(CSV_URL, dtype=str).fillna("")
        df_all.columns = pd.Index(df_all.columns).astype(str).str.replace("\u00A0", " ", regex=False).str.strip()
        df_all["__sheet__"] = "gid_0"
        fallback = True

    if df_all.empty:
        return df_all, fallback

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
    for c in list(df_all.columns):
        lc = c.lower().strip()
        if lc in ["qty purchased", "quantity purchased", "qty"]:
            rename_map[c] = "Qty Purchased"
    df_all = df_all.rename(columns=rename_map)

    if "Date" in df_all.columns:
        parsed, raw = parse_dates_smart(df_all["Date"], prefer=prefer)
        df_all["_Date_raw"] = raw
        df_all["Date"] = parsed

    for c in ["Qty Purchased", "Bonus"]:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c].astype(str).str.replace(",", "", regex=False),
                                      errors="coerce").fillna(0)

    if "Bonus %" in df_all.columns:
        df_all["Bonus %"] = (df_all["Bonus %"].astype(str)
                             .str.replace("%", "", regex=False)
                             .str.replace(",", "", regex=False))
        df_all["Bonus %"] = pd.to_numeric(df_all["Bonus %"], errors="coerce").fillna(0.0)

    return df_all, fallback

# --------- Range utilities ----------
def month_agg(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Filter df by [start_ts, end_ts], group monthly; include MonthLabel ('Jan'..'Dec')."""
    msk = df["Date"].between(start_ts, end_ts, inclusive="both")
    d = df.loc[msk].copy()
    if d.empty:
        return pd.DataFrame({
            "MonthIndex": [], "MonthStart": [], "MonthLabel": [],
            "Qty_Purchased": [], "Bonus_Received": [], "Bonus %": []
        })

    d["MonthStart"] = d["Date"].values.astype("datetime64[M]")
    g = (d.groupby("MonthStart", as_index=False)
           .agg(Qty_Purchased=("Qty Purchased", "sum"),
                Bonus_Received=("Bonus", "sum")))
    g["Bonus %"] = (g["Bonus_Received"] / g["Qty_Purchased"] * 100).replace([float("inf"), -float("inf")], 0).fillna(0.0)
    g = g.sort_values("MonthStart").reset_index(drop=True)
    g["MonthIndex"] = g.index + 1
    g["MonthLabel"] = g["MonthStart"].dt.strftime("%b")
    return g[["MonthIndex", "MonthStart", "MonthLabel", "Qty_Purchased", "Bonus_Received", "Bonus %"]]

def totals(df_month: pd.DataFrame):
    t_qty = int(df_month["Qty_Purchased"].sum()) if not df_month.empty else 0
    t_bonus = int(df_month["Bonus_Received"].sum()) if not df_month.empty else 0
    t_pct = (t_bonus / t_qty * 100) if t_qty > 0 else 0.0
    return t_qty, t_bonus, t_pct

def sku_agg_range(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Per-SKU totals (Qty, Bonus, Bonus %) in the given date range."""
    msk = df["Date"].between(start_ts, end_ts, inclusive="both")
    d = df.loc[msk].copy()
    if d.empty:
        return pd.DataFrame(columns=["Code", "Product", "Qty", "Bonus", "Bonus %"])
    g = (d.groupby(["Code", "Product"], as_index=False)
           .agg(Qty=("Qty Purchased", "sum"),
                Bonus=("Bonus", "sum")))
    g["Bonus %"] = (g["Bonus"] / g["Qty"] * 100).replace([float("inf"), -float("inf")], 0).fillna(0.0)
    return g

# ------------------------------ UI ------------------------------
st.title("🗓️ Year/Range Comparison — Purchase vs Bonus")

tx, used_fallback = load_all(prefer="mdy")
if used_fallback:
    st.warning("Workbook read fell back to first tab CSV (install `openpyxl` to load all tabs).")
if tx.empty or "Code" not in tx.columns or "Product" not in tx.columns:
    st.warning("No data or missing required columns (Code/Product).")
    st.stop()
if "Date" not in tx.columns or tx["Date"].notna().sum() == 0:
    st.info("No valid dates found. This page requires a Date column.")
    st.stop()

# Sidebar filters (supplier/search + optional SKU)
with st.sidebar:
    st.header("Filters")
    sup_series = tx.get("Supplier Name", pd.Series("", index=tx.index)).astype(str).str.strip()
    supplier_options = sorted([s for s in sup_series.unique() if s])
    selected_supplier = st.selectbox("Supplier", ["All suppliers"] + supplier_options, index=0)
    q = st.text_input("Search (code or product)", "")
    _ = st.checkbox("Include rows with missing Date (outside ranges)", value=False)  # informational

# Apply base mask first
mask = pd.Series(True, index=tx.index)
if selected_supplier != "All suppliers":
    mask &= (sup_series == selected_supplier)
if q.strip():
    ql = q.strip().lower()
    mask &= (
        tx.get("Product", pd.Series("", index=tx.index)).astype(str).str.lower().str.contains(ql, na=False)
        | tx.get("Code", pd.Series("", index=tx.index)).astype(str).str.contains(ql, na=False)
    )
tx_f = tx[mask & tx["Date"].notna()].copy()  # comparison needs valid dates
if tx_f.empty:
    st.info("No transactions match your supplier/search filters.")
    st.stop()

# ---- Compute safe defaults & render date pickers ----
dmin = tx_f["Date"].min().normalize()
dmax = tx_f["Date"].max().normalize()

latest_year = int(dmax.year)
prev_year   = latest_year - 1

def jan1(y):  return pd.Timestamp(y, 1, 1)
def dec31(y): return pd.Timestamp(y, 12, 31)

def clamp_range(start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                lo: pd.Timestamp, hi: pd.Timestamp):
    """Clamp [start_ts, end_ts] to [lo, hi]. If empty after clamp, fallback to [lo, hi]."""
    s = max(start_ts.normalize(), lo)
    e = min(end_ts.normalize(), hi)
    if s > e:
        s, e = lo, hi
    return s, e

raw_a = (jan1(prev_year),  dec31(prev_year))
raw_b = (jan1(latest_year), dec31(latest_year))

default_a_start, default_a_end = clamp_range(raw_a[0], raw_a[1], dmin, dmax)
default_b_start, default_b_end = clamp_range(raw_b[0], raw_b[1], dmin, dmax)

colA, colB = st.columns(2)
with colA:
    st.subheader("Range A")
    start_a_date, end_a_date = st.date_input(
        "Start / End (A)",
        value=(default_a_start.date(), default_a_end.date()),
        min_value=dmin.date(),
        max_value=dmax.date(),
        format="MM/DD/YYYY",
    )
with colB:
    st.subheader("Range B")
    start_b_date, end_b_date = st.date_input(
        "Start / End (B)",
        value=(default_b_start.date(), default_b_end.date()),
        min_value=dmin.date(),
        max_value=dmax.date(),
        format="MM/DD/YYYY",
    )

# Convert to timestamps & guard against inverted ranges
start_ts_a, end_ts_a = pd.Timestamp(start_a_date).normalize(), pd.Timestamp(end_a_date).normalize()
start_ts_b, end_ts_b = pd.Timestamp(start_b_date).normalize(), pd.Timestamp(end_b_date).normalize()
if start_ts_a > end_ts_a:
    start_ts_a, end_ts_a = end_ts_a, start_ts_a
if start_ts_b > end_ts_b:
    start_ts_b, end_ts_b = end_ts_b, start_ts_b

# Optional SKU picker AFTER we know the slice universe
sku_series = tx_f["Code"].astype(str) + " — " + tx_f["Product"].astype(str)
sku_list = ["(All filtered SKUs)"] + sorted(sku_series.unique().tolist())
with st.sidebar:
    selected_sku = st.selectbox("SKU (optional)", sku_list, index=0)

# Min combined quantity filter for the movers tables
with st.sidebar:
    overall_qty = int(tx_f.groupby(["Code","Product"])["Qty Purchased"].sum().max() or 0)
    slider_max = max(overall_qty, 1000)
    min_combined_qty = st.slider("Min combined Qty (A+B) for tables", 0, slider_max, 100, step=50)

if selected_sku != "(All filtered SKUs)":
    code_part = selected_sku.split(" — ", 1)[0]
    prod_part = selected_sku.split(" — ", 1)[1] if " — " in selected_sku else None
    sku_mask = tx_f["Code"].astype(str).eq(code_part)
    if prod_part:
        sku_mask &= tx_f["Product"].astype(str).eq(prod_part)
    tx_f = tx_f[sku_mask].copy()
    if tx_f.empty:
        st.info("No rows for the chosen SKU in the selected filters.")
        st.stop()

# Build month aggregations for both ranges
m_a = month_agg(tx_f, start_ts_a, end_ts_a)
m_b = month_agg(tx_f, start_ts_b, end_ts_b)

tqty_a, tbon_a, tpct_a = totals(m_a)
tqty_b, tbon_b, tpct_b = totals(m_b)

# KPI row
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric(f"Qty A ({start_ts_a.strftime(DATE_FMT_DISPLAY)} → {end_ts_a.strftime(DATE_FMT_DISPLAY)})", f"{tqty_a:,}")
k2.metric("Bonus A", f"{tbon_a:,}")
k3.metric("Bonus % A", f"{tpct_a:.1f}%")
k4.metric(f"Qty B ({start_ts_b.strftime(DATE_FMT_DISPLAY)} → {end_ts_b.strftime(DATE_FMT_DISPLAY)})",
          f"{tqty_b:,}", delta=f"{tqty_b - tqty_a:+,}")
k5.metric("Bonus B", f"{tbon_b:,}", delta=f"{tbon_b - tbon_a:+,}")
k6.metric("Bonus % B", f"{tpct_b:.1f}%", delta=f"{tpct_b - tpct_a:+.1f} pp")

st.divider()

# Charts — x = Month name in fixed calendar order
c1, c2 = st.columns(2)
with c1:
    st.subheader("Monthly Qty Purchased")
    dd_q = pd.concat([m_a.assign(Range="A"), m_b.assign(Range="B")], ignore_index=True)
    fig_q = px.line(
        dd_q, x="MonthLabel", y="Qty_Purchased", color="Range",
        markers=True, hover_data={"MonthLabel": True, "MonthStart": True},
        category_orders={"MonthLabel": MONTHS_ABBR}
    )
    fig_q.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Month")
    st.plotly_chart(fig_q, use_container_width=True)

with c2:
    st.subheader("Monthly Bonus")
    fig_b = px.line(
        dd_q, x="MonthLabel", y="Bonus_Received", color="Range",
        markers=True, hover_data={"MonthLabel": True, "MonthStart": True},
        category_orders={"MonthLabel": MONTHS_ABBR}
    )
    fig_b.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Month")
    st.plotly_chart(fig_b, use_container_width=True)

st.subheader("Monthly Bonus %")
dd_bp = pd.concat([m_a.assign(Range="A"), m_b.assign(Range="B")], ignore_index=True)
fig_bp = px.line(
    dd_bp, x="MonthLabel", y="Bonus %", color="Range", markers=True,
    hover_data={"MonthLabel": True, "MonthStart": True},
    category_orders={"MonthLabel": MONTHS_ABBR}
)
fig_bp.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Month")
st.plotly_chart(fig_bp, use_container_width=True)

st.divider()

# ===================== SKU Bonus % Movers =====================

# Per-SKU aggregates for each range
a_sku = sku_agg_range(tx_f, start_ts_a, end_ts_a).rename(
    columns={"Qty": "Qty A", "Bonus": "Bonus A", "Bonus %": "Bonus % A"}
)
b_sku = sku_agg_range(tx_f, start_ts_b, end_ts_b).rename(
    columns={"Qty": "Qty B", "Bonus": "Bonus B", "Bonus %": "Bonus % B"}
)

# Merge WITHOUT filling, then require presence in both periods
comp = a_sku.merge(b_sku, on=["Code", "Product"], how="outer")
both_mask = comp["Qty A"].notna() & comp["Qty B"].notna()
comp = comp.loc[both_mask].copy()

# Now fill NaNs for safe arithmetic
for c in ["Qty A","Bonus A","Bonus % A","Qty B","Bonus B","Bonus % B"]:
    if c in comp.columns:
        comp[c] = pd.to_numeric(comp[c], errors="coerce").fillna(0)

comp["Total Qty (A+B)"] = (comp["Qty A"] + comp["Qty B"]).astype(int)
comp["Δ Bonus % (pp)"] = (comp["Bonus % B"] - comp["Bonus % A"]).astype(float).round(1)

# Apply minimum combined quantity to reduce noise
min_combined_qty = int('min_combined_qty' in locals() and min_combined_qty or 100)
comp_f = comp[comp["Total Qty (A+B)"] >= min_combined_qty].copy()

# Separate by sign to avoid identical lists when deltas ~ 0
increases = comp_f[comp_f["Δ Bonus % (pp)"] > 0].sort_values("Δ Bonus % (pp)", ascending=False).head(25)
decreases = comp_f[comp_f["Δ Bonus % (pp)"] < 0].sort_values("Δ Bonus % (pp)", ascending=True).head(25)

st.subheader("🔼 SKUs with Bonus % Increased (Top 25)")
if increases.empty:
    st.info("No SKUs show an increase in Bonus % (present in both periods) under the current filters.")
else:
    st.dataframe(
        increases[[
            "Code","Product",
            "Qty A","Bonus A","Bonus % A",
            "Qty B","Bonus B","Bonus % B",
            "Total Qty (A+B)","Δ Bonus % (pp)"
        ]],
        use_container_width=True, hide_index=True,
        column_config={
            "Qty A": st.column_config.NumberColumn(format="%,d"),
            "Bonus A": st.column_config.NumberColumn(format="%,d"),
            "Bonus % A": st.column_config.NumberColumn(format="%.1f"),
            "Qty B": st.column_config.NumberColumn(format="%,d"),
            "Bonus B": st.column_config.NumberColumn(format="%,d"),
            "Bonus % B": st.column_config.NumberColumn(format="%.1f"),
            "Total Qty (A+B)": st.column_config.NumberColumn(format="%,d"),
            "Δ Bonus % (pp)": st.column_config.NumberColumn(format="%.1f"),
        },
    )

st.subheader("🔻 SKUs with Bonus % Decreased (Top 25)")
if decreases.empty:
    st.info("No SKUs show a decrease in Bonus % (present in both periods) under the current filters.")
else:
    st.dataframe(
        decreases[[
            "Code","Product",
            "Qty A","Bonus A","Bonus % A",
            "Qty B","Bonus B","Bonus % B",
            "Total Qty (A+B)","Δ Bonus % (pp)"
        ]],
        use_container_width=True, hide_index=True,
        column_config={
            "Qty A": st.column_config.NumberColumn(format="%,d"),
            "Bonus A": st.column_config.NumberColumn(format="%,d"),
            "Bonus % A": st.column_config.NumberColumn(format="%.1f"),
            "Qty B": st.column_config.NumberColumn(format="%,d"),
            "Bonus B": st.column_config.NumberColumn(format="%,d"),
            "Bonus % B": st.column_config.NumberColumn(format="%.1f"),
            "Total Qty (A+B)": st.column_config.NumberColumn(format="%,d"),
            "Δ Bonus % (pp)": st.column_config.NumberColumn(format="%.1f"),
        },
    )

st.divider()

# Quick link back to the main dashboard (adjust filename if needed)
try:
    st.page_link("app.py", label="⬅️ Back to Dashboard", icon="🏠")
except Exception:
    st.caption("Use the sidebar pages menu or browser Back to return to the dashboard.")
