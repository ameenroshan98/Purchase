# pages/Compare_Yearly_Patterns.py
import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import plotly.express as px
import re

st.set_page_config(page_title="YoY Compare — Purchase vs Bonus", layout="wide")

# --- Shared config (same as main)
SHEET_ID = "1R7o4xKMeYWcYWwAOorMyYDtjg0-74FqDK0xAFKN6Iuo"
XLSX_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=xlsx"
DATE_FMT_DISPLAY = "%m/%d/%Y"

# ---------------- Helpers (copied from main for self-containment) ----------------
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
            # prefer MDY on ambiguous
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

    if df_all.empty:
        return df_all

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

    return df_all

# ------------------------------ UI ------------------------------
st.title("🗓️ Year-over-Year Comparison — Purchase vs Bonus")

tx = load_all(prefer="mdy")
if tx.empty or "Code" not in tx.columns or "Product" not in tx.columns:
    st.warning("No data or missing required columns (Code/Product).")
    st.stop()

if "Date" not in tx.columns or tx["Date"].notna().sum() == 0:
    st.info("No valid dates found. This page requires a Date column.")
    st.stop()

tx["Year"] = tx["Date"].dt.year
tx["Month"] = tx["Date"].dt.month

# Sidebar filters
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

    # Optional SKU pick for focused compare
    # Build list after supplier/search filters are applied below
    st.divider()
    st.caption("Optional: lock to a single SKU for focused YoY view.")

# Apply base mask (supplier + search)
mask = pd.Series(True, index=tx.index)
if selected_supplier != "All suppliers":
    mask &= (sup_series == selected_supplier)
if q.strip():
    ql = q.strip().lower()
    mask &= (
        tx.get("Product", pd.Series("", index=tx.index)).astype(str).str.lower().str.contains(ql, na=False)
        | tx.get("Code", pd.Series("", index=tx.index)).astype(str).str.contains(ql, na=False)
    )

tx_f = tx[mask].copy()
if tx_f.empty:
    st.info("No transactions match your supplier/search filters.")
    st.stop()

# SKU list (after filter)
sku_series = tx_f["Code"].astype(str) + " — " + tx_f["Product"].astype(str)
sku_list = ["(All filtered SKUs)"] + sorted(sku_series.unique().tolist())

with st.sidebar:
    selected_sku = st.selectbox("SKU", sku_list, index=0)

if selected_sku != "(All filtered SKUs)":
    # parse "Code — Product"
    code_part = selected_sku.split(" — ", 1)[0]
    prod_part = selected_sku.split(" — ", 1)[1] if " — " in selected_sku else None
    mask2 = tx_f["Code"].astype(str).eq(code_part)
    if prod_part:
        mask2 &= tx_f["Product"].astype(str).eq(prod_part)
    tx_f = tx_f[mask2].copy()

# Year choices
years = sorted(tx_f["Year"].dropna().unique())
if len(years) < 2:
    st.info("Need at least two years of data to compare.")
    st.stop()

col_yr1, col_yr2 = st.columns(2)
with col_yr1:
    yr_a = st.selectbox("Year A", options=years, index=len(years)-1)
with col_yr2:
    # default to previous year if available
    default_b = years.index(yr_a)-1 if yr_a in years and years.index(yr_a) > 0 else 0
    yr_b = st.selectbox("Year B", options=[y for y in years if y != yr_a], index=default_b)

# Aggregate monthly for each year
def monthly_agg(df):
    g = (df.groupby(["Year", "Month"], as_index=False)
           .agg(Qty_Purchased=("Qty Purchased", "sum"),
                Bonus_Received=("Bonus", "sum")))
    g["Bonus %"] = (g["Bonus_Received"] / g["Qty_Purchased"] * 100).replace([pd.NA, float("inf")], 0).fillna(0.0)
    return g

m = monthly_agg(tx_f)
m_a = m[m["Year"] == yr_a].set_index("Month").reindex(range(1,13)).reset_index().fillna(0)
m_b = m[m["Year"] == yr_b].set_index("Month").reindex(range(1,13)).reset_index().fillna(0)

# KPIs
def totals_row(df, year):
    t_qty = int(df.loc[df["Year"]==year, "Qty_Purchased"].sum())
    t_bonus = int(df.loc[df["Year"]==year, "Bonus_Received"].sum())
    t_bpct = (t_bonus / t_qty * 100) if t_qty > 0 else 0
    return t_qty, t_bonus, t_bpct

tqty_a, tbon_a, tbp_a = totals_row(m, yr_a)
tqty_b, tbon_b, tbp_b = totals_row(m, yr_b)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric(f"Qty {yr_a}", f"{tqty_a:,}")
k2.metric(f"Bonus {yr_a}", f"{tbon_a:,}")
k3.metric(f"Bonus % {yr_a}", f"{tbp_a:.1f}%")
k4.metric(f"Qty {yr_b}", f"{tqty_b:,}", delta=f"{tqty_b - tqty_a:+,}")
k5.metric(f"Bonus {yr_b}", f"{tbon_b:,}", delta=f"{tbon_b - tbon_a:+,}")
k6.metric(f"Bonus % {yr_b}", f"{tbp_b:.1f}%", delta=f"{tbp_b - tbp_a:+.1f} pp")

st.divider()

# Charts — side-by-side lines for Qty and Bonus
c1, c2 = st.columns(2)
with c1:
    st.subheader("Monthly Qty Purchased")
    dd = pd.concat([
        m_a.assign(Year=str(yr_a)),
        m_b.assign(Year=str(yr_b)),
    ], ignore_index=True)
    fig_q = px.line(dd, x="Month", y="Qty_Purchased", color="Year",
                    markers=True)
    fig_q.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(tickmode="linear", tick0=1, dtick=1))
    st.plotly_chart(fig_q, use_container_width=True)

with c2:
    st.subheader("Monthly Bonus")
    fig_b = px.line(dd, x="Month", y="Bonus_Received", color="Year",
                    markers=True)
    fig_b.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(tickmode="linear", tick0=1, dtick=1))
    st.plotly_chart(fig_b, use_container_width=True)

st.subheader("Monthly Bonus %")
dd_bp = pd.concat([
    m_a.assign(Year=str(yr_a)),
    m_b.assign(Year=str(yr_b)),
], ignore_index=True)
fig_bp = px.line(dd_bp, x="Month", y="Bonus %", color="Year", markers=True)
fig_bp.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                     xaxis=dict(tickmode="linear", tick0=1, dtick=1))
st.plotly_chart(fig_bp, use_container_width=True)

st.divider()

# Bars — annual totals
st.subheader("Annual Totals Comparison")
tot_df = pd.DataFrame({
    "Year": [str(yr_a), str(yr_b)],
    "Qty Purchased": [tqty_a, tqty_b],
    "Bonus Received": [tbon_a, tbon_b],
    "Bonus %": [tbp_a, tbp_b],
})
fig_tot_q = px.bar(tot_df, x="Year", y="Qty Purchased", text="Qty Purchased")
fig_tot_q.update_traces(texttemplate="%{text:,}", textposition="outside")
fig_tot_q.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_tot_q, use_container_width=True)

fig_tot_b = px.bar(tot_df, x="Year", y="Bonus Received", text="Bonus Received")
fig_tot_b.update_traces(texttemplate="%{text:,}", textposition="outside")
fig_tot_b.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_tot_b, use_container_width=True)

st.divider()

# Table — month-by-month side-by-side
st.subheader("Month-by-Month (Side-by-Side)")

table = pd.DataFrame({
    "Month": range(1, 13),
    f"Qty {yr_a}": m_a["Qty_Purchased"].astype(int),
    f"Qty {yr_b}": m_b["Qty_Purchased"].astype(int),
    f"Bonus {yr_a}": m_a["Bonus_Received"].astype(int),
    f"Bonus {yr_b}": m_b["Bonus_Received"].astype(int),
    f"Bonus % {yr_a}": (m_a["Bonus %"]).round(1),
    f"Bonus % {yr_b}": (m_b["Bonus %"]).round(1),
    "Δ Qty": (m_b["Qty_Purchased"] - m_a["Qty_Purchased"]).astype(int),
    "Δ Bonus": (m_b["Bonus_Received"] - m_a["Bonus_Received"]).astype(int),
    "Δ Bonus % (pp)": (m_b["Bonus %"] - m_a["Bonus %"]).round(1),
})
st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    column_config={
        f"Qty {yr_a}": st.column_config.NumberColumn(format="%,d"),
        f"Qty {yr_b}": st.column_config.NumberColumn(format="%,d"),
        f"Bonus {yr_a}": st.column_config.NumberColumn(format="%,d"),
        f"Bonus {yr_b}": st.column_config.NumberColumn(format="%,d"),
        f"Bonus % {yr_a}": st.column_config.NumberColumn(format="%.1f"),
        f"Bonus % {yr_b}": st.column_config.NumberColumn(format="%.1f"),
        "Δ Qty": st.column_config.NumberColumn(format="%,d"),
        "Δ Bonus": st.column_config.NumberColumn(format="%,d"),
        "Δ Bonus % (pp)": st.column_config.NumberColumn(format="+.1f"),
    }
)

st.divider()
# Quick link back to the main dashboard (adjust if your main file name differs)
try:
    st.page_link("app.py", label="⬅️ Back to Dashboard", icon="🏠")
except Exception:
    st.caption("Use the sidebar pages menu or browser Back to return to the dashboard.")
