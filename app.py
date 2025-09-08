# app.py
import streamlit as st
st.set_page_config(page_title="Purchase Dashboard", layout="wide")  # must be first Streamlit call

# =============================
# Authentication (version-proof)
# =============================
import streamlit_authenticator as stauth

# ---- Secrets format expected (in Settings → Secrets):
# [cookie]
# name = "purchase_dash_auth"
# key = "your-long-random-secret"
# expiry_days = 14
#
# [credentials.usernames.admin]
# name = "Admin User"
# password = "$2b$12$BCRYPT_HASH..."
#
# [credentials.usernames.analyst]
# name = "Analyst"
# password = "$2b$12$BCRYPT_HASH..."

# Build credentials dict from st.secrets
_creds = {"usernames": {}}
for uname, u in st.secrets["credentials"]["usernames"].items():
    _creds["usernames"][uname] = {
        "name": str(u["name"]),
        "password": str(u["password"]),
    }

# Read cookie settings & coerce types
cookie_cfg = st.secrets.get("cookie", {})
cookie_name = str(cookie_cfg.get("name", "purchase_dash_auth"))
cookie_key = str(cookie_cfg.get("key", "change-this-secret"))
try:
    cookie_expiry_days = int(cookie_cfg.get("expiry_days", 14))
except Exception:
    cookie_expiry_days = 14


def build_authenticator(creds, cookie_name, cookie_key, expiry_days):
    """
    Try all supported Authenticate signatures across popular versions:
    1) credentials=..., cookie_name=..., key=..., cookie_expiry_days=...
    2) credentials=..., cookie_name=..., key=..., expiry_days=...
    3) (LEGACY positional) names, usernames, passwords, cookie_name, key, expiry_days
    """
    # 1) Newer style (most common)
    try:
        return stauth.Authenticate(
            credentials=creds,
            cookie_name=cookie_name,
            key=cookie_key,
            cookie_expiry_days=expiry_days,
        )
    except TypeError:
        pass

    # 2) Some versions use 'expiry_days' instead of 'cookie_expiry_days'
    try:
        return stauth.Authenticate(
            credentials=creds,
            cookie_name=cookie_name,
            key=cookie_key,
            expiry_days=expiry_days,
        )
    except TypeError:
        pass

    # 3) Very old legacy positional API
    names = [creds["usernames"][u]["name"] for u in creds["usernames"]]
    usernames = list(creds["usernames"].keys())
    passwords = [creds["usernames"][u]["password"] for u in creds["usernames"]]
    return stauth.Authenticate(
        names,
        usernames,
        passwords,
        cookie_name,
        cookie_key,
        expiry_days,
    )


_authenticator = build_authenticator(
    _creds, cookie_name, cookie_key, cookie_expiry_days
)

name, auth_status, username = _authenticator.login("Login", "main")

if auth_status is False:
    st.error("Username/password is incorrect.")
    st.stop()
elif auth_status is None:
    st.info("Please enter your username and password.")
    st.stop()

_authenticator.logout("Logout", "sidebar")
st.sidebar.caption(f"Signed in as **{name}**")

# =============================
# App proper
# =============================
import pandas as pd
import requests
from io import BytesIO
import plotly.express as px
import re

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

        if c < 100:
            c += 2000 if c < 70 else 1900

        if a > 12 and b <= 12:   # D/M/Y
            m, d, y = b, a, c
        elif b > 12 and a <= 12: # M/D/Y
            m, d, y = a, b, c
        elif a > 12 and b > 12:
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
# UI & Analytics (WITH DATE FILTER)
# -------------------------------------------------
tx = load_transactions(prefer="mdy")

st.title("📊 Purchase Dashboard")

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
    })
if "Date" in tx.columns and tx["Date"].isna().sum() > 0:
    bad = tx.loc[tx["Date"].isna(), "_Date_raw"].value_counts().head(10)
    with st.expander("⚠️ Examples of unparseable dates"):
        st.write(bad)

# Sidebar filters (now includes DATE RANGE)
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
            Last_Purchase=("Date", "max") if "Date" in tx_f.columns else ("Product","count")
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
agg = agg_base.merge(gap_series, on=["Code","Product"], how="left").merge(var_df, on=["Code","Product"], how="left")
agg["Bonus %"] = (agg["Bonus Received"] / agg["Qty Purchased"] * 100).replace([pd.NA, pd.NaT, float("inf")], 0).fillna(0).round(1)
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
BONUS_PROMO = 8.0      # %
BPR_PROMO   = 0.50     # share of orders with bonus
STALE_DAYS_MIN = 90
FACTOR_GAP     = 1.5

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
total_bonus = int(agg["Bonus Received"].sum())
overall_bonus_rate = (total_bonus / total_purchased * 100) if total_purchased > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Products (after filters)", f"{len(agg):,}")
c2.metric("Total Purchased", f"{total_purchased:,}")
c3.metric("Total Bonus", f"{total_bonus:,}")
c4.metric("Overall Bonus %", f"{overall_bonus_rate:.1f}%")

# ---- Status definitions (always visible, compact)
st.markdown(
    """
**Status definitions:**  
- 🟢 **Core** — Stable demand; buy steadily and negotiate base price.  
- 🟠 **Promo-timed** — Buy during bonus windows; avoid outside promos.  
- 🔴 **Review** — Dormant/anomalous; verify demand, data, or supplier terms.
"""
)

st.divider()

# Charts
chart_df = agg.sort_values("Qty Purchased", ascending=False)
if 'top_n' in locals() and top_n is not None:
    chart_df = chart_df.head(top_n)
chart_df["Label"] = chart_df.apply(lambda r: short_label(r["Code"], r["Product"]), axis=1)

top_qty   = chart_df["Qty Purchased"].sum()
top_bonus = chart_df["Bonus Received"].sum()
cov_qty   = (top_qty / total_purchased * 100) if total_purchased else 0
cov_bonus = (top_bonus / total_bonus * 100) if total_bonus else 0
st.caption(f"Bars cover {top_qty:,.0f} purchased ({cov_qty:.1f}% of total) "
           f"and {top_bonus:,.0f} bonus ({cov_bonus:.1f}% of total).")

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

colA, colB = st.columns(2)
with colA:
    st.subheader("Purchase Share — Top Products (Treemap)")
    if not chart_df.empty:
        fig_tree = px.treemap(
            chart_df,
            path=[px.Constant("Top-N" if ('top_n' in locals() and top_n is not None) else "All"), "Label"],
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
        # ---- Dynamic Y max: never exceed 500%, add ~15% headroom if small
        max_pct = float(pd.to_numeric(bub["Bonus %"], errors="coerce").max() or 0)
        y_upper = min(500.0, max(10.0, max_pct * 1.15))
        fig_bub.update_yaxes(range=[0, y_upper])
        fig_bub.add_hline(y=10, line_dash="dot", annotation_text="10% ref")
        fig_bub.update_layout(margin=dict(l=10, r=10, t=30, b=10),
                              xaxis_title="Qty Purchased", yaxis_title="Bonus %")
        st.plotly_chart(fig_bub, use_container_width=True)
    else:
        st.info("No rows for bubble chart.")

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

# Detailed Products table
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

# 🔎 SKU Drill-down (entire dataset; respects current filters)
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
        hist["Bonus % (tx)"] = ((pd.to_numeric(hist["Bonus"], errors="coerce") /
                                 pd.to_numeric(hist["Qty Purchased"], errors="coerce"))
                                .replace([float("inf"), -float("inf")], 0) * 100).fillna(0).round(1)

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

# Raw transactions (filtered table)
with st.expander("🧾 View all filtered transactions"):
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
