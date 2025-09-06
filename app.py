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
RANGE    = "Sheet1!A:G"  # pull ALL used rows in columns A..G (no hard row cap)
CSV_URL  = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

# Earliest available data + display format  (MM/DD/YYYY)
EARLIEST_AVAILABLE = pd.Timestamp(2024, 8, 1)   # 08/01/2024
DATE_FMT_DISPLAY   = "%m/%d/%Y"                 # mm/dd/yyyy

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

def parse_dates_force_mdy(raw_col: pd.Series):
    """
    Enforce MM/DD/YYYY semantics and handle:
      - '8/11/2024', '08-11-2024', '08.11.2024', '8/11/24'
      - ISO '2024-11-08'
      - Google/Excel serial numbers (e.g., 45678)
    Returns (parsed_Timestamps, original_strings)
    """
    s_raw = raw_col.astype(str)

    # Clean NBSP, trim, normalize separators
    s = (
        s_raw.str.replace("\u00A0", " ", regex=False)
             .str.strip()
             .str.replace(".", "/", regex=False)  # 08.11.2024 -> 08/11/2024
    )

    # Try month-first directly
    dt1 = pd.to_datetime(s, errors="coerce", dayfirst=False)

    # Retry after '-' -> '/'
    mask = dt1.isna()
    if mask.any():
        s2 = s.str.replace("-", "/", regex=False)
        dt2 = pd.to_datetime(s2, errors="coerce", dayfirst=False)
        dt1.loc[mask] = dt2.loc[mask]

    # Excel/Sheets serial numbers (vectorized)
    serial = pd.to_numeric(s, errors="coerce")
    serial_mask = serial.notna() & (dt1.isna())
    if serial_mask.any():
        dt_serial = pd.to_datetime(serial, unit="D", origin="1899-12-30", errors="coerce")
        dt1.loc[serial_mask] = dt_serial.loc[serial_mask]

    return dt1.dt.normalize(), s_raw

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
        # Fallback: entire tab via CSV
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
    # Tolerate header variants
    for c in list(df.columns):
        lc = c.lower().strip()
        if lc in ["qty purchased", "quantity purchased"]:
            rename_map[c] = "Qty Purchased"
    df = df.rename(columns=rename_map)

    # Parse Date (force MM/DD/YYYY semantics)
    if "Date" in df.columns:
        parsed, date_raw = parse_dates_force_mdy(df["Date"])
        df["_Date_raw"] = date_raw
        df["Date"] = parsed

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
st.caption("ℹ️ Data available from **08/01/2024** onward.")

if tx.empty or "Date" not in tx.columns:
    st.warning("No data found or 'Date' column missing. Check sharing, tab gid, or column names.")
    st.stop()

# ---- Diagnostics (how much got loaded + date parsing)
with st.expander("📊 Load diagnostics"):
    st.write({
        "Raw rows loaded": len(tx),
        "Rows with parseable Date": int(tx["Date"].notna().sum()),
        "Rows with missing/unparseable Date": int(tx["Date"].isna().sum()),
        "Earliest parsed date": str(tx["Date"].min()),
        "Latest parsed date": str(tx["Date"].max()),
        "Distinct suppliers": int(tx["Supplier Name"].nunique() if "Supplier Name" in tx else 0),
    })
parse_fail = tx["Date"].isna().sum()
if parse_fail > 0:
    bad = tx.loc[tx["Date"].isna(), "_Date_raw"].value_counts().head(10)
    with st.expander(f"⚠️ {parse_fail} rows have unparseable dates — examples"):
        st.write(bad)

# -------------------------------------------------
# Sidebar Filters (DATE + SUPPLIER dropdown + search + scope)
# -------------------------------------------------
with st.sidebar:
    st.header("Filters")

    # Date bounds (respect earliest available)
    data_min = tx_
