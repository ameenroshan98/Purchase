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
    if isinsta
