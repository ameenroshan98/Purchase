import streamlit as st
import pandas as pd

# Replace these with your own values
SHEET_ID = "YOUR_SPREADSHEET_ID"
RANGE = "Sheet1!A1:D20"
API_KEY = "YOUR_API_KEY"

URL = f"https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}/values/{RANGE}?key={API_KEY}"

st.title("Google Sheets Dashboard")

try:
    data = pd.read_json(URL)
    # Google Sheets API returns {"range":..., "majorDimension":..., "values":[...]}
    values = data["values"].tolist() if "values" in data else None

    if values:
        df = pd.DataFrame(values[1:], columns=values[0])  # headers in first row
        st.dataframe(df)
    else:
        st.error("No data found in sheet")

except Exception as e:
    st.error(f"⚠️ Error: {e}")
