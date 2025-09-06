import streamlit as st

# Set up the page
st.set_page_config(page_title="Purchase Dashboard", layout="wide")

# App title
st.title("📊 Purchase Dashboard")

# Intro text
st.write("""
Welcome to your **Purchase Dashboard**!  
This app is deployed directly from your GitHub repository using **Streamlit Cloud**.  

You can extend it by:
- Connecting to your Google Sheet or database
- Adding KPIs, charts, and filters
- Building interactive purchase and bonus analysis
""")

# Placeholder metric cards
col1, col2, col3 = st.columns(3)
col1.metric("Total Purchased", "0")
col2.metric("Total Bonus", "0")
col3.metric("Products", "0")

st.success("✅ Dashboard is running successfully!")
