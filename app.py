import streamlit as st

st.set_page_config(page_title="Live Indian Stock Screener", page_icon="📈", layout="wide")

pg = st.navigation([
    st.Page("main_screener.py", title="Stock Screener", icon="📈", default=True),
    st.Page("pages/2_Next_Day_Momentum.py", title="Next-Day Momentum", icon="🚀"),
    st.Page("pages/1_Next_Day_Backtest.py", title="Next-Day Backtest", icon="🧪"),
])
pg.run()
