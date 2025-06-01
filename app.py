import streamlit as st
import pandas as pd
import numpy as np
from strategy.data_collection import DataCollector
from strategy.signal_combiner import combine_signals

# Initialize DataCollector with Alpha Vantage API key
collector = DataCollector(api_key="ARUSM71L0ISPNU4F")

# Streamlit app
st.title("Trading Bot Dashboard")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")
symbol = st.sidebar.text_input("Symbol", "AAPL")
interval = st.sidebar.selectbox("Interval", ["1min", "5min", "15min", "30min", "60min", "daily"])
start_time = st.sidebar.date_input("Start Time", pd.Timestamp("2023-01-01"))
end_time = st.sidebar.date_input("End Time", pd.Timestamp("2023-01-31"))

# Fetch historical data
if st.sidebar.button("Fetch Data"):
    with st.spinner("Fetching data..."):
        ohlcv_data = collector.collect_ohlcv_data(symbol, interval, start_time, end_time)
        st.write("Historical Data:", ohlcv_data)

# Simulate signals from different strategies (placeholder)
signals = {
    'sentiment': 1,
    'statistical_arbitrage': 0,
    'market_microstructure': -1
}

# Combine signals using majority voting
final_signal = combine_signals(signals)
st.write("Final Signal:", final_signal)

# Display performance metrics (placeholder)
st.write("Performance Metrics:")
st.write("Sharpe Ratio: 1.5")
st.write("Max Drawdown: -0.1")
st.write("Total Return: 0.2") 