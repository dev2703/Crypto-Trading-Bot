import streamlit as st
import pandas as pd
import numpy as np
from strategy.data_collection import DataCollector
from strategy.signal_combiner import combine_signals
from strategy.real_time_streaming import RealTimeStreamer
import time
import threading
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
if not API_KEY:
    st.error("Please set the ALPHA_VANTAGE_API_KEY environment variable")
    st.stop()

# Initialize DataCollector and RealTimeStreamer
collector = DataCollector(api_key=API_KEY)
streamer = RealTimeStreamer(api_key=API_KEY)

# Streamlit app
st.title("Trading Bot Dashboard")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")
symbol = st.sidebar.text_input("Symbol", "AAPL")
interval = st.sidebar.selectbox("Interval", ["1min", "5min", "15min", "30min", "60min", "daily"])
start_time = st.sidebar.date_input("Start Time", pd.Timestamp("2023-01-01"))
end_time = st.sidebar.date_input("End Time", pd.Timestamp("2023-01-31"))

# Create placeholder for real-time data
real_time_data = st.empty()
historical_data = st.empty()
signals_display = st.empty()
metrics_display = st.empty()

# Initialize session state for real-time data
if 'latest_price' not in st.session_state:
    st.session_state.latest_price = None
if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False

def price_callback(data):
    """Callback function for real-time price updates."""
    st.session_state.latest_price = data

# Real-time streaming controls
st.sidebar.header("Real-time Controls")
if st.sidebar.button("Start Streaming"):
    try:
        streamer.connect()
        streamer.add_callback('price', price_callback)
        streamer.subscribe(symbol, ['price'])
        st.session_state.is_streaming = True
        st.sidebar.success("Streaming started!")
    except Exception as e:
        st.sidebar.error(f"Error starting stream: {str(e)}")

if st.sidebar.button("Stop Streaming"):
    streamer.disconnect()
    st.session_state.is_streaming = False
    st.sidebar.info("Streaming stopped")

# Fetch historical data
if st.sidebar.button("Fetch Historical Data"):
    with st.spinner("Fetching data..."):
        ohlcv_data = collector.collect_ohlcv_data(symbol, interval, start_time, end_time)
        historical_data.write("Historical Data:", ohlcv_data)

# Main loop for updating real-time data
while st.session_state.is_streaming:
    if st.session_state.latest_price:
        real_time_data.write("Real-time Price:", st.session_state.latest_price)
    time.sleep(1)

# Simulate signals from different strategies (placeholder)
signals = {
    'sentiment': 1,
    'statistical_arbitrage': 0,
    'market_microstructure': -1
}

# Combine signals using majority voting
final_signal = combine_signals(signals)
signals_display.write("Final Signal:", final_signal)

# Display performance metrics (placeholder)
metrics_display.write("Performance Metrics:")
metrics_display.write("Sharpe Ratio: 1.5")
metrics_display.write("Max Drawdown: -0.1")
metrics_display.write("Total Return: 0.2") 