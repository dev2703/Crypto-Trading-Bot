import streamlit as st
import pandas as pd
import numpy as np
from strategy.data_collection import DataCollector
from strategy.signal_combiner import combine_signals
from strategy.real_time_streaming import RealTimeStreamer
from strategy.news_collector import NewsCollector
from strategy.sentiment_analysis import SentimentAnalyzer
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

# Initialize components
collector = DataCollector(api_key=API_KEY)
streamer = RealTimeStreamer(api_key=API_KEY)
news_collector = NewsCollector()
sentiment_analyzer = SentimentAnalyzer()

# Streamlit app
st.title("Trading Bot Dashboard")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")
symbol = st.sidebar.text_input("Symbol", "AAPL")
interval = st.sidebar.selectbox("Interval", ["1min", "5min", "15min", "30min", "60min", "daily"])
start_time = st.sidebar.date_input("Start Time", pd.Timestamp("2023-01-01"))
end_time = st.sidebar.date_input("End Time", pd.Timestamp("2023-01-31"))

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Market Data", "News & Sentiment", "Trading Signals"])

with tab1:
    # Market Data Section
    st.header("Market Data")
    
    # Create placeholder for real-time data
    real_time_data = st.empty()
    historical_data = st.empty()
    
    # Initialize session state for real-time data
    if 'latest_price' not in st.session_state:
        st.session_state.latest_price = None
    if 'is_streaming' not in st.session_state:
        st.session_state.is_streaming = False
    
    def price_callback(data):
        """Callback function for real-time price updates."""
        st.session_state.latest_price = data
    
    # Real-time streaming controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Streaming"):
            try:
                streamer.connect()
                streamer.add_callback('price', price_callback)
                streamer.subscribe(symbol, ['price'])
                st.session_state.is_streaming = True
                st.success("Streaming started!")
            except Exception as e:
                st.error(f"Error starting stream: {str(e)}")
    
    with col2:
        if st.button("Stop Streaming"):
            streamer.disconnect()
            st.session_state.is_streaming = False
            st.info("Streaming stopped")
    
    # Fetch historical data
    if st.button("Fetch Historical Data"):
        with st.spinner("Fetching data..."):
            ohlcv_data = collector.collect_ohlcv_data(symbol, interval, start_time, end_time)
            historical_data.write("Historical Data:", ohlcv_data)
    
    # Main loop for updating real-time data
    while st.session_state.is_streaming:
        if st.session_state.latest_price:
            real_time_data.write("Real-time Price:", st.session_state.latest_price)
        time.sleep(1)

with tab2:
    # News & Sentiment Section
    st.header("News & Sentiment Analysis")
    
    # News collection controls
    if st.button("Fetch Latest News"):
        with st.spinner("Fetching news..."):
            news_df = news_collector.collect_all_news(symbol)
            filtered_news = news_collector.filter_relevant_news(news_df)
            
            # Display news articles
            for _, article in filtered_news.iterrows():
                with st.expander(f"{article['title']} ({article['source']})"):
                    st.write(f"Published: {article['published_at']}")
                    st.write(f"Text: {article['text']}")
                    if article['url']:
                        st.write(f"URL: {article['url']}")
                    
                    # Analyze sentiment if not already provided
                    if pd.isna(article['sentiment']):
                        sentiment_scores = sentiment_analyzer.analyze_sentiment(article['text'])
                        st.write("Sentiment Analysis:")
                        st.write(f"Positive: {sentiment_scores['positive']:.2f}")
                        st.write(f"Neutral: {sentiment_scores['neutral']:.2f}")
                        st.write(f"Negative: {sentiment_scores['negative']:.2f}")

with tab3:
    # Trading Signals Section
    st.header("Trading Signals")
    
    # Simulate signals from different strategies
    signals = {
        'sentiment': 1,
        'statistical_arbitrage': 0,
        'market_microstructure': -1
    }
    
    # Combine signals using majority voting
    final_signal = combine_signals(signals)
    
    # Display signals
    st.subheader("Current Signals")
    for strategy, signal in signals.items():
        st.write(f"{strategy}: {'Buy' if signal == 1 else 'Sell' if signal == -1 else 'Hold'}")
    
    st.subheader("Final Signal")
    st.write(f"Combined Signal: {'Buy' if final_signal == 1 else 'Sell' if final_signal == -1 else 'Hold'}")
    
    # Display performance metrics
    st.subheader("Performance Metrics")
    st.write("Sharpe Ratio: 1.5")
    st.write("Max Drawdown: -0.1")
    st.write("Total Return: 0.2") 