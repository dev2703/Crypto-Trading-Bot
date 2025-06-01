import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from strategy.data_collection import DataCollector
from strategy.signal_combiner import combine_signals
from strategy.real_time_streaming import RealTimeStreamer
from strategy.news_collector import NewsCollector
from strategy.sentiment_analysis import SentimentAnalyzer
from strategy.technical_analysis import TechnicalAnalyzer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize components
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
if not API_KEY:
    st.error("Please set the ALPHA_VANTAGE_API_KEY environment variable")
    st.stop()

collector = DataCollector(api_key=API_KEY)
streamer = RealTimeStreamer(api_key=API_KEY)
news_collector = NewsCollector()
sentiment_analyzer = SentimentAnalyzer()
technical_analyzer = TechnicalAnalyzer()

# Page config
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide"
)

# Sidebar
st.sidebar.title("Trading Bot Controls")

# Input parameters
symbol = st.sidebar.text_input("Symbol", "AAPL")
interval = st.sidebar.selectbox(
    "Interval",
    ["1min", "5min", "15min", "30min", "60min", "daily"]
)
start_date = st.sidebar.date_input(
    "Start Date",
    datetime.now() - timedelta(days=30)
)
end_date = st.sidebar.date_input(
    "End Date",
    datetime.now()
)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Market Data", "News & Sentiment", "Trading Signals"])

with tab1:
    st.header("Market Data")
    
    # Add error handling for API key
    if not API_KEY:
        st.error("Alpha Vantage API key is not set. Please add it to your .env file.")
        st.stop()
    
    # Add date range validation
    if end_date < start_date:
        st.error("End date must be after start date")
        st.stop()
    
    # Add date range limit warning
    date_range = (end_date - start_date).days
    if date_range > 30:
        st.warning("Large date ranges may be limited by Alpha Vantage API. Consider using a smaller range.")
    
    # Fetch historical data
    if st.button("Fetch Historical Data"):
        with st.spinner("Fetching historical data..."):
            try:
                # Add progress bar
                progress_bar = st.progress(0)
                
                # Fetch data
                df = collector.collect_ohlcv_data(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_date.strftime("%Y-%m-%d"),
                    end_time=end_date.strftime("%Y-%m-%d")
                )
                progress_bar.progress(50)
                
                if df.empty:
                    st.error("No data returned for the selected date range. Try adjusting the dates.")
                    st.stop()
                
                # Create candlestick chart
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close']
                )])
                
                fig.update_layout(
                    title=f"{symbol} Price Chart",
                    yaxis_title="Price",
                    xaxis_title="Date",
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                progress_bar.progress(75)
                
                # Display statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${df['close'].iloc[-1]:.2f}")
                with col2:
                    price_change = df['close'].iloc[-1] - df['close'].iloc[-2]
                    st.metric("24h Change", f"${price_change:.2f}", f"{price_change:.2f}%")
                with col3:
                    st.metric("24h High", f"${df['high'].iloc[-1]:.2f}")
                with col4:
                    st.metric("24h Low", f"${df['low'].iloc[-1]:.2f}")
                
                # Display raw data
                st.subheader("Raw Data")
                st.dataframe(df)
                
                progress_bar.progress(100)
                st.success("Data fetched successfully!")
                
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                st.info("Common issues:\n"
                       "1. API key not set or invalid\n"
                       "2. Rate limit exceeded (5 calls/minute, 500 calls/day)\n"
                       "3. Invalid date range\n"
                       "4. Invalid symbol")

with tab2:
    st.header("News & Sentiment Analysis")
    
    # News collection controls
    if st.button("Fetch Latest News"):
        with st.spinner("Fetching news..."):
            try:
                news_df = news_collector.collect_all_news(symbol)
                filtered_news = news_collector.filter_relevant_news(news_df)
                
                # Display news articles
                for _, article in filtered_news.iterrows():
                    with st.expander(f"{article['title']} ({article['source']})"):
                        st.write(f"Published: {article['published_at']}")
                        st.write(f"Text: {article['text']}")
                        if article['url']:
                            st.write(f"URL: {article['url']}")
                        
                        # Display sentiment if available
                        if article['sentiment'] is not None:
                            sentiment = article['sentiment']
                            sentiment_color = "green" if sentiment > 0 else "red" if sentiment < 0 else "gray"
                            st.markdown(f"Sentiment: <span style='color:{sentiment_color}'>{sentiment:.2f}</span>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error fetching news: {str(e)}")

with tab3:
    st.header("Trading Signals")
    
    # Generate trading signals
    if st.button("Generate Trading Signals"):
        with st.spinner("Generating signals..."):
            try:
                # Fetch historical data for technical analysis
                df = collector.collect_ohlcv_data(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_date.strftime("%Y-%m-%d"),
                    end_time=end_date.strftime("%Y-%m-%d")
                )
                
                if df.empty:
                    st.error("No data available for signal generation")
                    st.stop()
                
                # Generate technical signals
                technical_signals = technical_analyzer.generate_signals(df)
                
                # Calculate indicators for display
                rsi = technical_analyzer.calculate_rsi(df)
                macd_line, signal_line, histogram = technical_analyzer.calculate_macd(df)
                short_ma, long_ma = technical_analyzer.calculate_moving_averages(df)
                middle_band, upper_band, lower_band = technical_analyzer.calculate_bollinger_bands(df)
                
                # Display technical indicators
                st.subheader("Technical Indicators")
                
                # RSI Chart
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI'))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(title="RSI", height=300)
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # MACD Chart
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df.index, y=macd_line, name='MACD'))
                fig_macd.add_trace(go.Scatter(x=df.index, y=signal_line, name='Signal'))
                fig_macd.add_trace(go.Bar(x=df.index, y=histogram, name='Histogram'))
                fig_macd.update_layout(title="MACD", height=300)
                st.plotly_chart(fig_macd, use_container_width=True)
                
                # Moving Averages Chart
                fig_ma = go.Figure()
                fig_ma.add_trace(go.Scatter(x=df.index, y=df['close'], name='Price'))
                fig_ma.add_trace(go.Scatter(x=df.index, y=short_ma, name='20-day MA'))
                fig_ma.add_trace(go.Scatter(x=df.index, y=long_ma, name='50-day MA'))
                fig_ma.update_layout(title="Moving Averages", height=300)
                st.plotly_chart(fig_ma, use_container_width=True)
                
                # Bollinger Bands Chart
                fig_bb = go.Figure()
                fig_bb.add_trace(go.Scatter(x=df.index, y=df['close'], name='Price'))
                fig_bb.add_trace(go.Scatter(x=df.index, y=upper_band, name='Upper Band', line=dict(dash='dash')))
                fig_bb.add_trace(go.Scatter(x=df.index, y=middle_band, name='Middle Band'))
                fig_bb.add_trace(go.Scatter(x=df.index, y=lower_band, name='Lower Band', line=dict(dash='dash')))
                fig_bb.update_layout(title="Bollinger Bands", height=300)
                st.plotly_chart(fig_bb, use_container_width=True)
                
                # Display signals
                st.subheader("Trading Signals")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    rsi_signal = technical_signals.get('rsi', 0)
                    rsi_color = "green" if rsi_signal > 0 else "red" if rsi_signal < 0 else "gray"
                    st.metric("RSI Signal", rsi_signal, delta_color=rsi_color)
                
                with col2:
                    macd_signal = technical_signals.get('macd', 0)
                    macd_color = "green" if macd_signal > 0 else "red" if macd_signal < 0 else "gray"
                    st.metric("MACD Signal", macd_signal, delta_color=macd_color)
                
                with col3:
                    ma_signal = technical_signals.get('ma', 0)
                    ma_color = "green" if ma_signal > 0 else "red" if ma_signal < 0 else "gray"
                    st.metric("MA Signal", ma_signal, delta_color=ma_color)
                
                with col4:
                    bb_signal = technical_signals.get('bb', 0)
                    bb_color = "green" if bb_signal > 0 else "red" if bb_signal < 0 else "gray"
                    st.metric("BB Signal", bb_signal, delta_color=bb_color)
                
                # Calculate combined signal
                combined_signal = sum(technical_signals.values()) / len(technical_signals)
                
                # Display final signal
                signal_color = "green" if combined_signal > 0 else "red" if combined_signal < 0 else "gray"
                st.markdown(f"### Final Signal: <span style='color:{signal_color}'>{combined_signal:.2f}</span>", unsafe_allow_html=True)
                
                # Display signal interpretation
                if combined_signal > 0.5:
                    st.success("Strong Buy Signal")
                elif combined_signal > 0:
                    st.info("Weak Buy Signal")
                elif combined_signal < -0.5:
                    st.error("Strong Sell Signal")
                elif combined_signal < 0:
                    st.warning("Weak Sell Signal")
                else:
                    st.info("Neutral Signal")
                
            except Exception as e:
                st.error(f"Error generating signals: {str(e)}")
                st.info("Common issues:\n"
                       "1. Insufficient data for indicator calculation\n"
                       "2. Invalid date range\n"
                       "3. API rate limit exceeded")

# Footer
st.markdown("---")
st.markdown("Trading Bot Dashboard | Powered by Alpha Vantage API") 