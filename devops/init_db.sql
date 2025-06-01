-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create tables for different data types
CREATE TABLE IF NOT EXISTS ohlcv (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    quote_volume DOUBLE PRECISION NOT NULL,
    trades INTEGER NOT NULL,
    taker_buy_base DOUBLE PRECISION NOT NULL,
    taker_buy_quote DOUBLE PRECISION NOT NULL
);

-- Convert to hypertable
SELECT create_hypertable('ohlcv', 'timestamp');

-- Create order book table
CREATE TABLE IF NOT EXISTS order_book (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    quantity DOUBLE PRECISION NOT NULL
);

-- Convert to hypertable
SELECT create_hypertable('order_book', 'timestamp');

-- Create on-chain metrics table
CREATE TABLE IF NOT EXISTS onchain_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    asset VARCHAR(10) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    value DOUBLE PRECISION NOT NULL
);

-- Convert to hypertable
SELECT create_hypertable('onchain_metrics', 'timestamp');

-- Create sentiment data table
CREATE TABLE IF NOT EXISTS sentiment_data (
    timestamp TIMESTAMPTZ NOT NULL,
    source VARCHAR(20) NOT NULL,
    text TEXT NOT NULL,
    sentiment_score DOUBLE PRECISION,
    engagement_score DOUBLE PRECISION
);

-- Convert to hypertable
SELECT create_hypertable('sentiment_data', 'timestamp');

-- Create trades table
CREATE TABLE IF NOT EXISTS trades (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    quantity DOUBLE PRECISION NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    pnl DOUBLE PRECISION
);

-- Convert to hypertable
SELECT create_hypertable('trades', 'timestamp');

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv (symbol);
CREATE INDEX IF NOT EXISTS idx_order_book_symbol ON order_book (symbol);
CREATE INDEX IF NOT EXISTS idx_onchain_metrics_asset ON onchain_metrics (asset);
CREATE INDEX IF NOT EXISTS idx_sentiment_source ON sentiment_data (source);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol);

-- Create continuous aggregates for common queries
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS bucket,
    symbol,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume
FROM ohlcv
GROUP BY bucket, symbol;

-- Refresh policy for continuous aggregates
SELECT add_continuous_aggregate_policy('ohlcv_1h',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'); 