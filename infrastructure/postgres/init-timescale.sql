-- ============================================================
-- NeuralTrade - TimescaleDB Initialization Script
-- ============================================================
-- Creates hypertables for time-series market data
-- ============================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================================
-- OHLCV Market History (Candlestick Data)
-- ============================================================
CREATE TABLE IF NOT EXISTS ohlcv_data (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,      -- 1m, 5m, 15m, 1h, 4h, 1d
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(24, 8) NOT NULL,
    high DECIMAL(24, 8) NOT NULL,
    low DECIMAL(24, 8) NOT NULL,
    close DECIMAL(24, 8) NOT NULL,
    volume DECIMAL(24, 8) NOT NULL,
    quote_volume DECIMAL(24, 8),
    trades_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, exchange, timeframe, timestamp)
);

-- Convert to hypertable (partitioned by time)
SELECT create_hypertable(
    'ohlcv_data',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- Index for fast symbol+timeframe queries
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe
ON ohlcv_data (symbol, timeframe, timestamp DESC);

-- ============================================================
-- Price Ticks (Real-time Tick Stream)
-- ============================================================
CREATE TABLE IF NOT EXISTS price_ticks (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    price DECIMAL(24, 8) NOT NULL,
    quantity DECIMAL(24, 8) NOT NULL,
    side VARCHAR(4),                      -- 'buy' or 'sell'
    trade_id VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, exchange, timestamp, id)
);

-- Convert to hypertable with smaller chunks for high-frequency data
SELECT create_hypertable(
    'price_ticks',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 hour'
);

-- Index for real-time price lookups
CREATE INDEX IF NOT EXISTS idx_price_ticks_symbol
ON price_ticks (symbol, timestamp DESC);

-- ============================================================
-- Signal History (AI Trading Signals)
-- ============================================================
CREATE TABLE IF NOT EXISTS signal_history (
    id BIGSERIAL,
    signal_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    signal_type VARCHAR(20) NOT NULL,     -- 'BUY', 'SELL', 'HOLD'
    confidence DECIMAL(5, 4) NOT NULL,    -- 0.0000 to 1.0000
    model_name VARCHAR(100),
    strategy_name VARCHAR(100),
    features JSONB,                       -- Input features used
    reasoning TEXT,                       -- AI reasoning (if available)
    executed BOOLEAN DEFAULT FALSE,
    execution_price DECIMAL(24, 8),
    execution_time TIMESTAMPTZ,
    pnl DECIMAL(24, 8),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, timestamp, signal_id)
);

-- Convert to hypertable
SELECT create_hypertable(
    'signal_history',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- Index for signal analysis
CREATE INDEX IF NOT EXISTS idx_signal_symbol_type
ON signal_history (symbol, signal_type, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_signal_model
ON signal_history (model_name, timestamp DESC);

-- ============================================================
-- Retention Policies (Auto-delete old data)
-- ============================================================

-- Keep tick data for 7 days (high volume)
SELECT add_retention_policy(
    'price_ticks',
    INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Keep OHLCV data for 2 years
SELECT add_retention_policy(
    'ohlcv_data',
    INTERVAL '2 years',
    if_not_exists => TRUE
);

-- Keep signal history for 1 year
SELECT add_retention_policy(
    'signal_history',
    INTERVAL '1 year',
    if_not_exists => TRUE
);

-- ============================================================
-- Continuous Aggregates (Pre-computed rollups)
-- ============================================================

-- 1-hour OHLCV aggregate from 1-minute data
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1h
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    exchange,
    '1h' AS timeframe,
    time_bucket('1 hour', timestamp) AS timestamp,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trades_count) AS trades_count
FROM ohlcv_data
WHERE timeframe = '1m'
GROUP BY symbol, exchange, time_bucket('1 hour', timestamp)
WITH NO DATA;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('ohlcv_1h',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- ============================================================
-- Performance Tuning
-- ============================================================
-- Enable compression on older chunks
SELECT add_compression_policy('ohlcv_data', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('signal_history', INTERVAL '30 days', if_not_exists => TRUE);

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO neuraltrade;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO neuraltrade;

COMMENT ON TABLE ohlcv_data IS 'Market candlestick data (OHLCV) stored as TimescaleDB hypertable';
COMMENT ON TABLE price_ticks IS 'Real-time price tick stream for high-frequency data';
COMMENT ON TABLE signal_history IS 'AI trading signals with execution tracking';
