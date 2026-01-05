-- Week 1: Database Schema for ML Trading System
-- Run this in your Supabase SQL Editor

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Market Data Table
-- Stores historical OHLCV data from Polygon.io
CREATE TABLE IF NOT EXISTS market_data (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  ticker VARCHAR(10) NOT NULL,
  timeframe VARCHAR(10) NOT NULL, -- '1h', '1d', '1w', etc.
  timestamp TIMESTAMPTZ NOT NULL,
  open DECIMAL(12, 4) NOT NULL,
  high DECIMAL(12, 4) NOT NULL,
  low DECIMAL(12, 4) NOT NULL,
  close DECIMAL(12, 4) NOT NULL,
  volume BIGINT NOT NULL,
  source VARCHAR(50) DEFAULT 'polygon',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(ticker, timeframe, timestamp)
);

-- Create indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_market_data_ticker_timeframe_timestamp
  ON market_data(ticker, timeframe, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp
  ON market_data(timestamp DESC);

-- Features Table
-- Stores calculated technical indicators and features for ML
CREATE TABLE IF NOT EXISTS features (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  ticker VARCHAR(10) NOT NULL,
  timeframe VARCHAR(10) NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  feature_name VARCHAR(100) NOT NULL, -- 'rsi_14', 'macd', 'sma_20', etc.
  feature_value DECIMAL(12, 6) NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(ticker, timeframe, timestamp, feature_name)
);

CREATE INDEX IF NOT EXISTS idx_features_ticker_timeframe_timestamp
  ON features(ticker, timeframe, timestamp DESC);

-- Predictions Table
-- Stores ML model predictions
CREATE TABLE IF NOT EXISTS predictions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  ticker VARCHAR(10) NOT NULL,
  timeframe VARCHAR(10) NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  model_name VARCHAR(50) NOT NULL, -- 'xgboost', 'lstm', 'ensemble', etc.
  predicted_direction VARCHAR(10) NOT NULL, -- 'up', 'down', 'neutral'
  predicted_change DECIMAL(8, 4), -- Predicted percentage change
  confidence DECIMAL(5, 4), -- Model confidence (0-1)
  actual_direction VARCHAR(10), -- Actual direction after the fact
  actual_change DECIMAL(8, 4), -- Actual percentage change
  accuracy DECIMAL(5, 4), -- How accurate was the prediction
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(ticker, timeframe, timestamp, model_name)
);

CREATE INDEX IF NOT EXISTS idx_predictions_ticker_timeframe_timestamp
  ON predictions(ticker, timeframe, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model_name
  ON predictions(model_name);

-- Models Table
-- Stores ML model metadata and performance metrics
CREATE TABLE IF NOT EXISTS models (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name VARCHAR(50) UNIQUE NOT NULL,
  ticker VARCHAR(10) NOT NULL,
  timeframe VARCHAR(10) NOT NULL,
  algorithm VARCHAR(50) NOT NULL, -- 'xgboost', 'lstm', 'random_forest', etc.
  version INTEGER DEFAULT 1,
  parameters JSONB, -- Model hyperparameters
  features JSONB, -- List of features used
  accuracy DECIMAL(5, 4), -- Overall accuracy
  precision_score DECIMAL(5, 4),
  recall DECIMAL(5, 4),
  f1_score DECIMAL(5, 4),
  training_date TIMESTAMPTZ,
  training_samples INTEGER,
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_models_ticker_timeframe
  ON models(ticker, timeframe);
CREATE INDEX IF NOT EXISTS idx_models_is_active
  ON models(is_active);

-- Trades Table (for future use - Week 11)
-- Stores actual trades executed
CREATE TABLE IF NOT EXISTS trades (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  ticker VARCHAR(10) NOT NULL,
  timeframe VARCHAR(10) NOT NULL,
  entry_timestamp TIMESTAMPTZ NOT NULL,
  exit_timestamp TIMESTAMPTZ,
  direction VARCHAR(10) NOT NULL, -- 'long' or 'short'
  entry_price DECIMAL(12, 4) NOT NULL,
  exit_price DECIMAL(12, 4),
  quantity INTEGER NOT NULL,
  stop_loss DECIMAL(12, 4),
  take_profit DECIMAL(12, 4),
  pnl DECIMAL(12, 4), -- Profit/Loss
  pnl_percent DECIMAL(8, 4), -- P/L percentage
  status VARCHAR(20) DEFAULT 'open', -- 'open', 'closed', 'stopped'
  model_name VARCHAR(50), -- Which model triggered this
  prediction_id UUID REFERENCES predictions(id),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_ticker_timestamp
  ON trades(ticker, entry_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_status
  ON trades(status);

-- Portfolio Table (for future use - Week 11)
-- Tracks overall portfolio performance
CREATE TABLE IF NOT EXISTS portfolio (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  timestamp TIMESTAMPTZ NOT NULL,
  total_value DECIMAL(12, 2) NOT NULL,
  cash DECIMAL(12, 2) NOT NULL,
  positions_value DECIMAL(12, 2) NOT NULL,
  daily_pnl DECIMAL(12, 2),
  total_pnl DECIMAL(12, 2),
  total_pnl_percent DECIMAL(8, 4),
  num_positions INTEGER DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp
  ON portfolio(timestamp DESC);

-- Data Ingestion Log Table
-- Tracks data fetching jobs
CREATE TABLE IF NOT EXISTS ingestion_log (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  ticker VARCHAR(10) NOT NULL,
  timeframe VARCHAR(10) NOT NULL,
  start_date TIMESTAMPTZ NOT NULL,
  end_date TIMESTAMPTZ NOT NULL,
  bars_fetched INTEGER NOT NULL,
  bars_inserted INTEGER NOT NULL,
  bars_skipped INTEGER NOT NULL,
  status VARCHAR(20) NOT NULL, -- 'success', 'error', 'partial'
  error_message TEXT,
  duration_ms INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ingestion_log_ticker_timeframe
  ON ingestion_log(ticker, timeframe, created_at DESC);

-- Enable Row Level Security (RLS) - Optional, for production
-- For now, we'll keep it simple, but you can enable this later
-- ALTER TABLE market_data ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE features ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE models ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE portfolio ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE ingestion_log ENABLE ROW LEVEL SECURITY;

-- Create a view for recent market data summary
CREATE OR REPLACE VIEW market_data_summary AS
SELECT
  ticker,
  timeframe,
  MAX(timestamp) as latest_timestamp,
  COUNT(*) as total_bars,
  MIN(timestamp) as earliest_timestamp,
  MAX(timestamp) - MIN(timestamp) as date_range
FROM market_data
GROUP BY ticker, timeframe;

-- Create a view for model performance summary
CREATE OR REPLACE VIEW model_performance AS
SELECT
  m.id,
  m.name,
  m.ticker,
  m.timeframe,
  m.algorithm,
  m.accuracy,
  COUNT(p.id) as total_predictions,
  SUM(CASE WHEN p.actual_direction IS NOT NULL THEN 1 ELSE 0 END) as verified_predictions,
  AVG(p.accuracy) as avg_prediction_accuracy,
  AVG(p.confidence) as avg_confidence
FROM models m
LEFT JOIN predictions p ON m.name = p.model_name
WHERE m.is_active = true
GROUP BY m.id, m.name, m.ticker, m.timeframe, m.algorithm, m.accuracy;

COMMENT ON TABLE market_data IS 'Historical OHLCV data from Polygon.io';
COMMENT ON TABLE features IS 'Calculated technical indicators for ML training';
COMMENT ON TABLE predictions IS 'ML model predictions and their accuracy';
COMMENT ON TABLE models IS 'ML model metadata and performance metrics';
COMMENT ON TABLE trades IS 'Actual trades executed (Week 11)';
COMMENT ON TABLE portfolio IS 'Portfolio performance tracking (Week 11)';
COMMENT ON TABLE ingestion_log IS 'Data fetching job history';
