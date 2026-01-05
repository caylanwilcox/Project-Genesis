-- Create FVG Detections Table
-- Fair Value Gap pattern detection following Fabio Valentini methodology

CREATE TABLE IF NOT EXISTS fvg_detections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ticker VARCHAR(10) NOT NULL,
  timeframe VARCHAR(10) NOT NULL,
  detected_at TIMESTAMPTZ(6) NOT NULL,

  -- FVG Type and Direction
  fvg_type VARCHAR(20) NOT NULL, -- 'bullish' or 'bearish'
  trading_mode VARCHAR(20) NOT NULL, -- 'scalping', 'intraday', 'daily', etc.

  -- 3-Candle Pattern Data
  candle1_timestamp TIMESTAMPTZ(6) NOT NULL,
  candle1_high DECIMAL(12, 4) NOT NULL,
  candle1_low DECIMAL(12, 4) NOT NULL,

  candle2_timestamp TIMESTAMPTZ(6) NOT NULL,
  candle2_high DECIMAL(12, 4) NOT NULL,
  candle2_low DECIMAL(12, 4) NOT NULL,

  candle3_timestamp TIMESTAMPTZ(6) NOT NULL,
  candle3_high DECIMAL(12, 4) NOT NULL,
  candle3_low DECIMAL(12, 4) NOT NULL,

  -- Gap Metrics
  gap_high DECIMAL(12, 4) NOT NULL,
  gap_low DECIMAL(12, 4) NOT NULL,
  gap_size DECIMAL(12, 4) NOT NULL,
  gap_size_pct DECIMAL(8, 4) NOT NULL,

  -- Entry and Exit Levels
  entry_price DECIMAL(12, 4) NOT NULL,
  stop_loss DECIMAL(12, 4) NOT NULL,
  take_profit1 DECIMAL(12, 4) NOT NULL,
  take_profit2 DECIMAL(12, 4) NOT NULL,
  take_profit3 DECIMAL(12, 4) NOT NULL,

  -- Fabio Valentini Validation Metrics
  volume_profile VARCHAR(20), -- 'bell_curve', 'skewed', etc.
  market_structure VARCHAR(50), -- 'balance_to_imbalance', 'trending', etc.
  validation_score DECIMAL(5, 4), -- 0-1 confidence

  -- Outcome Labels (for ML training)
  filled BOOLEAN DEFAULT FALSE,
  filled_at TIMESTAMPTZ(6),
  hit_tp1 BOOLEAN DEFAULT FALSE,
  hit_tp1_at TIMESTAMPTZ(6),
  hit_tp2 BOOLEAN DEFAULT FALSE,
  hit_tp2_at TIMESTAMPTZ(6),
  hit_tp3 BOOLEAN DEFAULT FALSE,
  hit_tp3_at TIMESTAMPTZ(6),
  hit_stop_loss BOOLEAN DEFAULT FALSE,
  hit_stop_loss_at TIMESTAMPTZ(6),
  hold_time_mins INTEGER,
  final_outcome VARCHAR(20), -- 'tp1', 'tp2', 'tp3', 'stop_loss', 'pending'

  -- ML Prediction (added later in Week 3+)
  predicted_win_rate DECIMAL(5, 4),
  predicted_hold_time INTEGER,
  model_name VARCHAR(50),

  created_at TIMESTAMPTZ(6) DEFAULT NOW()
);

-- Create indexes for efficient querying
CREATE UNIQUE INDEX IF NOT EXISTS fvg_detections_unique_idx
  ON fvg_detections(ticker, timeframe, detected_at, fvg_type);

CREATE INDEX IF NOT EXISTS fvg_detections_ticker_timeframe_idx
  ON fvg_detections(ticker, timeframe, detected_at DESC);

CREATE INDEX IF NOT EXISTS fvg_detections_trading_mode_idx
  ON fvg_detections(trading_mode, detected_at DESC);

CREATE INDEX IF NOT EXISTS fvg_detections_outcome_idx
  ON fvg_detections(fvg_type, final_outcome);

CREATE INDEX IF NOT EXISTS fvg_detections_timestamp_idx
  ON fvg_detections(detected_at DESC);

-- Add comments for documentation
COMMENT ON TABLE fvg_detections IS 'Fair Value Gap pattern detections using Fabio Valentini methodology';
COMMENT ON COLUMN fvg_detections.fvg_type IS 'Direction: bullish (buy signal) or bearish (sell signal)';
COMMENT ON COLUMN fvg_detections.trading_mode IS 'Trading mode: scalping, intraday, daily, swing, weekly, biweekly, monthly';
COMMENT ON COLUMN fvg_detections.gap_size IS 'Absolute gap size in price units';
COMMENT ON COLUMN fvg_detections.gap_size_pct IS 'Gap size as percentage of price';
COMMENT ON COLUMN fvg_detections.validation_score IS 'Fabio Valentini validation confidence (0-1)';
COMMENT ON COLUMN fvg_detections.final_outcome IS 'Actual outcome: tp1, tp2, tp3, stop_loss, or pending';
