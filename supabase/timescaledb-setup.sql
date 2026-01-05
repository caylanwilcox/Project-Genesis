-- ============================================================================
-- TimescaleDB Setup for ML Trading System
-- Run this in Supabase SQL Editor to enable time-series optimizations
-- ============================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================================================
-- CONVERT TABLES TO HYPERTABLES
-- ============================================================================

-- Convert market_data to hypertable (partitioned by timestamp)
-- Chunk interval: 1 week (optimal for trading data)
SELECT create_hypertable(
  'market_data',
  'timestamp',
  chunk_time_interval => INTERVAL '1 week',
  if_not_exists => TRUE
);

-- Convert features to hypertable
SELECT create_hypertable(
  'features',
  'timestamp',
  chunk_time_interval => INTERVAL '1 week',
  if_not_exists => TRUE
);

-- Convert predictions to hypertable
SELECT create_hypertable(
  'predictions',
  'timestamp',
  chunk_time_interval => INTERVAL '1 month',
  if_not_exists => TRUE
);

-- Convert portfolio to hypertable (for historical tracking)
SELECT create_hypertable(
  'portfolio',
  'timestamp',
  chunk_time_interval => INTERVAL '1 month',
  if_not_exists => TRUE
);

-- ============================================================================
-- COMPRESSION POLICIES
-- ============================================================================

-- Enable compression on market_data
-- Compress data older than 7 days to save storage
ALTER TABLE market_data SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'ticker,timeframe',
  timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('market_data', INTERVAL '7 days');

-- Enable compression on features
ALTER TABLE features SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'ticker,timeframe',
  timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('features', INTERVAL '7 days');

-- Enable compression on predictions
ALTER TABLE predictions SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'ticker,timeframe,model_name',
  timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('predictions', INTERVAL '30 days');

-- ============================================================================
-- RETENTION POLICIES (Optional - keep data for specific time periods)
-- ============================================================================

-- Keep market_data for 5 years (adjust as needed)
-- Uncomment to enable:
-- SELECT add_retention_policy('market_data', INTERVAL '5 years');

-- Keep features for 2 years
-- SELECT add_retention_policy('features', INTERVAL '2 years');

-- Keep predictions for 2 years
-- SELECT add_retention_policy('predictions', INTERVAL '2 years');

-- ============================================================================
-- CONTINUOUS AGGREGATES (Pre-computed rollups for fast queries)
-- ============================================================================

-- Create daily OHLCV aggregate from minute data
-- This speeds up queries for daily/weekly/monthly charts
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_market_summary
WITH (timescaledb.continuous) AS
SELECT
  ticker,
  timeframe,
  time_bucket('1 day', timestamp) AS day,
  first(open, timestamp) AS open,
  max(high) AS high,
  min(low) AS low,
  last(close, timestamp) AS close,
  sum(volume) AS volume,
  count(*) AS bar_count
FROM market_data
GROUP BY ticker, timeframe, day;

-- Add refresh policy for continuous aggregate
-- Refresh every hour, covering last 7 days
SELECT add_continuous_aggregate_policy(
  'daily_market_summary',
  start_offset => INTERVAL '7 days',
  end_offset => INTERVAL '1 hour',
  schedule_interval => INTERVAL '1 hour'
);

-- Create weekly summary
CREATE MATERIALIZED VIEW IF NOT EXISTS weekly_market_summary
WITH (timescaledb.continuous) AS
SELECT
  ticker,
  timeframe,
  time_bucket('1 week', timestamp) AS week,
  first(open, timestamp) AS open,
  max(high) AS high,
  min(low) AS low,
  last(close, timestamp) AS close,
  sum(volume) AS volume,
  count(*) AS bar_count
FROM market_data
GROUP BY ticker, timeframe, week;

SELECT add_continuous_aggregate_policy(
  'weekly_market_summary',
  start_offset => INTERVAL '1 month',
  end_offset => INTERVAL '1 day',
  schedule_interval => INTERVAL '1 day'
);

-- ============================================================================
-- PERFORMANCE INDEXES (Additional to Prisma-generated indexes)
-- ============================================================================

-- Composite index for common query pattern: ticker + timeframe + timestamp range
CREATE INDEX IF NOT EXISTS idx_market_data_ticker_timeframe_ts
ON market_data (ticker, timeframe, timestamp DESC);

-- Index for feature queries
CREATE INDEX IF NOT EXISTS idx_features_ticker_timeframe_ts
ON features (ticker, timeframe, timestamp DESC);

-- Index for prediction accuracy lookups
CREATE INDEX IF NOT EXISTS idx_predictions_ticker_model_ts
ON predictions (ticker, model_name, timestamp DESC);

-- Index for ingestion log queries
CREATE INDEX IF NOT EXISTS idx_ingestion_ticker_tf_created
ON ingestion_log (ticker, timeframe, created_at DESC);

-- ============================================================================
-- STATISTICS & MONITORING FUNCTIONS
-- ============================================================================

-- Function to check compression status
CREATE OR REPLACE FUNCTION get_compression_stats()
RETURNS TABLE (
  hypertable_name TEXT,
  total_chunks BIGINT,
  compressed_chunks BIGINT,
  uncompressed_size TEXT,
  compressed_size TEXT,
  compression_ratio NUMERIC
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    h.hypertable_name::TEXT,
    COUNT(*)::BIGINT AS total_chunks,
    COUNT(*) FILTER (WHERE c.compressed_chunk_id IS NOT NULL)::BIGINT AS compressed_chunks,
    pg_size_pretty(SUM(pg_total_relation_size(format('%I.%I', c.chunk_schema, c.chunk_name)::regclass))) AS uncompressed_size,
    pg_size_pretty(SUM(pg_total_relation_size(format('%I.%I', c.chunk_schema, c.chunk_name)::regclass))
      FILTER (WHERE c.compressed_chunk_id IS NOT NULL)) AS compressed_size,
    ROUND(
      SUM(pg_total_relation_size(format('%I.%I', c.chunk_schema, c.chunk_name)::regclass))::NUMERIC /
      NULLIF(SUM(pg_total_relation_size(format('%I.%I', c.chunk_schema, c.chunk_name)::regclass))
        FILTER (WHERE c.compressed_chunk_id IS NOT NULL), 0),
      2
    ) AS compression_ratio
  FROM timescaledb_information.hypertables h
  LEFT JOIN timescaledb_information.chunks c ON h.hypertable_name = c.hypertable_name
  GROUP BY h.hypertable_name;
END;
$$ LANGUAGE plpgsql;

-- Function to get hypertable stats
CREATE OR REPLACE FUNCTION get_hypertable_stats()
RETURNS TABLE (
  table_name TEXT,
  total_size TEXT,
  table_size TEXT,
  index_size TEXT,
  row_count BIGINT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    schemaname || '.' || tablename AS table_name,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) AS index_size,
    n_live_tup AS row_count
  FROM pg_stat_user_tables
  WHERE schemaname = 'public'
    AND tablename IN ('market_data', 'features', 'predictions', 'portfolio')
  ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Check if TimescaleDB is enabled
SELECT * FROM pg_extension WHERE extname = 'timescaledb';

-- List all hypertables
SELECT * FROM timescaledb_information.hypertables;

-- Check compression policies
SELECT * FROM timescaledb_information.jobs
WHERE proc_name LIKE '%compression%';

-- Check continuous aggregates
SELECT * FROM timescaledb_information.continuous_aggregates;

-- Get compression stats
SELECT * FROM get_compression_stats();

-- Get hypertable stats
SELECT * FROM get_hypertable_stats();

-- ============================================================================
-- USAGE EXAMPLES
-- ============================================================================

-- Query using time_bucket for efficient aggregation
-- Example: Get hourly OHLCV for SPY from last 24 hours
SELECT
  time_bucket('1 hour', timestamp) AS hour,
  first(open, timestamp) AS open,
  max(high) AS high,
  min(low) AS low,
  last(close, timestamp) AS close,
  sum(volume) AS volume
FROM market_data
WHERE ticker = 'SPY'
  AND timeframe = '1m'
  AND timestamp > NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour DESC;

-- Query continuous aggregate (much faster than raw data)
SELECT * FROM daily_market_summary
WHERE ticker = 'SPY'
  AND day > NOW() - INTERVAL '30 days'
ORDER BY day DESC;

-- ============================================================================
-- CLEANUP (Optional - only if you need to start over)
-- ============================================================================

-- WARNING: These commands will delete all data and hypertable configurations
-- Uncomment only if you need to reset everything

-- DROP MATERIALIZED VIEW IF EXISTS daily_market_summary CASCADE;
-- DROP MATERIALIZED VIEW IF EXISTS weekly_market_summary CASCADE;
-- DROP FUNCTION IF EXISTS get_compression_stats();
-- DROP FUNCTION IF EXISTS get_hypertable_stats();

-- ============================================================================
-- NOTES
-- ============================================================================

-- 1. Compression happens automatically based on policies
-- 2. Compressed chunks are read-only but queries work normally
-- 3. Continuous aggregates refresh automatically
-- 4. Monitor compression with: SELECT * FROM get_compression_stats();
-- 5. Check query performance with EXPLAIN ANALYZE
-- 6. Adjust chunk_time_interval based on your data volume

-- Expected benefits:
-- - 10x faster time-series queries
-- - 70-90% storage reduction with compression
-- - Automatic data partitioning
-- - Fast aggregations with continuous aggregates
