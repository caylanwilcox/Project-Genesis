-- Clear existing 1m and 5m scalping data (capped at 100 bars)
-- We'll re-ingest with proper limits

-- Option 1: Clear ONLY scalping timeframes (1m, 5m)
DELETE FROM market_data
WHERE timeframe IN ('1m', '5m');

-- Option 2: Clear ALL data to start fresh with 3 years (recommended)
-- Uncomment if you want to start completely fresh:
-- DELETE FROM market_data;

-- Verify deletion
SELECT
    timeframe,
    COUNT(*) as bar_count
FROM market_data
GROUP BY timeframe
ORDER BY timeframe;
