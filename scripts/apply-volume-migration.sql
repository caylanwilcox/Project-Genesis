-- Check current column type
SELECT column_name, data_type, character_maximum_length, numeric_precision, numeric_scale
FROM information_schema.columns
WHERE table_name = 'market_data' AND column_name = 'volume';

-- Alter the volume column type (handles existing data safely)
ALTER TABLE market_data
ALTER COLUMN volume TYPE DECIMAL(18, 2) USING volume::DECIMAL(18, 2);

-- Add comment
COMMENT ON COLUMN market_data.volume IS 'Trading volume - DECIMAL(18,2) to support fractional volumes (e.g., UVXY)';

-- Verify the change
SELECT column_name, data_type, numeric_precision, numeric_scale
FROM information_schema.columns
WHERE table_name = 'market_data' AND column_name = 'volume';

-- Check data integrity
SELECT
  ticker,
  COUNT(*) as total_rows,
  MIN(volume) as min_volume,
  MAX(volume) as max_volume
FROM market_data
GROUP BY ticker
ORDER BY ticker;
