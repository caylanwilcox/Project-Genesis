-- Migration: Change volume column from BIGINT to DECIMAL
-- Reason: UVXY and some ETFs have fractional volume values
-- Date: November 6, 2025

-- Step 1: Alter the volume column type
ALTER TABLE market_data
ALTER COLUMN volume TYPE DECIMAL(18, 2);

-- Step 2: Verify the change
COMMENT ON COLUMN market_data.volume IS 'Trading volume - changed from BIGINT to DECIMAL to support fractional volumes (e.g., UVXY)';

-- Step 3: Optional - Update any existing NULL volumes to 0
UPDATE market_data
SET volume = 0
WHERE volume IS NULL;

-- Verification query
SELECT
  ticker,
  COUNT(*) as total_rows,
  MIN(volume) as min_volume,
  MAX(volume) as max_volume,
  AVG(volume) as avg_volume
FROM market_data
GROUP BY ticker;
