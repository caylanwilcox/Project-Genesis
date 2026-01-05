# TimescaleDB Setup Guide

This guide walks you through enabling TimescaleDB optimizations for your ML trading database.

---

## What is TimescaleDB?

TimescaleDB is a PostgreSQL extension that provides:
- **10x faster time-series queries**
- **70-90% storage reduction** with compression
- **Automatic data partitioning** (chunks)
- **Continuous aggregates** for instant rollups

---

## Prerequisites

✅ Supabase project running
✅ Database password from Supabase dashboard
✅ All tables created (from Day 1 & 2)

---

## Step-by-Step Instructions

### Step 1: Open Supabase SQL Editor

1. Go to your Supabase project: https://app.supabase.com/project/yvrfkqggtxmfhmqjzulh
2. Click on **SQL Editor** in the left sidebar
3. Click **New Query**

### Step 2: Run TimescaleDB Setup Script

1. Open the file: [`supabase/timescaledb-setup.sql`](./supabase/timescaledb-setup.sql)
2. Copy the entire contents
3. Paste into the Supabase SQL Editor
4. Click **Run** (or press Cmd+Enter)

### Step 3: Verify Installation

Run these verification queries in the SQL Editor:

```sql
-- Check if TimescaleDB is enabled
SELECT * FROM pg_extension WHERE extname = 'timescaledb';
-- Should return 1 row

-- List all hypertables
SELECT * FROM timescaledb_information.hypertables;
-- Should show: market_data, features, predictions, portfolio

-- Check compression policies
SELECT * FROM timescaledb_information.jobs WHERE proc_name LIKE '%compression%';
-- Should show 3 compression policies

-- Check continuous aggregates
SELECT * FROM timescaledb_information.continuous_aggregates;
-- Should show: daily_market_summary, weekly_market_summary
```

---

## What Gets Enabled

### 1. Hypertables (Automatic Partitioning)

Tables converted to hypertables:
- ✅ `market_data` - 1 week chunks
- ✅ `features` - 1 week chunks
- ✅ `predictions` - 1 month chunks
- ✅ `portfolio` - 1 month chunks

**Benefit:** Queries automatically use only relevant chunks, 10x faster

### 2. Compression Policies

Automatic compression for old data:
- `market_data`: Compress after 7 days
- `features`: Compress after 7 days
- `predictions`: Compress after 30 days

**Benefit:** 70-90% storage reduction, queries still work normally

### 3. Continuous Aggregates

Pre-computed rollups:
- `daily_market_summary`: Daily OHLCV from minute data
- `weekly_market_summary`: Weekly OHLCV from minute data

**Benefit:** Instant queries for daily/weekly charts (no computation needed)

### 4. Performance Indexes

Additional indexes for common query patterns:
- Ticker + Timeframe + Timestamp (composite)
- Feature queries
- Prediction accuracy lookups
- Ingestion log queries

---

## Expected Performance Improvements

| Query Type | Before TimescaleDB | After TimescaleDB | Improvement |
|------------|-------------------|-------------------|-------------|
| Last 100 bars | 50ms | 5ms | **10x faster** |
| 1 year of data | 500ms | 50ms | **10x faster** |
| Daily aggregates | 2000ms | 10ms | **200x faster** |
| Storage (1M bars) | 500MB | 50-150MB | **70-90% reduction** |

---

## Monitoring

### Check Compression Status

```sql
SELECT * FROM get_compression_stats();
```

Example output:
```
hypertable_name | total_chunks | compressed_chunks | uncompressed_size | compressed_size | compression_ratio
----------------|--------------|-------------------|-------------------|-----------------|------------------
market_data     | 52           | 48                | 450 MB            | 45 MB           | 10.0
features        | 52           | 48                | 320 MB            | 35 MB           | 9.1
predictions     | 12           | 10                | 120 MB            | 15 MB           | 8.0
```

### Check Hypertable Stats

```sql
SELECT * FROM get_hypertable_stats();
```

Example output:
```
table_name   | total_size | table_size | index_size | row_count
-------------|------------|------------|------------|----------
market_data  | 450 MB     | 380 MB     | 70 MB      | 2000000
features     | 320 MB     | 270 MB     | 50 MB      | 1500000
predictions  | 120 MB     | 100 MB     | 20 MB      | 500000
```

---

## Using TimescaleDB Features

### Time Bucket Queries (Fast Aggregation)

```sql
-- Get hourly OHLCV from minute data (super fast!)
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
```

### Query Continuous Aggregates (Instant Results)

```sql
-- Get daily data (instant - pre-computed!)
SELECT * FROM daily_market_summary
WHERE ticker = 'SPY'
  AND day > NOW() - INTERVAL '30 days'
ORDER BY day DESC;

-- Get weekly data (instant!)
SELECT * FROM weekly_market_summary
WHERE ticker = 'SPY'
  AND week > NOW() - INTERVAL '6 months'
ORDER BY week DESC;
```

---

## Troubleshooting

### Issue: "extension timescaledb does not exist"

**Solution:** TimescaleDB might not be enabled on your Supabase plan.

1. Check your Supabase plan (should be Pro or higher)
2. Contact Supabase support to enable TimescaleDB
3. Alternative: Continue without TimescaleDB (slower but works)

### Issue: "ERROR: relation is already a hypertable"

**Solution:** You've already run the setup script.

- This is not an error if you're re-running
- Tables are already optimized
- Continue to next step

### Issue: Compression not working

**Solution:** Check compression policies:

```sql
-- List all jobs
SELECT * FROM timescaledb_information.jobs;

-- Manually trigger compression
CALL run_job(JOB_ID); -- Replace JOB_ID from jobs query
```

---

## Optional: Retention Policies

If you want to automatically delete old data, uncomment these lines in the setup script:

```sql
-- Keep market_data for 5 years (delete older data)
SELECT add_retention_policy('market_data', INTERVAL '5 years');

-- Keep features for 2 years
SELECT add_retention_policy('features', INTERVAL '2 years');

-- Keep predictions for 2 years
SELECT add_retention_policy('predictions', INTERVAL '2 years');
```

**Warning:** This will permanently delete old data. Only enable if you're sure.

---

## Cleanup (If Needed)

If you need to remove TimescaleDB optimizations:

```sql
-- Drop continuous aggregates
DROP MATERIALIZED VIEW IF EXISTS daily_market_summary CASCADE;
DROP MATERIALIZED VIEW IF EXISTS weekly_market_summary CASCADE;

-- Drop helper functions
DROP FUNCTION IF EXISTS get_compression_stats();
DROP FUNCTION IF EXISTS get_hypertable_stats();

-- Note: Cannot easily revert hypertables - would need to recreate tables
```

---

## Next Steps

After enabling TimescaleDB:

1. ✅ Verify hypertables are created
2. ✅ Check compression policies are active
3. ✅ Test query performance
4. ✅ Monitor compression stats
5. ✅ Continue to Day 3 (Historical Data Ingestion)

---

## Performance Testing

Test query performance before and after:

```sql
-- Test 1: Query 100K rows
EXPLAIN ANALYZE
SELECT * FROM market_data
WHERE ticker = 'SPY' AND timeframe = '1m'
ORDER BY timestamp DESC
LIMIT 100000;

-- Test 2: Aggregation query
EXPLAIN ANALYZE
SELECT
  ticker,
  time_bucket('1 day', timestamp) AS day,
  avg(close) AS avg_close
FROM market_data
WHERE ticker = 'SPY'
  AND timestamp > NOW() - INTERVAL '1 year'
GROUP BY ticker, day;
```

Look for "Execution Time" - should be significantly faster with TimescaleDB.

---

## Support

- **TimescaleDB Docs:** https://docs.timescale.com/
- **Supabase Support:** https://supabase.com/support
- **Issue Tracker:** Create issue in project repository

---

## Summary Checklist

- [ ] Opened Supabase SQL Editor
- [ ] Ran `timescaledb-setup.sql` script
- [ ] Verified TimescaleDB extension enabled
- [ ] Confirmed 4 hypertables created
- [ ] Checked 3 compression policies active
- [ ] Verified 2 continuous aggregates created
- [ ] Tested query performance
- [ ] Monitored compression stats

---

**Status:** Ready to use TimescaleDB ✅

**Next:** Continue to Day 3 - Historical Data Ingestion
