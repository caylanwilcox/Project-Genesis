# Week 1: Days 4-5 Execution Guide

**Status:** Ready to Execute
**Estimated Time:** 30 min (minimum) to 2.5 hours (complete)

---

## Quick Summary

Week 1 Days 1-3 are **COMPLETE** ✅. Days 4-5 have been **DESIGNED and SCRIPTED** ✅.

**What's Done:**
- ✅ Prisma schema updated (volume: BigInt → Decimal)
- ✅ Code updated (removed BigInt conversions)
- ✅ Migration SQL created
- ✅ Test scripts created
- ✅ Benchmark scripts created
- ✅ Documentation complete

**What Remains:**
- ⏳ **Execute** the migration (5 min manual step)
- ⏳ **Run** the test scripts (automated)
- ⏳ **Run** the benchmarks (automated)
- ⏳ **(Optional)** Add scalping data (2 hours)

---

## Option A: Minimum Path (30 minutes) ✅ RECOMMENDED

**Goal:** Complete critical Week 1 items, ready for Week 2

### Step 1: Apply Volume Migration (5 min) 🔴 MANUAL

1. **Copy the migration SQL:**
   ```bash
   cat /Users/it/Documents/mvp_coder_starter_kit\ \(2\)/mvp-trading-app/scripts/apply-volume-migration.sql
   ```

2. **Open Supabase SQL Editor:**
   - Go to: https://supabase.com/dashboard
   - Navigate to your project
   - Click "SQL Editor" in left sidebar

3. **Paste and run:**
   - Paste the entire migration SQL
   - Click "Run" or press Cmd+Enter
   - Verify output shows `volume | numeric | 18 | 2`

4. **Expected output:**
   ```
   ALTER TABLE
   column_name | data_type | numeric_precision | numeric_scale
   -----------+-----------+-------------------+---------------
   volume      | numeric   | 18                | 2
   ```

### Step 2: Test UVXY Ingestion (2 min)

```bash
cd /Users/it/Documents/mvp_coder_starter_kit\ \(2\)/mvp-trading-app
npx ts-node scripts/test-uvxy-ingestion.ts
```

**Expected output:**
```
======================================
Testing UVXY Data Ingestion
======================================

Test 1: UVXY 1h timeframe (30 days)
-----------------------------------
✅ SUCCESS: 100+ bars inserted
   Fetched: 100+
   Skipped: 0
   Duration: 3000ms

Test 2: UVXY 1d timeframe (2 years)
-----------------------------------
✅ SUCCESS: 500+ bars inserted
   Fetched: 500+
   Skipped: 0
   Duration: 2500ms

✅ Fractional volume detected: 5140.20

======================================
UVXY Ingestion Test Complete!
======================================
```

### Step 3: Run Benchmarks (2 min)

```bash
npx ts-node scripts/benchmark-database.ts
```

**Expected output:**
```
======================================
Benchmark Summary
======================================

Total tests: 5
✅ Passed: 5
❌ Failed: 0
Pass rate: 100.0%

✅ Bulk Insert (10K rows): 1243ms (target: < 2000ms)
✅ Query (1000 rows): 142ms (target: < 200ms)
✅ Latest Bar Query: 3.21ms (target: < 10ms)
✅ Aggregation Query: 67ms (target: < 100ms)
✅ Complex Filter Query: 198ms (target: < 300ms)
```

### Step 4: Update Documentation (20 min)

1. Mark todos as complete
2. Update [WEEK_1_FINAL_STATUS.md](WEEK_1_FINAL_STATUS.md)
3. Celebrate! 🎉

---

## Option B: Complete Path (2.5 hours)

**Goal:** Fully complete Week 1, including scalping data

### Steps 1-3: Same as Option A (30 min)

### Step 4: Add Scalping Timeframes (2 hours)

```bash
npx ts-node scripts/add-scalping-timeframes.ts
```

**What it does:**
- Adds 1m and 5m data for 4 tickers (SPY, QQQ, IWM, UVXY)
- Total: ~18,680 bars
- Takes 2 hours due to Polygon.io rate limits (13s delay × 8 jobs)

**Expected output:**
```
======================================
Adding Scalping Timeframes (1m, 5m)
======================================

📈 Processing SPY
-----------------------------------
⏱️  1-minute candles (7 days)
✅ SUCCESS
   Bars inserted: 2,730
   Duration: 3.2s
   ⏳ Waiting 13s for rate limit...

⏱️  5-minute candles (30 days)
✅ SUCCESS
   Bars inserted: 2,340
   Duration: 2.8s
   ⏳ Waiting 13s for rate limit...

[... repeats for QQQ, IWM, UVXY ...]

======================================
Summary
======================================

Total jobs: 8
✅ Successful: 8
❌ Failed: 0
📊 Total bars inserted: 18,680
```

### Step 5: Final Documentation (20 min)

Same as Option A Step 4

---

## What Each Script Does

### 1. apply-volume-migration.sql
**Purpose:** Change volume column from BigInt to Decimal(18,2)

**Why needed:** UVXY has fractional volumes (e.g., 5140.2) which BigInt cannot store

**Safe to run:** Yes, uses `USING volume::DECIMAL(18,2)` to convert existing data

### 2. test-uvxy-ingestion.ts
**Purpose:** Verify UVXY data can now be ingested

**Tests:**
- Ingest 30 days of UVXY 1h data
- Ingest 2 years of UVXY 1d data
- Check for fractional volumes in database

**Duration:** 30 seconds (includes 13s rate limit wait)

### 3. add-scalping-timeframes.ts
**Purpose:** Add 1m and 5m data for all tickers

**Why needed:** Week 2 scalping mode requires minute-level data

**Duration:** ~2 hours (8 ingestion jobs × 13s rate limit = 104s per ticker)

**Can skip:** Yes, can add later when actually building scalping mode

### 4. benchmark-database.ts
**Purpose:** Verify database performance meets Week 1 targets

**Tests:**
- Bulk insert speed
- Query performance
- Index efficiency
- Aggregation speed

**Duration:** 10 seconds

**Cleans up:** Yes, deletes benchmark data after testing

---

## Troubleshooting

### Issue: Migration fails with "column already altered"

**Solution:** Column was already migrated, skip to Step 2

### Issue: UVXY test fails with same BigInt error

**Solution:** Migration didn't apply. Verify in Supabase:
```sql
SELECT column_name, data_type, numeric_precision, numeric_scale
FROM information_schema.columns
WHERE table_name = 'market_data' AND column_name = 'volume';
```

Should show: `volume | numeric | 18 | 2`

### Issue: Scalping script fails on first ticker

**Solution:** Polygon.io rate limit hit. Wait 60 seconds, restart script

### Issue: Benchmarks fail with performance targets

**Solution:** Database might be under load. Try again in a few minutes.

---

## Files Created for Days 4-5

### Scripts (4 files)
1. `scripts/apply-volume-migration.sql` - Migration + verification
2. `scripts/test-uvxy-ingestion.ts` - UVXY test suite
3. `scripts/add-scalping-timeframes.ts` - 1m/5m data ingestion
4. `scripts/benchmark-database.ts` - Performance tests

### Documentation (3 files)
5. `ML Plan/Week 01/DAY_4_STATUS.md` - Day 4 status
6. `ML Plan/Week 01/DAY_5_STATUS.md` - Day 5 status
7. `ML Plan/Week 01/DAYS_4-5_EXECUTION_GUIDE.md` - This file

### Migrations (1 file)
8. `supabase/migrations/002_volume_to_decimal.sql` - Migration script

**Total:** 8 files, ~1,000 lines

---

## Checklist

### Pre-Flight ✅
- [x] Prisma schema updated
- [x] Code updated (no BigInt conversions)
- [x] Prisma client regenerated
- [x] Scripts created and ready
- [x] Documentation complete

### Execution ⏳
- [ ] Migration applied in Supabase
- [ ] UVXY test passed
- [ ] Benchmarks passed
- [ ] Documentation updated
- [ ] (Optional) Scalping data added

### Post-Flight
- [ ] Week 1 marked complete
- [ ] Ready to start Week 2
- [ ] FVG detection can begin

---

## Recommendation

**Start with Option A** (30 min minimum path)

**Reasoning:**
1. ✅ Fixes critical UVXY blocker
2. ✅ Validates performance
3. ✅ Gets you to Week 2 fastest
4. ⏳ Can add scalping data later (during Week 2 Day 2)

**When to do Option B:**
- If you want Week 1 100% complete before moving on
- If you have 2 hours available now
- If you want to test scalping mode immediately in Week 2

---

## Commands Summary

```bash
# Step 1: Copy migration SQL (run this, then paste in Supabase)
cat scripts/apply-volume-migration.sql

# Step 2: Test UVXY
npx ts-node scripts/test-uvxy-ingestion.ts

# Step 3: Benchmarks
npx ts-node scripts/benchmark-database.ts

# Step 4 (Optional): Scalping data
npx ts-node scripts/add-scalping-timeframes.ts
```

---

**Ready to execute?** Follow Option A (30 min) to complete Week 1! 🚀

**Last Updated:** November 6, 2025
