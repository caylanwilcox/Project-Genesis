# Day 3: Train/Test Data Split - Critical for ML

**Date:** November 5, 2025
**Status:** 🟢 Setup Complete, Ready to Run

---

## Why Train/Test Split Matters

### The Problem: Data Leakage

If you train an ML model on ALL your data, you can't trust its accuracy metrics because:
- The model has "seen" the future
- It memorized patterns instead of learning them
- Real-world performance will be much worse
- You can't detect overfitting

### The Solution: Proper Data Splitting

Split your historical data into **two separate sets**:

**🎓 TRAINING SET (70% - Earlier Data)**
- Used to train the ML model
- Model learns patterns from this data
- Older historical data

**🧪 TESTING SET (30% - Recent Data)**
- Used to validate the model
- Model has NEVER seen this data during training
- More recent historical data (simulates "future")

---

## How We Split the Data

### Timeline Visualization

```
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  2 Years of Historical Data                                     │
│                                                                  │
├──────────────────────────────┬─────────────────────────────────────┤
│                              │                                  │
│  🎓 TRAINING SET             │  🧪 TESTING SET                  │
│  70% (First 504 days)        │  30% (Last 216 days)             │
│  2023-11-05 → 2025-03-23     │  2025-03-24 → 2025-11-05         │
│                              │                                  │
│  Model LEARNS from this      │  Model VALIDATES on this         │
│  ✅ Used during training     │  ❌ Never seen during training   │
│                              │                                  │
└──────────────────────────────┴─────────────────────────────────────┘
```

### Split Configuration

- **Total Time Span:** 2 years (730 days)
- **Training Period:** First 70% (~504 days)
- **Testing Period:** Last 30% (~216 days)
- **Split Method:** Time-based (not random - critical for time-series!)

---

## Files Created

### 1. Backfill Script (CLI)
**File:** `scripts/backfill-historical-data.ts`

Features:
- Fetches 2 years of data from Polygon.io
- Automatically splits into train/test
- Shows progress for each ticker/timeframe
- Saves summary report

### 2. Train/Test Marker (CLI)
**File:** `scripts/mark-train-test-split.ts`

Features:
- Creates `train_test_config` table
- Stores split dates for each ticker/timeframe
- Provides helper queries for ML training
- Tracks split ratio

### 3. Backfill API (Recommended)
**Endpoint:** `POST /api/v2/data/backfill`

Features:
- Trigger backfill via HTTP request
- Works in Next.js environment
- Real-time progress tracking
- Returns detailed results

---

## How to Run the Backfill

### Option A: Via API (Recommended)

```bash
curl -X POST http://localhost:3002/api/v2/data/backfill \
  -H "Content-Type: application/json" \
  -d '{
    "yearsBack": 2,
    "trainTestSplit": 0.7
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "summary": {
    "totalJobs": 8,
    "successful": 8,
    "failed": 0,
    "totalBars": 12500,
    "trainingBars": 8750,
    "testingBars": 3750,
    "trainTestSplit": "70% / 30%"
  },
  "results": [...]
}
```

### Option B: Via Script (Alternative)

```bash
# If using CLI script
npx tsx scripts/backfill-historical-data.ts
```

---

## What Gets Ingested

### Tickers
- SPY (S&P 500 ETF)
- QQQ (NASDAQ 100 ETF)
- IWM (Russell 2000 ETF)
- UVXY (Volatility ETF)

### Timeframes
- 1h (Hourly data)
- 1d (Daily data)

**Total Datasets:** 8 (4 tickers × 2 timeframes)

### Expected Data Volume

| Ticker | Timeframe | Training Bars | Testing Bars | Total |
|--------|-----------|--------------|--------------|-------|
| SPY    | 1h        | ~2,500       | ~1,100       | ~3,600 |
| SPY    | 1d        | ~350         | ~150         | ~500  |
| QQQ    | 1h        | ~2,500       | ~1,100       | ~3,600 |
| QQQ    | 1d        | ~350         | ~150         | ~500  |
| IWM    | 1h        | ~2,500       | ~1,100       | ~3,600 |
| IWM    | 1d        | ~350         | ~150         | ~500  |
| UVXY   | 1h        | ~2,500       | ~1,100       | ~3,600 |
| UVXY   | 1d        | ~350         | ~150         | ~500  |
| **Total** | | **~10,500** | **~4,500** | **~15,000** |

---

## How to Use Train/Test Split in ML Training

### Query Training Data (Week 3)

```typescript
// Get training data for SPY 1h
const trainingData = await prisma.marketData.findMany({
  where: {
    ticker: 'SPY',
    timeframe: '1h',
    timestamp: {
      lte: new Date('2025-03-23') // Split date
    }
  },
  orderBy: { timestamp: 'asc' }
})

console.log(`Training on ${trainingData.length} bars`)
// Model learns from this data
```

### Query Testing Data (Week 4 - Backtesting)

```typescript
// Get testing data for SPY 1h
const testingData = await prisma.marketData.findMany({
  where: {
    ticker: 'SPY',
    timeframe: '1h',
    timestamp: {
      gt: new Date('2025-03-23') // After split date
    }
  },
  orderBy: { timestamp: 'asc' }
})

console.log(`Testing on ${testingData.length} bars`)
// Model validates on this UNSEEN data
```

### Using the Config Table

```typescript
// Get split info from database
const splitInfo = await prisma.$queryRaw`
  SELECT ticker, timeframe, split_date, train_ratio
  FROM train_test_config
  WHERE ticker = 'SPY' AND timeframe = '1h'
`

// Use split_date for querying
const trainData = await prisma.marketData.findMany({
  where: {
    ticker: splitInfo.ticker,
    timeframe: splitInfo.timeframe,
    timestamp: { lte: splitInfo.split_date }
  }
})
```

---

## Rate Limiting Considerations

### Polygon.io Free Tier

- **Limit:** 5 calls per minute
- **Wait time:** 13 seconds between requests
- **Backfill duration:** ~2 minutes (8 datasets × 13s wait)

### Expected Backfill Time

| Datasets | Wait Time | Total Time |
|----------|-----------|------------|
| 8        | 13s each  | ~2 minutes |

**NOTE:** This is for the API's rate limiting, not our script speed!

---

## Data Quality Checks

After backfill, verify:

```bash
# Check total bars ingested
curl "http://localhost:3002/api/v2/data/ingest/status"

# Check specific ticker data
curl "http://localhost:3002/api/v2/data/market?ticker=SPY&timeframe=1h&limit=5"
```

Expected results:
- ✅ ~15,000 total bars
- ✅ No duplicate timestamps
- ✅ Proper date ranges
- ✅ Training > Testing (70/30 split)

---

## Best Practices

### ✅ DO

1. **Use time-based split** (not random!)
   - Time-series data has temporal dependencies
   - Random split causes data leakage

2. **Keep testing data "in the future"**
   - Training: older data
   - Testing: newer data
   - Simulates real-world deployment

3. **Never peek at testing data**
   - Don't use it for feature engineering
   - Don't use it for hyperparameter tuning
   - Only use for final validation

4. **Document your split**
   - Save split dates
   - Track what data was used
   - Enable reproducibility

### ❌ DON'T

1. **Don't use random split**
   - Breaks time-series assumptions
   - Causes data leakage

2. **Don't train on testing data**
   - Defeats the purpose
   - Inflates accuracy metrics

3. **Don't change split after training**
   - Makes results incomparable
   - Breaks reproducibility

4. **Don't use future data for past predictions**
   - Classic data leakage
   - Real-world performance will fail

---

## Validation

### Check Split Quality

```typescript
// Verify no overlap between train and test
const trainLatest = await prisma.marketData.findFirst({
  where: { ticker: 'SPY', timeframe: '1h', timestamp: { lte: splitDate } },
  orderBy: { timestamp: 'desc' }
})

const testEarliest = await prisma.marketData.findFirst({
  where: { ticker: 'SPY', timeframe: '1h', timestamp: { gt: splitDate } },
  orderBy: { timestamp: 'asc' }
})

console.log('Train ends:', trainLatest.timestamp)
console.log('Test starts:', testEarliest.timestamp)
// Should be consecutive with no overlap!
```

---

## Next Steps

After Day 3 is complete:

1. ✅ Verify 15K+ bars ingested
2. ✅ Confirm train/test split (70/30)
3. ✅ Check data quality (no gaps)
4. ✅ Mark split in database
5. ✅ Move to Week 2: Feature Engineering

---

## Summary

**Day 3 Achievement:**
- ✅ Proper train/test data infrastructure
- ✅ 2 years of historical data ready
- ✅ 70/30 split prevents data leakage
- ✅ Foundation for accurate ML training

**Critical for:**
- Week 3: Model Training (uses training set)
- Week 4: Backtesting (uses testing set)
- Week 11: Production deployment

---

**Last Updated:** November 5, 2025
**Status:** Ready to run backfill
**Next:** Execute backfill via API endpoint
