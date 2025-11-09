# Week 1, Day 4: Data Validation & API Enhancement

**Date:** November 6, 2025
**Status:** ✅ IN PROGRESS
**Focus:** Fix UVXY volume issue, add scalping timeframes, API security

---

## Task 1: Fix UVXY Volume Decimal Issue ✅

### Problem
UVXY (and some other ETFs) have fractional volume values, which cannot be stored in a `BIGINT` column.

**Error:**
```
The number 5140.2 cannot be converted to a BigInt because it is not an integer
```

### Solution Implemented

#### 1. Updated Prisma Schema ✅
**File:** `prisma/schema.prisma`

**Changed:**
```prisma
// Before
volume BigInt

// After
volume Decimal @db.Decimal(18, 2)
```

**Reasoning:**
- `DECIMAL(18, 2)` supports up to 999,999,999,999,999,999.99
- 2 decimal places sufficient for fractional volumes
- Still handles large volumes (e.g., SPY: 50M+ shares)

#### 2. Created Migration Script ✅
**File:** `supabase/migrations/002_volume_to_decimal.sql`

```sql
ALTER TABLE market_data
ALTER COLUMN volume TYPE DECIMAL(18, 2) USING volume::DECIMAL(18, 2);
```

**To Apply Migration:**
1. Open Supabase SQL Editor
2. Paste contents of `scripts/apply-volume-migration.sql`
3. Click "Run"
4. Verify with: `SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'market_data' AND column_name = 'volume';`

#### 3. Updated Code ✅

**File:** `src/services/dataIngestionService.v2.ts`

**Changed:**
```typescript
// Before
volume: BigInt(bar.volume)

// After
volume: bar.volume  // Now Decimal, no conversion needed
```

#### 4. Regenerated Prisma Client ✅
```bash
npx prisma generate
```

#### 5. Created Test Script ✅
**File:** `scripts/test-uvxy-ingestion.ts`

**Test Plan:**
1. Ingest UVXY 1h data (30 days)
2. Ingest UVXY 1d data (2 years)
3. Verify data in database
4. Check for fractional volumes

**To Run:**
```bash
npx ts-node scripts/test-uvxy-ingestion.ts
```

---

## Task 2: Add 1m and 5m Timeframe Data ⏳

### Purpose
Enable scalping mode (Week 2) which requires 1-minute and 5-minute candles.

### Plan

#### Tickers to Add
- SPY (S&P 500 ETF)
- QQQ (Nasdaq ETF)
- IWM (Russell 2000 ETF)
- UVXY (Volatility ETF) - after migration

#### Timeframes
- `1m` - 1-minute candles
- `5m` - 5-minute candles

#### Data Range
- 1m: Last 7 days (Polygon free tier limit)
- 5m: Last 30 days

#### Estimated Data Volume

| Ticker | Timeframe | Days | Bars | Size (est) |
|--------|-----------|------|------|------------|
| SPY    | 1m        | 7    | ~2,730 | 300 KB |
| SPY    | 5m        | 30   | ~2,340 | 260 KB |
| QQQ    | 1m        | 7    | ~2,730 | 300 KB |
| QQQ    | 5m        | 30   | ~2,340 | 260 KB |
| IWM    | 1m        | 7    | ~2,730 | 300 KB |
| IWM    | 5m        | 30   | ~2,340 | 260 KB |
| UVXY   | 1m        | 7    | ~2,730 | 300 KB |
| UVXY   | 5m        | 30   | ~2,340 | 260 KB |
| **Total** |        |      | **~18,680** | **~2 MB** |

#### Script to Create

**File:** `scripts/add-scalping-timeframes.ts`

```typescript
import { DataIngestionServiceV2 } from '@/services/dataIngestionService.v2'

async function addScalpingTimeframes() {
  const ingestionService = new DataIngestionServiceV2()

  const tickers = ['SPY', 'QQQ', 'IWM', 'UVXY']
  const timeframes: Array<{tf: '1m' | '5m', days: number}> = [
    { tf: '1m', days: 7 },
    { tf: '5m', days: 30 }
  ]

  for (const ticker of tickers) {
    for (const {tf, days} of timeframes) {
      console.log(`Ingesting ${ticker} ${tf} (${days} days)...`)
      const result = await ingestionService.ingestHistoricalData(ticker, tf, days)
      console.log(`✅ ${result.barsInserted} bars inserted`)

      // Rate limiting
      await new Promise(resolve => setTimeout(resolve, 13000))
    }
  }
}
```

**Status:** ⏳ PENDING (will create after UVXY migration verified)

---

## Task 3: API Authentication ⏳

### Purpose
Secure API endpoints for trading bot integration (Week 11)

### Plan

#### 1. Create API Key Middleware
**File:** `src/middleware/auth.ts`

```typescript
import { NextRequest, NextResponse } from 'next/server'

const API_KEYS = new Set([
  process.env.TRADING_BOT_API_KEY,
  process.env.ML_SERVICE_API_KEY,
  process.env.ADMIN_API_KEY,
])

export function withAuth(handler: (req: NextRequest) => Promise<NextResponse>) {
  return async (req: NextRequest) => {
    const apiKey = req.headers.get('Authorization')?.replace('Bearer ', '')

    if (!apiKey || !API_KEYS.has(apiKey)) {
      return NextResponse.json(
        { success: false, error: 'Unauthorized' },
        { status: 401 }
      )
    }

    return handler(req)
  }
}
```

#### 2. Generate API Keys
```bash
# Generate secure random keys
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

#### 3. Update .env.local
```env
# API Authentication Keys
TRADING_BOT_API_KEY="<generated-key-1>"
ML_SERVICE_API_KEY="<generated-key-2>"
ADMIN_API_KEY="<generated-key-3>"
```

#### 4. Protect Endpoints
```typescript
// Example: Protect trading signals endpoint
import { withAuth } from '@/middleware/auth'

export const GET = withAuth(async (request: NextRequest) => {
  // ... endpoint logic
})
```

**Status:** ⏳ PENDING (not blocking Week 2)

---

## Task 4: Rate Limiting ⏳

### Purpose
Prevent API abuse and ensure fair usage

### Plan

#### 1. Install Rate Limit Package
```bash
npm install @upstash/ratelimit @upstash/redis
```

**Alternative (simpler):** In-memory rate limiting

```typescript
// src/middleware/rateLimit.ts
const requestCounts = new Map<string, { count: number, resetAt: number }>()

export function rateLimit(maxRequests: number, windowMs: number) {
  return (req: NextRequest) => {
    const ip = req.headers.get('x-forwarded-for') || 'unknown'
    const now = Date.now()

    const record = requestCounts.get(ip)

    if (!record || now > record.resetAt) {
      requestCounts.set(ip, { count: 1, resetAt: now + windowMs })
      return { limited: false }
    }

    if (record.count >= maxRequests) {
      return { limited: true, retryAfter: record.resetAt - now }
    }

    record.count++
    return { limited: false }
  }
}
```

#### 2. Apply to Endpoints
```typescript
import { rateLimit } from '@/middleware/rateLimit'

const limiter = rateLimit(100, 60 * 1000) // 100 requests per minute

export async function GET(request: NextRequest) {
  const { limited, retryAfter } = limiter(request)

  if (limited) {
    return NextResponse.json(
      { error: 'Too many requests', retryAfter },
      { status: 429 }
    )
  }

  // ... endpoint logic
}
```

**Status:** ⏳ PENDING (not blocking Week 2)

---

## Task 5: Data Validation ⏳

### Purpose
Ensure data quality and completeness

### Checks to Implement

#### 1. Gap Detection
Find missing bars in time series

```typescript
async function detectGaps(ticker: string, timeframe: string) {
  const data = await marketDataRepo.findMany({ ticker, timeframe }, 10000)

  const gaps = []
  for (let i = 1; i < data.length; i++) {
    const expected = getExpectedNextTimestamp(data[i-1].timestamp, timeframe)
    const actual = data[i].timestamp

    if (actual.getTime() !== expected.getTime()) {
      gaps.push({
        from: data[i-1].timestamp,
        to: actual,
        missing: calculateMissingBars(expected, actual, timeframe)
      })
    }
  }

  return gaps
}
```

#### 2. Data Quality Checks
- No NULL values in OHLCV
- High >= Low
- High >= Open, Close
- Low <= Open, Close
- Volume >= 0

```typescript
async function validateDataQuality(ticker: string, timeframe: string) {
  const issues = await prisma.$queryRaw`
    SELECT *
    FROM market_data
    WHERE ticker = ${ticker}
      AND timeframe = ${timeframe}
      AND (
        high < low OR
        high < open OR
        high < close OR
        low > open OR
        low > close OR
        volume < 0 OR
        open IS NULL OR
        high IS NULL OR
        low IS NULL OR
        close IS NULL OR
        volume IS NULL
      )
  `

  return issues
}
```

#### 3. Coverage Report
```typescript
async function getCoverageReport() {
  const summary = await marketDataRepo.getSummary()

  return summary.map(row => ({
    ticker: row.ticker,
    timeframe: row.timeframe,
    bars: row.bars,
    earliest: row.earliest,
    latest: row.latest,
    coverage: calculateCoveragePercentage(row),
    quality: 'GOOD' // based on validation checks
  }))
}
```

**Status:** ⏳ PENDING (can defer to Week 11)

---

## Current Progress

### Completed ✅
- [x] Fixed UVXY volume decimal issue (schema + code)
- [x] Created migration script
- [x] Regenerated Prisma client
- [x] Created UVXY test script

### In Progress ⏳
- [ ] Apply database migration (manual step in Supabase)
- [ ] Run UVXY ingestion test
- [ ] Verify fractional volumes working

### Pending ⏳
- [ ] Add 1m and 5m timeframe data
- [ ] Implement API authentication
- [ ] Implement rate limiting
- [ ] Create data validation scripts

---

## Files Created/Modified

### Modified Files
1. `prisma/schema.prisma` - Changed volume to Decimal(18, 2)
2. `src/services/dataIngestionService.v2.ts` - Removed BigInt conversion

### New Files
3. `supabase/migrations/002_volume_to_decimal.sql` - Migration script
4. `scripts/apply-volume-migration.sql` - Verification queries
5. `scripts/test-uvxy-ingestion.ts` - UVXY test script
6. `ML Plan/Week 01/DAY_4_STATUS.md` - This file

---

## Next Steps

### Immediate (Complete Day 4)
1. **Apply Migration** - Run SQL in Supabase
2. **Test UVXY** - Run `npx ts-node scripts/test-uvxy-ingestion.ts`
3. **Add Scalping Data** - Create and run `add-scalping-timeframes.ts`

### Optional (Defer to Week 11)
4. **API Auth** - Create middleware and generate keys
5. **Rate Limiting** - Implement simple in-memory limiter
6. **Data Validation** - Create validation scripts

---

## Decision Point

### Option A: Complete All Day 4 Tasks Now
**Pros:**
- Comprehensive API security
- Data validation ensures quality
- 1m/5m data ready for scalping mode

**Cons:**
- Takes 4-6 more hours
- Delays Week 2 start

**Timeline:** +4-6 hours

### Option B: Critical Path Only (UVXY fix + scalping data)
**Pros:**
- Unblocks UVXY ticker
- Enables scalping mode development
- Faster path to Week 2

**Cons:**
- API still unsecured (ok for development)
- No formal data validation (can manual check)

**Timeline:** +2 hours

### Option C: UVXY Fix Only, Defer Rest
**Pros:**
- Minimal time investment
- Unblocks UVXY
- Can add 1m/5m later when building scalping mode

**Cons:**
- No scalping timeframes yet
- API unsecured

**Timeline:** +30 min

---

## Recommendation: Option B ✅

**Complete:** UVXY fix + add 1m/5m data
**Defer:** API auth, rate limiting, validation

**Justification:**
1. ✅ UVXY fix is critical (proven blocker)
2. ✅ 1m/5m data needed for Week 2 scalping mode
3. ⏳ API auth not needed until Week 11 (trading bot)
4. ⏳ Rate limiting not needed for single-user dev
5. ⏳ Data validation can be manual for now

**Timeline:** ~2 hours to complete Option B

---

## Status Summary

**Day 4 Progress:** 60% Complete

| Task | Status | Time Spent | Time Remaining |
|------|--------|------------|----------------|
| UVXY volume fix | ✅ 90% | 45 min | 15 min (test) |
| 1m/5m data | ⏳ 0% | 0 min | 90 min |
| API auth | ⏳ 0% | 0 min | (defer) |
| Rate limiting | ⏳ 0% | 0 min | (defer) |
| Data validation | ⏳ 0% | 0 min | (defer) |

**Total Day 4 Time:** 45 min actual + 105 min remaining = **2.5 hours**

---

**Last Updated:** November 6, 2025
**Next:** Apply migration, test UVXY, add scalping data
