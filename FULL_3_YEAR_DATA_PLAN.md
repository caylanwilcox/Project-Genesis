# Full 3-Year Dataset Loading Plan

## Current Status

✅ **Database cleared** - Ready for fresh 3-year data
✅ **Code fixed** - Removed 100-bar cap in `calculateLimit()` function
⚠️ **Server crashes** - Large batch inserts causing memory issues

### Data Loaded So Far
- SPY 1d: 1,095 bars ✅
- QQQ 1d: 1,095 bars ✅
- IWM 1d: Failed (server crash)
- UVXY 1d: Not attempted

## The Problem

Loading 3 years of data causes the Next.js dev server to crash due to:
1. **Memory constraints** - Inserting 1,095+ bars in single transactions
2. **Database connection pooling** - Too many concurrent Prisma queries
3. **Next.js hot reload** - Dev server not optimized for bulk operations

## Solutions (Choose One)

### Option A: Load Data via Supabase SQL Editor (RECOMMENDED)

**Fastest and most reliable - 5 minutes**

1. Export data from Polygon.io using their bulk download API
2. Format as SQL INSERT statements
3. Run directly in Supabase SQL Editor
4. No server needed, no crashes

### Option B: Use Production Build Instead of Dev Server

**More stable for bulk operations - 15 minutes**

```bash
# Build for production
npm run build

# Run production server (more stable)
npm start

# Then run Python script
python3 scripts/load-full-dataset.py
```

Production build has:
- Better memory management
- No hot reload overhead
- Optimized database connections

### Option C: Load Data in Smaller Batches

**Slower but works with dev server - 30 minutes**

Modify script to load smaller chunks:
- Instead of 1095 days at once, load 365 days × 3 iterations
- Add sleep delays between batches
- Monitor server health

### Option D: Use Direct Database Connection (Best Long-term)

**Create standalone Node.js script - 10 minutes setup**

Create `scripts/bulk-load.ts` that:
- Connects directly to PostgreSQL (bypass Next.js)
- Uses batch inserts with proper transaction management
- Handles errors and retries gracefully
- Shows progress bar

```typescript
// Example structure
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

async function bulkLoad() {
  // Load data in batches of 100 bars
  // Commit every batch
  // Show progress
}
```

## Recommendation

**Use Option B (Production Build)** for now:

1. Kill all dev servers
2. Build production: `npm run build`
3. Start production: `npm start` (runs on port 3000)
4. Run: `python3 scripts/load-full-dataset.py`

This should complete all 16 ingestion jobs (4 tickers × 4 timeframes) in ~10-15 minutes without crashes.

## Expected Final Dataset

| Timeframe | Days Back | Bars per Ticker | Total (4 tickers) |
|-----------|-----------|-----------------|-------------------|
| 1d        | 1,095     | ~1,095          | ~4,380            |
| 1h        | 1,095     | ~7,117          | ~28,468           |
| 5m        | 30        | ~2,340          | ~9,360            |
| 1m        | 7         | ~2,730          | ~10,920           |
| **TOTAL** |           |                 | **~53,128 bars**  |

### Train/Test Split
- **Training**: 2/3 = ~730 days (May 2022 - May 2024)
- **Testing**: 1/3 = ~365 days (May 2024 - Nov 2025)

## Files Modified

1. ✅ [dataIngestionService.v2.ts](src/services/dataIngestionService.v2.ts#L195-L218) - Fixed `calculateLimit()`
2. ✅ [load-full-dataset.py](scripts/load-full-dataset.py) - Python ingestion script
3. ✅ [.env.local](.env.local) - Set `NEXT_PUBLIC_POLYGON_PLAN=starter`

## Next Steps After Data Loading

1. Verify data: `curl http://localhost:3000/api/v2/data/ingest/status`
2. Check train/test split dates
3. Update Week 1 documentation
4. Begin Week 2: Feature Engineering
