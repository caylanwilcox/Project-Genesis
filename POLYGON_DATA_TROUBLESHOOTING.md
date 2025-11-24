# Polygon Data Accuracy Troubleshooting Guide

## 🔍 Common Issues with SPY & Intervals

### Issue 1: Incorrect Ticker Symbol
**Problem**: Polygon requires exact ticker symbols
**Solution**: Always use uppercase (SPY not spy)
**Status**: ✅ Already handled in code (`symbol?.toUpperCase()`)

### Issue 2: Timeframe Mapping Mismatch
**Problem**: UI labels don't match API timeframes
**Check**: Verify the mapping in `timeframePolicy.ts`

Current mapping:
```typescript
'1D'  → '5m'   (5-minute bars)
'5D'  → '30m'  (30-minute bars)
'1M'  → '1h'   (1-hour bars)
'3M'  → '4h'   (4-hour bars)
'6M'  → '1d'   (daily bars)
'YTD' → '1d'   (daily bars)
'1Y'  → '1d'   (daily bars)
'5Y'  → '1w'   (weekly bars)
```

**Interval dropdown mapping:**
```typescript
'1 min'   → '1m'
'5 min'   → '5m'
'15 min'  → '15m'
'30 min'  → '30m'
'1 hour'  → '1h'
'2 hour'  → '2h'
'4 hour'  → '4h'
'1 day'   → '1d'
'1 week'  → '1w'
'1 month' → '1M'
```

### Issue 3: Market Hours vs Extended Hours
**Problem**: SPY trades during regular hours only (9:30 AM - 4:00 PM ET)
**Impact**: Data outside these hours will be sparse or missing

**Check if you're seeing**:
- Gaps in intraday data
- Missing bars during lunch
- No data after 4 PM ET

**Solutions**:
1. Filter to regular market hours only
2. Adjust expectations for after-hours data
3. Use daily bars for after-hours analysis

### Issue 4: Date Range Calculation
**Problem**: Requesting data from wrong date range
**Debug**: Check console for these logs:
```
[PolygonService] Requesting {limit} bars of {timeframe} data for SPY from {from} to {to}
```

**Common date range issues**:
- Weekends included (markets closed)
- Holidays included (markets closed)
- Asking for 100 1-minute bars but spanning multiple days
- Not accounting for trading hours (6.5 hours = 390 1-min bars)

### Issue 5: Bar Limit vs Available Data
**Problem**: Asking for more bars than exist
**Example**:
- Request: 1000 1-minute bars for SPY
- Reality: Only 390 bars exist per trading day
- Result: Only get 390 bars, not 1000

**Check**: `getBarLimit()` function in ticker page
```typescript
const getBarLimit = (tf: Timeframe, displayTf: string): number =>
  recommendedBarLimit(tf, displayTf) + additionalBarsToLoad
```

**Recommended limits** (from `timeframePolicy.ts`):
- 1D + 5m bars = 78 bars (6.5 trading hours)
- 5D + 30m bars = 65 bars
- 1M + 1h bars = 140 bars
- etc.

### Issue 6: Delayed Data (Free Tier)
**Problem**: Polygon free tier has 15-minute delay
**Impact**: "Current" data is actually 15 minutes old
**Solution**: Upgrade to paid tier for real-time data

## 🛠️ Debugging Steps

### Step 1: Check Browser Console
Look for these log messages:
```
[usePolygonData] Fetching data for SPY - Timeframe: 5m, Limit: 78, Display: 1D
[PolygonService] Requesting 78 bars of 5m data for SPY from 2024-11-18 to 2024-11-24
[PolygonService] API Response: { resultsCount: 78, ... }
[usePolygonData] Fetched 78 bars for SPY
```

**Red flags**:
- ❌ `resultsCount: 0` - No data returned
- ❌ `Error: 429` - Rate limit exceeded
- ❌ `Error: 403` - API key invalid
- ❌ Wrong date range

### Step 2: Verify API Key
**Check**: `.env.local` file
```bash
NEXT_PUBLIC_POLYGON_API_KEY=your_key_here
NEXT_PUBLIC_POLYGON_PLAN=advanced  # or free/starter/developer
```

**Test API key**:
```bash
curl "https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2024-11-01/2024-11-23?apiKey=YOUR_KEY"
```

### Step 3: Check Timeframe State
Add console logging to track state:
```typescript
// In ticker page
console.log('Current timeframe:', timeframe, 'Display:', displayTimeframe)
```

**Expected flow**:
1. Click "1D" → timeframe='5m', displayTimeframe='1D'
2. Click "15 min" → timeframe='15m', displayTimeframe='Custom'
3. Data fetches with correct timeframe

### Step 4: Inspect Network Tab
1. Open DevTools → Network tab
2. Filter: `polygon.io`
3. Look for requests to:
   - `/v2/aggs/ticker/SPY/...`
   - `/v2/snapshot/...`
   - `/v2/aggs/previous-close/...`

**Check**:
- ✅ Request URL has correct multiplier/timespan
- ✅ Date range makes sense
- ✅ Response status 200
- ✅ Response has `results` array with data

### Step 5: Test with Known Good Data
**Try this in browser console**:
```javascript
fetch('https://api.polygon.io/v2/aggs/ticker/SPY/range/5/minute/2024-11-22/2024-11-23?adjusted=true&sort=asc&limit=500&apiKey=YOUR_KEY')
  .then(r => r.json())
  .then(d => console.log('Data:', d))
```

**Expected**: Array of bars with OHLCV data

## 🔧 Common Fixes

### Fix 1: Wrong Interval for SPY
**Problem**: Requesting sub-1-minute data (not available for SPY on free tier)
**Solution**: Use 1-minute or higher intervals

### Fix 2: Requesting Too Much Historical Data
**Problem**: Free tier has limits on how far back you can query
**Solution**: Reduce bar limit or upgrade plan

### Fix 3: Weekend/Holiday Data
**Problem**: No trading on weekends/holidays
**Solution**: Date range calculation should skip non-trading days

Current code handles this in `calculateDateRange()`:
```typescript
// Adjust for weekends and holidays by going back further
const daysMultiplier = isIntraday ? 3.0 : 1.5;
```

### Fix 4: Stale Cache
**Problem**: Seeing old data even after timeframe change
**Solution**: Cache key includes timeframe + limit + display

Cache key format:
```typescript
`aggs_${ticker}_${timeframe}_${limit}_${displayTimeframe || 'default'}`
```

**To clear cache**: Hard refresh (Cmd+Shift+R / Ctrl+Shift+F5)

### Fix 5: Rate Limiting
**Problem**: Making too many requests
**Solution**: Queue system with rate limiting

Current settings:
- Free tier: 13 seconds between requests
- Paid tier: No delay

## 📊 Verification Checklist

- [ ] API key is valid and in `.env.local`
- [ ] Ticker symbol is uppercase (SPY)
- [ ] Timeframe mapping is correct
- [ ] Date range includes trading days
- [ ] Bar limit is reasonable
- [ ] Console shows successful data fetch
- [ ] Network tab shows 200 responses
- [ ] Chart displays non-zero number of candles

## 🚀 Quick Test

1. Open chart for SPY
2. Open browser console
3. Click "1D" timeframe
4. Look for:
   ```
   [usePolygonData] Fetching data for SPY - Timeframe: 5m, Limit: 78
   [usePolygonData] Fetched 78 bars for SPY
   ```
5. Chart should show ~78 5-minute candles

## 📞 Still Having Issues?

**Debug output to share**:
1. Console logs (especially `[PolygonService]` and `[usePolygonData]`)
2. Network request URL
3. API response (first few bars)
4. Timeframe you're trying to view
5. Plan tier (free/paid)

**Check**:
- Polygon.io dashboard: https://polygon.io/dashboard
- API status: https://status.polygon.io/
- Documentation: https://polygon.io/docs/stocks/getting-started
