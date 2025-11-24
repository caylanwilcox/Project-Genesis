# Polygon Data Accuracy Fixes for SPY

## 🐛 Identified Issues

### 1. **Initialization Mismatch**
**Problem**: Chart component defaults to '1D' but parent state might differ
**Impact**: First render might show wrong data until state synchronizes

**Current Flow**:
```
TickerPage initializes:
  - displayTimeframe = '1D'
  - timeframe = '5m' (from resolveDisplayToData('1D'))

ProfessionalChart/useTimeframeState initializes:
  - displayTimeframe = '1D'
  - dataTimeframe = '5m'

✅ These match, so no issue here
```

### 2. **Interval Dropdown Default Mismatch**
**Problem**: Chart shows '1 hour' in dropdown but should show '5 min' for 1D timeframe

**Fix Applied**: Updated default interval in useTimeframeState to match mapped value

### 3. **Console Logging Missing**
**Problem**: Hard to debug what data is being requested
**Solution**: Added comprehensive logging

## ✅ Fixes Applied

### Fix 1: Added Debug Logging to useTimeframeState
```typescript
// Now logs when timeframe/interval changes
console.log('[useTimeframeState] Timeframe clicked:', tf)
console.log('[useTimeframeState] Interval changed to:', newInterval)
```

### Fix 2: Ensured Interval Resets on Timeframe Change
```typescript
handleIntervalChange() now:
  - Sets displayTimeframe to 'Custom'
  - Calls onResetScales()
  - Propagates to parent via onTimeframeChange()
```

### Fix 3: Proper Viewport Reset
```typescript
resetScales() now:
  - Resets panOffset to 0
  - Resets priceOffset to 0
  - Resets priceScale to 1
  - Resets timeScale to 1
```

## 🔍 How to Verify Data is Accurate

### Step 1: Check Browser Console
After clicking "1D" timeframe, you should see:
```
[useTimeframeState] Timeframe clicked: 1D
[TickerPage] Timeframe changed to: 5m Display: 1D
[usePolygonData] Fetching data for SPY - Timeframe: 5m, Limit: 78, Display: 1D
[PolygonService] Requesting 78 bars of 5m data for SPY from YYYY-MM-DD to YYYY-MM-DD
[PolygonService] API Response: { resultsCount: 78, ...}
[usePolygonData] Fetched 78 bars for SPY
```

### Step 2: Verify Data Points
For SPY with 5-minute bars on a trading day:
- **Expected**: ~78 bars (6.5 hours × 12 five-minute periods)
- **Timestamps**: Should be 5 minutes apart
- **Time range**: 9:30 AM - 4:00 PM ET

### Step 3: Check Candle Accuracy
Compare chart with TradingView or Yahoo Finance:
- **Open/High/Low/Close** should match
- **Volume** should be similar
- **Patterns** should align

## 🚨 Common SPY Data Issues

### Issue: "Only showing 20 candles instead of 78"
**Cause**: Viewport calculation limiting visible range
**Check**: `useVisibleRange` hook
**Fix**: Adjust `baseCandlesInView` for '1D' + '5m' combination

### Issue: "Candles show wrong times"
**Cause**: Timezone mismatch
**Fix**: Polygon returns timestamps in milliseconds (Unix time)
**Verify**: Check if `time` field is being converted correctly

### Issue: "Missing recent data"
**Cause 1**: Free tier has 15-minute delay
**Solution**: Upgrade to paid plan

**Cause 2**: Market is closed
**Solution**: Check market hours (9:30 AM - 4:00 PM ET, Mon-Fri)

**Cause 3**: Cache is stale
**Solution**: Hard refresh (Cmd+Shift+R)

### Issue: "Candles don't match other platforms"
**Possible causes**:
1. **Adjusted vs Unadjusted**: Polygon uses adjusted prices by default
2. **Extended hours**: Other platforms might include pre/post market
3. **Data provider differences**: Different data feeds have slight variations
4. **Aggregation method**: How bars are calculated (time-based vs tick-based)

## 📊 Expected Data for SPY

### 1D Timeframe (5-minute bars)
```
Date range: Today (market open to close)
Bars: ~78
First bar: 9:30 AM ET
Last bar: 4:00 PM ET
Interval: 5 minutes
```

### 5D Timeframe (30-minute bars)
```
Date range: Last 5 trading days
Bars: ~65 (5 days × 13 bars per day)
Interval: 30 minutes
```

### 1M Timeframe (1-hour bars)
```
Date range: Last ~21 trading days
Bars: ~140
Interval: 1 hour
```

## 🧪 Test Scenarios

### Test 1: Fresh Page Load
1. Navigate to `/ticker/SPY`
2. Default should be "1D" timeframe
3. Console should show fetching 5m data
4. Chart should display ~78 candles

### Test 2: Timeframe Change
1. Click "5D" button
2. Console should show fetching 30m data
3. Chart should display ~65 candles
4. Pan/zoom should reset

### Test 3: Interval Change
1. Click interval dropdown
2. Select "15 min"
3. Console should show fetching 15m data
4. Display should show "Custom"
5. Chart should update with 15-minute candles

### Test 4: Infinite Scroll
1. Pan left to view older data
2. When reaching left edge, should trigger load more
3. Console should show fetching additional 100 bars
4. Chart should smoothly append older data

## 🔧 Debug Commands

### Check Current State (Browser Console)
```javascript
// In browser console while on chart page
console.log('Chart state:', {
  displayTimeframe: document.querySelector('[class*="timeframe"]')?.textContent,
  // Add more state checks as needed
})
```

### Test API Directly
```bash
# Test 5-minute bars for SPY (replace YOUR_KEY and dates)
curl "https://api.polygon.io/v2/aggs/ticker/SPY/range/5/minute/2024-11-22/2024-11-23?adjusted=true&sort=asc&limit=500&apiKey=YOUR_KEY"
```

### Expected Response
```json
{
  "ticker": "SPY",
  "status": "OK",
  "queryCount": 78,
  "resultsCount": 78,
  "results": [
    {
      "v": 1234567,        // volume
      "vw": 445.6789,      // volume weighted average
      "o": 445.50,         // open
      "c": 445.75,         // close
      "h": 445.80,         // high
      "l": 445.45,         // low
      "t": 1700654400000,  // timestamp (Unix ms)
      "n": 1000            // number of transactions
    },
    // ... more bars
  ]
}
```

## ✅ Verification Checklist

After making changes, verify:

- [ ] Console shows correct timeframe being fetched
- [ ] Number of bars matches expected count
- [ ] Timestamps are sequential and correct interval
- [ ] OHLCV data looks reasonable
- [ ] Chart renders all fetched candles
- [ ] Timeframe button shows correct selection
- [ ] Interval dropdown shows correct value
- [ ] Pan/zoom resets on timeframe change
- [ ] Data updates when changing intervals
- [ ] No console errors

## 📞 If Data Still Inaccurate

**Collect this information**:
1. Symbol: SPY (or other)
2. Timeframe clicked: (e.g., "1D")
3. Expected interval: (e.g., "5 min")
4. Actual data received: (check console logs)
5. Number of bars: (expected vs actual)
6. Screenshot of chart
7. Console logs (especially Polygon API responses)

**Common root causes**:
- API key issues
- Rate limiting
- Market closed (no recent data)
- Wrong date range calculation
- Cache serving stale data
- Timeframe mapping misconfigured
