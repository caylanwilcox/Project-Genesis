# 🔍 Data Accuracy Diagnostic Steps

## Enhanced Logging Added

I've added comprehensive logging to track the data flow. When you open the chart, you'll now see:

```javascript
[useChartOrchestrator] Received externalData: 78 bars, currentPrice: 445.20
[useChartData] Setting external data: 78 bars
[useChartOrchestrator] Using real data: 78 bars
```

OR if using mock data:
```javascript
[useChartOrchestrator] Received externalData: 0 bars, currentPrice: 445.20
[useChartOrchestrator] No data available, using mock data
```

## How to Diagnose the Issue

### Step 1: Open Browser Console
1. Press `F12` (or `Cmd+Option+I` on Mac)
2. Go to the "Console" tab
3. Clear console (trash icon)
4. Navigate to `/ticker/SPY`

### Step 2: Check the Log Sequence

**Look for this complete flow**:

```
1. [usePolygonData] Fetching data for SPY - Timeframe: 5m, Limit: 78, Display: 1D
2. [PolygonService] Requesting 78 bars of 5m data for SPY from 2024-11-22 to 2024-11-24
3. [PolygonService] API Response: { status: 'OK', resultsCount: 78, ... }
4. [usePolygonData] Fetched 78 bars for SPY
5. [useChartOrchestrator] Received externalData: 78 bars, currentPrice: 445.20
6. [useChartData] Setting external data: 78 bars
7. [useChartOrchestrator] Using real data: 78 bars
```

### Step 3: Identify the Problem

#### Scenario A: Mock Data is Being Used
**Logs show**:
```
[useChartOrchestrator] Received externalData: 0 bars, currentPrice: 445.20
[useChartOrchestrator] No data available, using mock data
```

**Cause**: Polygon API is not returning data
**Check**:
- API key in `.env.local`
- Network tab for API request failures
- Console for error messages
- Market is open (not weekend/holiday)

#### Scenario B: Data is Fetched but Incorrect
**Logs show**:
```
[usePolygonData] Fetched 78 bars for SPY
[useChartOrchestrator] Using real data: 78 bars
```

**But chart shows wrong data**

**Possible causes**:
1. **Wrong timeframe** - Check if requested timeframe matches display
2. **Stale cache** - Try hard refresh (Cmd+Shift+R)
3. **Data transformation issue** - Check data normalization

**To verify**:
```javascript
// In browser console, run:
window.__polygonDebugData = true

// Then check the data structure
```

#### Scenario C: No Logs Appear
**Nothing shows up**

**Cause**: Page not loading correctly
**Check**:
- Console for JavaScript errors
- Network tab for failed requests
- Dev server is running

### Step 4: Inspect Actual Data

**In browser console, run**:
```javascript
// This will show you the first 3 bars
const event = new Event('inspectChartData')
window.dispatchEvent(event)
```

**Or manually check**:
```javascript
// Find the chart data in React DevTools
// Look for ProfessionalChart component
// Check the 'data' prop
```

### Step 5: Compare with Polygon API Directly

**Test the API directly** (replace YOUR_KEY and dates):
```bash
curl -s "https://api.polygon.io/v2/aggs/ticker/SPY/range/5/minute/2024-11-22/2024-11-23?adjusted=true&sort=asc&limit=500&apiKey=YOUR_KEY" | jq '.results[0:3]'
```

**Expected response**:
```json
[
  {
    "v": 1234567,
    "vw": 445.6789,
    "o": 445.50,
    "c": 445.75,
    "h": 445.80,
    "l": 445.45,
    "t": 1700654400000,
    "n": 1000
  },
  // ... more bars
]
```

**Compare this with what the chart is showing**

## Common Issues & Fixes

### Issue 1: Showing Mock Data Instead of Real Data

**Symptom**: Chart shows ~120 randomly generated candles
**Log**: `[useChartOrchestrator] No data available, using mock data`

**Fix**:
1. Check `.env.local` has valid API key
2. Check console for API errors
3. Verify market is open (not weekend)
4. Check Network tab for 403/429 errors

### Issue 2: Data is Delayed by 15+ Minutes

**Symptom**: Latest candle is from 15 minutes ago
**Cause**: Polygon free tier has 15-minute delay

**Fix**: Upgrade to paid plan for real-time data

### Issue 3: Wrong Number of Candles

**Symptom**: Expected 78 bars, seeing 120
**Log**: `[useChartOrchestrator] Using real data: 120 bars`

**Cause**: Bar limit calculation is off

**Fix**: Check `getBarLimit()` function in ticker page

### Issue 4: Candle Timestamps are Wrong

**Symptom**: Candles show at wrong times
**Cause**: Timezone conversion issue

**Check**:
```javascript
// In console, inspect first bar
console.log(new Date(chartData[0].time))
// Should show time in your timezone
```

### Issue 5: OHLCV Values Don't Match

**Symptom**: Prices different from TradingView/Yahoo Finance
**Causes**:
1. **Adjusted vs Unadjusted** - Polygon uses adjusted by default
2. **Different data providers** - Slight variations are normal
3. **Extended hours** - Pre/post market data

**Verify**:
- Compare with Polygon's own charts
- Check if `adjusted=true` in API request
- Ensure comparing same time period

## Debug Checklist

Run through this checklist:

- [ ] Browser console is open
- [ ] No JavaScript errors in console
- [ ] API key is set in `.env.local`
- [ ] Dev server is running (port 3000 or 3003)
- [ ] Navigated to `/ticker/SPY`
- [ ] Console shows `[usePolygonData]` logs
- [ ] Console shows `[useChartOrchestrator]` logs
- [ ] Network tab shows API requests to polygon.io
- [ ] API responses have status 200
- [ ] `externalData` count matches `resultsCount`
- [ ] Chart is NOT showing mock data
- [ ] Candle times match expected interval
- [ ] Candle OHLCV values look reasonable

## What to Share for Help

If data is still incorrect, share:

**1. Console Logs** (copy full output):
```
[usePolygonData] Fetching data for SPY - Timeframe: 5m, Limit: 78, Display: 1D
[PolygonService] Requesting 78 bars of 5m data for SPY from 2024-11-22 to 2024-11-24
...
```

**2. Network Request**:
- URL that was called
- Response status
- Response body (first 3 bars)

**3. What You're Seeing**:
- Number of candles displayed
- Time of first candle
- Time of last candle
- OHLC values of latest candle

**4. What You Expected**:
- Based on TradingView or another source
- Time range
- Interval

**5. Environment**:
- Polygon plan (free/paid)
- Time you checked (market open/closed)
- Any error messages

## Quick Verification

**Run this in browser console**:
```javascript
// Check if real data is being used
const chartContainer = document.querySelector('[class*="chartContainer"]')
if (chartContainer) {
  console.log('✅ Chart found')
  // Data logs will show if using real or mock data
} else {
  console.log('❌ Chart not found - page not loaded correctly')
}
```

The enhanced logging will make it very clear where the data pipeline breaks!
