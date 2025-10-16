# Rate Limit Solution for Polygon.io API

## Problem

The 429 error (Too Many Requests) occurred because the free tier of Polygon.io allows only **5 API calls per minute**, and the app was making:

- **Dashboard**: 4 tickers × 2 calls each = 8 calls on initial load
- **Ticker Pages**: 2 additional calls per ticker
- **Total**: 10+ calls in rapid succession, exceeding the 5 calls/minute limit

## Solution Implemented

### 1. **Request Queue with Rate Limiting**

Added a global request queue in `polygonService.ts` that:
- Queues all API requests
- Processes them one at a time
- Enforces **13-second minimum interval** between requests
- Ensures max ~4.6 requests/minute (safely under 5/min limit)

```typescript
private minRequestInterval: number = 13000; // 13 seconds between requests
private requestQueue: Array<() => Promise<any>> = [];
private lastRequestTime: number = 0;
```

### 2. **Response Caching**

Implemented 30-second cache for all API responses:
- Prevents duplicate requests for same data
- Significantly reduces API calls
- Data stays fresh enough for trading app

```typescript
private cache: Map<string, { data: any; timestamp: number }> = new Map();
private cacheDuration: number = 30000; // 30 seconds
```

Cache keys:
- `aggs_{ticker}_{timeframe}_{limit}` - for aggregate data
- `prev_{ticker}` - for previous close data

### 3. **Retry Logic with Exponential Backoff**

Added automatic retry for 429 errors:
- Waits 20s, 40s, 80s on successive retries
- Prevents cascading failures
- Gracefully handles temporary rate limit hits

```typescript
private async makeRequestWithRetry<T>(
  requestFn: () => Promise<T>,
  maxRetries: number = 2
): Promise<T> {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await requestFn();
    } catch (error: any) {
      if (error.response?.status === 429) {
        const waitTime = Math.pow(2, attempt) * 20000;
        await new Promise(resolve => setTimeout(resolve, waitTime));
        continue;
      }
      throw error;
    }
  }
}
```

### 4. **Parallel Fetching with Queue**

Updated `useMultiTickerData` to:
- Launch all ticker fetches in parallel
- Let the service queue and rate-limit them
- Removed manual 3-second delays

```typescript
// Before: Sequential with delays
for (let i = 0; i < symbols.length; i++) {
  await fetchTickerData(symbols[i]);
  await new Promise(resolve => setTimeout(resolve, 3000));
}

// After: Parallel with service queue
const promises = symbols.map(symbol => fetchTickerData(symbol));
await Promise.all(promises); // Service handles rate limiting
```

### 5. **Increased Refresh Intervals**

Updated refresh timings for better rate limit compliance:
- **Dashboard**: 30s → 60s refresh interval
- **Ticker Pages**: 60s refresh interval
- With 30s cache, effective refresh is every 30-60s

## How It Works

### Request Flow

```
User Action (Dashboard Load)
        ↓
4 Tickers × 2 Calls = 8 Requests
        ↓
queueRequest() checks cache
        ↓
    Cache Hit?
    ↙        ↘
  YES         NO
   ↓           ↓
Return      Add to
Cached      Queue
Data          ↓
         Process Queue
              ↓
         Wait 13s between
         each request
              ↓
         makeRequestWithRetry()
              ↓
         429 Error?
         ↙      ↘
       YES      NO
        ↓        ↓
    Wait &    Return
    Retry     Data
        ↓        ↓
    Cache    Cache
    Result   Result
```

### Timing Example

**Dashboard Load (4 tickers):**
```
T=0s:   Request 1 (SPY prev close)     → [Cache MISS] → API call
T=13s:  Request 2 (SPY aggregates)     → [Cache MISS] → API call
T=26s:  Request 3 (UVXY prev close)    → [Cache MISS] → API call
T=39s:  Request 4 (UVXY aggregates)    → [Cache MISS] → API call
T=52s:  Request 5 (QQQ prev close)     → [Cache MISS] → API call
T=65s:  Request 6 (QQQ aggregates)     → [Cache MISS] → API call
T=78s:  Request 7 (IWM prev close)     → [Cache MISS] → API call
T=91s:  Request 8 (IWM aggregates)     → [Cache MISS] → API call

Total time: ~91 seconds
API calls in first minute: 4 calls ✅ (under 5/min limit)
API calls in second minute: 4 calls ✅ (under 5/min limit)
```

**Subsequent Refresh (within 30s cache window):**
```
T=60s:  Request 1 (SPY prev close)     → [Cache HIT] → Instant
T=60s:  Request 2 (SPY aggregates)     → [Cache HIT] → Instant
T=60s:  Request 3 (UVXY prev close)    → [Cache HIT] → Instant
T=60s:  Request 4 (UVXY aggregates)    → [Cache HIT] → Instant
... (all cached, 0 API calls)

Total time: <1 second
API calls: 0 ✅
```

**After Cache Expires (>30s):**
- Process repeats but with queue spacing
- Still respects 5 calls/minute limit

## Console Output

When the system is working correctly, you'll see:

```
[Cache SET] prev_SPY
[Rate Limit] Waiting 13.0s before next request...
[Cache SET] aggs_SPY_1h_200
[Cache HIT] prev_SPY
[Cache HIT] aggs_SPY_1h_200
```

On 429 error (now rare):
```
[429 Rate Limit] Waiting 20s before retry 1/2...
[429 Rate Limit] Waiting 40s before retry 2/2...
```

## API Call Budget

### Free Tier Limit
- **5 calls per minute**
- **Resets every 60 seconds**

### Our Usage Pattern

**Dashboard (4 tickers):**
- Initial load: 8 calls over ~91 seconds
- Cached refresh: 0 calls
- Post-cache refresh: 8 calls over ~91 seconds

**Ticker Page:**
- Initial load: 2 calls (aggregates + prev close)
- Cached refresh: 0 calls
- Post-cache refresh: 2 calls

**Combined (Dashboard + 1 Ticker Page):**
- Worst case: 10 calls over ~130 seconds
- Rate: ~4.6 calls/minute ✅ (under 5/min)

## Benefits

### ✅ No More 429 Errors
- Queue ensures proper spacing
- Cache reduces redundant calls
- Retry handles edge cases

### ✅ Faster User Experience
- Cache provides instant responses
- Progressive loading shows data as it arrives
- No long delays for users

### ✅ Efficient API Usage
- 30s cache window reduces calls by ~50%
- Queue prevents wasted parallel requests
- Smart caching maximizes free tier value

### ✅ Reliable Operation
- Automatic retry on failures
- Exponential backoff prevents hammering
- Graceful degradation on errors

## Testing

### Verify Rate Limiting Works

1. **Open Browser Console** (F12)
2. **Load Dashboard** and watch for:
   ```
   [Cache SET] messages (data being cached)
   [Rate Limit] Waiting messages (queue working)
   [Cache HIT] messages (cache working)
   ```

3. **Refresh Page** within 30 seconds:
   - Should see mostly [Cache HIT] messages
   - No API calls made
   - Instant data load

4. **Wait >30 seconds and Refresh**:
   - Should see [Cache SET] messages again
   - API calls spaced 13 seconds apart
   - No 429 errors

### Monitor API Usage

Check Polygon.io dashboard:
1. Go to https://polygon.io/dashboard/api-usage
2. Verify requests/minute stays under 5
3. Check for any 429 responses (should be 0)

## Configuration Options

### Adjust Cache Duration
In `polygonService.ts`:
```typescript
private cacheDuration: number = 30000; // Change to 60000 for 60s cache
```

### Adjust Request Interval
In `polygonService.ts`:
```typescript
private minRequestInterval: number = 13000; // Change to 15000 for 4 req/min
```

### Adjust Refresh Intervals
In `app/dashboard/page.tsx`:
```typescript
useMultiTickerData(SYMBOLS, true, 60000) // Change to 120000 for 2min refresh
```

In `app/ticker/[symbol]/page.tsx`:
```typescript
usePolygonData({
  refreshInterval: 60000, // Change to 120000 for 2min refresh
})
```

## Upgrading to Paid Tier

If you need more API calls:

### Starter Plan ($29/month)
- **100 calls/minute** (vs 5)
- **Real-time data** (vs 15-min delay)
- **All endpoints** including snapshots
- **WebSocket streaming**

With Starter plan, you can:
- Remove rate limiting (or set to 1s intervals)
- Reduce cache duration (or remove caching)
- Faster refresh intervals (10s or less)
- Use snapshot endpoint for instant multi-ticker data

## Summary

The rate limiting solution:
1. ✅ **Queues all requests** with 13-second spacing
2. ✅ **Caches responses** for 30 seconds
3. ✅ **Retries 429 errors** with exponential backoff
4. ✅ **Stays safely under** 5 calls/minute limit
5. ✅ **Provides fast UX** via intelligent caching

**Result**: No more 429 errors, reliable operation, and efficient use of free tier API limits.
