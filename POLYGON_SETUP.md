# Polygon.io Integration Setup Guide

This application is now integrated with Polygon.io for real-time and historical market data.

## Setup Instructions

### 1. Get a Polygon.io API Key

1. Visit [polygon.io](https://polygon.io/)
2. Sign up for a free account (or paid plan for more features)
3. Navigate to your Dashboard
4. Copy your API key

### 2. Configure Environment Variables

1. Copy the `.env.example` file to `.env.local`:
   ```bash
   cp .env.example .env.local
   ```

2. Open `.env.local` and add your Polygon.io API key:
   ```
   NEXT_PUBLIC_POLYGON_API_KEY=your_actual_api_key_here
   ```

3. Restart your development server for changes to take effect

### 3. Verify Integration

1. Start your development server:
   ```bash
   npm run dev
   ```

2. Navigate to a ticker page (e.g., `/ticker/SPY`)
3. If configured correctly, you should see:
   - Real-time price data from Polygon.io
   - A green checkmark: "✓ Live data from Polygon.io"
   - Historical candlestick chart with actual market data

4. If the API key is not configured, the app will:
   - Display a warning message
   - Fall back to simulated data
   - Still function normally for testing purposes

## Features

### Chart Data Integration

The `ProfessionalChart` component now supports:
- **Real-time data**: Automatic data refresh every 60 seconds
- **Historical data**: Fetches up to 100 candles based on selected timeframe
- **Fallback mode**: Works with simulated data if API is unavailable
- **Multiple timeframes**: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 1d, 1w, 1M

### Available Polygon.io Services

#### `polygonService`
Located in `src/services/polygonService.ts`

Methods:
- `getAggregates(ticker, timeframe, limit)` - Get OHLCV bars/candles
- `getPreviousClose(ticker)` - Get previous day's close data
- `getSnapshot(ticker)` - Get real-time snapshot
- `getIntradayData(ticker, interval)` - Get today's intraday data

#### `usePolygonData` Hook
Located in `src/hooks/usePolygonData.ts`

React hook for fetching and managing Polygon.io data with:
- Automatic data fetching
- Loading and error states
- Auto-refresh capabilities
- Price change calculations

Example usage:
```tsx
const {
  data,
  currentPrice,
  priceChange,
  priceChangePercent,
  isLoading,
  error,
  refetch,
} = usePolygonData({
  ticker: 'SPY',
  timeframe: '1h',
  limit: 100,
  autoRefresh: true,
  refreshInterval: 60000,
})
```

## Data Format

The chart expects data in the following format:

```typescript
interface CandleData {
  time: number      // Unix timestamp in milliseconds
  open: number      // Opening price
  high: number      // Highest price
  low: number       // Lowest price
  close: number     // Closing price
  volume: number    // Trading volume
}
```

## Polygon.io API Limits

### Free Plan
- 5 API calls per minute
- Delayed data (15 minutes)
- Limited to 2 years of historical data

### Starter Plan ($29/month)
- 100 API calls per minute
- Real-time data
- Full historical data access

### Developer Plan ($99/month)
- 500 API calls per minute
- Real-time data
- WebSocket streaming support

For more details, visit: https://polygon.io/pricing

## Troubleshooting

### "API key not configured" Error
- Ensure you've created a `.env.local` file
- Verify the variable name is exactly `NEXT_PUBLIC_POLYGON_API_KEY`
- Restart your development server after adding the key

### No Data Returned
- Check that your API key is valid
- Verify you're not exceeding rate limits
- Ensure the ticker symbol is valid (e.g., 'SPY', 'AAPL')
- Check browser console for detailed error messages

### Chart Shows Mock Data
- This is expected when:
  - API key is not configured
  - Rate limit exceeded
  - Market is closed (for real-time data)
  - Network connection issues
- The app will automatically fall back to simulated data

## Development Notes

### Adding New Timeframes

To add new timeframes, update `src/types/polygon.ts`:

```typescript
export type Timeframe = '1m' | '5m' | ... | 'your_new_timeframe'

export const TIMEFRAME_CONFIGS: Record<Timeframe, Partial<TimeframeConfig>> = {
  // ... existing configs
  'your_new_timeframe': { multiplier: X, timespan: 'minute' | 'hour' | 'day' }
}
```

### Using Polygon Data in Other Components

Import and use the service or hook:

```typescript
import { polygonService } from '@/services/polygonService'
import { usePolygonData } from '@/hooks/usePolygonData'

// Direct service usage
const data = await polygonService.getAggregates('AAPL', '1d', 30)

// Hook usage in React components
const { data, isLoading, error } = usePolygonData({ ticker: 'AAPL' })
```

## Next Steps

Consider implementing:
- WebSocket integration for real-time streaming
- Technical indicators (RSI, MACD, Moving Averages)
- Multiple chart types (line, area, heikin-ashi)
- Volume profile analysis
- Drawing tools and annotations

## Support

For Polygon.io API questions:
- Documentation: https://polygon.io/docs
- Support: support@polygon.io

For application issues:
- Check the browser console for errors
- Review the network tab for API responses
- Ensure all dependencies are installed
