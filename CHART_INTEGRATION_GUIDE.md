# Chart Integration Guide

## Overview

The chart system has been completely refactored to support real-time data from Polygon.io while maintaining backward compatibility with mock data for development and testing.

## Architecture

### Components

#### 1. **ProfessionalChart** (`src/components/ProfessionalChart.tsx`)
The main chart component using custom Canvas API for high-performance rendering.

**Features:**
- Candlestick visualization with OHLCV data
- Volume bars
- Interactive crosshair
- Price levels (stop loss, targets, entry points)
- Real-time price updates
- Responsive design

**Props:**
```typescript
interface ProfessionalChartProps {
  symbol: string                    // Ticker symbol
  currentPrice?: number             // Current/live price
  stopLoss?: number                 // Stop loss price level
  targets?: number[]                // Target price levels
  entryPoint?: number               // Entry price level
  data?: CandleData[]               // External chart data (Polygon.io)
  onDataUpdate?: (data) => void     // Callback when data updates
}
```

**Usage:**
```tsx
import { ProfessionalChart } from '@/components/ProfessionalChart'

<ProfessionalChart
  symbol="SPY"
  currentPrice={445.20}
  stopLoss={443.00}
  targets={[447.50, 450.00, 453.50]}
  entryPoint={444.75}
  data={polygonData}  // Pass real data or omit for mock data
/>
```

### Services

#### 2. **PolygonService** (`src/services/polygonService.ts`)
Service layer for Polygon.io API integration.

**Key Methods:**
- `getAggregates(ticker, timeframe, limit)` - Fetch OHLCV bars
- `getPreviousClose(ticker)` - Get previous day data
- `getSnapshot(ticker)` - Get real-time snapshot
- `getIntradayData(ticker, interval)` - Get today's data

**Example:**
```typescript
import { polygonService } from '@/services/polygonService'

const data = await polygonService.getAggregates('SPY', '1h', 100)
```

#### 3. **usePolygonData Hook** (`src/hooks/usePolygonData.ts`)
React hook for managing Polygon.io data with state management.

**Features:**
- Automatic data fetching
- Loading and error states
- Auto-refresh support
- Price change calculations

**Example:**
```typescript
import { usePolygonData } from '@/hooks/usePolygonData'

const {
  data,
  currentPrice,
  priceChange,
  priceChangePercent,
  isLoading,
  error,
  refetch
} = usePolygonData({
  ticker: 'SPY',
  timeframe: '1h',
  limit: 100,
  autoRefresh: true,
  refreshInterval: 60000
})
```

### Type Definitions

#### 4. **Polygon Types** (`src/types/polygon.ts`)
TypeScript definitions for Polygon.io data structures.

**Key Types:**
```typescript
// Normalized chart data format
interface NormalizedChartData {
  time: number      // Unix timestamp
  open: number
  high: number
  low: number
  close: number
  volume: number
}

// Supported timeframes
type Timeframe = '1m' | '5m' | '15m' | '30m' | '1h' | '2h' | '4h' | '1d' | '1w' | '1M'
```

## Data Flow

```
Polygon.io API
      ↓
polygonService.getAggregates()
      ↓
usePolygonData() hook
      ↓
Normalized data format
      ↓
ProfessionalChart component
      ↓
Canvas rendering
```

## Integration Examples

### Basic Integration

```tsx
'use client'

import { ProfessionalChart } from '@/components/ProfessionalChart'
import { usePolygonData } from '@/hooks/usePolygonData'

export default function MyChart() {
  const { data, currentPrice, isLoading } = usePolygonData({
    ticker: 'AAPL',
    timeframe: '1h'
  })

  if (isLoading) return <div>Loading...</div>

  return (
    <ProfessionalChart
      symbol="AAPL"
      currentPrice={currentPrice}
      data={data}
    />
  )
}
```

### Advanced Integration with Trading Levels

```tsx
'use client'

import { ProfessionalChart } from '@/components/ProfessionalChart'
import { usePolygonData } from '@/hooks/usePolygonData'

export default function TradingChart() {
  const { data, currentPrice } = usePolygonData({
    ticker: 'SPY',
    timeframe: '1h',
    limit: 100,
    autoRefresh: true,
    refreshInterval: 60000
  })

  return (
    <ProfessionalChart
      symbol="SPY"
      currentPrice={currentPrice ?? 445.20}
      stopLoss={443.00}
      targets={[447.50, 450.00, 453.50]}
      entryPoint={444.75}
      data={data}
    />
  )
}
```

### Error Handling

```tsx
const {
  data,
  currentPrice,
  error,
  isLoading
} = usePolygonData({
  ticker: 'SPY',
  timeframe: '1h'
})

if (error) {
  return (
    <div>
      <p>Error: {error.message}</p>
      {/* Chart will fall back to mock data */}
      <ProfessionalChart
        symbol="SPY"
        currentPrice={445.20}
      />
    </div>
  )
}
```

## Customization

### Adding New Timeframes

1. Update `src/types/polygon.ts`:
```typescript
export type Timeframe = '1m' | '5m' | ... | '3h' // Add '3h'

export const TIMEFRAME_CONFIGS: Record<Timeframe, Partial<TimeframeConfig>> = {
  // ... existing configs
  '3h': { multiplier: 3, timespan: 'hour' }
}
```

2. Use in your component:
```typescript
const { data } = usePolygonData({
  ticker: 'SPY',
  timeframe: '3h' // Now available
})
```

### Customizing Chart Appearance

The chart uses canvas rendering. To customize colors, edit `ProfessionalChart.tsx`:

```typescript
// Line 179 - Candle colors
ctx.strokeStyle = isGreen ? '#22c55e' : '#ef4444'  // Wick colors
ctx.fillStyle = isGreen ? '#22c55e' : '#ef4444'     // Body colors

// Line 200-238 - Price level colors
drawPriceLine(stopLoss, '#ef444488', 'SL', true)    // Stop loss
drawPriceLine(entryPoint, '#06b6d488', 'ENTRY', false) // Entry
drawPriceLine(target, '#22c55e88', `T${i + 1}`, false) // Targets
```

### Adding Technical Indicators

To add indicators like Moving Averages:

1. Calculate indicator in your component:
```typescript
const calculateSMA = (data: CandleData[], period: number) => {
  return data.map((candle, i) => {
    if (i < period - 1) return null
    const sum = data.slice(i - period + 1, i + 1)
      .reduce((acc, c) => acc + c.close, 0)
    return sum / period
  })
}
```

2. Pass to chart as additional data
3. Modify chart rendering code to draw indicator lines

## Performance Considerations

### Canvas Rendering
- The chart uses direct Canvas API for optimal performance
- Can handle 1000+ candles without lag
- Uses device pixel ratio for sharp rendering on high-DPI screens

### Data Fetching
- Implements auto-refresh with configurable intervals
- Debounced to prevent excessive API calls
- Caches data in component state

### Optimization Tips
1. Set reasonable refresh intervals (60000ms minimum recommended)
2. Limit data points (100-200 candles optimal)
3. Use appropriate timeframes for your use case
4. Implement error boundaries for production

## Testing

### Without Polygon.io API Key
The chart works perfectly without an API key:
- Displays warning message
- Falls back to simulated data
- All features remain functional

### With Polygon.io API Key
1. Configure `.env.local` with your key
2. Chart will automatically fetch real data
3. Shows confirmation: "✓ Live data from Polygon.io"

### Mock Data Generation
For testing, the chart can generate realistic mock data:
- 50 candles by default
- Realistic price movements
- Volume simulation
- Based on current price prop

## Troubleshooting

### Issue: Chart Not Rendering
**Solution:** Ensure the component is wrapped with `dynamic` import:
```typescript
const ProfessionalChart = dynamic(
  () => import('@/components/ProfessionalChart').then(m => m.ProfessionalChart),
  { ssr: false }
)
```

### Issue: Data Not Updating
**Solution:** Check these items:
1. API key is configured correctly
2. Not exceeding rate limits (5 calls/min on free plan)
3. Ticker symbol is valid
4. Market is open (for real-time data)

### Issue: TypeScript Errors
**Solution:** Ensure proper imports:
```typescript
import { NormalizedChartData } from '@/types/polygon'
import { usePolygonData } from '@/hooks/usePolygonData'
```

## Best Practices

1. **Always handle loading states**
   ```typescript
   if (isLoading) return <Skeleton />
   ```

2. **Always handle errors**
   ```typescript
   if (error) return <ErrorMessage error={error} />
   ```

3. **Use appropriate timeframes**
   - Intraday trading: 1m, 5m, 15m
   - Day trading: 30m, 1h, 2h
   - Swing trading: 4h, 1d
   - Long-term: 1w, 1M

4. **Set reasonable refresh rates**
   - Free API: 60000ms (1 minute) minimum
   - Paid API: 5000ms (5 seconds) recommended

5. **Implement proper error boundaries**
   ```typescript
   <ErrorBoundary fallback={<ErrorUI />}>
     <ProfessionalChart {...props} />
   </ErrorBoundary>
   ```

## Future Enhancements

Potential improvements:
- [ ] WebSocket integration for real-time streaming
- [ ] Technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Multiple chart types (line, area, heikin-ashi)
- [ ] Drawing tools (trend lines, rectangles)
- [ ] Chart annotations and alerts
- [ ] Export to image/PDF
- [ ] Multi-timeframe analysis
- [ ] Comparison with other symbols

## Resources

- [Polygon.io Documentation](https://polygon.io/docs)
- [Canvas API Reference](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)
- [React Hooks Guide](https://react.dev/reference/react)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)

## Support

For issues or questions:
1. Check the browser console for errors
2. Review the network tab for API responses
3. Ensure all environment variables are set
4. Verify dependencies are installed
5. Restart the development server after config changes
