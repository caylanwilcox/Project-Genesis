# Professional Chart UX Specification

## Overview
This document defines the requirements and best practices for the ProfessionalChart component, ensuring smooth zoom, pan, multi-day analysis, interval switching, and FVG pattern visualization.

---

## Current Implementation Status

### Replay Page Chart UI
The replay page (`/replay`) includes a professional chart with the following features:

#### Ticker Selection
- Click any ticker card (SPY, QQQ, IWM) to open chart view
- Selected ticker highlighted with cyan color
- Close button (X) to dismiss chart

#### Timeframe Selector
```
┌─────────────────────────────────────────────────────┐
│ SPY  [1m] [5m] [15m] [1h]                      [X] │
└─────────────────────────────────────────────────────┘
```
- 4 interval buttons: 1m, 5m, 15m, 1h
- Active interval highlighted with cyan background
- Switching intervals fetches new data automatically

#### Key Levels Legend
Below the chart, key levels are displayed:
```
┌─────────────────────────────────────────────────────┐
│ ■ R1: $XXX.XX  ■ Pivot: $XXX.XX  ■ S1: $XXX.XX    │
│                    FVG patterns shown • Click to close │
└─────────────────────────────────────────────────────┘
```
- R1 (red) - Resistance level
- Pivot (cyan) - Central pivot point
- S1 (green) - Support level

---

## Core Requirements

### 1. Data Management

#### Single-Day Mode (Current - Replay)
- Fetch data for the selected replay date only
- Filter by endTime for point-in-time replay
- Limit: 500 bars per request
- API endpoint: `/api/v2/data/market?ticker=X&timeframe=Y&date=Z&endTime=HH:MM`

#### Multi-Day Mode (Future - Dashboard)
- Fetch historical data spanning multiple days
- Support zoom-out to see bigger picture
- Incremental loading: fetch more data when user pans to edge
- API endpoint: `/api/v2/data/market?ticker=X&timeframe=Y&fromDate=A&date=B&limit=2000`

```typescript
// Data fetch strategy
interface ChartDataConfig {
  mode: 'single-day' | 'multi-day'
  ticker: string
  timeframe: '1m' | '5m' | '15m' | '1h' | '4h' | '1d'

  // Single-day mode
  date?: string        // YYYY-MM-DD
  endTime?: string     // HH:MM

  // Multi-day mode
  fromDate?: string    // Start date
  toDate?: string      // End date
  limit?: number       // Max bars
}
```

### 2. Timeframe/Interval Cycling

#### Available Intervals
| Interval | Use Case | Bars per Day |
|----------|----------|--------------|
| 1m | Scalping, precise entries | 390 |
| 5m | Day trading, FVG detection | 78 |
| 15m | Swing setups | 26 |
| 1h | Trend analysis | 7 |
| 4h | Multi-day trends | 2 |
| 1d | Long-term view | 1 |

#### Interval Switching Rules
1. **Preserve time context**: When switching intervals, keep the same time window visible
2. **Reset zoom**: Set timeScale to 1.0 on interval change
3. **Reset pan**: Set panOffset to 0 (show most recent data)
4. **Cache previous data**: Keep last 2 timeframes in memory for quick switching

```typescript
// Interval state management
interface TimeframeState {
  dataTimeframe: Timeframe      // Actual data resolution
  displayTimeframe: string      // UI display label
  cachedData: Map<Timeframe, CandleData[]>
}
```

### 3. Zoom Behavior

#### Zoom Mechanics
- **Multiplicative zoom**: TradingView-style (not additive)
- **Zoom factor**: 1.15x per scroll step (1.05x with Ctrl held)
- **Zoom limits**: 0.05 (zoom out max) to 5.0 (zoom in max)
- **Anchor point**: Zoom centered on mouse cursor position

```typescript
// Zoom calculation
const handleWheel = (deltaY: number, pointerRatio: number) => {
  const zoomFactor = ctrlKey ? 1.05 : 1.15
  const newScale = deltaY < 0
    ? currentScale * zoomFactor   // Scroll up = zoom in
    : currentScale / zoomFactor   // Scroll down = zoom out

  // Adjust pan to keep pointer position stable
  adjustPanForZoom(currentScale, newScale, pointerRatio)
  setTimeScale(clamp(newScale, 0.05, 5.0))
}
```

#### Visible Range Calculation
```typescript
// At 1x zoom: show baseBars candles
// At 2x zoom: show baseBars/2 candles (zoomed in)
// At 0.5x zoom: show baseBars*2 candles (zoomed out)
const effectiveBars = Math.round(baseBars / timeScale)
const clampedBars = Math.max(10, Math.min(data.length, effectiveBars))

// panOffset = 0 means show most recent data
const end = data.length - panOffset
const start = end - clampedBars
```

### 4. Pan Behavior

#### Pan Mechanics
- **Direction**: Drag right = scroll back in time (see older data)
- **Sensitivity**: 2.0x multiplier for comfortable panning
- **Limits**:
  - Left limit: Can scroll back through all historical data
  - Right limit: Can scroll 50 bars into empty future space

```typescript
// Pan calculation
const candlesPerPixel = visibleCandles / chartWidth
const candleDelta = deltaX * candlesPerPixel * panSensitivity
const newOffset = clamp(currentOffset + candleDelta, -50, data.length)
```

#### Edge Loading (Multi-Day Mode)
When user pans near the left edge (oldest data):
1. Trigger `onReachLeftEdge` callback
2. Fetch older historical data
3. Prepend to existing data array
4. Adjust panOffset to maintain visual position

### 5. FVG (Fair Value Gap) Visualization

#### Detection Rules
```typescript
interface FVGPattern {
  type: 'bullish' | 'bearish'
  startIndex: number      // Bar where gap starts
  gapTop: number          // Upper price of gap
  gapBottom: number       // Lower price of gap
  gapSize: number         // Percentage size
  filled: boolean         // Has price returned to fill gap?
  fillIndex?: number      // Bar where gap was filled
}

// Bullish FVG: bar[i-2].high < bar[i].low (gap up)
// Bearish FVG: bar[i-2].low > bar[i].high (gap down)
```

#### Visual Rendering
- **Bullish FVG**: Green rectangle with 20% opacity
- **Bearish FVG**: Red rectangle with 20% opacity
- **Minimum size**: Configurable (default 0.2% of price)
- **Extend**: Draw rectangle from gap bar to current bar (or fill bar)

#### FVG Settings
```typescript
interface FVGConfig {
  enabled: boolean
  minPercentage: number   // Minimum gap size (0.1 - 1.0%)
  showFilled: boolean     // Show gaps that have been filled
  maxAge: number          // Max bars to look back (default: 100)
}
```

---

## State Management

### Chart Orchestrator Pattern
```typescript
function useChartOrchestrator(props) {
  // 1. Data layer
  const { data } = useChartData(externalData)
  const chartData = useMemo(() => filterMarketHours(data), [data])

  // 2. Viewport layer
  const viewport = useChartViewport(chartData.length)

  // 3. Scaling layer
  const scaling = useChartScaling()

  // 4. Timeframe layer
  const timeframe = useTimeframeState()

  // 5. Visible range (derived)
  const visibleRange = useVisibleRange(
    chartData,
    viewport.panOffset,
    scaling.timeScale,
    timeframe.displayTimeframe,
    timeframe.dataTimeframe
  )

  // 6. Interaction handlers
  const interaction = useChartInteraction(...)

  return { chartData, visibleRange, interaction, ... }
}
```

### Reset Conditions
| Event | Reset Pan | Reset Zoom | Reset Price |
|-------|-----------|------------|-------------|
| Ticker change | Yes | Yes | Yes |
| Timeframe change | Yes | Yes | Yes |
| Date change | Yes | No | Yes |
| Data length change >50% | Yes | No | No |

---

## Performance Best Practices

### 1. Rendering Optimization
- Use `useMemo` for visible range (synchronous updates)
- Canvas rendering for candles/volume (not SVG)
- Batch DOM updates with `requestAnimationFrame`
- Debounce resize observers (500ms)

### 2. Data Optimization
- Limit bars per request (500 for intraday, 1000 for daily)
- Cache timeframe data in memory
- Use `sort=asc` from API (oldest first)
- Filter market hours client-side

### 3. Interaction Optimization
- Native wheel event listener (passive: false)
- Throttle mouse move updates (16ms = 60fps)
- Use refs for frequently-changing values

---

## Error Handling

### Data Fetch Errors
```typescript
try {
  const data = await fetchChartData(config)
  if (data.length === 0) {
    // Show "No data available" message
    // Keep previous data visible (don't flash empty)
  }
} catch (error) {
  // Show error toast
  // Retry with exponential backoff
  // Fall back to cached data if available
}
```

### Edge Cases
1. **Weekend/holiday dates**: Show message "Market closed"
2. **Pre-market time**: Show data up to market open
3. **Future dates**: Show "No data available"
4. **API timeout**: Retry 3x, then show cached data

---

## Accessibility

- Keyboard navigation: Arrow keys for pan, +/- for zoom
- Screen reader: Announce price changes
- High contrast mode: Ensure candle colors are distinguishable
- Touch support: Pinch zoom, two-finger pan

---

## Testing Requirements

See `CHART_UX_TESTS.md` for comprehensive test cases.
