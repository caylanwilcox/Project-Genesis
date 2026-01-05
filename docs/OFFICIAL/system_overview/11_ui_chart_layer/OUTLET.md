# UI Chart Layer - System Connections

## The Body Metaphor

The UI Chart Layer is the **visual cortex** of the trading platform. Just as the visual cortex processes raw retinal signals into coherent spatial perception, the chart layer transforms raw market data into visual patterns that traders can interpret.

The chart is where all system components become visible:
- **Candles**: Raw market data (price, volume, time)
- **Price Lines**: Northstar key levels (R1, Pivot, S1)
- **FVG Patterns**: Pattern recognition engine output
- **ML Overlays**: V6 prediction visualizations
- **Crosshair**: Real-time price/time feedback

If the chart fails, traders lose their primary window into the market. No chart means no visual decision-making.

---

## Upstream Data Sources

### What Feeds the Chart

| Data Source | What It Provides | Update Frequency |
|-------------|------------------|------------------|
| **Polygon API** | OHLCV candle data | 1s-5m depending on timeframe |
| **Northstar Phase 1** | Key levels (pivot, R1, S1, recent high/low) | On each bar close |
| **V6 ML Model** | Prediction direction and probability | On each bar close |
| **FVG ML Model** | Pattern fill probability | On pattern detection |
| **Data Ingestion Service** | Cached/normalized bar data | From database |

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     POLYGON API                              │
│              (real-time bars, quotes)                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              DATA INGESTION SERVICE                          │
│        (fetch, validate, cache, normalize)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  NORTHSTAR  │   │   V6 MODEL  │   │  FVG MODEL  │
│   Phase 1   │   │  Predictor  │   │  Predictor  │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       │    Key Levels   │  Predictions    │  Fill Prob
       └────────────────┐│┌────────────────┘
                        ▼▼▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃              PROFESSIONAL CHART                              ┃
┃                                                              ┃
┃   Candles ──► Price Lines ──► FVG Patterns ──► ML Overlays  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                            │
│        (Dashboard, Ticker Detail, Replay Mode)               │
└─────────────────────────────────────────────────────────────┘
```

---

## Interface Contracts

### Candle Data Contract

```typescript
// From Data Ingestion Service → Chart
interface CandleData {
  time: number    // Unix timestamp (ms)
  open: number    // Opening price
  high: number    // High price
  low: number     // Low price
  close: number   // Closing price
  volume: number  // Volume
}

// Expectations:
// - Sorted by time ascending
// - No duplicate timestamps
// - No gaps in intraday data (market hours only)
// - All numbers positive
```

### Key Levels Contract

```typescript
// From Northstar Phase 1 → Chart
interface KeyLevels {
  pivot_r1: number      // Resistance 1
  pivot: number         // Central pivot
  pivot_s1: number      // Support 1
  recent_high: number   // 30-bar high
  recent_low: number    // 30-bar low
  mid_point: number     // (high + low) / 2
  current_price: number
  today_open: number
  prev_close: number
}

// Chart maps these to:
// targets[] ← pivot_r1 (green lines)
// entryPoint ← pivot (cyan line)
// stopLoss ← pivot_s1 (red dashed line)
```

### V6 Prediction Contract (Future)

```typescript
// From V6 ML Model → Chart (not yet implemented)
interface V6Prediction {
  direction: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
  probability_a: number  // P(close > open)
  probability_b: number  // P(close > 11am)
  confidence: number     // max(prob, 1-prob)
  timestamp: number      // When prediction was made
}
```

### FVG Prediction Contract

```typescript
// From FVG ML Model → Chart
interface FvgPrediction {
  pattern_id: string
  fill_probability: number  // 0-1
  expected_fill_bars: number
  confidence: 'HIGH' | 'MEDIUM' | 'LOW'
}
```

---

## Consumer Pages

### Dashboard (`/dashboard`)

| What Chart Receives | Source | Purpose |
|---------------------|--------|---------|
| Live candle data | Polygon WebSocket | Real-time chart updates |
| Northstar key levels | `/api/v2/northstar` | Draw pivot lines |
| FVG patterns | Local detection | Pattern visualization |
| V6 predictions | `/api/v2/ml/daily-signals` | (Not yet drawn on chart) |

### Ticker Detail (`/ticker/[symbol]`)

| What Chart Receives | Source | Purpose |
|---------------------|--------|---------|
| Historical + live data | Polygon API | Full chart history |
| Configurable timeframes | User selection | Zoom levels |
| FVG patterns | Local detection | Pattern visualization |

### Replay Mode (`/replay`)

| What Chart Receives | Source | Purpose |
|---------------------|--------|---------|
| Historical data (filtered) | `/api/v2/data/market` | Show only bars up to replay time |
| Historical key levels | Northstar @ replay time | Draw pivot lines as they were |
| Historical predictions | V6 @ replay time | (Not yet implemented) |

---

## Event Flow

### User Interaction → System Response

```
User Action          Chart Event              System Response
───────────────────────────────────────────────────────────────
Click ticker     →   setSelectedTicker()   →  Fetch chart data
Pan left         →   onPan()               →  Shift visible range
Zoom in          →   onTimeScaleChange()   →  Increase candle width
Click FVG dot    →   onFvgClick()          →  Toggle pattern expansion
Change timeframe →   onTimeframeChange()   →  Fetch new data
Hover            →   onMouseMove()         →  Update crosshair
```

### Data Update → Chart Response

```
Data Event           Chart Response
───────────────────────────────────────────────────────────────
New bar arrives   →  Append to data[], shift visible range right
Key levels update →  Redraw price lines at new levels
Prediction update →  Update ML overlay (when implemented)
FVG detected      →  Draw new pattern, request ML prediction
```

---

## Failure Modes and Impact

| Chart Failure | User Impact | System Response |
|---------------|-------------|-----------------|
| Candle data stale | Chart shows outdated prices | Show "Data Delayed" warning |
| Key levels missing | No pivot lines drawn | Fall back to no lines (silent) |
| FVG detection fails | No patterns shown | Disable FVG toggle |
| Canvas error | Blank chart area | Show error message, retry |
| WebSocket disconnect | No live updates | Reconnect with backfill |

### Graceful Degradation

```
Full Functionality
       │
       ▼ (FVG ML fails)
Chart + Candles + Lines + FVG (no predictions)
       │
       ▼ (Key levels missing)
Chart + Candles + FVG
       │
       ▼ (FVG fails)
Chart + Candles only
       │
       ▼ (Candle data fails)
"Unable to load chart data" message
```

---

## Performance Contracts

| Metric | Target | Current |
|--------|--------|---------|
| Initial render | < 100ms | ~200ms |
| Pan response | < 16ms (60fps) | ~30ms |
| Zoom response | < 16ms (60fps) | ~50ms |
| Data update | < 50ms | ~100ms |
| FVG detection | < 50ms per 500 bars | ~100ms |

---

## Integration Points

### With Northstar Panel

The chart and Northstar panel share:
- Ticker selection state
- Key levels data
- Direction/bias indicators

When chart shows key level lines, Northstar panel shows the same levels in text form.

### With Trading Directions Component

The chart should eventually show:
- Entry zones from TradingDirections
- Stop loss levels from TradingDirections
- Target zones from TradingDirections

### With Replay Mode Controls

The chart must respond to:
- Time slider changes (filter data)
- Play/pause state (auto-advance)
- Speed changes (update rate)

---

## Visual Cortex Analogy

### Primary Visual Processing (Candles)
Like V1 detecting edges and orientations, the chart renders:
- Price direction (up/down)
- Body size (conviction)
- Wick length (rejection)

### Pattern Recognition (FVG)
Like V2/V4 detecting complex shapes, the chart identifies:
- Gap patterns
- Failed breakouts
- Support/resistance tests

### Object Recognition (Key Levels)
Like inferotemporal cortex recognizing objects, the chart marks:
- Significant price levels
- Historical pivots
- Predicted targets

### Action Planning (ML Overlay)
Like parietal cortex planning movements, the chart shows:
- Prediction direction
- Confidence zones
- Entry/exit guidance

---

*The chart is the trader's window into the market. Its fidelity determines the quality of visual decision-making.*
