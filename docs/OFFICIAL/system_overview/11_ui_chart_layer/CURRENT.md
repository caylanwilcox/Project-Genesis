# UI Chart Layer - Current State

## Overview

The UI Chart Layer provides interactive candlestick charting with ML analysis overlays. The primary component is `ProfessionalChart`, a custom canvas-based charting solution that renders price data, key levels, and Fair Value Gap (FVG) patterns.

## Core Responsibility

The chart layer visualizes market data and ML predictions. It answers: "What does the current market structure look like?" and "Where are the key levels and patterns?"

## What the Chart Layer Owns

| Ownership | Description |
|-----------|-------------|
| Candlestick rendering | OHLCV bars with proper coloring |
| Volume visualization | Volume bars below price chart |
| Price lines | Horizontal lines for key levels (R1, Pivot, S1) |
| FVG pattern detection | Fair Value Gap highlighting and ML predictions |
| Time/price grid | Visual gridlines and axis labels |
| Crosshair | Mouse-following price/time indicator |
| Pan and zoom | Interactive chart navigation |
| High/low markers | Triangle markers at extremes (Webull-style) |

## What the Chart Layer Does NOT Own

| Exclusion | Belongs To |
|-----------|------------|
| Data fetching | Data Ingestion Service |
| Price level calculation | Northstar Pipeline (Phase 1) |
| ML predictions | V6 Model / FVG ML Model |
| Trade execution | Policy Engine |
| Signal health scoring | RPE Phase 2 |

---

## Component Architecture

```
ProfessionalChart (RootChart.tsx)
├── ChartHeader.tsx          # Symbol, timeframe selector, fullscreen toggle
├── ChartCanvas.tsx          # Container for canvas elements
│   └── MainChart.tsx        # Primary rendering component
│       ├── canvasRendering.ts    # Canvas setup, clearing
│       ├── gridDrawing.ts        # Price/time grids
│       ├── candleDrawing.ts      # OHLCV candles
│       ├── priceLines.ts         # Horizontal level lines
│       ├── fvgDrawing.ts         # FVG pattern detection/rendering
│       └── Crosshair.tsx         # Mouse crosshair overlay
├── ChartFooter.tsx          # Current time display
└── core/
    └── state/
        └── useChartOrchestrator.ts  # State coordination
```

---

## Props Interface

```typescript
interface ProfessionalChartProps {
  symbol: string              // Ticker symbol (e.g., "SPY")
  currentPrice?: number       // Current price for live indicator
  data?: CandleData[]         // External candle data (overrides internal fetch)

  // Price Lines - drawn as horizontal lines on chart
  stopLoss?: number           // Red dashed line (support level)
  entryPoint?: number         // Cyan solid line (pivot point)
  targets?: number[]          // Green solid lines (resistance levels)

  // FVG Configuration
  showFvg?: boolean           // Enable FVG pattern detection
  fvgPercentage?: number      // Minimum gap % for FVG (0.2 = 0.2%)

  // Callbacks
  onDataUpdate?: (data: CandleData[]) => void
  onTimeframeChange?: (tf: string, displayTf: string, intervalLabel?: string) => void
  onFvgCountChange?: (count: number) => void
  onVisibleBarCountChange?: (count: number, visibleData: CandleData[]) => void
  onLoadMoreData?: () => void

  // State
  isLoadingMore?: boolean
  isTimeframeCached?: (displayTf: string) => boolean
}

interface CandleData {
  time: number    // Unix timestamp in milliseconds
  open: number
  high: number
  low: number
  close: number
  volume: number
}
```

---

## Key Level Mapping

When integrating with Northstar, map key levels to price lines:

| Northstar Level | Chart Prop | Line Color | Line Style |
|-----------------|------------|------------|------------|
| `pivot_r1` | `targets` | Green | Solid |
| `pivot` | `entryPoint` | Cyan | Solid |
| `pivot_s1` | `stopLoss` | Red | Dashed |

Example:
```tsx
<ProfessionalChart
  symbol="SPY"
  data={candleData}
  stopLoss={northstar.phase1.key_levels.pivot_s1}
  entryPoint={northstar.phase1.key_levels.pivot}
  targets={[northstar.phase1.key_levels.pivot_r1]}
  showFvg={true}
/>
```

---

## FVG Pattern Detection

Fair Value Gaps are detected using a 3-candle pattern:

```
Bullish FVG:
  Candle 1 high < Candle 3 low = gap exists

Bearish FVG:
  Candle 1 low > Candle 3 high = gap exists
```

### FVG ML Integration

When `showFvg={true}`, the chart:
1. Detects FVG patterns in visible data
2. Requests ML predictions via `/api/v2/fvg/predict`
3. Displays prediction confidence on each FVG
4. Highlights high-confidence patterns differently

---

## Rendering Pipeline

```
1. Canvas Setup (setupCanvas)
   ├── Get device pixel ratio
   ├── Scale canvas for retina displays
   └── Return 2D context

2. Clear & Calculate (clearCanvas, calculatePadding, calculatePriceRange)
   ├── Clear previous frame
   ├── Calculate chart dimensions
   └── Determine visible price range

3. Draw Grid (drawPriceGrid, drawTimeGrid)
   ├── Draw horizontal price lines
   ├── Draw vertical time lines
   └── Add axis labels

4. Draw Candles (drawCandles)
   ├── Calculate candle width
   ├── Draw wicks (high-low)
   └── Draw bodies (open-close with fill)

5. Draw Volume (drawVolumeBars)
   ├── Scale to max volume
   └── Color by price direction

6. Draw FVG Patterns (drawFvgPatterns) - if enabled
   ├── Detect patterns
   ├── Draw gap rectangles
   └── Add ML prediction indicators

7. Draw Price Lines (drawPriceLines, drawCurrentPriceLine)
   ├── Draw target/entry/stop lines
   ├── Draw current price line
   └── Create overlay tags for labels

8. Draw Markers (drawHighPriceMarker, drawLowPriceMarker)
   ├── Find highest/lowest visible candle
   └── Draw triangle markers with price labels
```

---

## Current Implementation Files

| Module | Location | Lines | Purpose |
|--------|----------|-------|---------|
| RootChart.tsx | `/src/components/ProfessionalChart/` | ~100 | Root orchestrator |
| MainChart.tsx | `/src/components/ProfessionalChart/` | ~320 | Canvas rendering controller |
| candleDrawing.ts | `/src/components/ProfessionalChart/` | ~150 | Candle/volume rendering |
| priceLines.ts | `/src/components/ProfessionalChart/` | ~160 | Price level lines |
| fvgDrawing.ts | `/src/components/ProfessionalChart/` | ~400 | FVG detection/rendering |
| gridDrawing.ts | `/src/components/ProfessionalChart/` | ~200 | Grid lines |
| hooks.ts | `/src/components/ProfessionalChart/` | ~215 | Chart data/state hooks |
| types.ts | `/src/components/ProfessionalChart/` | ~120 | TypeScript interfaces |
| useChartOrchestrator.ts | `/src/components/ProfessionalChart/core/state/` | ~200 | State coordination |

---

## Usage Locations

| Page | Purpose | Data Source |
|------|---------|-------------|
| Dashboard (`/dashboard`) | Live market view | Polygon API real-time |
| Ticker Detail (`/ticker/[symbol]`) | Full-screen chart | Polygon API |
| Replay Mode (`/replay`) | Historical replay | Filtered historical data |

---

## Known Issues

1. **Glitchy rendering on rapid zoom** - Canvas can flicker when zooming quickly
2. **FVG patterns not persisting across timeframes** - Patterns recalculate on timeframe change
3. **No ML analysis overlay** - V6 predictions not displayed on chart
4. **Missing key level lines** - Recent high/low not drawn as lines (only markers)
5. **Performance on large datasets** - Slowdown with 1000+ candles visible

---

*Last updated: 2026-01-03*
