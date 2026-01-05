# UI Chart Layer - Architectural Roadmap

## Purpose

This document describes structural improvements required for the ProfessionalChart component to achieve production-grade stability, performance, and feature completeness.

---

## Gap 1: Rendering Stability (Glitchy Behavior)

### Current State
The chart exhibits visual glitches during:
- Rapid pan/zoom gestures
- Window resize events
- Timeframe switching
- Data updates during active interaction

### Root Cause
1. Canvas redraws are not synchronized with requestAnimationFrame
2. State updates trigger multiple re-renders before settling
3. No debouncing on interaction events
4. Device pixel ratio handling inconsistent across browsers

### Required Improvement

```typescript
// 1. Use requestAnimationFrame for all canvas updates
const rafId = useRef<number>()
useEffect(() => {
  const draw = () => {
    // All canvas operations here
    rafId.current = requestAnimationFrame(draw)
  }
  rafId.current = requestAnimationFrame(draw)
  return () => cancelAnimationFrame(rafId.current!)
}, [dependencies])

// 2. Debounce resize events
const debouncedResize = useDebouncedCallback(handleResize, 100)

// 3. Use double-buffering for smooth transitions
// Draw to offscreen canvas, then copy to visible canvas
```

### Priority: P0 (Critical)

---

## Gap 2: ML Analysis Overlay

### Current State
V6 ML predictions are displayed in the Northstar panel but NOT on the chart itself. Users cannot see:
- Prediction direction (bullish/bearish) on the chart
- Confidence level visualization
- Historical prediction accuracy at similar points

### Why This Matters
Traders need to see predictions in context with price action. Separating predictions from the chart creates cognitive load and slows decision-making.

### Required Improvement

```
ML Overlay Components:
┌─────────────────────────────────────────────────────────────┐
│  1. Direction Banner (top of chart)                          │
│     ┌─────────────────────────────────────────────────┐     │
│     │  ▲ BULLISH 78% | Target B: Close > 11AM         │     │
│     └─────────────────────────────────────────────────┘     │
├─────────────────────────────────────────────────────────────┤
│  2. Prediction Zones (shaded regions)                        │
│     - Green zone above current price = bullish target        │
│     - Red zone below current price = bearish target          │
│     - Opacity indicates confidence level                     │
├─────────────────────────────────────────────────────────────┤
│  3. Historical Prediction Markers                            │
│     - Small icons at past prediction points                  │
│     - Green checkmark = correct prediction                   │
│     - Red X = incorrect prediction                           │
├─────────────────────────────────────────────────────────────┤
│  4. Session Divider Lines                                    │
│     - Vertical line at 11 AM (Target B reference)            │
│     - Vertical line at market open (Target A reference)      │
└─────────────────────────────────────────────────────────────┘
```

### New Props Required

```typescript
interface MLOverlayProps {
  v6Prediction?: {
    direction: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
    probability_a: number  // Close vs Open
    probability_b: number  // Close vs 11AM
    confidence: number
  }
  showMLOverlay?: boolean
  showSessionLines?: boolean
  price11am?: number  // For Target B reference line
}
```

### Priority: P0 (Critical)

---

## Gap 3: Key Level Lines Enhancement

### Current State
Only `stopLoss`, `entryPoint`, and `targets` are drawn as horizontal lines. Missing:
- Recent high/low lines (currently only markers)
- Previous day high/low/close
- VWAP bands
- 11 AM price line for Target B

### Required Improvement

```typescript
interface KeyLevels {
  // Current props
  stopLoss?: number
  entryPoint?: number
  targets?: number[]

  // New props for complete Northstar integration
  recentHigh?: number       // 30-bar high - red dashed
  recentLow?: number        // 30-bar low - green dashed
  prevClose?: number        // Yesterday close - white dotted
  vwap?: number            // VWAP line - yellow
  vwapUpper?: number       // VWAP +1 std - yellow dashed
  vwapLower?: number       // VWAP -1 std - yellow dashed
  price11am?: number       // 11 AM price - cyan dotted
  todayOpen?: number       // Today's open - white solid
}
```

### Priority: P1 (High)

---

## Gap 4: Performance Optimization

### Current State
Chart slows down with:
- 500+ visible candles
- 50+ FVG patterns
- Frequent data updates (1s intervals)

### Root Cause
1. Full canvas redraw on every frame
2. FVG detection runs on every render
3. No virtualization of off-screen elements
4. No caching of computed values

### Required Improvement

```typescript
// 1. Dirty rectangle tracking - only redraw changed regions
interface DirtyRegion {
  x: number; y: number; width: number; height: number;
}
const dirtyRegions = useRef<DirtyRegion[]>([])

// 2. Memoize FVG detection
const fvgPatterns = useMemo(() => detectFvgPatterns(data), [data])

// 3. Canvas layer separation
// Layer 1: Static grid (rarely redraws)
// Layer 2: Candles (redraws on pan/zoom)
// Layer 3: Overlays (redraws on data update)
// Layer 4: Crosshair (redraws on mouse move)

// 4. Off-screen culling
const visibleCandles = data.slice(visibleRange.start, visibleRange.end)
```

### Priority: P1 (High)

---

## Gap 5: Replay Mode Chart Synchronization

### Current State
In Replay mode, the chart:
- Fetches all historical data then filters client-side
- Does not update as the time slider moves
- Key levels are for current time, not replay time

### Required Improvement

```typescript
// 1. Server-side filtering for replay data
// GET /api/v2/replay/bars?ticker=SPY&date=2025-12-30&time=14:30

// 2. Reactive chart updates
useEffect(() => {
  if (replayMode && replayTime) {
    fetchChartData(ticker, replayDate, replayTime)
  }
}, [replayTime])  // Update chart when slider moves

// 3. Time-synchronized key levels
// Key levels should be calculated as of replay time, not current time
```

### Priority: P1 (High)

---

## Gap 6: Accessibility

### Current State
Chart has no accessibility features:
- No keyboard navigation
- No screen reader support
- Color-only information (red/green candles)

### Required Improvement

```typescript
// 1. ARIA labels for chart container
<div role="img" aria-label="Candlestick chart for SPY showing price from $590 to $595">

// 2. Keyboard navigation
onKeyDown={(e) => {
  if (e.key === 'ArrowLeft') panLeft()
  if (e.key === 'ArrowRight') panRight()
  if (e.key === '+') zoomIn()
  if (e.key === '-') zoomOut()
}}

// 3. Pattern indicators beyond color
// Bullish: filled body + ▲ marker
// Bearish: hollow body + ▼ marker
```

### Priority: P3 (Low)

---

## Gap 7: Touch Support

### Current State
Touch interactions are not optimized:
- Pinch-to-zoom is jerky
- Pan gestures conflict with page scroll
- No touch-specific affordances

### Required Improvement

```typescript
// 1. Use touch events with proper handling
onTouchStart={(e) => {
  e.preventDefault()  // Prevent scroll
  if (e.touches.length === 2) startPinch(e)
  else startPan(e)
}}

// 2. Inertial scrolling
const velocity = useRef({ x: 0, y: 0 })
// Apply momentum after touch end

// 3. Touch-friendly hit targets
// Increase FVG dot click radius for touch
const touchRadius = 'ontouchstart' in window ? 30 : 15
```

### Priority: P2 (Medium)

---

## Priority Matrix

| Gap | Severity | Effort | Priority |
|-----|----------|--------|----------|
| Rendering Stability | Critical | Medium | P0 |
| ML Analysis Overlay | Critical | High | P0 |
| Key Level Lines Enhancement | High | Low | P1 |
| Performance Optimization | High | High | P1 |
| Replay Mode Synchronization | High | Medium | P1 |
| Accessibility | Low | Medium | P3 |
| Touch Support | Medium | Medium | P2 |

---

## Implementation Order

1. **P0: Rendering Stability** - Fix the glitches first so other features work correctly
2. **P0: ML Analysis Overlay** - Core value proposition for traders
3. **P1: Key Level Lines** - Complete Northstar integration
4. **P1: Performance** - Enable larger datasets
5. **P1: Replay Sync** - Make replay mode fully functional
6. **P2: Touch Support** - Mobile/tablet users
7. **P3: Accessibility** - Compliance and inclusion

---

*This is an architectural roadmap, not a task list. Implementation details and timelines are not specified.*
