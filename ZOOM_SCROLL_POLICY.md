# Chart Zoom and Scroll Policy

## Core Principles

1. **CONSISTENCY**: All timeframes follow the same zoom/scroll rules
2. **PREDICTABILITY**: User interactions produce expected results
3. **AUTO-FIT FIRST**: Charts always start by showing ALL data
4. **PRESERVE USER INTENT**: Once user zooms/pans, maintain that state until reset

---

## Policy Rules

### 1. Initial Load Behavior

**RULE**: When a chart first loads OR when user switches timeframes, the chart MUST:
- Show ALL available data fitted to screen width
- Display from first bar to last bar
- No whitespace on left or right edges
- Reset all zoom/pan state

**Implementation**:
```typescript
timeScale = 1.0        // No zoom applied
panOffset = 0          // No scroll offset
autoFit = true         // Auto-fit mode enabled
```

**Detection**: Chart is in auto-fit mode when:
```typescript
timeScale === 1.0 && panOffset === 0
```

---

### 2. Auto-Fit Mode (Default State)

**RULE**: In auto-fit mode, the chart MUST:
- Calculate `candlesInView = data.length` (show ALL bars)
- Set `visibleRange = { start: 0, end: data.length }`
- Ignore base candle calculations
- Fit all data edge-to-edge

**Canvas Drawing**:
```typescript
effectiveWidth = data.length  // Use actual data length
candleWidth = chartWidth / effectiveWidth
// Draw from x=0 to x=chartWidth with no overflow
```

---

### 3. User Zoom Behavior

**RULE**: When user zooms (mouse wheel OR pinch):
- Exit auto-fit mode immediately
- Apply zoom factor to `timeScale`
- Maintain center point of zoom
- Allow zoom range: 0.2x (zoomed out) to 5.0x (zoomed in)

**Implementation**:
```typescript
// Wheel zoom
const sensitivity = 0.0015
const zoomDelta = -e.deltaY * sensitivity
timeScale *= Math.exp(zoomDelta)
timeScale = clamp(timeScale, 0.2, 5.0)

// This exits auto-fit mode (timeScale !== 1.0)
```

**Manual Zoom Mode**:
```typescript
baseCandlesInView = 100  // Fixed baseline
candlesInView = baseCandlesInView / timeScale
// At 1.0x: 100 candles visible
// At 2.0x: 50 candles visible (zoomed in)
// At 0.5x: 200 candles visible (zoomed out)
```

---

### 4. User Pan/Scroll Behavior

**RULE**: When user pans (mouse drag OR touch drag):
- Exit auto-fit mode if not already exited
- Move through historical data
- Allow scrolling into past (positive offset)
- Prevent scrolling beyond oldest bar
- Allow minimal future whitespace (for UX)

**Implementation**:
```typescript
// Pan calculation
const candlesPerPixel = actualCandlesInView / chartWidth
const candleDelta = deltaX * candlesPerPixel * 2
newPanOffset = currentPanOffset + candleDelta

// Bounds
const maxOffset = Math.max(0, data.length - actualCandlesInView)
const minOffset = 0  // No future whitespace in auto-fit
panOffset = clamp(newPanOffset, minOffset, maxOffset)
```

---

### 5. Visible Range Calculation

**RULE**: Calculate visible range differently based on mode:

**Auto-Fit Mode** (timeScale=1.0, panOffset=0):
```typescript
start = 0
end = data.length
candlesInView = data.length
```

**Manual Mode** (any other state):
```typescript
baseCandlesInView = 100
candlesInView = baseCandlesInView / timeScale
scrollBack = panOffset  // How far back in time
end = data.length - scrollBack
start = Math.max(0, end - candlesInView)
```

---

### 6. Timeframe Change Behavior

**RULE**: When user clicks a new timeframe button:
- Reset to auto-fit mode
- Fetch new data for that timeframe
- Show ALL new data fitted to screen
- Clear any zoom/pan state from previous timeframe

**Implementation**:
```typescript
const handleTimeframeClick = (tf: string) => {
  setTimeframe(tf)
  setPriceScale(1.0)     // Reset vertical zoom
  setTimeScale(1.0)      // Reset horizontal zoom = AUTO-FIT
  setPanOffset(0)        // Reset scroll = AUTO-FIT
  // Fetch new data...
}
```

---

### 7. Interval Change Behavior

**RULE**: When user changes interval (not timeframe display):
- Reset to auto-fit mode
- Fetch new data with new interval
- Show ALL new data fitted to screen

**Implementation**:
```typescript
const handleIntervalChange = (interval: string) => {
  setInterval(interval)
  setTimeScale(1.0)      // Reset to AUTO-FIT
  setPanOffset(0)        // Reset to AUTO-FIT
  // Fetch new data...
}
```

---

### 8. Canvas Drawing Consistency

**RULE**: Candle drawing MUST respect visible range:

```typescript
// Always use data length for width in auto-fit
const effectiveWidth = shouldAutoFit
  ? data.length
  : Math.max(baseWidth, data.length)

const candleWidth = chartWidth / effectiveWidth

// Boundary check - prevent overflow
data.forEach((candle, i) => {
  const x = padding.left + i * candleWidth + candleWidth / 2
  if (x < padding.left || x > padding.left + chartWidth) return
  // Draw candle...
})
```

---

## Implementation Checklist

- [ ] Set initial state to auto-fit: `panOffset=0`, `timeScale=1.0`
- [ ] Update auto-fit detection: check `timeScale === 1.0 && panOffset === 0`
- [ ] Update visible range hook to handle both modes
- [ ] Update canvas drawing to use `data.length` in auto-fit
- [ ] Update timeframe change to reset to auto-fit
- [ ] Update interval change to reset to auto-fit
- [ ] Remove any hardcoded whitespace in auto-fit mode
- [ ] Test all timeframes (1D, 5D, 1M, 3M, 6M, YTD, 1Y, 5Y)
- [ ] Test zoom in/out maintains consistency
- [ ] Test pan/scroll maintains consistency
- [ ] Test switching between timeframes resets properly

---

## Expected User Experience

1. **First Load**: User sees SPY 1D chart with full trading day (6-7 bars) fitted to screen
2. **Zoom In**: User scrolls wheel, chart zooms to show fewer bars with more detail
3. **Pan**: User drags chart left, scrolls back in time to see historical data
4. **Switch Timeframe**: User clicks "5D", chart resets to show all 5 days fitted to screen
5. **Change Interval**: User changes from "1 hour" to "15 min", chart resets to show all bars
6. **Zoom Out**: User scrolls wheel opposite direction, chart shows more bars (up to all data)

---

## Summary of Key Changes

| Aspect | Old Behavior | New Behavior |
|--------|--------------|--------------|
| Initial state | `panOffset=-15` (15% whitespace) | `panOffset=0` (no whitespace) |
| Auto-fit detection | `timeScale===1.0 && panOffset===-15` | `timeScale===1.0 && panOffset===0` |
| Auto-fit candles | `baseCandlesInView/0.85` | `data.length` exactly |
| Manual mode candles | `100 / timeScale` with whitespace | `100 / timeScale` clean |
| Timeframe reset | Reset with whitespace | Reset to perfect auto-fit |
| Canvas width | Fixed `baseWidth=100` | `data.length` in auto-fit |

---

**Version**: 1.0
**Date**: 2025-11-01
**Status**: APPROVED - Ready for Implementation
