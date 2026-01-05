# Chart Scrolling & Interaction Documentation

This document explains how user interactions (mouse, trackpad, touch) control the trading chart navigation and zooming.

---

## Overview

The chart supports three primary interaction modes:
1. **Dragging (Grabbing Tool)** - Click and drag to pan in any direction
2. **Horizontal Scrolling** - Scroll left/right to navigate through time
3. **Vertical Scrolling/Pinching** - Scroll up/down or pinch to zoom in/out

---

## 1. Dragging (Grabbing Tool)

**How to trigger:**
- Click and hold anywhere on the chart
- Move mouse/finger while holding

**Behavior:**
- **Horizontal drag**: Pan through time axis
  - Drag left → See older data (go back in time)
  - Drag right → See newer data (go forward in time)
- **Vertical drag**: Pan through price axis
  - Drag up → See lower prices
  - Drag down → See higher prices

**Technical Details:**
- Managed by `useChartInteraction` hook in [useChartInteraction.ts](src/components/ProfessionalChart/useChartInteraction.ts)
- State tracked via `isPanning` flag
- Cursor changes to `grabbing` during drag (see [ProfessionalChart.module.css](src/components/ProfessionalChart.module.css))
- Panning sensitivity is based on visible candles, not total data length
- Prevents infinite data loading triggers via 2-second debounce

**Code Location:**
- `handleMouseDown` (line 20-24)
- `handleMouseMove` (line 26-70)
- `handleMouseUp` (line 72-75)

---

## 2. Horizontal Scrolling (Time Navigation)

**How to trigger:**
- Trackpad: Two-finger swipe left/right
- Mouse: Shift + scroll wheel
- Result: Pan through time WITHOUT zooming

**Behavior:**
- Scroll left → Go back in time (see older candles)
- Scroll right → Go forward in time (see newer candles)
- **CRITICAL**: Horizontal scrolling NEVER triggers zoom

**Detection Logic:**
```typescript
const isHorizontalScroll = absX > 0.05 && absX > absY
```
- Must have meaningful horizontal movement (> 0.05 pixels)
- Horizontal delta must be greater than vertical delta
- Pinch gesture (Ctrl/Cmd) disables horizontal panning

**Technical Details:**
- Implemented in wheel event handler in [ProfessionalChart.tsx](src/components/ProfessionalChart.tsx) (line 194-203)
- Updates `panOffset` state which controls visible time window
- Clamped to valid range: `[0, data.length]`
- Ignored during active drag/pan (`isPanning === true`)

---

## 3. Vertical Scrolling (Zoom)

**How to trigger:**
- Trackpad: Two-finger swipe up/down
- Mouse: Scroll wheel up/down
- Result: Zoom in/out while keeping cursor position fixed

**Behavior:**
- Scroll up → Zoom in (see fewer candles, more detail)
- Scroll down → Zoom out (see more candles, less detail)
- Candle under cursor stays in same position (zoom-to-cursor)

**Detection Logic:**
```typescript
const isVerticalScroll = absY >= 4 && absY > absX
```
- Must have meaningful vertical movement (≥ 4 pixels)
- Vertical delta must be greater than horizontal delta

**Technical Details:**
- Implemented in wheel event handler (line 208-230)
- Adjusts `timeScale` (higher = zoom in, lower = zoom out)
- Range: `[0.1, 10]` (10x zoom out to 10x zoom in)
- Simultaneously adjusts `panOffset` to keep cursor-focused candle in place

**Zoom-to-Cursor Math:**
```typescript
const candleDifference = oldVisibleCandles - newVisibleCandles
const offsetAdjustment = candleDifference * (1 - cursorRatio)
```
- `cursorRatio`: 0 = left edge, 1 = right edge
- Adjustment keeps the candle under cursor stationary

---

## 4. Pinch Zoom (Trackpad Gesture)

**How to trigger:**
- Trackpad: Pinch gesture (two fingers moving apart/together)
- Browser translates this to: `Ctrl+scroll` or `Cmd+scroll`

**Behavior:**
- Same as vertical scrolling
- Pinch in → Zoom out
- Pinch out → Zoom in

**Detection Logic:**
```typescript
const isPinchGesture = e.ctrlKey || e.metaKey
```

**Technical Details:**
- Shares zoom logic with vertical scrolling
- Overrides horizontal scrolling when active
- Commonly used on MacBook trackpads

---

## 5. Touch Gestures (Mobile/Tablet)

**Supported Gestures:**
1. **Single finger drag**: Pan in both directions (same as mouse drag)
2. **Two finger pinch**: Zoom in/out

**Technical Details:**
- Implemented in `handleTouchStart`, `handleTouchMove`, `handleTouchEnd`
- Pinch detection via distance calculation:
  ```typescript
  const distance = Math.hypot(
    touch2.clientX - touch1.clientX,
    touch2.clientY - touch1.clientY
  )
  ```
- Automatically loads more historical data when panning near left edge

---

## Critical Rules & Constraints

### Rule 1: Horizontal vs Vertical Separation
- **Horizontal scrolling** → ONLY pans time axis (NEVER zooms)
- **Vertical scrolling** → ONLY zooms (NEVER pans time)
- Detection is mutually exclusive via delta comparison

### Rule 2: Pan During Drag
- While `isPanning === true`, wheel events are completely ignored
- Prevents conflicting interactions
- User must release mouse/touch before scrolling

### Rule 3: Event Prevention
- All wheel events call `preventDefault()` to avoid page scrolling
- Listener uses `{ passive: false }` to allow preventDefault
- Events are stopped from propagating up DOM tree

### Rule 4: Zoom Limits
- **timeScale range**: `[0.1, 10]`
- **priceScale range**: Unlimited (user can pan vertically forever)
- **panOffset range**: `[0, data.length]` (can't scroll beyond available data)

### Rule 5: Infinite Scroll
- Triggered when panning left reaches within 20 candles of oldest data
- Debounced to 2 seconds to prevent rapid API calls
- Callback: `onReachLeftEdge()` → `onLoadMoreData()`

---

## State Management

### Key State Variables

| Variable | Type | Purpose | Range |
|----------|------|---------|-------|
| `panOffset` | number | How far back in time we're viewing | `[0, data.length]` |
| `priceOffset` | number | Vertical price axis offset | Unlimited |
| `timeScale` | number | Zoom level (higher = more zoomed in) | `[0.1, 10]` |
| `isPanning` | boolean | Is user currently dragging? | true/false |
| `mousePos` | {x, y} | Current mouse position for crosshair | null or coords |

### Derived Values

| Variable | Calculation | Purpose |
|----------|-------------|---------|
| `visibleRange` | Based on `panOffset` and `timeScale` | Determines which candles to render |
| `candlesPerPixel` | `visibleCandles / chartWidth` | Converts pixel movement to candle movement |
| `cursorRatio` | `mouseX / chartWidth` | Where cursor is for zoom-to-cursor |

---

## File Structure

```
src/components/ProfessionalChart/
├── ProfessionalChart.tsx          # Main component with wheel event handler
├── useChartInteraction.ts         # Mouse/touch drag handling
├── useChartScaling.ts             # Scale state management
├── hooks.ts                       # Visible range calculation
├── MainChart.tsx                  # Canvas rendering
└── types.ts                       # TypeScript interfaces
```

### Key Functions

1. **Wheel Handler** ([ProfessionalChart.tsx](src/components/ProfessionalChart.tsx):151-234)
   - Detects gesture type (pan vs zoom)
   - Calculates deltaX/deltaY
   - Updates panOffset or timeScale

2. **useChartInteraction** ([useChartInteraction.ts](src/components/ProfessionalChart/useChartInteraction.ts))
   - Handles mouse down/move/up
   - Manages `isPanning` state
   - Triggers infinite scroll callback

3. **useVisibleRange** ([hooks.ts](src/components/ProfessionalChart/hooks.ts))
   - Calculates which candles to display
   - Based on `panOffset`, `timeScale`, and timeframe

---

## Debugging Tips

### Console Logs
The wheel handler emits detailed logs:
```javascript
[Scroll] Horizontal pan: deltaX=5.2, deltaY=0.1
[Scroll] Zoom: deltaY=8.5, isPinch=false
[Zoom] scale: 1.00 → 1.05, visible: 120 → 114
```

### Common Issues

**Problem**: Horizontal scroll causes zoom
- **Cause**: `absY > absX` condition passing incorrectly
- **Fix**: Ensure `isHorizontalScroll` checks `absX > absY`

**Problem**: Can't scroll while dragging
- **Cause**: `isPanning === true` blocks wheel events
- **Expected**: This is intentional to prevent conflicts

**Problem**: Zoom doesn't center on cursor
- **Cause**: `offsetAdjustment` calculation incorrect
- **Fix**: Verify `cursorRatio` and `candleDifference` math

---

## Future Enhancements

Potential improvements:
1. **Momentum scrolling**: Continue panning after quick drag release
2. **Smooth zoom animation**: Interpolate timeScale changes
3. **Custom scroll sensitivity**: User-configurable pan/zoom speed
4. **Keyboard shortcuts**: Arrow keys for panning, +/- for zoom
5. **Mini-map navigator**: Overview of entire dataset with visible window indicator

---

## Summary

The chart scrolling system is designed to be intuitive and non-conflicting:

✅ **Drag with mouse** = Pan freely in any direction
✅ **Scroll left/right** = Navigate time (NO zoom)
✅ **Scroll up/down** = Zoom in/out (cursor-focused)
✅ **Pinch gesture** = Zoom in/out (trackpad)
✅ **Touch drag** = Pan on mobile
✅ **Touch pinch** = Zoom on mobile

All interactions are mutually exclusive and properly separated to avoid confusion or conflicts.
