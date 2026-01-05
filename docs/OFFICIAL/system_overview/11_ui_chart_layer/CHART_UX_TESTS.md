# Chart UX Test Plan

## Test Categories

1. **Data Loading** - Correct data for date/time/interval
2. **Zoom** - Smooth multiplicative zoom with cursor anchoring
3. **Pan** - Natural scroll direction, edge limits
4. **Interval Switching** - Quick transitions, state preservation
5. **FVG Patterns** - Detection accuracy, visual rendering
6. **Edge Cases** - Error handling, empty states

---

## Manual Test Checklist

### Phase 1: Basic Functionality

#### T1.1 - Data Loading (Single Day)
- [ ] Select SPY on Dec 30, 2025 at 11:00 AM
- [ ] Chart shows data from 9:30 AM to 11:00 AM
- [ ] Time axis shows correct date (Dec 30)
- [ ] No data from previous days visible
- [ ] Console shows: `[Chart] Loaded X 5m bars for 2025-12-30`

#### T1.2 - Interval Switching
- [ ] Switch from 5m to 1m: More candles appear
- [ ] Switch from 5m to 15m: Fewer, wider candles
- [ ] Switch from 5m to 1h: ~2 candles visible for 1.5 hours
- [ ] Each switch resets view to show most recent data
- [ ] No flash of old data during switch

#### T1.3 - Zoom In
- [ ] Scroll up on trackpad: Candles get wider
- [ ] Zoom anchors at cursor position
- [ ] Zoom stops at ~10 candles visible minimum
- [ ] Price axis adjusts to visible range
- [ ] Smooth animation (no jank)

#### T1.4 - Zoom Out
- [ ] Scroll down on trackpad: Candles get thinner
- [ ] Can see all loaded data when fully zoomed out
- [ ] Zoom stops when all data visible
- [ ] Candles don't disappear or clip

#### T1.5 - Pan Left (History)
- [ ] Click and drag right: See older data
- [ ] Can scroll back to 9:30 AM (market open)
- [ ] Cannot scroll past oldest bar
- [ ] Time axis updates while panning

#### T1.6 - Pan Right (Recent)
- [ ] Click and drag left: See newer data
- [ ] Most recent bar stays at right edge
- [ ] Can scroll ~50 bars into empty future space
- [ ] Empty space shows grid but no candles

---

### Phase 2: FVG Patterns

#### T2.1 - FVG Detection
- [ ] Enable FVG toggle (showFvg=true)
- [ ] Bullish FVGs appear as green rectangles
- [ ] Bearish FVGs appear as red rectangles
- [ ] Gaps smaller than 0.2% are filtered out

#### T2.2 - FVG Visual Quality
- [ ] Gap rectangle spans from gap bar to current bar
- [ ] Rectangle height matches actual gap (high to low)
- [ ] Opacity is ~20% (not too distracting)
- [ ] Gaps don't overlap price candles badly

#### T2.3 - FVG with Zoom/Pan
- [ ] FVGs stay aligned when zooming
- [ ] FVGs stay aligned when panning
- [ ] FVGs update when new data loads

---

### Phase 3: Multi-Day Mode (Future)

#### T3.1 - Multi-Day Data Loading
- [ ] Fetch 5 trading days of data
- [ ] Data sorted oldest to newest
- [ ] Initial view shows most recent day
- [ ] Zoom out reveals previous days

#### T3.2 - Day Boundaries
- [ ] Gaps visible between trading days
- [ ] Time axis shows date changes
- [ ] No overnight/weekend data

#### T3.3 - Edge Loading
- [ ] Pan to oldest data triggers load
- [ ] New data prepends without jump
- [ ] Loading indicator appears
- [ ] At least 2 second debounce

---

### Phase 4: Edge Cases

#### T4.1 - Empty Data
- [ ] No data for date: Show "No data available"
- [ ] API error: Show error message
- [ ] Keep previous data while loading

#### T4.2 - Weekend/Holiday
- [ ] Select Saturday: "Market closed"
- [ ] Show nearest trading day's data

#### T4.3 - Rapid Interactions
- [ ] Rapid zoom in/out: No crash
- [ ] Rapid interval switch: No stale data
- [ ] Rapid pan: Smooth performance

#### T4.4 - Window Resize
- [ ] Chart resizes with container
- [ ] Candles scale proportionally
- [ ] No clipping at edges

---

## Automated Tests

### Unit Tests (Jest)

```typescript
// hooks.test.ts
describe('getDefaultBarsForView', () => {
  it('returns 78 for 1D display with 5m data', () => {
    expect(getDefaultBarsForView('1D', '5m', 100)).toBe(78)
  })

  it('returns dataLength for unknown display', () => {
    expect(getDefaultBarsForView(undefined, undefined, 150)).toBe(150)
  })
})

describe('useVisibleRange', () => {
  it('shows last N bars when panOffset=0', () => {
    const data = Array(200).fill({ time: 0 })
    const range = calculateVisibleRange(data, 0, 1.0, '1D', '5m')
    expect(range.end).toBe(200)
    expect(range.start).toBe(200 - 78)
  })

  it('shifts range when panning left', () => {
    const data = Array(200).fill({ time: 0 })
    const range = calculateVisibleRange(data, 50, 1.0, '1D', '5m')
    expect(range.end).toBe(150)
  })

  it('clamps start to 0', () => {
    const data = Array(50).fill({ time: 0 })
    const range = calculateVisibleRange(data, 100, 1.0, '1D', '5m')
    expect(range.start).toBe(0)
  })
})

describe('useChartViewport', () => {
  it('resets panOffset when data length changes >50%', () => {
    // Test with renderHook
  })
})
```

### Integration Tests (Playwright)

```typescript
// chart.spec.ts
test('chart loads data for selected date', async ({ page }) => {
  await page.goto('/replay')
  await page.fill('input[type="date"]', '2025-12-30')
  await page.click('text=SPY')

  // Wait for chart to load
  await expect(page.locator('[data-testid="chart-canvas"]')).toBeVisible()

  // Check console for data load message
  const logs = await page.evaluate(() => window.consoleLogs)
  expect(logs).toContain('[Chart] Loaded')
})

test('zoom changes visible bar count', async ({ page }) => {
  await page.goto('/replay')
  // ... setup

  const initialBars = await page.evaluate(() => window.visibleBarCount)

  // Scroll to zoom in
  await page.locator('[data-testid="chart-canvas"]').hover()
  await page.mouse.wheel(0, -100)

  const zoomedBars = await page.evaluate(() => window.visibleBarCount)
  expect(zoomedBars).toBeLessThan(initialBars)
})

test('interval switch updates timeframe', async ({ page }) => {
  await page.goto('/replay')
  // ... setup

  await page.click('button:has-text("1m")')
  await expect(page.locator('text=1m bars')).toBeVisible()

  await page.click('button:has-text("1h")')
  await expect(page.locator('text=1h bars')).toBeVisible()
})
```

---

## Performance Benchmarks

| Metric | Target | Measure |
|--------|--------|---------|
| Initial render | <100ms | `performance.mark('chart-render')` |
| Zoom response | <16ms | FPS counter during zoom |
| Pan response | <16ms | FPS counter during pan |
| Interval switch | <200ms | Time from click to render |
| Data fetch | <500ms | Network timing |

---

## Regression Prevention

### Before Each PR
1. Run manual checklist (Phase 1-2)
2. Run unit tests: `npm test -- hooks.test.ts`
3. Run integration tests: `npx playwright test chart.spec.ts`

### After Each Release
1. Full manual test (Phase 1-4)
2. Performance benchmark comparison
3. Cross-browser testing (Chrome, Safari, Firefox)

---

## Known Issues & Workarounds

| Issue | Status | Workaround |
|-------|--------|------------|
| Passive wheel listener warning | Fixed | Native event listener with passive:false |
| Zoom not anchoring correctly | Fixed | Use pointerRatio in adjustPanForZoom |
| Candles disappear on far left pan | Fixed | Clamp actualStart to 0 |
| Data from wrong date showing | Fixed | Reset panOffset on data change |

---

## Future Enhancements

1. **Keyboard shortcuts**: Arrow keys for pan, +/- for zoom
2. **Drawing tools**: Trend lines, horizontal lines
3. **Indicators**: Moving averages, RSI
4. **Multi-chart layout**: Compare tickers side by side
5. **Save/load views**: Remember zoom/pan state per ticker
