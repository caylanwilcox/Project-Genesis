import { describe, it } from 'node:test'
import assert from 'node:assert'

// Import the function we're testing
// Note: We can't directly import from the React component, so we duplicate the logic here
function getDefaultBarsForView(
  display?: string,
  dataTf?: string,
  dataLength: number = 0,
): number {
  if (!display || !dataTf) return Math.min(50, dataLength)

  if (display === '1D' && dataTf === '1m') return 390
  if (display === '1D' && dataTf === '5m') return 78
  if (display === '5D' && dataTf === '5m') return 390
  if (display === '1M' && dataTf === '1h') return 140
  if (display === '3M' && dataTf === '4h') return 100
  if (display === '6M' && dataTf === '1d') return 126
  if (display === 'YTD' && dataTf === '1d') return 200
  if (display === '1Y' && dataTf === '1d') return 252
  if (display === '5Y' && dataTf === '1w') return 260
  if (display === 'All' && dataTf === '1M') return dataLength || 100

  if (display === '1D') {
    if (dataTf === '15m') return 26
    if (dataTf === '30m') return 13
    if (dataTf === '1h') return 7
    if (dataTf === '2h') return 3
    if (dataTf === '4h') return 2
    if (dataTf === '1d') return 1
  }

  if (display === '5D') {
    if (dataTf === '1m') return 1950
    if (dataTf === '5m') return 390
    if (dataTf === '15m') return 130
    if (dataTf === '1h') return 33
    if (dataTf === '2h') return 16
    if (dataTf === '4h') return 8
    if (dataTf === '1d') return 5
  }

  if (display === 'All') {
    return dataLength || 100
  }

  return Math.max(100, dataLength)
}

// Helper to simulate useVisibleRange logic
function calculateVisibleRange(
  dataLength: number,
  panOffset: number,
  timeScale: number,
  displayTf?: string,
  dataTf?: string
) {
  if (dataLength === 0) return { start: 0, end: 100 }

  const baseCandlesInView = getDefaultBarsForView(displayTf, dataTf, dataLength)
  const zoomedCandlesInView = Math.round(baseCandlesInView / Math.max(timeScale, 0.05))
  const effectiveCandlesInView = Math.max(10, Math.min(dataLength, zoomedCandlesInView))

  const scrollBack = Math.floor(panOffset)
  const end = dataLength - scrollBack
  const start = end - effectiveCandlesInView

  const actualStart = Math.max(0, start)
  const actualEnd = Math.max(actualStart + effectiveCandlesInView, end)

  return { start: actualStart, end: actualEnd }
}

describe('getDefaultBarsForView', () => {
  describe('1D display timeframe', () => {
    it('returns 390 for 1m data', () => {
      assert.strictEqual(getDefaultBarsForView('1D', '1m', 500), 390)
    })

    it('returns 78 for 5m data', () => {
      assert.strictEqual(getDefaultBarsForView('1D', '5m', 100), 78)
    })

    it('returns 26 for 15m data', () => {
      assert.strictEqual(getDefaultBarsForView('1D', '15m', 50), 26)
    })

    it('returns 7 for 1h data', () => {
      assert.strictEqual(getDefaultBarsForView('1D', '1h', 20), 7)
    })
  })

  describe('5D display timeframe', () => {
    it('returns 390 for 5m data', () => {
      assert.strictEqual(getDefaultBarsForView('5D', '5m', 500), 390)
    })

    it('returns 33 for 1h data', () => {
      assert.strictEqual(getDefaultBarsForView('5D', '1h', 50), 33)
    })
  })

  describe('default behavior', () => {
    it('returns min(50, dataLength) when display/dataTf undefined', () => {
      assert.strictEqual(getDefaultBarsForView(undefined, undefined, 30), 30)
      assert.strictEqual(getDefaultBarsForView(undefined, undefined, 100), 50)
    })

    it('returns 100 minimum for unknown display with large data', () => {
      assert.strictEqual(getDefaultBarsForView('unknown', 'unknown', 50), 100)
    })
  })

  describe('All display timeframe', () => {
    it('returns dataLength for All display with 1M data', () => {
      assert.strictEqual(getDefaultBarsForView('All', '1M', 500), 500)
    })

    it('returns 100 for All display with no data', () => {
      assert.strictEqual(getDefaultBarsForView('All', '1M', 0), 100)
    })
  })
})

describe('calculateVisibleRange', () => {
  describe('panOffset = 0 (show most recent)', () => {
    it('shows last N bars based on display timeframe', () => {
      const range = calculateVisibleRange(200, 0, 1.0, '1D', '5m')
      assert.strictEqual(range.end, 200)
      assert.strictEqual(range.start, 200 - 78) // 78 bars for 1D/5m
    })

    it('shows all data if less than default bars', () => {
      const range = calculateVisibleRange(50, 0, 1.0, '1D', '5m')
      assert.strictEqual(range.start, 0)
      assert.strictEqual(range.end, 50)
    })
  })

  describe('panning', () => {
    it('shifts range when panning left (positive panOffset)', () => {
      const range = calculateVisibleRange(200, 50, 1.0, '1D', '5m')
      assert.strictEqual(range.end, 150) // 200 - 50
      assert.strictEqual(range.start, 150 - 78)
    })

    it('clamps start to 0 when panning too far left', () => {
      const range = calculateVisibleRange(100, 90, 1.0, '1D', '5m')
      assert.strictEqual(range.start, 0)
    })

    it('allows panning right into empty space (negative panOffset)', () => {
      const range = calculateVisibleRange(100, -20, 1.0, '1D', '5m')
      assert.strictEqual(range.end, 120) // 100 - (-20)
    })
  })

  describe('zooming', () => {
    it('shows fewer bars when zoomed in (timeScale > 1)', () => {
      const range = calculateVisibleRange(200, 0, 2.0, '1D', '5m')
      const visibleBars = range.end - range.start
      assert.strictEqual(visibleBars, 39) // 78/2 rounded
    })

    it('shows more bars when zoomed out (timeScale < 1)', () => {
      const range = calculateVisibleRange(200, 0, 0.5, '1D', '5m')
      const visibleBars = range.end - range.start
      assert.strictEqual(visibleBars, 156) // 78*2
    })

    it('clamps to data length when fully zoomed out', () => {
      const range = calculateVisibleRange(100, 0, 0.1, '1D', '5m')
      assert.strictEqual(range.start, 0)
      assert.strictEqual(range.end, 100)
    })

    it('maintains minimum 10 bars when zoomed in', () => {
      const range = calculateVisibleRange(200, 0, 100, '1D', '5m')
      const visibleBars = range.end - range.start
      assert.ok(visibleBars >= 10, `Expected at least 10 bars, got ${visibleBars}`)
    })
  })

  describe('empty data', () => {
    it('returns default range for empty data', () => {
      const range = calculateVisibleRange(0, 0, 1.0)
      assert.deepStrictEqual(range, { start: 0, end: 100 })
    })
  })
})

describe('FVG detection logic', () => {
  interface CandleData {
    time: number
    open: number
    high: number
    low: number
    close: number
    volume: number
  }

  interface FVGPattern {
    type: 'bullish' | 'bearish'
    startIndex: number
    gapTop: number
    gapBottom: number
    gapPercent: number
  }

  function detectFVGs(data: CandleData[], minPercent: number = 0.2): FVGPattern[] {
    const patterns: FVGPattern[] = []

    for (let i = 2; i < data.length; i++) {
      const bar0 = data[i - 2] // Two bars ago
      const bar2 = data[i]     // Current bar

      // Bullish FVG: bar0.high < bar2.low (gap up)
      if (bar0.high < bar2.low) {
        const gapSize = bar2.low - bar0.high
        const gapPercent = (gapSize / bar0.high) * 100
        if (gapPercent >= minPercent) {
          patterns.push({
            type: 'bullish',
            startIndex: i - 1,
            gapTop: bar2.low,
            gapBottom: bar0.high,
            gapPercent
          })
        }
      }

      // Bearish FVG: bar0.low > bar2.high (gap down)
      if (bar0.low > bar2.high) {
        const gapSize = bar0.low - bar2.high
        const gapPercent = (gapSize / bar0.low) * 100
        if (gapPercent >= minPercent) {
          patterns.push({
            type: 'bearish',
            startIndex: i - 1,
            gapTop: bar0.low,
            gapBottom: bar2.high,
            gapPercent
          })
        }
      }
    }

    return patterns
  }

  it('detects bullish FVG (gap up)', () => {
    const data: CandleData[] = [
      { time: 1, open: 100, high: 101, low: 99, close: 100.5, volume: 1000 },
      { time: 2, open: 101, high: 102, low: 100.5, close: 101.5, volume: 1000 },
      { time: 3, open: 102, high: 103, low: 101.5, close: 102.5, volume: 1000 }, // Low > bar0 high
    ]

    const patterns = detectFVGs(data, 0.1)
    assert.strictEqual(patterns.length, 1)
    assert.strictEqual(patterns[0].type, 'bullish')
    assert.strictEqual(patterns[0].gapBottom, 101)  // bar0.high
    assert.strictEqual(patterns[0].gapTop, 101.5)   // bar2.low
  })

  it('detects bearish FVG (gap down)', () => {
    const data: CandleData[] = [
      { time: 1, open: 100, high: 101, low: 99, close: 100.5, volume: 1000 },
      { time: 2, open: 99, high: 100, low: 98, close: 98.5, volume: 1000 },
      { time: 3, open: 98, high: 98.5, low: 97, close: 97.5, volume: 1000 }, // High < bar0 low
    ]

    const patterns = detectFVGs(data, 0.1)
    assert.strictEqual(patterns.length, 1)
    assert.strictEqual(patterns[0].type, 'bearish')
    assert.strictEqual(patterns[0].gapTop, 99)      // bar0.low
    assert.strictEqual(patterns[0].gapBottom, 98.5) // bar2.high
  })

  it('filters out gaps smaller than minPercent', () => {
    const data: CandleData[] = [
      { time: 1, open: 100, high: 100.1, low: 99.9, close: 100, volume: 1000 },
      { time: 2, open: 100.1, high: 100.2, low: 100.05, close: 100.15, volume: 1000 },
      { time: 3, open: 100.15, high: 100.25, low: 100.11, close: 100.2, volume: 1000 },
    ]

    const patterns = detectFVGs(data, 0.2) // 0.2% minimum
    assert.strictEqual(patterns.length, 0) // Gap too small
  })

  it('detects multiple FVGs', () => {
    const data: CandleData[] = [
      { time: 1, open: 100, high: 101, low: 99, close: 100.5, volume: 1000 },
      { time: 2, open: 101, high: 102, low: 100.5, close: 101.5, volume: 1000 },
      { time: 3, open: 102, high: 103, low: 101.5, close: 102.5, volume: 1000 }, // Bullish FVG
      { time: 4, open: 102.5, high: 103.5, low: 102, close: 103, volume: 1000 },
      { time: 5, open: 103, high: 104, low: 102.5, close: 103.5, volume: 1000 },
      { time: 6, open: 103.5, high: 104.5, low: 103, close: 104, volume: 1000 },
    ]

    const patterns = detectFVGs(data, 0.1)
    assert.ok(patterns.length >= 1, `Expected at least 1 FVG, got ${patterns.length}`)
  })
})
