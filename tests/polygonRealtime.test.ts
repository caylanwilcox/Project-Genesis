import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { mergeCandlesReplacing } from '@/hooks/polygonRealtimeUtils'
import type { NormalizedChartData, Timeframe } from '@/types/polygon'

const INTERVAL_MS: Record<Timeframe, number> = {
  '1m': 60 * 1000,
  '5m': 5 * 60 * 1000,
  '15m': 15 * 60 * 1000,
  '30m': 30 * 60 * 1000,
  '1h': 60 * 60 * 1000,
  '2h': 2 * 60 * 60 * 1000,
  '4h': 4 * 60 * 60 * 1000,
  '1d': 24 * 60 * 60 * 1000,
  '1w': 7 * 24 * 60 * 60 * 1000,
  '1M': 30 * 24 * 60 * 60 * 1000,
}

const TIMEFRAMES = Object.keys(INTERVAL_MS) as Timeframe[]
const BASE_TIME = Date.UTC(2024, 10, 22, 20, 0, 0)

const createBar = (time: number, close: number): NormalizedChartData => ({
  time,
  open: close - 0.25,
  high: close + 0.5,
  low: close - 0.75,
  close,
  volume: 10_000,
})

describe('Polygon realtime candle reconciliation', () => {
  TIMEFRAMES.forEach((timeframe) => {
    it(`keeps realtime patch as latest ${timeframe} candle`, () => {
      const interval = INTERVAL_MS[timeframe]
      const historical: NormalizedChartData[] = [
        createBar(BASE_TIME - interval * 3, 100),
        createBar(BASE_TIME - interval * 2, 101),
        createBar(BASE_TIME - interval, 102),
      ]

      // Patch mirrors Polygon's realtime feed: first bar overwrites cached candle,
      // second bar represents the newest trade for this interval.
      const realtimePatch: NormalizedChartData[] = [
        createBar(BASE_TIME - interval, 150),
        createBar(BASE_TIME, 151),
      ]

      const merged = mergeCandlesReplacing(historical, realtimePatch)
      const latest = merged[merged.length - 1]
      const expectedLatest = realtimePatch[realtimePatch.length - 1]

      assert.equal(
        latest.time,
        expectedLatest.time,
        `latest ${timeframe} candle should come from realtime patch`
      )
      assert.equal(
        latest.close,
        expectedLatest.close,
        'latest candle price should mirror Polygon realtime close'
      )

      const duplicateCount = merged.filter((bar) => bar.time === realtimePatch[0].time).length
      assert.equal(
        duplicateCount,
        1,
        'patched candle should replace (not duplicate) the cached timestamp'
      )

      merged.forEach((bar, index) => {
        if (index === 0) return
        assert.ok(
          merged[index - 1].time < bar.time,
          'candles should remain sorted after patching'
        )
      })
    })
  })
})

