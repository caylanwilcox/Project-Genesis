import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { shouldPatchIntradaySession } from '@/hooks/polygonRealtimeUtils'
import { NormalizedChartData } from '@/types/polygon'

const makeBar = (time: number): NormalizedChartData => ({
  time,
  open: 100,
  high: 101,
  low: 99,
  close: 100.5,
  volume: 10_000,
})

describe('shouldPatchIntradaySession', () => {
  it('patches when latest bar is from previous trading day', () => {
    const bars = [makeBar(Date.UTC(2024, 10, 21, 21, 0, 0))]
    const shouldPatch = shouldPatchIntradaySession('1h', bars, {
      now: new Date('2024-11-22T15:00:00-05:00'),
      utcNowMs: Date.UTC(2024, 10, 22, 20, 0, 0),
    })
    assert.equal(shouldPatch, true)
  })

  it('patches when latest bar is same day but stale beyond 1.5x interval', () => {
    const bars = [makeBar(Date.UTC(2024, 10, 22, 15, 0, 0))]
    const shouldPatch = shouldPatchIntradaySession('1h', bars, {
      now: new Date('2024-11-22T15:30:00-05:00'),
      utcNowMs: Date.UTC(2024, 10, 22, 21, 30, 0),
    })
    assert.equal(shouldPatch, true)
  })

  it('skips patch when latest bar is fresh for the current day', () => {
    const bars = [makeBar(Date.UTC(2024, 10, 22, 20, 0, 0))]
    const shouldPatch = shouldPatchIntradaySession('1h', bars, {
      now: new Date('2024-11-22T16:30:00-05:00'),
      utcNowMs: Date.UTC(2024, 10, 22, 21, 30, 0),
    })
    assert.equal(shouldPatch, false)
  })
})

