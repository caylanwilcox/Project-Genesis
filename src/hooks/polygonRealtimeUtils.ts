import { NormalizedChartData, Timeframe } from '@/types/polygon'

export const INTRADAY_TIMEFRAMES: Timeframe[] = ['1m', '5m', '15m', '30m', '1h', '2h', '4h']

export const FALLBACK_INTRADAY_INTERVAL: Partial<Record<Timeframe, '1' | '5' | '15' | '30'>> = {
  '1m': '1',
  '5m': '5',
  '15m': '15',
  '30m': '30',
  '1h': '30',
  '2h': '30',
  '4h': '30',
}

export const TIMEFRAME_IN_MS: Record<Timeframe, number> = {
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

const MIN_ET_HOUR_FOR_PATCH = 6 // 6:00am ET - avoid patching overnight when no data exists

export function getCurrentEtDate(): Date {
  return new Date(new Date().toLocaleString('en-US', { timeZone: 'America/New_York' }))
}

export function formatEtDateKey(timestamp: number): string {
  return new Date(timestamp).toLocaleDateString('en-CA', { timeZone: 'America/New_York' })
}

interface PatchOptions {
  now?: Date
  utcNowMs?: number
}

export function shouldPatchIntradaySession(
  timeframe: Timeframe,
  aggregates: NormalizedChartData[],
  options: PatchOptions = {},
): boolean {
  if (!INTRADAY_TIMEFRAMES.includes(timeframe)) return false
  if (!aggregates || aggregates.length === 0) return false

  const latestBar = aggregates[aggregates.length - 1]
  if (!latestBar) return false

  const etNow = options.now ?? getCurrentEtDate()
  const utcNow = options.utcNowMs ?? Date.now()
  const isWeekday = etNow.getDay() !== 0 && etNow.getDay() !== 6
  if (!isWeekday) return false

  const etHour = etNow.getHours() + etNow.getMinutes() / 60
  if (etHour < MIN_ET_HOUR_FOR_PATCH) return false

  const todayKey = formatEtDateKey(etNow.getTime())
  const latestKey = formatEtDateKey(latestBar.time)
  if (todayKey !== latestKey) {
    return true
  }

  const intervalMs = TIMEFRAME_IN_MS[timeframe] ?? 0
  if (intervalMs === 0) return false
  const maxStaleness = Math.max(intervalMs * 1.5, 30 * 60 * 1000)
  const isStaleByTime = utcNow - latestBar.time > maxStaleness
  return isStaleByTime
}

export function mergeCandlesReplacing(
  original: NormalizedChartData[],
  patch: NormalizedChartData[],
): NormalizedChartData[] {
  if (patch.length === 0) return original
  const patchTimes = new Set(patch.map((bar) => bar.time))
  const filtered = original.filter((bar) => !patchTimes.has(bar.time))
  return [...filtered, ...patch].sort((a, b) => a.time - b.time)
}

export function aggregateBarsToDuration(
  bars: NormalizedChartData[],
  bucketMs: number,
): NormalizedChartData[] {
  if (bars.length === 0) return []

  const aggregated: NormalizedChartData[] = []
  let current: NormalizedChartData | null = null
  let bucketStart = 0

  for (const bar of bars) {
    if (!current) {
      current = { ...bar }
      bucketStart = bar.time
      continue
    }

    if (bar.time - bucketStart >= bucketMs) {
      aggregated.push(current)
      current = { ...bar }
      bucketStart = bar.time
      continue
    }

    current.high = Math.max(current.high, bar.high)
    current.low = Math.min(current.low, bar.low)
    current.close = bar.close
    current.volume += bar.volume
  }

  if (current) {
    aggregated.push(current)
  }

  return aggregated
}

