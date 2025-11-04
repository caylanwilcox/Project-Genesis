import { Timeframe } from '@/types/polygon'

// Canonical display timeframes used across the UI
export const DISPLAY_TIMEFRAMES: string[] = [
  '1D', '5D', '1M', '3M', '6M', 'YTD', '1Y', '5Y', 'All',
]

// Map UI display timeframe to API timeframe and human label
const displayToDataTimeframeMap: Record<string, { timeframe: Timeframe; intervalLabel: string }> = {
  // Use finer intraday granularity for better fidelity while staying performant
  '1D':  { timeframe: '15m', intervalLabel: '15 min' },
  '5D':  { timeframe: '1h',  intervalLabel: '1 hour' },
  '1M':  { timeframe: '4h',  intervalLabel: '4 hour' },
  '3M':  { timeframe: '1d',  intervalLabel: '1 day' },
  '6M':  { timeframe: '1d',  intervalLabel: '1 day' },
  'YTD': { timeframe: '1d',  intervalLabel: '1 day' },
  '1Y':  { timeframe: '1d',  intervalLabel: '1 day' },
  '5Y':  { timeframe: '1w',  intervalLabel: '1 week' },
  'All': { timeframe: '1M',  intervalLabel: '1 month' },
}

// Convert dropdown interval label to API timeframe
const intervalLabelToTimeframeMap: Record<string, Timeframe> = {
  '1 min': '1m',
  '5 min': '5m',
  '15 min': '15m',
  '30 min': '30m',
  '1 hour': '1h',
  '2 hour': '2h',
  '4 hour': '4h',
  '1 day': '1d',
  '1 week': '1w',
  '1 month': '1M',
}

export function resolveDisplayToData(displayTimeframe: string): { timeframe: Timeframe; intervalLabel: string } {
  return displayToDataTimeframeMap[displayTimeframe] ?? { timeframe: '1h', intervalLabel: '1 hour' }
}

export function intervalLabelToTimeframe(label: string): Timeframe | undefined {
  return intervalLabelToTimeframeMap[label]
}

// Recommended polling cadence based on API timeframe (ms)
export function recommendedRefreshMs(timeframe: Timeframe): number {
  switch (timeframe) {
    case '1m':
    case '5m':
    case '15m':
      return 15000; // 15s for faster intraday updates
    case '30m':
    case '1h':
    case '2h':
    case '4h':
      return 60000; // 60s
    case '1d':
      return 5 * 60 * 1000; // 5m
    case '1w':
    case '1M':
      return 15 * 60 * 1000; // 15m
    default:
      return 60000;
  }
}

// Recommended bar limits based on timeframe and selected display range
export function recommendedBarLimit(timeframe: Timeframe, displayTimeframe: string): number {
  // Minute resolutions
  if (timeframe === '1m') return 390 // full trading day
  if (timeframe === '5m') return 390 // ~5 days coverage
  if (timeframe === '15m') return 200 // ~2-3 weeks
  if (timeframe === '30m') return 200

  // Hour resolutions
  if (timeframe === '1h') {
    if (displayTimeframe === '1D') return 10 // 1 day + buffer
    if (displayTimeframe === '5D') return 150 // ~1 month for scroll headroom
    if (displayTimeframe === '1M') return 150 // ~1 month
    return 200
  }
  if (timeframe === '2h') return 200
  if (timeframe === '4h') return 200

  // Daily and above
  if (timeframe === '1d') {
    if (displayTimeframe === '1M') return 30
    if (displayTimeframe === '3M') return 90
    if (displayTimeframe === '6M') return 180
    if (displayTimeframe === 'YTD') return 300
    if (displayTimeframe === '1Y') return 365
    return 200
  }
  if (timeframe === '1w') return 260 // ~5 years
  if (timeframe === '1M') return 120 // ~10 years

  return 200
}

export const INTERVAL_LABELS: string[] = [
  '1 min', '5 min', '15 min', '30 min', '1 hour', '2 hour', '4 hour', '1 day', '1 week', '1 month',
]


