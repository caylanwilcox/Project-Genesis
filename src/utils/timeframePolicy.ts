import { Timeframe } from '@/types/polygon'

// Canonical display timeframes used across the UI
export const DISPLAY_TIMEFRAMES: string[] = [
  '1D', '5D', '1M', '3M', '6M', 'YTD', '1Y', '5Y', 'All',
]

// Map UI display timeframe to API timeframe and human label
// Goal: Show appropriate granularity for each time period
const displayToDataTimeframeMap: Record<string, { timeframe: Timeframe; intervalLabel: string }> = {
  '1D':  { timeframe: '1m',  intervalLabel: '1 min' },   // 1 day: 1-min bars (390 bars per trading day)
  '5D':  { timeframe: '5m',  intervalLabel: '5 min' },   // 5 days: 5-min bars (390 bars total)
  '1M':  { timeframe: '1h',  intervalLabel: '1 hour' },   // 1 month: hourly bars (~140 bars)
  '3M':  { timeframe: '1h',  intervalLabel: '1 hour' },   // 3 months: hourly bars (~410 bars)
  '6M':  { timeframe: '4h',  intervalLabel: '4 hour' },   // 6 months: 4-hour bars (~315 bars)
  'YTD': { timeframe: '1d',  intervalLabel: '1 day' },   // Year to date: daily bars
  '1Y':  { timeframe: '1d',  intervalLabel: '1 day' },   // 1 year: daily bars (252 trading days)
  '5Y':  { timeframe: '1w',  intervalLabel: '1 week' },  // 5 years: weekly bars (260 bars)
  'All': { timeframe: '1M',  intervalLabel: '1 month' }, // All time: monthly bars
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

const intervalLabelToDisplayMap: Record<string, string> = {
  '1 min': '1D',
  '5 min': '1D',
  '15 min': '5D',
  '30 min': '1M',
  '1 hour': '1M',
  '2 hour': '3M',
  '4 hour': '3M',
  '1 day': '6M',
  '1 week': '1Y',
  '1 month': '5Y',
}

export function intervalLabelToDisplayTimeframe(label: string): string | undefined {
  return intervalLabelToDisplayMap[label]
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

// Recommended bar limits based on interval and display timeframe
// Strategy: Load 3-10x the visible bars for smooth panning
export function recommendedBarLimit(timeframe: Timeframe, displayTimeframe: string): number {
  // Calculate base visible bars for this combination
  const getVisibleBars = (interval: Timeframe, display: string): number => {
    const tradingHoursPerDay = 6.5;

    // 1D display timeframe
    if (display === '1D') {
      if (interval === '1m') return 390;   // 6.5 hours × 60
      if (interval === '5m') return 78;    // 6.5 hours × 12
      if (interval === '15m') return 26;   // 6.5 hours × 4
      if (interval === '30m') return 13;   // 6.5 hours × 2
      if (interval === '1h') return 7;     // 6.5 hours
      if (interval === '2h') return 3;     // 6.5 / 2
      if (interval === '4h') return 2;     // 6.5 / 4
      if (interval === '1d') return 1;
      return 78;
    }

    // 5D display timeframe
    if (display === '5D') {
      if (interval === '1m') return 1950;  // 5 days × 390
      if (interval === '5m') return 390;   // 5 days × 78
      if (interval === '15m') return 130;  // 5 days × 26
      if (interval === '30m') return 65;   // 5 days × 13
      if (interval === '1h') return 33;    // 5 days × 6.5
      if (interval === '2h') return 16;    // 5 days × 3.25
      if (interval === '4h') return 8;     // 5 days × 1.625
      if (interval === '1d') return 5;
      return 65;
    }

    // 1M display timeframe (~21 trading days)
    if (display === '1M') {
      if (interval === '1m') return 8190;  // 21 × 390
      if (interval === '5m') return 1638;  // 21 × 78
      if (interval === '15m') return 546;  // 21 × 26
      if (interval === '30m') return 273;  // 21 × 13
      if (interval === '1h') return 140;   // 21 × 6.5 (rounded)
      if (interval === '2h') return 68;    // 21 × 3.25
      if (interval === '4h') return 34;    // 21 × 1.625
      if (interval === '1d') return 21;
      return 140;
    }

    // 3M display timeframe (~63 trading days)
    if (display === '3M') {
      if (interval === '1m') return 24570; // 63 × 390
      if (interval === '5m') return 4914;  // 63 × 78
      if (interval === '15m') return 1638; // 63 × 26
      if (interval === '30m') return 819;  // 63 × 13
      if (interval === '1h') return 410;   // 63 × 6.5
      if (interval === '2h') return 205;   // 63 × 3.25
      if (interval === '4h') return 100;   // 63 × 1.625 (rounded)
      if (interval === '1d') return 63;
      if (interval === '1w') return 13;
      return 100;
    }

    // 6M display timeframe (~126 trading days)
    if (display === '6M') {
      if (interval === '2h') return 410;   // 126 × 3.25
      if (interval === '4h') return 205;   // 126 × 1.625
      if (interval === '1d') return 126;
      if (interval === '1w') return 26;
      return 126;
    }

    // 1Y display timeframe (252 trading days)
    if (display === '1Y') {
      if (interval === '4h') return 410;   // 252 × 1.625
      if (interval === '1d') return 252;
      if (interval === '1w') return 52;
      return 252;
    }

    // 5Y display timeframe
    if (display === '5Y') {
      if (interval === '1d') return 1260;  // 1260 trading days
      if (interval === '1w') return 260;   // 260 weeks
      if (interval === '1M') return 60;    // 60 months
      return 260;
    }

    // YTD - varies, use generous estimate
    if (display === 'YTD') {
      if (interval === '4h') return 350;
      if (interval === '1d') return 200;
      if (interval === '1w') return 40;
      return 200;
    }

    // All - maximum data
    if (display === 'All') {
      if (interval === '1w') return 520;   // 10 years
      if (interval === '1M') return 240;   // 20 years
      if (interval === '1d') return 2520;  // 10 years
      return 520;
    }

    return 100; // fallback
  };

  const visibleBars = getVisibleBars(timeframe, displayTimeframe);

  // Apply 3x buffer for smooth panning (except for very long timeframes)
  const bufferMultiplier = displayTimeframe === 'All' || displayTimeframe === '5Y' ? 1.5 : 3;

  return Math.ceil(visibleBars * bufferMultiplier);
}

export const INTERVAL_LABELS: string[] = [
  '1 min', '5 min', '15 min', '30 min', '1 hour', '2 hour', '4 hour', '1 day', '1 week', '1 month',
]

// Ordered display timeframes from shortest to longest duration
export const TIMEFRAME_PROGRESSION: string[] = [
  '1D', '5D', '1M', '3M', '6M', 'YTD', '1Y', '5Y', 'All',
]

/**
 * Get the next coarser (longer) timeframe when zooming out
 * Returns undefined if already at the coarsest timeframe
 */
export function getNextCoarserTimeframe(currentDisplayTimeframe: string): string | undefined {
  const currentIndex = TIMEFRAME_PROGRESSION.indexOf(currentDisplayTimeframe)
  if (currentIndex === -1 || currentIndex >= TIMEFRAME_PROGRESSION.length - 1) {
    return undefined
  }
  return TIMEFRAME_PROGRESSION[currentIndex + 1]
}

/**
 * Get the next finer (shorter) timeframe when zooming in
 * Returns undefined if already at the finest timeframe
 */
export function getNextFinerTimeframe(currentDisplayTimeframe: string): string | undefined {
  const currentIndex = TIMEFRAME_PROGRESSION.indexOf(currentDisplayTimeframe)
  if (currentIndex <= 0) {
    return undefined
  }
  return TIMEFRAME_PROGRESSION[currentIndex - 1]
}

/**
 * Get zoom thresholds for transitioning between timeframes
 * Returns the timeScale values at which we should transition to coarser/finer timeframes
 */
export function getZoomTransitionThresholds(_displayTimeframe: string): {
  zoomOutThreshold: number  // When timeScale drops below this, go to coarser timeframe
  zoomInThreshold: number   // When timeScale exceeds this, go to finer timeframe
} {
  // Default thresholds - transition when zoomed out to show 3x more candles (scale 0.33)
  // or zoomed in to show 3x fewer candles (scale 3.0)
  // Note: _displayTimeframe can be used in the future for timeframe-specific thresholds
  return {
    zoomOutThreshold: 0.25,  // Very zoomed out - show coarser timeframe
    zoomInThreshold: 2.5,    // Very zoomed in - show finer timeframe
  }
}


