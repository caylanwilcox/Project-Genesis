import { Timeframe } from '@/types/polygon'

export interface FvgGapSettings {
  min: number
  max: number
  defaultValue: number
  step: number
}

/**
 * Get FVG gap threshold settings based on timeframe
 * Different timeframes need different gap thresholds due to price volatility
 */
export function getFvgGapSettingsForTimeframe(timeframe: Timeframe): FvgGapSettings {
  switch (timeframe) {
    case '1m':
    case '5m':
      // Intraday minute charts: very tight gaps (0.1% - 2%)
      return {
        min: 0.1,
        max: 2.0,
        defaultValue: 0.3,
        step: 0.1,
      }

    case '15m':
    case '30m':
      // Short intraday: tight gaps (0.2% - 3%)
      return {
        min: 0.2,
        max: 3.0,
        defaultValue: 0.5,
        step: 0.1,
      }

    case '1h':
    case '2h':
      // Hourly charts: moderate gaps (0.3% - 4%)
      return {
        min: 0.3,
        max: 4.0,
        defaultValue: 0.8,
        step: 0.1,
      }

    case '4h':
      // 4-hour charts: moderate gaps (0.5% - 5%)
      return {
        min: 0.5,
        max: 5.0,
        defaultValue: 1.0,
        step: 0.1,
      }

    case '1d':
      // Daily charts: larger gaps (1% - 10%)
      return {
        min: 1.0,
        max: 10.0,
        defaultValue: 2.0,
        step: 0.5,
      }

    case '1w':
      // Weekly charts: very large gaps (2% - 15%)
      return {
        min: 2.0,
        max: 15.0,
        defaultValue: 3.0,
        step: 0.5,
      }

    case '1M':
      // Monthly charts: massive gaps (5% - 25%)
      return {
        min: 5.0,
        max: 25.0,
        defaultValue: 8.0,
        step: 1.0,
      }

    default:
      // Default: moderate settings
      return {
        min: 0.5,
        max: 5.0,
        defaultValue: 1.0,
        step: 0.1,
      }
  }
}

/**
 * Check if a gap percentage is valid for the given timeframe
 */
export function isValidFvgGapForTimeframe(gapPercent: number, timeframe: Timeframe): boolean {
  const settings = getFvgGapSettingsForTimeframe(timeframe)
  return gapPercent >= settings.min && gapPercent <= settings.max
}

/**
 * Clamp a gap percentage to valid range for timeframe
 */
export function clampFvgGapForTimeframe(gapPercent: number, timeframe: Timeframe): number {
  const settings = getFvgGapSettingsForTimeframe(timeframe)
  return Math.max(settings.min, Math.min(settings.max, gapPercent))
}
