/**
 * Fair Value Gap (FVG) Detection Service
 *
 * Implements Fabio Valentini's Fair Value Gap methodology:
 * 1. 3-Candle Pattern: Identifies gaps between candle 1 and candle 3
 * 2. Market Structure: Validates balance → imbalance transitions
 * 3. Volume Profile: Checks for bell curve distribution
 * 4. Entry/Exit Levels: Calculates Fibonacci-based take profits
 *
 * Trading Modes:
 * - scalping: 1-15 minute trades
 * - intraday: 1-4 hour trades
 * - daily: 4-24 hour trades
 * - swing: 2-3 day trades
 * - weekly: 5-7 day trades
 * - biweekly: 10-14 day trades
 * - monthly: 20-30 day trades
 */

import { Decimal } from '@prisma/client/runtime/library'

export interface MarketBar {
  timestamp: Date
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface FvgPattern {
  // Pattern identification
  fvgType: 'bullish' | 'bearish'
  tradingMode: TradingMode
  detectedAt: Date

  // 3-Candle pattern data
  candle1: {
    timestamp: Date
    high: number
    low: number
  }
  candle2: {
    timestamp: Date
    high: number
    low: number
  }
  candle3: {
    timestamp: Date
    high: number
    low: number
  }

  // Gap metrics
  gapHigh: number
  gapLow: number
  gapSize: number
  gapSizePct: number

  // Entry and exit levels
  entryPrice: number
  stopLoss: number
  takeProfit1: number  // 38.2% Fibonacci
  takeProfit2: number  // 61.8% Fibonacci
  takeProfit3: number  // 100% (full gap fill)

  // Validation metrics
  volumeProfile?: 'bell_curve' | 'front_loaded' | 'back_loaded' | 'flat'
  marketStructure?: 'balance_to_imbalance' | 'trending' | 'ranging' | 'choppy'
  validationScore: number  // 0-1 confidence
}

export type TradingMode =
  | 'scalping'   // 1-15 min
  | 'intraday'   // 1-4 hours
  | 'daily'      // 4-24 hours
  | 'swing'      // 2-3 days
  | 'weekly'     // 5-7 days
  | 'biweekly'   // 10-14 days
  | 'monthly'    // 20-30 days

export interface FvgDetectionConfig {
  minGapSizePct: number      // Minimum gap size as % of price (default: 0.1%)
  maxGapSizePct: number      // Maximum gap size as % of price (default: 5%)
  requireVolumeConfirmation: boolean  // Require volume spike on candle 2
  minValidationScore: number  // Minimum validation score (default: 0.6)
  fibonacciLevels: {
    tp1: number  // Default: 0.382
    tp2: number  // Default: 0.618
    tp3: number  // Default: 1.0
  }
}

const DEFAULT_CONFIG: FvgDetectionConfig = {
  minGapSizePct: 0.1,
  maxGapSizePct: 5.0,
  requireVolumeConfirmation: true,
  minValidationScore: 0.6,
  fibonacciLevels: {
    tp1: 0.382,
    tp2: 0.618,
    tp3: 1.0,
  },
}

export class FvgDetectionService {
  private config: FvgDetectionConfig

  constructor(config?: Partial<FvgDetectionConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config }
  }

  /**
   * Scan historical data for FVG patterns
   * @param bars - Array of OHLCV bars (must be sorted by timestamp ascending)
   * @param tradingMode - Which trading mode to optimize for
   * @returns Array of detected FVG patterns
   */
  detectFvgs(bars: MarketBar[], tradingMode: TradingMode): FvgPattern[] {
    const patterns: FvgPattern[] = []

    // Need at least 3 candles for FVG detection
    if (bars.length < 3) {
      return patterns
    }

    // Scan through bars looking for 3-candle FVG patterns
    for (let i = 0; i < bars.length - 2; i++) {
      const candle1 = bars[i]
      const candle2 = bars[i + 1]
      const candle3 = bars[i + 2]

      // Check for bullish FVG (gap up)
      const bullishFvg = this.detectBullishFvg(candle1, candle2, candle3, tradingMode)
      if (bullishFvg) {
        patterns.push(bullishFvg)
      }

      // Check for bearish FVG (gap down)
      const bearishFvg = this.detectBearishFvg(candle1, candle2, candle3, tradingMode)
      if (bearishFvg) {
        patterns.push(bearishFvg)
      }
    }

    return patterns
  }

  /**
   * Detect bullish FVG pattern
   * Bullish FVG: Candle 3 low > Candle 1 high (gap between them)
   */
  private detectBullishFvg(
    candle1: MarketBar,
    candle2: MarketBar,
    candle3: MarketBar,
    tradingMode: TradingMode
  ): FvgPattern | null {
    // Check if there's a gap: candle3.low > candle1.high
    if (candle3.low <= candle1.high) {
      return null
    }

    const gapLow = candle1.high
    const gapHigh = candle3.low
    const gapSize = gapHigh - gapLow
    const gapSizePct = (gapSize / candle2.close) * 100

    // Validate gap size
    if (gapSizePct < this.config.minGapSizePct || gapSizePct > this.config.maxGapSizePct) {
      return null
    }

    // Calculate entry and exit levels
    const entryPrice = gapHigh  // Enter at top of gap (conservative)
    const stopLoss = gapLow - (gapSize * 0.1)  // 10% buffer below gap
    const takeProfit1 = gapLow + (gapSize * this.config.fibonacciLevels.tp1)
    const takeProfit2 = gapLow + (gapSize * this.config.fibonacciLevels.tp2)
    const takeProfit3 = gapLow  // Full gap fill

    // Calculate validation score
    const validationScore = this.calculateValidationScore(candle1, candle2, candle3, 'bullish')

    // Check minimum validation score
    if (validationScore < this.config.minValidationScore) {
      return null
    }

    // Analyze volume profile
    const volumeProfile = this.analyzeVolumeProfile([candle1, candle2, candle3])

    // Analyze market structure
    const marketStructure = this.analyzeMarketStructure([candle1, candle2, candle3], 'bullish')

    return {
      fvgType: 'bullish',
      tradingMode,
      detectedAt: candle3.timestamp,
      candle1: {
        timestamp: candle1.timestamp,
        high: candle1.high,
        low: candle1.low,
      },
      candle2: {
        timestamp: candle2.timestamp,
        high: candle2.high,
        low: candle2.low,
      },
      candle3: {
        timestamp: candle3.timestamp,
        high: candle3.high,
        low: candle3.low,
      },
      gapHigh,
      gapLow,
      gapSize,
      gapSizePct,
      entryPrice,
      stopLoss,
      takeProfit1,
      takeProfit2,
      takeProfit3,
      volumeProfile,
      marketStructure,
      validationScore,
    }
  }

  /**
   * Detect bearish FVG pattern
   * Bearish FVG: Candle 3 high < Candle 1 low (gap between them)
   */
  private detectBearishFvg(
    candle1: MarketBar,
    candle2: MarketBar,
    candle3: MarketBar,
    tradingMode: TradingMode
  ): FvgPattern | null {
    // Check if there's a gap: candle3.high < candle1.low
    if (candle3.high >= candle1.low) {
      return null
    }

    const gapHigh = candle1.low
    const gapLow = candle3.high
    const gapSize = gapHigh - gapLow
    const gapSizePct = (gapSize / candle2.close) * 100

    // Validate gap size
    if (gapSizePct < this.config.minGapSizePct || gapSizePct > this.config.maxGapSizePct) {
      return null
    }

    // Calculate entry and exit levels
    const entryPrice = gapLow  // Enter at bottom of gap (conservative)
    const stopLoss = gapHigh + (gapSize * 0.1)  // 10% buffer above gap
    const takeProfit1 = gapHigh - (gapSize * this.config.fibonacciLevels.tp1)
    const takeProfit2 = gapHigh - (gapSize * this.config.fibonacciLevels.tp2)
    const takeProfit3 = gapHigh  // Full gap fill

    // Calculate validation score
    const validationScore = this.calculateValidationScore(candle1, candle2, candle3, 'bearish')

    // Check minimum validation score
    if (validationScore < this.config.minValidationScore) {
      return null
    }

    // Analyze volume profile
    const volumeProfile = this.analyzeVolumeProfile([candle1, candle2, candle3])

    // Analyze market structure
    const marketStructure = this.analyzeMarketStructure([candle1, candle2, candle3], 'bearish')

    return {
      fvgType: 'bearish',
      tradingMode,
      detectedAt: candle3.timestamp,
      candle1: {
        timestamp: candle1.timestamp,
        high: candle1.high,
        low: candle1.low,
      },
      candle2: {
        timestamp: candle2.timestamp,
        high: candle2.high,
        low: candle2.low,
      },
      candle3: {
        timestamp: candle3.timestamp,
        high: candle3.high,
        low: candle3.low,
      },
      gapHigh,
      gapLow,
      gapSize,
      gapSizePct,
      entryPrice,
      stopLoss,
      takeProfit1,
      takeProfit2,
      takeProfit3,
      volumeProfile,
      marketStructure,
      validationScore,
    }
  }

  /**
   * Calculate validation score based on Fabio Valentini's rules
   * Returns 0-1 score (higher is better)
   */
  private calculateValidationScore(
    candle1: MarketBar,
    candle2: MarketBar,
    candle3: MarketBar,
    direction: 'bullish' | 'bearish'
  ): number {
    let score = 0.5  // Base score

    // Rule 1: Volume confirmation (candle 2 should have higher volume)
    if (this.config.requireVolumeConfirmation) {
      const avgVolume = (candle1.volume + candle3.volume) / 2
      if (candle2.volume > avgVolume * 1.2) {
        score += 0.2  // +20% for volume spike
      }
    }

    // Rule 2: Candle 2 momentum (should be strong directional move)
    const candle2Range = candle2.high - candle2.low
    const candle2Body = Math.abs(candle2.close - candle2.open)
    const bodyRatio = candle2Body / candle2Range

    if (bodyRatio > 0.7) {
      score += 0.15  // +15% for strong momentum candle
    }

    // Rule 3: Directional confirmation
    if (direction === 'bullish') {
      // Bullish: candle2 should be green and candle3 should be green
      if (candle2.close > candle2.open) score += 0.1
      if (candle3.close > candle3.open) score += 0.05
    } else {
      // Bearish: candle2 should be red and candle3 should be red
      if (candle2.close < candle2.open) score += 0.1
      if (candle3.close < candle3.open) score += 0.05
    }

    // Ensure score stays in 0-1 range
    return Math.min(1.0, Math.max(0.0, score))
  }

  /**
   * Analyze volume profile across 3 candles
   * Fabio Valentini looks for bell curve distribution
   */
  private analyzeVolumeProfile(candles: MarketBar[]): 'bell_curve' | 'front_loaded' | 'back_loaded' | 'flat' {
    if (candles.length !== 3) return 'flat'

    const [v1, v2, v3] = candles.map(c => c.volume)
    const total = v1 + v2 + v3
    const avg = total / 3

    // Bell curve: middle candle has highest volume
    if (v2 > v1 && v2 > v3 && v2 > avg * 1.2) {
      return 'bell_curve'
    }

    // Front loaded: first candle has highest volume
    if (v1 > v2 && v1 > v3) {
      return 'front_loaded'
    }

    // Back loaded: last candle has highest volume
    if (v3 > v1 && v3 > v2) {
      return 'back_loaded'
    }

    return 'flat'
  }

  /**
   * Analyze market structure
   * Fabio Valentini emphasizes balance → imbalance transitions
   */
  private analyzeMarketStructure(
    candles: MarketBar[],
    direction: 'bullish' | 'bearish'
  ): 'balance_to_imbalance' | 'trending' | 'ranging' | 'choppy' {
    if (candles.length !== 3) return 'choppy'

    const [c1, c2, c3] = candles

    // Calculate ranges
    const range1 = c1.high - c1.low
    const range2 = c2.high - c2.low
    const range3 = c3.high - c3.low

    // Balance to imbalance: candle 2 range > 2x average of candle 1 and 3
    const avgSideRange = (range1 + range3) / 2
    if (range2 > avgSideRange * 2) {
      return 'balance_to_imbalance'
    }

    // Trending: consistent direction
    if (direction === 'bullish') {
      if (c1.close < c2.close && c2.close < c3.close) {
        return 'trending'
      }
    } else {
      if (c1.close > c2.close && c2.close > c3.close) {
        return 'trending'
      }
    }

    // Ranging: similar highs and lows
    const totalRange = Math.max(c1.high, c2.high, c3.high) - Math.min(c1.low, c2.low, c3.low)
    if (totalRange < avgSideRange * 2) {
      return 'ranging'
    }

    return 'choppy'
  }

  /**
   * Get optimal trading mode based on timeframe
   */
  static getTradingModeForTimeframe(timeframe: string): TradingMode {
    const lower = timeframe.toLowerCase()

    if (lower.includes('1m') || lower.includes('5m')) return 'scalping'
    if (lower.includes('15m') || lower.includes('30m')) return 'scalping'
    if (lower.includes('1h') || lower.includes('2h')) return 'intraday'
    if (lower.includes('4h')) return 'daily'
    if (lower.includes('1d')) return 'swing'
    if (lower.includes('1w')) return 'weekly'

    return 'daily'  // Default
  }

  /**
   * Format FVG pattern for display
   */
  static formatPattern(pattern: FvgPattern): string {
    const direction = pattern.fvgType === 'bullish' ? '📈 BULLISH' : '📉 BEARISH'
    return `
${direction} FVG detected at ${pattern.detectedAt.toISOString()}
Gap: $${pattern.gapLow.toFixed(2)} - $${pattern.gapHigh.toFixed(2)} (${pattern.gapSizePct.toFixed(2)}%)
Entry: $${pattern.entryPrice.toFixed(2)}
Stop Loss: $${pattern.stopLoss.toFixed(2)}
TP1 (38.2%): $${pattern.takeProfit1.toFixed(2)}
TP2 (61.8%): $${pattern.takeProfit2.toFixed(2)}
TP3 (100%): $${pattern.takeProfit3.toFixed(2)}
Validation Score: ${(pattern.validationScore * 100).toFixed(1)}%
Volume Profile: ${pattern.volumeProfile || 'N/A'}
Market Structure: ${pattern.marketStructure || 'N/A'}
    `.trim()
  }
}
