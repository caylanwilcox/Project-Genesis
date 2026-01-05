/**
 * Advanced FVG Trading Strategy Service
 *
 * Gap-Zone Fibonacci FVG Strategy for 2:1 to 5:1 Returns
 *
 * Core Principles:
 * 1. Trade WITHIN the Fair Value Gap using Fibonacci levels for entry and targets
 * 2. Larger Gaps (1.5%+): Target patterns that signal major moves (2:1 to 5:1 R:R)
 * 3. Fibonacci-Based Targets: Enter at Fib retracements, target gap fill + extensions for 5% returns
 * 4. Multi-Timeframe Confirmation: Only trade FVGs that align with higher timeframe bias
 * 5. Quality Over Quantity: High-confidence patterns only (validation score ≥ 75%)
 * 6. Adaptive Risk Management: Position sizing based on pattern quality and market conditions
 */

import { CandleData } from '@/components/ProfessionalChart/types'
import { detectFvgPatterns, FvgPattern } from '@/components/ProfessionalChart/fvgDrawing'

export interface FvgStrategySignal {
  // Signal identification
  id: string
  timestamp: number
  type: 'long' | 'short'
  pattern: FvgPattern

  // Entry details
  entryPrice: number
  entryZoneLow: number
  entryZoneHigh: number

  // Exit levels
  stopLoss: number
  tp1: number  // 0.5:1 R:R - Partial profit (50% position)
  tp2: number  // 1:1 R:R - Partial profit (30% position)
  tp3: number  // 2:1 R:R - Final target (20% position)
  trailStopAfterTP1: boolean

  // Risk metrics
  riskAmount: number
  riskRewardRatio: number
  positionSizePercent: number  // % of capital to risk

  // Signal quality
  confidence: number  // 0-100
  strength: 'high' | 'medium' | 'low'

  // Multi-timeframe context
  htfBias?: 'bullish' | 'bearish' | 'neutral'  // Higher timeframe bias
  mtfConfirmation?: boolean  // Multi-timeframe confirmation

  // Pattern characteristics
  gapSizePercent: number
  volumeProfile: string
  marketStructure: string

  // Trade management
  status: 'active' | 'tp1_hit' | 'tp2_hit' | 'tp3_hit' | 'stopped_out' | 'expired'
  exitPrice?: number
  exitTime?: number
  profitLoss?: number
}

export interface FvgStrategyConfig {
  // Pattern filtering
  minConfidence: number  // Default: 75%
  minGapSizePercent: number  // Default: 0.15%
  maxGapSizePercent: number  // Default: 3.0%

  // Multi-timeframe settings
  requireHTFConfirmation: boolean  // Default: true
  htfLookback: number  // Bars to analyze for HTF bias (default: 20)

  // Risk management
  baseRiskPercent: number  // Base risk per trade (default: 1%)
  maxRiskPercent: number  // Maximum risk per trade (default: 2%)
  adjustRiskByConfidence: boolean  // Scale risk by signal confidence

  // Entry management
  entryMode: 'limit' | 'market' | 'stop'  // Entry order type
  entryZonePercent: number  // % of gap to use as entry zone (default: 50%)
  maxBarsToWaitForEntry: number  // Cancel signal after N bars (default: 10)

  // Exit management
  useTrailingStop: boolean  // Trail stop after TP1 hit
  trailStopPercent: number  // Trailing stop distance (default: 0.5%)
  partialProfits: boolean  // Take partial profits at TP1/TP2
}

const DEFAULT_CONFIG: FvgStrategyConfig = {
  minConfidence: 75,
  minGapSizePercent: 1.5,  // LARGER gaps (1.5%+) signal institutional activity and 2:1 to 5:1 move potential
  maxGapSizePercent: 8.0,  // Allow larger gaps for high volatility plays targeting 5% returns
  requireHTFConfirmation: true,
  htfLookback: 20,
  baseRiskPercent: 1.0,
  maxRiskPercent: 2.0,
  adjustRiskByConfidence: true,
  entryMode: 'limit',
  entryZonePercent: 50,  // Enter at 50% Fibonacci retracement within gap
  maxBarsToWaitForEntry: 10,
  useTrailingStop: true,
  trailStopPercent: 0.5,
  partialProfits: true,
}

/**
 * Analyze higher timeframe bias from recent price action
 */
function analyzeHTFBias(data: CandleData[], lookback: number): 'bullish' | 'bearish' | 'neutral' {
  if (data.length < lookback) return 'neutral'

  const recentData = data.slice(-lookback)

  // Calculate trend using EMAs
  const ema9 = calculateEMA(recentData.map(d => d.close), 9)
  const ema21 = calculateEMA(recentData.map(d => d.close), 21)

  // Price action analysis
  const currentPrice = recentData[recentData.length - 1].close
  const priceVsEMA9 = ((currentPrice - ema9[ema9.length - 1]) / currentPrice) * 100
  const priceVsEMA21 = ((currentPrice - ema21[ema21.length - 1]) / currentPrice) * 100

  // Higher highs / lower lows
  const highs = recentData.map(d => d.high)
  const lows = recentData.map(d => d.low)
  const recentHigh = Math.max(...highs.slice(-5))
  const priorHigh = Math.max(...highs.slice(-10, -5))
  const recentLow = Math.min(...lows.slice(-5))
  const priorLow = Math.min(...lows.slice(-10, -5))

  const higherHighs = recentHigh > priorHigh
  const higherLows = recentLow > priorLow
  const lowerHighs = recentHigh < priorHigh
  const lowerLows = recentLow < priorLow

  // Bullish bias: Price above EMAs, higher highs & higher lows
  if (priceVsEMA9 > 0 && priceVsEMA21 > 0 && higherHighs && higherLows) {
    return 'bullish'
  }

  // Bearish bias: Price below EMAs, lower highs & lower lows
  if (priceVsEMA9 < 0 && priceVsEMA21 < 0 && lowerHighs && lowerLows) {
    return 'bearish'
  }

  return 'neutral'
}

/**
 * Calculate Exponential Moving Average
 */
function calculateEMA(prices: number[], period: number): number[] {
  const ema: number[] = []
  const multiplier = 2 / (period + 1)

  // Start with SMA
  let sum = 0
  for (let i = 0; i < period; i++) {
    sum += prices[i]
  }
  ema[period - 1] = sum / period

  // Calculate EMA
  for (let i = period; i < prices.length; i++) {
    ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]
  }

  return ema
}

/**
 * Calculate position size based on confidence and risk settings
 */
function calculatePositionSize(
  confidence: number,
  config: FvgStrategyConfig
): number {
  if (!config.adjustRiskByConfidence) {
    return config.baseRiskPercent
  }

  // Scale risk from baseRisk to maxRisk based on confidence
  // 75% confidence = baseRisk, 100% confidence = maxRisk
  const minConfidence = config.minConfidence
  const confidenceRange = 100 - minConfidence
  const confidenceScore = (confidence - minConfidence) / confidenceRange

  const riskRange = config.maxRiskPercent - config.baseRiskPercent
  const positionSize = config.baseRiskPercent + (confidenceScore * riskRange)

  return Math.min(positionSize, config.maxRiskPercent)
}

/**
 * Generate FVG trading signals with advanced filtering
 */
export function generateFvgSignals(
  data: CandleData[],
  config: Partial<FvgStrategyConfig> = {}
): FvgStrategySignal[] {
  const cfg = { ...DEFAULT_CONFIG, ...config }
  const signals: FvgStrategySignal[] = []

  if (data.length < cfg.htfLookback + 10) {
    return signals
  }

  // Detect all FVG patterns
  const patterns = detectFvgPatterns(data, {
    recentOnly: true,
    minGapPct: cfg.minGapSizePercent,
    maxGapPct: cfg.maxGapSizePercent,
  })

  // Analyze higher timeframe bias
  const htfBias = analyzeHTFBias(data, cfg.htfLookback)

  // Filter and enhance patterns
  for (const pattern of patterns) {
    const confidence = pattern.validationScore * 100

    // Filter by minimum confidence
    if (confidence < cfg.minConfidence) {
      continue
    }

    // Check HTF confirmation if required
    const mtfConfirmation =
      (pattern.type === 'bullish' && htfBias === 'bullish') ||
      (pattern.type === 'bearish' && htfBias === 'bearish')

    if (cfg.requireHTFConfirmation && !mtfConfirmation) {
      continue
    }

    // Calculate Fibonacci entry zones WITHIN the gap
    // For bullish: Enter on retracement into gap (Fib 38.2% to 61.8%)
    // For bearish: Enter on bounce into gap (Fib 38.2% to 61.8%)
    const gapSize = pattern.gapHigh - pattern.gapLow
    const fib382 = pattern.type === 'bullish'
      ? pattern.gapLow + (gapSize * 0.382)
      : pattern.gapHigh - (gapSize * 0.382)
    const fib618 = pattern.type === 'bullish'
      ? pattern.gapLow + (gapSize * 0.618)
      : pattern.gapHigh - (gapSize * 0.618)

    // Entry zone is between Fibonacci 38.2% and 61.8% within the gap
    const entryZoneLow = pattern.type === 'bullish' ? fib382 : fib618
    const entryZoneHigh = pattern.type === 'bullish' ? fib618 : fib382

    // Calculate position size
    const positionSize = calculatePositionSize(confidence, cfg)

    // Determine signal strength
    let strength: 'high' | 'medium' | 'low' = 'low'
    if (confidence >= 85 && mtfConfirmation) {
      strength = 'high'
    } else if (confidence >= 75 || mtfConfirmation) {
      strength = 'medium'
    }

    // Calculate R:R ratio (based on TP3)
    const riskAmount = Math.abs(pattern.entryPrice - pattern.stopLoss)
    const rewardAmount = Math.abs(pattern.tp3 - pattern.entryPrice)
    const riskRewardRatio = rewardAmount / riskAmount

    // Create signal
    const signal: FvgStrategySignal = {
      id: `fvg_${pattern.type}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
      type: pattern.type === 'bullish' ? 'long' : 'short',
      pattern,

      entryPrice: pattern.entryPrice,
      entryZoneLow,
      entryZoneHigh,

      stopLoss: pattern.stopLoss,
      tp1: pattern.tp1,
      tp2: pattern.tp2,
      tp3: pattern.tp3,
      trailStopAfterTP1: cfg.useTrailingStop,

      riskAmount,
      riskRewardRatio,
      positionSizePercent: positionSize,

      confidence,
      strength,

      htfBias,
      mtfConfirmation,

      gapSizePercent: pattern.gapSizePct,
      volumeProfile: pattern.volumeProfile,
      marketStructure: pattern.marketStructure,

      status: 'active',
    }

    signals.push(signal)
  }

  // Sort by confidence (highest first)
  signals.sort((a, b) => b.confidence - a.confidence)

  return signals
}

/**
 * Get the best signal (highest confidence with HTF confirmation)
 */
export function getBestSignal(signals: FvgStrategySignal[]): FvgStrategySignal | null {
  if (signals.length === 0) return null

  // Prefer signals with HTF confirmation
  const confirmedSignals = signals.filter(s => s.mtfConfirmation)
  if (confirmedSignals.length > 0) {
    return confirmedSignals[0] // Already sorted by confidence
  }

  return signals[0]
}

/**
 * Format signal for display
 */
export function formatSignal(signal: FvgStrategySignal): string {
  const direction = signal.type.toUpperCase()
  const entry = signal.entryPrice.toFixed(2)
  const sl = signal.stopLoss.toFixed(2)
  const tp1 = signal.tp1.toFixed(2)
  const tp2 = signal.tp2.toFixed(2)
  const tp3 = signal.tp3.toFixed(2)
  const riskPct = signal.positionSizePercent.toFixed(1)
  const confidence = signal.confidence.toFixed(0)

  return `
${direction} Signal - ${signal.strength.toUpperCase()} Quality (${confidence}% confidence)
${signal.mtfConfirmation ? '✓' : '✗'} HTF Confirmation (${signal.htfBias || 'neutral'})

Entry Zone: $${signal.entryZoneLow.toFixed(2)} - $${signal.entryZoneHigh.toFixed(2)}
Entry Price: $${entry}
Stop Loss: $${sl}

Take Profit Levels:
  TP1 (50%): $${tp1} - Close 50% position
  TP2 (30%): $${tp2} - Close 30% position
  TP3 (20%): $${tp3} - Close final 20% position
${signal.trailStopAfterTP1 ? '  Trail stop after TP1 hit' : ''}

Risk Management:
  Risk/Reward: ${signal.riskRewardRatio.toFixed(2)}:1
  Position Size: ${riskPct}% of capital
  Risk per share: $${signal.riskAmount.toFixed(2)}

Pattern Analysis:
  Gap Size: ${signal.gapSizePercent.toFixed(2)}%
  Volume Profile: ${signal.volumeProfile}
  Market Structure: ${signal.marketStructure}
  `.trim()
}

/**
 * Get signal color for UI display
 */
export function getSignalColor(signal: FvgStrategySignal): string {
  if (signal.strength === 'high') {
    return signal.type === 'long' ? 'text-green-400' : 'text-red-400'
  } else if (signal.strength === 'medium') {
    return signal.type === 'long' ? 'text-green-500' : 'text-red-500'
  }
  return signal.type === 'long' ? 'text-green-600' : 'text-red-600'
}

/**
 * Get strategy statistics from recent signals
 */
export interface StrategyStats {
  totalSignals: number
  highQuality: number
  mediumQuality: number
  lowQuality: number
  avgConfidence: number
  htfConfirmationRate: number
  avgRiskReward: number
  bullishSignals: number
  bearishSignals: number
}

export function calculateStrategyStats(signals: FvgStrategySignal[]): StrategyStats {
  if (signals.length === 0) {
    return {
      totalSignals: 0,
      highQuality: 0,
      mediumQuality: 0,
      lowQuality: 0,
      avgConfidence: 0,
      htfConfirmationRate: 0,
      avgRiskReward: 0,
      bullishSignals: 0,
      bearishSignals: 0,
    }
  }

  const highQuality = signals.filter(s => s.strength === 'high').length
  const mediumQuality = signals.filter(s => s.strength === 'medium').length
  const lowQuality = signals.filter(s => s.strength === 'low').length

  const avgConfidence = signals.reduce((sum, s) => sum + s.confidence, 0) / signals.length

  const htfConfirmed = signals.filter(s => s.mtfConfirmation).length
  const htfConfirmationRate = (htfConfirmed / signals.length) * 100

  const avgRiskReward = signals.reduce((sum, s) => sum + s.riskRewardRatio, 0) / signals.length

  const bullishSignals = signals.filter(s => s.type === 'long').length
  const bearishSignals = signals.filter(s => s.type === 'short').length

  return {
    totalSignals: signals.length,
    highQuality,
    mediumQuality,
    lowQuality,
    avgConfidence,
    htfConfirmationRate,
    avgRiskReward,
    bullishSignals,
    bearishSignals,
  }
}
