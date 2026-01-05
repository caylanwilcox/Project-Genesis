/**
 * FVG Backtesting Service
 * Analyzes historical FVG patterns to calculate win rates and trading performance
 */

import { CandleData } from '@/components/ProfessionalChart/types'
import { detectFvgPatterns, FvgPattern } from '@/components/ProfessionalChart/fvgDrawing'
import { filterMarketHoursData } from '@/utils/marketHours'

export interface FvgTradeResult {
  pattern: FvgPattern
  detectionIndex: number
  detectionTime: number

  // Entry execution
  entryHit: boolean
  entryTime?: number
  entryIndex?: number

  // Target results
  tp1Hit: boolean
  tp1Time?: number
  tp1Index?: number

  tp2Hit: boolean
  tp2Time?: number
  tp2Index?: number

  tp3Hit: boolean
  tp3Time?: number
  tp3Index?: number

  // Stop loss
  stopLossHit: boolean
  stopLossTime?: number
  stopLossIndex?: number

  // Trade outcome
  outcome: 'win' | 'loss' | 'no_entry' | 'incomplete' | 'expired'
  profitPct?: number
  barsToEntry?: number
  barsToOutcome?: number
}

export interface BacktestResults {
  totalPatterns: number
  patternsWithEntry: number

  // Overall stats
  totalTrades: number
  wins: number
  losses: number
  incomplete: number
  winRate: number

  // Target stats
  tp1HitRate: number
  tp2HitRate: number
  tp3HitRate: number

  // Performance by type
  bullishWinRate: number
  bearishWinRate: number

  // Performance by confidence
  highConfidenceWinRate: number // >= 0.85
  medConfidenceWinRate: number  // >= 0.65
  lowConfidenceWinRate: number   // < 0.65

  // Time to outcome
  avgBarsToEntry: number
  avgBarsToWin: number
  avgBarsToLoss: number

  // Detailed results
  trades: FvgTradeResult[]
}

export interface BacktestOptions {
  minGapPct?: number
  maxGapPct?: number
  lookAheadBars?: number // How many bars to look ahead for entry/TP/SL (default: 50)
  entryTolerance?: number // How close price must get to entry (as % of gap size, default: 0.05 = 5%)
  marketHoursOnly?: boolean // Only backtest during market hours 9:30am-4pm (default: true)
  entryStrategy?: 'continuation' | 'immediate' // Entry strategy (default: 'continuation')
}

/**
 * Backtest FVG patterns on historical data
 */
export function backtestFvgPatterns(
  data: CandleData[],
  options: BacktestOptions = {}
): BacktestResults {
  const {
    minGapPct = 0.25,
    maxGapPct = 5.0,
    lookAheadBars = 50,
    entryTolerance = 0.05,
    marketHoursOnly = true,
    entryStrategy = 'continuation'
  } = options

  // Filter to market hours only (9:30am-4pm) if requested
  const backtestData = marketHoursOnly ? filterMarketHoursData(data) : data

  console.log(`[FVG Backtest] Original data: ${data.length} bars, After market hours filter: ${backtestData.length} bars, Market hours only: ${marketHoursOnly}`)

  // Detect all FVG patterns on filtered data
  const patterns = detectFvgPatterns(backtestData, { minGapPct, maxGapPct, recentOnly: false })

  console.log(`[FVG Backtest] Detected ${patterns.length} patterns with minGap: ${minGapPct}%, maxGap: ${maxGapPct}%`)

  const trades: FvgTradeResult[] = []

  // Analyze each pattern
  patterns.forEach(pattern => {
    const detectionIndex = pattern.startIndex + 2 // Pattern detected at 3rd candle

    // Don't backtest patterns too close to the end (need lookAheadBars for analysis)
    if (detectionIndex + lookAheadBars >= backtestData.length) {
      return
    }

    const trade: FvgTradeResult = {
      pattern,
      detectionIndex,
      detectionTime: backtestData[detectionIndex].time,
      entryHit: false,
      tp1Hit: false,
      tp2Hit: false,
      tp3Hit: false,
      stopLossHit: false,
      outcome: 'no_entry'
    }

    const gapSize = Math.abs(pattern.gapHigh - pattern.gapLow)
    const entryBuffer = gapSize * entryTolerance

    // IMMEDIATE ENTRY: Enter right after FVG detection (next candle)
    if (entryStrategy === 'immediate') {
      trade.entryHit = true
      trade.entryTime = backtestData[detectionIndex + 1]?.time
      trade.entryIndex = detectionIndex + 1
      trade.barsToEntry = 1
    }

    // Look ahead to see if entry, TPs, or SL were hit
    const endIndex = Math.min(detectionIndex + lookAheadBars, backtestData.length)

    for (let i = detectionIndex + 1; i < endIndex; i++) {
      const candle = backtestData[i]

      // CONTINUATION STRATEGY: Check for entry hit - price must TOUCH the gap zone entry level
      if (entryStrategy === 'continuation' && !trade.entryHit) {
        if (pattern.type === 'bullish') {
          // Bullish FVG CONTINUATION: LONG entry at gapLow (bottom of gap) when price retraces down
          // Entry triggered when candle touches gapLow (entry level)
          const entryTouched = candle.low <= pattern.entryPrice && candle.high >= pattern.entryPrice
          if (entryTouched) {
            trade.entryHit = true
            trade.entryTime = candle.time
            trade.entryIndex = i
            trade.barsToEntry = i - detectionIndex
          }
        } else {
          // Bearish FVG CONTINUATION: SHORT entry at gapHigh (top of gap) when price retraces up
          // Entry triggered when candle touches gapHigh (entry level)
          const entryTouched = candle.low <= pattern.entryPrice && candle.high >= pattern.entryPrice
          if (entryTouched) {
            trade.entryHit = true
            trade.entryTime = candle.time
            trade.entryIndex = i
            trade.barsToEntry = i - detectionIndex
          }
        }
        continue // Don't check TPs/SL until entry is triggered
      }

      // After entry is triggered, check for TP/SL hits
      // CRITICAL: For immediate entry, skip the entry candle itself to avoid forward-looking bias
      // Only start checking TPs/SLs from the NEXT candle after entry
      if (trade.entryHit && i === trade.entryIndex && entryStrategy === 'immediate') {
        continue // Skip checking TPs/SLs on the entry candle for immediate strategy
      }

      // Use conservative in-bar sequencing: for LONG trades assume Low before High (SL first)
      // For SHORT trades assume High before Low (SL first)
      if (pattern.type === 'bullish') {
        // Bullish FVG CONTINUATION: LONG trade (entry at gapLow, expecting UP continuation above gap)
        // Conservative in-bar order: LOW before HIGH
        // So check SL (below) before TP (above)

        // Check stop loss FIRST (price goes DOWN beyond SL - invalidates continuation)
        if (!trade.stopLossHit && candle.low <= pattern.stopLoss) {
          trade.stopLossHit = true
          trade.stopLossTime = candle.time
          trade.stopLossIndex = i
          trade.outcome = 'loss'
          // LONG trade: bought at entryPrice, sold at stopLoss (which is lower)
          trade.profitPct = ((pattern.stopLoss - pattern.entryPrice) / pattern.entryPrice) * 100
          trade.barsToOutcome = i - (trade.entryIndex || detectionIndex)
          break // Trade closed - SL hit
        }

        // Check TP3 (highest continuation target - 2:1 R:R above gap)
        if (!trade.tp3Hit && candle.high >= pattern.tp3) {
          trade.tp3Hit = true
          trade.tp3Time = candle.time
          trade.tp3Index = i
          trade.tp2Hit = true // TP3 hit means TP2 and TP1 also hit
          trade.tp1Hit = true
          trade.outcome = 'win'
          trade.profitPct = ((pattern.tp3 - pattern.entryPrice) / pattern.entryPrice) * 100
          trade.barsToOutcome = i - (trade.entryIndex || detectionIndex)
          break // Trade closed at TP3
        }

        // Check TP2 (1:1 R:R)
        if (!trade.tp2Hit && candle.high >= pattern.tp2) {
          trade.tp2Hit = true
          trade.tp2Time = candle.time
          trade.tp2Index = i
          trade.tp1Hit = true // TP2 hit means TP1 also hit
        }

        // Check TP1 (0.5:1 R:R)
        if (!trade.tp1Hit && candle.high >= pattern.tp1) {
          trade.tp1Hit = true
          trade.tp1Time = candle.time
          trade.tp1Index = i
        }

      } else {
        // Bearish FVG CONTINUATION: SHORT trade (entry at gapHigh, expecting DOWN continuation below gap)
        // Conservative in-bar order: HIGH before LOW
        // So check SL (above) before TP (below)

        // Check stop loss FIRST (price goes UP beyond SL - invalidates continuation)
        if (!trade.stopLossHit && candle.high >= pattern.stopLoss) {
          trade.stopLossHit = true
          trade.stopLossTime = candle.time
          trade.stopLossIndex = i
          trade.outcome = 'loss'
          // SHORT trade: sold at entryPrice, bought back at stopLoss (which is higher)
          trade.profitPct = ((pattern.entryPrice - pattern.stopLoss) / pattern.entryPrice) * 100
          trade.barsToOutcome = i - (trade.entryIndex || detectionIndex)
          break // Trade closed - SL hit
        }

        // Check TP3 (lowest continuation target - 2:1 R:R below gap)
        if (!trade.tp3Hit && candle.low <= pattern.tp3) {
          trade.tp3Hit = true
          trade.tp3Time = candle.time
          trade.tp3Index = i
          trade.tp2Hit = true // TP3 hit means TP2 and TP1 also hit
          trade.tp1Hit = true
          trade.outcome = 'win'
          trade.profitPct = ((pattern.entryPrice - pattern.tp3) / pattern.entryPrice) * 100
          trade.barsToOutcome = i - (trade.entryIndex || detectionIndex)
          break // Trade closed at TP3
        }

        // Check TP2 (1:1 R:R)
        if (!trade.tp2Hit && candle.low <= pattern.tp2) {
          trade.tp2Hit = true
          trade.tp2Time = candle.time
          trade.tp2Index = i
          trade.tp1Hit = true // TP2 hit means TP1 also hit
        }

        // Check TP1 (0.5:1 R:R)
        if (!trade.tp1Hit && candle.low <= pattern.tp1) {
          trade.tp1Hit = true
          trade.tp1Time = candle.time
          trade.tp1Index = i
        }
      }
    }

    // Determine final outcome based on what happened
    if (!trade.entryHit) {
      // Entry was never triggered within lookAheadBars
      trade.outcome = 'expired'
    } else if (trade.outcome === 'no_entry') {
      // Entry was hit but no TP or SL was reached
      if (trade.tp1Hit) {
        trade.outcome = 'win' // At least TP1 hit
        const exitPrice = trade.tp3Hit ? pattern.tp3 : (trade.tp2Hit ? pattern.tp2 : pattern.tp1)
        if (pattern.type === 'bullish') {
          // LONG trade: bought at entryPrice, sold at exitPrice (which is higher)
          trade.profitPct = ((exitPrice - pattern.entryPrice) / pattern.entryPrice) * 100
        } else {
          // SHORT trade: sold at entryPrice, bought back at exitPrice (which is lower)
          trade.profitPct = ((pattern.entryPrice - exitPrice) / pattern.entryPrice) * 100
        }
      } else {
        trade.outcome = 'incomplete'
      }
    }

    trades.push(trade)
  })

  // Calculate statistics
  const tradesWithEntry = trades.filter(t => t.entryHit)
  const completedTrades = trades.filter(t => t.outcome === 'win' || t.outcome === 'loss')
  const wins = trades.filter(t => t.outcome === 'win')
  const losses = trades.filter(t => t.outcome === 'loss')

  const bullishTrades = completedTrades.filter(t => t.pattern.type === 'bullish')
  const bearishTrades = completedTrades.filter(t => t.pattern.type === 'bearish')
  const bullishWins = bullishTrades.filter(t => t.outcome === 'win')
  const bearishWins = bearishTrades.filter(t => t.outcome === 'win')

  const highConfTrades = completedTrades.filter(t => t.pattern.validationScore >= 0.85)
  const medConfTrades = completedTrades.filter(t => t.pattern.validationScore >= 0.65 && t.pattern.validationScore < 0.85)
  const lowConfTrades = completedTrades.filter(t => t.pattern.validationScore < 0.65)

  const highConfWins = highConfTrades.filter(t => t.outcome === 'win')
  const medConfWins = medConfTrades.filter(t => t.outcome === 'win')
  const lowConfWins = lowConfTrades.filter(t => t.outcome === 'win')

  const expiredTrades = trades.filter(t => t.outcome === 'expired')

  return {
    totalPatterns: patterns.length,
    patternsWithEntry: tradesWithEntry.length,

    totalTrades: completedTrades.length,
    wins: wins.length,
    losses: losses.length,
    incomplete: trades.filter(t => t.outcome === 'incomplete').length + expiredTrades.length, // Include expired in incomplete
    winRate: completedTrades.length > 0 ? (wins.length / completedTrades.length) * 100 : 0,

    tp1HitRate: tradesWithEntry.length > 0 ? (trades.filter(t => t.tp1Hit).length / tradesWithEntry.length) * 100 : 0,
    tp2HitRate: tradesWithEntry.length > 0 ? (trades.filter(t => t.tp2Hit).length / tradesWithEntry.length) * 100 : 0,
    tp3HitRate: tradesWithEntry.length > 0 ? (trades.filter(t => t.tp3Hit).length / tradesWithEntry.length) * 100 : 0,

    bullishWinRate: bullishTrades.length > 0 ? (bullishWins.length / bullishTrades.length) * 100 : 0,
    bearishWinRate: bearishTrades.length > 0 ? (bearishWins.length / bearishTrades.length) * 100 : 0,

    highConfidenceWinRate: highConfTrades.length > 0 ? (highConfWins.length / highConfTrades.length) * 100 : 0,
    medConfidenceWinRate: medConfTrades.length > 0 ? (medConfWins.length / medConfTrades.length) * 100 : 0,
    lowConfidenceWinRate: lowConfTrades.length > 0 ? (lowConfWins.length / lowConfTrades.length) * 100 : 0,

    avgBarsToEntry: tradesWithEntry.length > 0
      ? tradesWithEntry.reduce((sum, t) => sum + (t.barsToEntry || 0), 0) / tradesWithEntry.length
      : 0,
    avgBarsToWin: wins.length > 0
      ? wins.reduce((sum, t) => sum + (t.barsToOutcome || 0), 0) / wins.length
      : 0,
    avgBarsToLoss: losses.length > 0
      ? losses.reduce((sum, t) => sum + (t.barsToOutcome || 0), 0) / losses.length
      : 0,

    trades
  }
}

/**
 * Format backtest results for display
 */
export function formatBacktestResults(results: BacktestResults): string {
  return `
FVG Backtest Results
====================

Pattern Detection:
  Total Patterns: ${results.totalPatterns}
  Patterns with Entry: ${results.patternsWithEntry} (${((results.patternsWithEntry / results.totalPatterns) * 100).toFixed(1)}%)

Overall Performance:
  Total Trades: ${results.totalTrades}
  Wins: ${results.wins}
  Losses: ${results.losses}
  Incomplete: ${results.incomplete}
  Win Rate: ${results.winRate.toFixed(1)}%

Target Hit Rates:
  TP1 Hit Rate: ${results.tp1HitRate.toFixed(1)}%
  TP2 Hit Rate: ${results.tp2HitRate.toFixed(1)}%
  TP3 Hit Rate: ${results.tp3HitRate.toFixed(1)}%

Performance by Type:
  Bullish Win Rate: ${results.bullishWinRate.toFixed(1)}%
  Bearish Win Rate: ${results.bearishWinRate.toFixed(1)}%

Performance by Confidence:
  High Confidence (≥85%): ${results.highConfidenceWinRate.toFixed(1)}% win rate
  Med Confidence (65-84%): ${results.medConfidenceWinRate.toFixed(1)}% win rate
  Low Confidence (<65%): ${results.lowConfidenceWinRate.toFixed(1)}% win rate

Time to Outcome:
  Avg Bars to Entry: ${results.avgBarsToEntry.toFixed(1)}
  Avg Bars to Win: ${results.avgBarsToWin.toFixed(1)}
  Avg Bars to Loss: ${results.avgBarsToLoss.toFixed(1)}
  `.trim()
}
