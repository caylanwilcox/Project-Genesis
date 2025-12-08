/**
 * FVG Outcome Labeling Service
 *
 * Labels historical FVG patterns with outcomes by analyzing
 * subsequent price action to determine:
 * - Did price fill the gap?
 * - Did it hit TP1, TP2, or TP3?
 * - Did it hit stop loss?
 * - How long did it take (hold time)?
 *
 * This is critical for ML training - we need labeled data
 * to train models that predict FVG success rates.
 */

import { marketDataRepo } from '@/repositories'
import { fvgDetectionRepo } from '@/repositories'
import type { FvgDetection } from '@prisma/client'

export interface LabelingResult {
  fvgId: string
  ticker: string
  fvgType: string
  filled: boolean
  filledAt?: Date
  hitTp1: boolean
  hitTp1At?: Date
  hitTp2: boolean
  hitTp2At?: Date
  hitTp3: boolean
  hitTp3At?: Date
  hitStopLoss: boolean
  hitStopLossAt?: Date
  holdTimeMins?: number
  finalOutcome: 'tp1' | 'tp2' | 'tp3' | 'stop_loss' | 'expired' | 'pending'
}

export interface LabelingSummary {
  totalProcessed: number
  tp1Count: number
  tp2Count: number
  tp3Count: number
  stopLossCount: number
  expiredCount: number
  pendingCount: number
  tp1WinRate: number
  tp2WinRate: number
  tp3WinRate: number
  avgHoldTimeMins: number
}

// Max bars to look ahead when checking outcomes
const MAX_BARS_LOOKUP: Record<string, number> = {
  '1m': 60 * 8,      // 8 hours of 1m bars
  '5m': 12 * 8,      // 8 hours of 5m bars
  '15m': 4 * 24,     // 24 hours of 15m bars
  '30m': 2 * 48,     // 48 hours of 30m bars
  '1h': 24 * 5,      // 5 days of 1h bars
  '4h': 6 * 10,      // 10 days of 4h bars
  '1d': 30,          // 30 days of daily bars
}

export class FvgOutcomeLabelingService {
  /**
   * Label a single FVG detection by analyzing subsequent price action
   */
  async labelFvg(fvg: FvgDetection): Promise<LabelingResult> {
    const ticker = fvg.ticker
    const timeframe = fvg.timeframe
    const entryPrice = Number(fvg.entryPrice)
    const stopLoss = Number(fvg.stopLoss)
    const tp1 = Number(fvg.takeProfit1)
    const tp2 = Number(fvg.takeProfit2)
    const tp3 = Number(fvg.takeProfit3)
    const isBullish = fvg.fvgType === 'bullish'

    // Get bars after the FVG was detected
    const maxBars = MAX_BARS_LOOKUP[timeframe] || 120
    const startDate = new Date(fvg.detectedAt)
    const endDate = new Date(startDate)
    endDate.setDate(endDate.getDate() + 30) // Max 30 days lookforward

    const bars = await marketDataRepo.getOHLCV(
      ticker,
      timeframe,
      startDate,
      endDate,
      maxBars
    )

    // Initialize result
    const result: LabelingResult = {
      fvgId: fvg.id,
      ticker,
      fvgType: fvg.fvgType,
      filled: false,
      hitTp1: false,
      hitTp2: false,
      hitTp3: false,
      hitStopLoss: false,
      finalOutcome: 'pending',
    }

    if (bars.length === 0) {
      result.finalOutcome = 'expired'
      return result
    }

    // Analyze each bar to see what happened
    for (let i = 0; i < bars.length; i++) {
      const bar = bars[i]
      const barTime = new Date(bar.time)
      const high = bar.high
      const low = bar.low

      // Skip the detection bar itself
      if (barTime.getTime() <= fvg.detectedAt.getTime()) {
        continue
      }

      // For bullish FVG: price needs to go DOWN to entry, then UP to TPs
      // For bearish FVG: price needs to go UP to entry, then DOWN to TPs
      if (isBullish) {
        // Check if entry was filled (price came down to entry zone)
        if (!result.filled && low <= entryPrice) {
          result.filled = true
          result.filledAt = barTime
        }

        // After entry, check if stop loss hit (price went below SL)
        if (result.filled && !result.hitStopLoss && low <= stopLoss) {
          result.hitStopLoss = true
          result.hitStopLossAt = barTime
          result.holdTimeMins = Math.round((barTime.getTime() - result.filledAt!.getTime()) / 60000)
          result.finalOutcome = 'stop_loss'
          break // Stop loss hit, exit
        }

        // Check take profits (price went up)
        if (result.filled) {
          if (!result.hitTp1 && high >= tp1) {
            result.hitTp1 = true
            result.hitTp1At = barTime
            if (!result.finalOutcome || result.finalOutcome === 'pending') {
              result.finalOutcome = 'tp1'
              result.holdTimeMins = Math.round((barTime.getTime() - result.filledAt!.getTime()) / 60000)
            }
          }
          if (!result.hitTp2 && high >= tp2) {
            result.hitTp2 = true
            result.hitTp2At = barTime
            result.finalOutcome = 'tp2'
            result.holdTimeMins = Math.round((barTime.getTime() - result.filledAt!.getTime()) / 60000)
          }
          if (!result.hitTp3 && high >= tp3) {
            result.hitTp3 = true
            result.hitTp3At = barTime
            result.finalOutcome = 'tp3'
            result.holdTimeMins = Math.round((barTime.getTime() - result.filledAt!.getTime()) / 60000)
            break // Full target hit, exit
          }
        }
      } else {
        // Bearish FVG: price needs to go UP to entry, then DOWN to TPs
        if (!result.filled && high >= entryPrice) {
          result.filled = true
          result.filledAt = barTime
        }

        // After entry, check if stop loss hit (price went above SL)
        if (result.filled && !result.hitStopLoss && high >= stopLoss) {
          result.hitStopLoss = true
          result.hitStopLossAt = barTime
          result.holdTimeMins = Math.round((barTime.getTime() - result.filledAt!.getTime()) / 60000)
          result.finalOutcome = 'stop_loss'
          break
        }

        // Check take profits (price went down)
        if (result.filled) {
          if (!result.hitTp1 && low <= tp1) {
            result.hitTp1 = true
            result.hitTp1At = barTime
            if (!result.finalOutcome || result.finalOutcome === 'pending') {
              result.finalOutcome = 'tp1'
              result.holdTimeMins = Math.round((barTime.getTime() - result.filledAt!.getTime()) / 60000)
            }
          }
          if (!result.hitTp2 && low <= tp2) {
            result.hitTp2 = true
            result.hitTp2At = barTime
            result.finalOutcome = 'tp2'
            result.holdTimeMins = Math.round((barTime.getTime() - result.filledAt!.getTime()) / 60000)
          }
          if (!result.hitTp3 && low <= tp3) {
            result.hitTp3 = true
            result.hitTp3At = barTime
            result.finalOutcome = 'tp3'
            result.holdTimeMins = Math.round((barTime.getTime() - result.filledAt!.getTime()) / 60000)
            break
          }
        }
      }
    }

    // If filled but no outcome determined, mark as expired
    if (result.filled && result.finalOutcome === 'pending') {
      result.finalOutcome = 'expired'
    }

    return result
  }

  /**
   * Label all unlabeled FVG detections for a ticker
   */
  async labelAllFvgs(
    ticker: string,
    timeframe: string
  ): Promise<LabelingSummary> {
    // Get all FVGs that haven't been labeled yet
    const fvgs = await fvgDetectionRepo.findMany(
      { ticker, timeframe },
      10000  // Get all
    )

    // Filter to only unlabeled ones
    const unlabeledFvgs = fvgs.filter(fvg => !fvg.finalOutcome || fvg.finalOutcome === 'pending')

    console.log(`[FVG Labeling] Found ${unlabeledFvgs.length} unlabeled FVGs for ${ticker} ${timeframe}`)

    const results: LabelingResult[] = []

    for (let i = 0; i < unlabeledFvgs.length; i++) {
      const fvg = unlabeledFvgs[i]

      try {
        const result = await this.labelFvg(fvg)
        results.push(result)

        // Update the database
        await fvgDetectionRepo.updateOutcome(fvg.id, {
          filled: result.filled,
          filledAt: result.filledAt,
          hitTp1: result.hitTp1,
          hitTp1At: result.hitTp1At,
          hitTp2: result.hitTp2,
          hitTp2At: result.hitTp2At,
          hitTp3: result.hitTp3,
          hitTp3At: result.hitTp3At,
          hitStopLoss: result.hitStopLoss,
          hitStopLossAt: result.hitStopLossAt,
          holdTimeMins: result.holdTimeMins,
          finalOutcome: result.finalOutcome,
        })

        if ((i + 1) % 10 === 0) {
          console.log(`[FVG Labeling] Processed ${i + 1}/${unlabeledFvgs.length}`)
        }
      } catch (error) {
        console.error(`[FVG Labeling] Error processing FVG ${fvg.id}:`, error)
      }
    }

    // Calculate summary stats
    const summary = this.calculateSummary(results)
    console.log(`[FVG Labeling] Complete for ${ticker} ${timeframe}:`, summary)

    return summary
  }

  /**
   * Calculate summary statistics from labeling results
   */
  private calculateSummary(results: LabelingResult[]): LabelingSummary {
    const filled = results.filter(r => r.filled)
    const tp1 = results.filter(r => r.hitTp1)
    const tp2 = results.filter(r => r.hitTp2)
    const tp3 = results.filter(r => r.hitTp3)
    const stopLoss = results.filter(r => r.hitStopLoss)
    const expired = results.filter(r => r.finalOutcome === 'expired')
    const pending = results.filter(r => r.finalOutcome === 'pending')

    const holdTimes = results
      .filter(r => r.holdTimeMins !== undefined)
      .map(r => r.holdTimeMins!)

    const avgHoldTime = holdTimes.length > 0
      ? holdTimes.reduce((sum, t) => sum + t, 0) / holdTimes.length
      : 0

    const filledCount = filled.length

    return {
      totalProcessed: results.length,
      tp1Count: tp1.length,
      tp2Count: tp2.length,
      tp3Count: tp3.length,
      stopLossCount: stopLoss.length,
      expiredCount: expired.length,
      pendingCount: pending.length,
      tp1WinRate: filledCount > 0 ? (tp1.length / filledCount) * 100 : 0,
      tp2WinRate: filledCount > 0 ? (tp2.length / filledCount) * 100 : 0,
      tp3WinRate: filledCount > 0 ? (tp3.length / filledCount) * 100 : 0,
      avgHoldTimeMins: Math.round(avgHoldTime),
    }
  }
}

export const fvgOutcomeLabelingService = new FvgOutcomeLabelingService()
