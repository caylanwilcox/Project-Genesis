/**
 * FVG Detection Repository
 *
 * Data access layer for Fair Value Gap detections
 * Handles CRUD operations and queries for FVG patterns
 */

import { prisma } from '@/lib/prisma'
import type { FvgDetection, Prisma } from '@prisma/client'
import type { FvgPattern, TradingMode } from '@/services/fvgDetectionService'

export interface FvgDetectionCreateInput {
  ticker: string
  timeframe: string
  detectedAt: Date
  fvgType: 'bullish' | 'bearish'
  tradingMode: string

  // 3-Candle pattern
  candle1Timestamp: Date
  candle1High: number
  candle1Low: number
  candle2Timestamp: Date
  candle2High: number
  candle2Low: number
  candle3Timestamp: Date
  candle3High: number
  candle3Low: number

  // Gap metrics
  gapHigh: number
  gapLow: number
  gapSize: number
  gapSizePct: number

  // Entry/Exit levels
  entryPrice: number
  stopLoss: number
  takeProfit1: number
  takeProfit2: number
  takeProfit3: number

  // Validation
  volumeProfile?: string
  marketStructure?: string
  validationScore: number
}

export interface FvgDetectionFilters {
  ticker?: string
  timeframe?: string
  tradingMode?: TradingMode
  fvgType?: 'bullish' | 'bearish'
  startDate?: Date
  endDate?: Date
  minValidationScore?: number
  finalOutcome?: string
}

export interface FvgWinRateStats {
  tradingMode: string
  fvgType: string
  totalDetections: number
  filledCount: number
  tp1Count: number
  tp2Count: number
  tp3Count: number
  stopLossCount: number
  tp1WinRate: number
  tp2WinRate: number
  tp3WinRate: number
  avgHoldTimeMins: number
}

export class FvgDetectionRepository {
  /**
   * Create a single FVG detection
   */
  async create(data: FvgDetectionCreateInput): Promise<FvgDetection> {
    return await prisma.fvgDetection.create({
      data: {
        ticker: data.ticker,
        timeframe: data.timeframe,
        detectedAt: data.detectedAt,
        fvgType: data.fvgType,
        tradingMode: data.tradingMode,
        candle1Timestamp: data.candle1Timestamp,
        candle1High: data.candle1High,
        candle1Low: data.candle1Low,
        candle2Timestamp: data.candle2Timestamp,
        candle2High: data.candle2High,
        candle2Low: data.candle2Low,
        candle3Timestamp: data.candle3Timestamp,
        candle3High: data.candle3High,
        candle3Low: data.candle3Low,
        gapHigh: data.gapHigh,
        gapLow: data.gapLow,
        gapSize: data.gapSize,
        gapSizePct: data.gapSizePct,
        entryPrice: data.entryPrice,
        stopLoss: data.stopLoss,
        takeProfit1: data.takeProfit1,
        takeProfit2: data.takeProfit2,
        takeProfit3: data.takeProfit3,
        volumeProfile: data.volumeProfile,
        marketStructure: data.marketStructure,
        validationScore: data.validationScore,
      },
    })
  }

  /**
   * Create multiple FVG detections in bulk
   */
  async createMany(patterns: FvgPattern[], ticker: string, timeframe: string): Promise<number> {
    const operations = patterns.map(pattern =>
      this.create({
        ticker,
        timeframe,
        detectedAt: pattern.detectedAt,
        fvgType: pattern.fvgType,
        tradingMode: pattern.tradingMode,
        candle1Timestamp: pattern.candle1.timestamp,
        candle1High: pattern.candle1.high,
        candle1Low: pattern.candle1.low,
        candle2Timestamp: pattern.candle2.timestamp,
        candle2High: pattern.candle2.high,
        candle2Low: pattern.candle2.low,
        candle3Timestamp: pattern.candle3.timestamp,
        candle3High: pattern.candle3.high,
        candle3Low: pattern.candle3.low,
        gapHigh: pattern.gapHigh,
        gapLow: pattern.gapLow,
        gapSize: pattern.gapSize,
        gapSizePct: pattern.gapSizePct,
        entryPrice: pattern.entryPrice,
        stopLoss: pattern.stopLoss,
        takeProfit1: pattern.takeProfit1,
        takeProfit2: pattern.takeProfit2,
        takeProfit3: pattern.takeProfit3,
        volumeProfile: pattern.volumeProfile,
        marketStructure: pattern.marketStructure,
        validationScore: pattern.validationScore,
      })
    )

    const results = await Promise.allSettled(operations)
    const successful = results.filter(r => r.status === 'fulfilled').length
    return successful
  }

  /**
   * Find FVG detections with filters
   */
  async findMany(filters: FvgDetectionFilters, limit: number = 100): Promise<FvgDetection[]> {
    const where: Prisma.FvgDetectionWhereInput = {}

    if (filters.ticker) where.ticker = filters.ticker
    if (filters.timeframe) where.timeframe = filters.timeframe
    if (filters.tradingMode) where.tradingMode = filters.tradingMode
    if (filters.fvgType) where.fvgType = filters.fvgType
    if (filters.finalOutcome) where.finalOutcome = filters.finalOutcome

    if (filters.startDate || filters.endDate) {
      where.detectedAt = {}
      if (filters.startDate) where.detectedAt.gte = filters.startDate
      if (filters.endDate) where.detectedAt.lte = filters.endDate
    }

    if (filters.minValidationScore !== undefined) {
      where.validationScore = { gte: filters.minValidationScore }
    }

    return await prisma.fvgDetection.findMany({
      where,
      orderBy: { detectedAt: 'desc' },
      take: limit,
    })
  }

  /**
   * Get latest unfilled FVG detections (for real-time trading)
   */
  async getUnfilledDetections(
    ticker: string,
    tradingMode: TradingMode,
    limit: number = 10
  ): Promise<FvgDetection[]> {
    return await prisma.fvgDetection.findMany({
      where: {
        ticker,
        tradingMode,
        filled: false,
      },
      orderBy: { detectedAt: 'desc' },
      take: limit,
    })
  }

  /**
   * Update FVG detection outcome (after monitoring if targets hit)
   */
  async updateOutcome(
    id: string,
    outcome: {
      filled?: boolean
      filledAt?: Date
      hitTp1?: boolean
      hitTp1At?: Date
      hitTp2?: boolean
      hitTp2At?: Date
      hitTp3?: boolean
      hitTp3At?: Date
      hitStopLoss?: boolean
      hitStopLossAt?: Date
      holdTimeMins?: number
      finalOutcome?: 'tp1' | 'tp2' | 'tp3' | 'stop_loss' | 'pending'
    }
  ): Promise<FvgDetection> {
    return await prisma.fvgDetection.update({
      where: { id },
      data: outcome,
    })
  }

  /**
   * Calculate win rate statistics per trading mode
   * Critical for ML training - this tells us which FVG setups are profitable
   */
  async getWinRateStats(
    ticker: string,
    tradingMode: TradingMode,
    fvgType?: 'bullish' | 'bearish'
  ): Promise<FvgWinRateStats | null> {
    const where: Prisma.FvgDetectionWhereInput = {
      ticker,
      tradingMode,
      filled: true,  // Only completed FVGs
    }

    if (fvgType) {
      where.fvgType = fvgType
    }

    const detections = await prisma.fvgDetection.findMany({ where })

    if (detections.length === 0) return null

    const stats = {
      tradingMode,
      fvgType: fvgType || 'all',
      totalDetections: detections.length,
      filledCount: detections.filter(d => d.filled).length,
      tp1Count: detections.filter(d => d.hitTp1).length,
      tp2Count: detections.filter(d => d.hitTp2).length,
      tp3Count: detections.filter(d => d.hitTp3).length,
      stopLossCount: detections.filter(d => d.hitStopLoss).length,
      tp1WinRate: 0,
      tp2WinRate: 0,
      tp3WinRate: 0,
      avgHoldTimeMins: 0,
    }

    stats.tp1WinRate = (stats.tp1Count / stats.filledCount) * 100
    stats.tp2WinRate = (stats.tp2Count / stats.filledCount) * 100
    stats.tp3WinRate = (stats.tp3Count / stats.filledCount) * 100

    const holdTimes = detections.filter(d => d.holdTimeMins !== null).map(d => d.holdTimeMins!)
    stats.avgHoldTimeMins = holdTimes.length > 0
      ? holdTimes.reduce((sum, t) => sum + t, 0) / holdTimes.length
      : 0

    return stats
  }

  /**
   * Get FVG detection summary by trading mode
   */
  async getSummaryByMode(ticker: string, timeframe: string): Promise<Record<string, number>> {
    const detections = await prisma.fvgDetection.findMany({
      where: { ticker, timeframe },
      select: { tradingMode: true },
    })

    const summary: Record<string, number> = {}
    detections.forEach(d => {
      summary[d.tradingMode] = (summary[d.tradingMode] || 0) + 1
    })

    return summary
  }

  /**
   * Get all FVG detections for ML training dataset
   * Returns only filled FVGs with outcomes
   */
  async getTrainingDataset(
    ticker: string,
    tradingMode: TradingMode,
    startDate?: Date,
    endDate?: Date
  ): Promise<FvgDetection[]> {
    const where: Prisma.FvgDetectionWhereInput = {
      ticker,
      tradingMode,
      filled: true,
      finalOutcome: { not: 'pending' },
    }

    if (startDate || endDate) {
      where.detectedAt = {}
      if (startDate) where.detectedAt.gte = startDate
      if (endDate) where.detectedAt.lte = endDate
    }

    return await prisma.fvgDetection.findMany({
      where,
      orderBy: { detectedAt: 'asc' },
    })
  }

  /**
   * Delete old FVG detections (cleanup)
   */
  async deleteOlderThan(date: Date): Promise<number> {
    const result = await prisma.fvgDetection.deleteMany({
      where: {
        detectedAt: { lt: date },
      },
    })

    return result.count
  }

  /**
   * Get total count
   */
  async count(filters?: FvgDetectionFilters): Promise<number> {
    const where: Prisma.FvgDetectionWhereInput = {}

    if (filters?.ticker) where.ticker = filters.ticker
    if (filters?.timeframe) where.timeframe = filters.timeframe
    if (filters?.tradingMode) where.tradingMode = filters.tradingMode
    if (filters?.fvgType) where.fvgType = filters.fvgType

    return await prisma.fvgDetection.count({ where })
  }
}

// Export singleton instance
export const fvgDetectionRepo = new FvgDetectionRepository()
