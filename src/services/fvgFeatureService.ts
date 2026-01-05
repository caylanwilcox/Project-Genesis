/**
 * FVG Feature Service
 *
 * Enriches FVG detections with technical indicator features for ML training.
 * Combines FVG pattern data with market context at time of detection.
 */

import { marketDataRepo, fvgDetectionRepo } from '@/repositories'
import { technicalIndicatorsService, AllIndicators } from './technicalIndicatorsService'
import type { FvgDetection } from '@prisma/client'

export interface FvgMLFeatures {
  // FVG Pattern Features
  fvg_id: string
  ticker: string
  timeframe: string
  fvg_type: 'bullish' | 'bearish'
  trading_mode: string
  detected_at: string

  // Gap Characteristics
  gap_size: number
  gap_size_pct: number
  validation_score: number
  volume_profile: string | null
  market_structure: string | null

  // Technical Indicators at Detection Time
  rsi_14: number | null
  macd: number | null
  macd_signal: number | null
  macd_histogram: number | null
  atr_14: number | null
  sma_20: number | null
  sma_50: number | null
  ema_12: number | null
  ema_26: number | null
  bb_upper: number | null
  bb_middle: number | null
  bb_lower: number | null
  bb_bandwidth: number | null
  volume_ratio: number | null

  // Derived Features
  price_vs_sma20: number | null      // Price position relative to SMA20
  price_vs_sma50: number | null      // Price position relative to SMA50
  rsi_zone: string | null            // oversold, neutral, overbought
  macd_trend: string | null          // bullish, bearish, neutral
  volatility_regime: string | null    // low, medium, high (based on ATR)

  // Time Features
  hour_of_day: number
  day_of_week: number
  is_market_open: boolean

  // Outcome Labels (for training)
  filled: boolean
  hit_tp1: boolean
  hit_tp2: boolean
  hit_tp3: boolean
  hit_stop_loss: boolean
  final_outcome: string | null
  hold_time_mins: number | null
}

export class FvgFeatureService {
  /**
   * Get ML features for a single FVG detection
   */
  async getFeaturesForFvg(fvg: FvgDetection): Promise<FvgMLFeatures | null> {
    const ticker = fvg.ticker
    const timeframe = fvg.timeframe
    const detectedAt = fvg.detectedAt

    // Get market data around detection time (need ~60 bars before for indicators)
    const startDate = new Date(detectedAt)
    startDate.setDate(startDate.getDate() - 30) // 30 days before for indicator warmup

    const data = await marketDataRepo.getOHLCV(
      ticker,
      timeframe,
      startDate,
      detectedAt,
      200
    )

    if (data.length < 50) {
      console.log(`[FVG Features] Insufficient data for ${fvg.id}`)
      return null
    }

    // Calculate indicators
    const indicators = technicalIndicatorsService.calculateAllIndicators(data)

    // Find the indicator values at detection time
    const detectionTime = detectedAt.getTime()
    let closestIndicator: AllIndicators | null = null
    let minTimeDiff = Infinity

    for (const ind of indicators) {
      const timeDiff = Math.abs(ind.time - detectionTime)
      if (timeDiff < minTimeDiff) {
        minTimeDiff = timeDiff
        closestIndicator = ind
      }
    }

    if (!closestIndicator) {
      console.log(`[FVG Features] No indicators found for ${fvg.id}`)
      return null
    }

    // Get the price at detection time
    const detectionBar = data.find(d => Math.abs(d.time - detectionTime) < 3600000) // Within 1 hour
    const price = detectionBar?.close || Number(fvg.entryPrice)

    // Calculate derived features
    const priceVsSma20 = closestIndicator.sma_20
      ? ((price - closestIndicator.sma_20) / closestIndicator.sma_20) * 100
      : null

    const priceVsSma50 = closestIndicator.sma_50
      ? ((price - closestIndicator.sma_50) / closestIndicator.sma_50) * 100
      : null

    const rsiZone = closestIndicator.rsi_14
      ? closestIndicator.rsi_14 < 30 ? 'oversold'
        : closestIndicator.rsi_14 > 70 ? 'overbought'
          : 'neutral'
      : null

    const macdTrend = closestIndicator.macd_histogram
      ? closestIndicator.macd_histogram > 0.5 ? 'bullish'
        : closestIndicator.macd_histogram < -0.5 ? 'bearish'
          : 'neutral'
      : null

    // Volatility regime based on ATR as % of price
    const atrPct = closestIndicator.atr_14 ? (closestIndicator.atr_14 / price) * 100 : null
    const volatilityRegime = atrPct
      ? atrPct < 0.5 ? 'low'
        : atrPct > 1.5 ? 'high'
          : 'medium'
      : null

    // Time features
    const hour = detectedAt.getUTCHours()
    const dayOfWeek = detectedAt.getUTCDay()
    const isMarketOpen = hour >= 13 && hour <= 21 && dayOfWeek >= 1 && dayOfWeek <= 5

    return {
      // FVG Pattern Features
      fvg_id: fvg.id,
      ticker: fvg.ticker,
      timeframe: fvg.timeframe,
      fvg_type: fvg.fvgType as 'bullish' | 'bearish',
      trading_mode: fvg.tradingMode,
      detected_at: fvg.detectedAt.toISOString(),

      // Gap Characteristics
      gap_size: Number(fvg.gapSize),
      gap_size_pct: Number(fvg.gapSizePct),
      validation_score: Number(fvg.validationScore),
      volume_profile: fvg.volumeProfile,
      market_structure: fvg.marketStructure,

      // Technical Indicators
      rsi_14: closestIndicator.rsi_14 ?? null,
      macd: closestIndicator.macd ?? null,
      macd_signal: closestIndicator.macd_signal ?? null,
      macd_histogram: closestIndicator.macd_histogram ?? null,
      atr_14: closestIndicator.atr_14 ?? null,
      sma_20: closestIndicator.sma_20 ?? null,
      sma_50: closestIndicator.sma_50 ?? null,
      ema_12: closestIndicator.ema_12 ?? null,
      ema_26: closestIndicator.ema_26 ?? null,
      bb_upper: closestIndicator.bb_upper ?? null,
      bb_middle: closestIndicator.bb_middle ?? null,
      bb_lower: closestIndicator.bb_lower ?? null,
      bb_bandwidth: closestIndicator.bb_bandwidth ?? null,
      volume_ratio: closestIndicator.volume_ratio ?? null,

      // Derived Features
      price_vs_sma20: priceVsSma20 ? Math.round(priceVsSma20 * 100) / 100 : null,
      price_vs_sma50: priceVsSma50 ? Math.round(priceVsSma50 * 100) / 100 : null,
      rsi_zone: rsiZone,
      macd_trend: macdTrend,
      volatility_regime: volatilityRegime,

      // Time Features
      hour_of_day: hour,
      day_of_week: dayOfWeek,
      is_market_open: isMarketOpen,

      // Outcome Labels
      filled: fvg.filled,
      hit_tp1: fvg.hitTp1,
      hit_tp2: fvg.hitTp2,
      hit_tp3: fvg.hitTp3,
      hit_stop_loss: fvg.hitStopLoss,
      final_outcome: fvg.finalOutcome,
      hold_time_mins: fvg.holdTimeMins,
    }
  }

  /**
   * Generate ML training dataset for all FVGs
   */
  async generateMLDataset(
    ticker: string,
    timeframe: string
  ): Promise<{ features: FvgMLFeatures[]; summary: any }> {
    // Get all labeled FVGs
    const fvgs = await fvgDetectionRepo.findMany(
      { ticker, timeframe },
      10000
    )

    // Filter to only labeled FVGs (have outcome)
    const labeledFvgs = fvgs.filter(
      f => f.finalOutcome && f.finalOutcome !== 'pending'
    )

    console.log(`[FVG Features] Processing ${labeledFvgs.length} labeled FVGs for ${ticker} ${timeframe}`)

    const features: FvgMLFeatures[] = []

    for (let i = 0; i < labeledFvgs.length; i++) {
      const fvg = labeledFvgs[i]

      try {
        const fvgFeatures = await this.getFeaturesForFvg(fvg)
        if (fvgFeatures) {
          features.push(fvgFeatures)
        }

        if ((i + 1) % 10 === 0) {
          console.log(`[FVG Features] Processed ${i + 1}/${labeledFvgs.length}`)
        }
      } catch (error) {
        console.error(`[FVG Features] Error processing ${fvg.id}:`, error)
      }
    }

    // Calculate summary statistics
    const summary = {
      totalFvgs: labeledFvgs.length,
      featuresGenerated: features.length,
      outcomeDistribution: {
        tp1: features.filter(f => f.final_outcome === 'tp1').length,
        tp2: features.filter(f => f.final_outcome === 'tp2').length,
        tp3: features.filter(f => f.final_outcome === 'tp3').length,
        stop_loss: features.filter(f => f.final_outcome === 'stop_loss').length,
        expired: features.filter(f => f.final_outcome === 'expired').length,
      },
      avgRsi: features.filter(f => f.rsi_14).reduce((sum, f) => sum + (f.rsi_14 || 0), 0) / features.length,
      avgGapSizePct: features.reduce((sum, f) => sum + f.gap_size_pct, 0) / features.length,
    }

    return { features, summary }
  }
}

export const fvgFeatureService = new FvgFeatureService()
