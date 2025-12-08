/**
 * FVG Detection API Endpoint
 * POST /api/v2/fvg/detect
 *
 * Scans historical market data for Fair Value Gap patterns
 * Stores detected patterns in database for ML training
 */

import { NextRequest, NextResponse } from 'next/server'
import { marketDataRepo, fvgDetectionRepo } from '@/repositories'
import { mlPredictionService } from '@/services/mlPredictionService'
import { technicalIndicatorsService } from '@/services/technicalIndicatorsService'

// Force dynamic rendering - don't pre-render at build time
export const dynamic = 'force-dynamic'
import { FvgDetectionService, TradingMode } from '@/services/fvgDetectionService'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    // Validate required parameters
    const { ticker, timeframe, tradingMode, daysBack } = body
    if (!ticker || !timeframe) {
      return NextResponse.json(
        { error: 'Missing required fields: ticker, timeframe' },
        { status: 400 }
      )
    }

    const mode: TradingMode = tradingMode || FvgDetectionService.getTradingModeForTimeframe(timeframe)
    const days = daysBack || 30

    // Calculate date range
    const endDate = new Date()
    const startDate = new Date()
    startDate.setDate(startDate.getDate() - days)

    // Fetch historical market data - use OHLCV for larger limits
    const marketData = await marketDataRepo.getOHLCV(
      ticker.toUpperCase(),
      timeframe,
      startDate,
      endDate,
      10000  // Allow up to 10k bars for historical scanning
    )

    if (marketData.length < 3) {
      return NextResponse.json(
        { error: 'Insufficient market data (need at least 3 bars)' },
        { status: 400 }
      )
    }

    // Convert to MarketBar format (getOHLCV returns time in ms, need Date)
    const bars = marketData.map(bar => ({
      timestamp: new Date(bar.time),
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
      volume: bar.volume,
    }))

    // Detect FVG patterns
    const fvgService = new FvgDetectionService({
      minGapSizePct: body.minGapSizePct || 0.25,
      maxGapSizePct: body.maxGapSizePct || 5.0,
      requireVolumeConfirmation: body.requireVolumeConfirmation ?? true,
      minValidationScore: body.minValidationScore || 0.6,
    })

    const patterns = fvgService.detectFvgs(bars, mode)

    if (patterns.length === 0) {
      return NextResponse.json({
        success: true,
        message: 'No FVG patterns detected',
        ticker: ticker.toUpperCase(),
        timeframe,
        tradingMode: mode,
        daysScanned: days,
        barsScanned: bars.length,
        patternsDetected: 0,
        patterns: [],
      })
    }

    // Store patterns in database
    const savedCount = await fvgDetectionRepo.createMany(
      patterns,
      ticker.toUpperCase(),
      timeframe
    )

    // Separate bullish and bearish
    const bullishPatterns = patterns.filter(p => p.fvgType === 'bullish')
    const bearishPatterns = patterns.filter(p => p.fvgType === 'bearish')

    // Calculate indicators for ML predictions if requested
    const includePredictions = body.includePredictions !== false
    let mlServerOnline = false
    let patternsWithPredictions = patterns.map(p => ({
      fvgType: p.fvgType,
      detectedAt: p.detectedAt,
      gapSize: p.gapSize,
      gapSizePct: p.gapSizePct,
      entryPrice: p.entryPrice,
      stopLoss: p.stopLoss,
      takeProfit1: p.takeProfit1,
      takeProfit2: p.takeProfit2,
      takeProfit3: p.takeProfit3,
      validationScore: p.validationScore,
      volumeProfile: p.volumeProfile,
      marketStructure: p.marketStructure,
      mlPrediction: null as null | { winProbability: number; confidence: number; recommendation: string },
    }))

    if (includePredictions) {
      mlServerOnline = await mlPredictionService.isAvailable()

      if (mlServerOnline) {
        // Calculate indicators
        const ohlcvData = marketData.map(bar => ({
          time: bar.time,
          open: bar.open,
          high: bar.high,
          low: bar.low,
          close: bar.close,
          volume: bar.volume,
        }))
        const indicators = technicalIndicatorsService.calculateAllIndicators(ohlcvData)

        // Get predictions for each pattern
        for (let i = 0; i < patterns.length; i++) {
          const pattern = patterns[i]
          const detectionTime = pattern.detectedAt.getTime()

          // Find closest indicator
          let closestIndicator = indicators[0]
          let minDiff = Infinity
          for (const ind of indicators) {
            const diff = Math.abs(ind.time - detectionTime)
            if (diff < minDiff) {
              minDiff = diff
              closestIndicator = ind
            }
          }

          // Build feature set
          const price = pattern.entryPrice
          const priceVsSma20 = closestIndicator.sma_20
            ? ((price - closestIndicator.sma_20) / closestIndicator.sma_20) * 100
            : 0
          const priceVsSma50 = closestIndicator.sma_50
            ? ((price - closestIndicator.sma_50) / closestIndicator.sma_50) * 100
            : 0
          const rsiZone = closestIndicator.rsi_14
            ? closestIndicator.rsi_14 < 30 ? 'oversold'
              : closestIndicator.rsi_14 > 70 ? 'overbought' : 'neutral'
            : 'neutral'
          const macdTrend = closestIndicator.macd_histogram
            ? closestIndicator.macd_histogram > 0.5 ? 'bullish'
              : closestIndicator.macd_histogram < -0.5 ? 'bearish' : 'neutral'
            : 'neutral'
          const atrPct = closestIndicator.atr_14 ? (closestIndicator.atr_14 / price) * 100 : 0
          const volatilityRegime = atrPct < 0.5 ? 'low' : atrPct > 1.5 ? 'high' : 'medium'

          const features = {
            fvg_id: `pattern_${i}`,
            gap_size_pct: pattern.gapSizePct,
            validation_score: pattern.validationScore,
            rsi_14: closestIndicator.rsi_14 ?? 50,
            macd: closestIndicator.macd ?? 0,
            macd_signal: closestIndicator.macd_signal ?? 0,
            macd_histogram: closestIndicator.macd_histogram ?? 0,
            atr_14: closestIndicator.atr_14 ?? 0,
            sma_20: closestIndicator.sma_20 ?? price,
            sma_50: closestIndicator.sma_50 ?? price,
            ema_12: closestIndicator.ema_12 ?? price,
            ema_26: closestIndicator.ema_26 ?? price,
            bb_bandwidth: closestIndicator.bb_bandwidth ?? 0.02,
            volume_ratio: closestIndicator.volume_ratio ?? 1,
            price_vs_sma20: priceVsSma20,
            price_vs_sma50: priceVsSma50,
            hour_of_day: pattern.detectedAt.getUTCHours(),
            day_of_week: pattern.detectedAt.getUTCDay(),
            fvg_type: pattern.fvgType,
            volume_profile: pattern.volumeProfile || 'medium',
            market_structure: pattern.marketStructure || 'neutral',
            rsi_zone: rsiZone,
            macd_trend: macdTrend,
            volatility_regime: volatilityRegime,
          }

          const prediction = await mlPredictionService.predict(features as any)
          if (prediction) {
            const rec = mlPredictionService.getRecommendation(prediction)
            patternsWithPredictions[i].mlPrediction = {
              winProbability: prediction.winProbability,
              confidence: prediction.confidence,
              recommendation: rec.reason,
            }
          }
        }
      }
    }

    return NextResponse.json({
      success: true,
      message: `Detected ${patterns.length} FVG patterns`,
      ticker: ticker.toUpperCase(),
      timeframe,
      tradingMode: mode,
      daysScanned: days,
      barsScanned: bars.length,
      patternsDetected: patterns.length,
      patternsSaved: savedCount,
      bullishCount: bullishPatterns.length,
      bearishCount: bearishPatterns.length,
      mlServerOnline,
      patterns: patternsWithPredictions,
    })
  } catch (error) {
    console.error('FVG detection error:', error)
    return NextResponse.json(
      {
        error: 'Failed to detect FVG patterns',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}

/**
 * GET /api/v2/fvg/detect?ticker=SPY&timeframe=1h&tradingMode=intraday
 * Retrieve detected FVG patterns
 */
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const ticker = searchParams.get('ticker')
    const timeframe = searchParams.get('timeframe')
    const tradingMode = searchParams.get('tradingMode') as TradingMode | null
    const fvgType = searchParams.get('fvgType') as 'bullish' | 'bearish' | null
    const limit = parseInt(searchParams.get('limit') || '50')

    // Build filters
    const filters: any = {}
    if (ticker) filters.ticker = ticker.toUpperCase()
    if (timeframe) filters.timeframe = timeframe
    if (tradingMode) filters.tradingMode = tradingMode
    if (fvgType) filters.fvgType = fvgType

    // Fetch detections
    const detections = await fvgDetectionRepo.findMany(filters, limit)

    // Convert Decimal to number for JSON serialization
    const serializedDetections = detections.map(d => ({
      id: d.id,
      ticker: d.ticker,
      timeframe: d.timeframe,
      detectedAt: d.detectedAt,
      fvgType: d.fvgType,
      tradingMode: d.tradingMode,
      gapSize: Number(d.gapSize),
      gapSizePct: Number(d.gapSizePct),
      entryPrice: Number(d.entryPrice),
      stopLoss: Number(d.stopLoss),
      takeProfit1: Number(d.takeProfit1),
      takeProfit2: Number(d.takeProfit2),
      takeProfit3: Number(d.takeProfit3),
      validationScore: Number(d.validationScore),
      volumeProfile: d.volumeProfile,
      marketStructure: d.marketStructure,
      filled: d.filled,
      hitTp1: d.hitTp1,
      hitTp2: d.hitTp2,
      hitTp3: d.hitTp3,
      hitStopLoss: d.hitStopLoss,
      finalOutcome: d.finalOutcome,
      holdTimeMins: d.holdTimeMins,
      createdAt: d.createdAt,
    }))

    return NextResponse.json({
      success: true,
      count: detections.length,
      detections: serializedDetections,
    })
  } catch (error) {
    console.error('FVG retrieval error:', error)
    return NextResponse.json(
      {
        error: 'Failed to retrieve FVG detections',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}
