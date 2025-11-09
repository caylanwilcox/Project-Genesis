/**
 * FVG Detection API Endpoint
 * POST /api/v2/fvg/detect
 *
 * Scans historical market data for Fair Value Gap patterns
 * Stores detected patterns in database for ML training
 */

import { NextRequest, NextResponse } from 'next/server'
import { marketDataRepo, fvgDetectionRepo } from '@/repositories'
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

    // Fetch historical market data
    const marketData = await marketDataRepo.findMany({
      ticker: ticker.toUpperCase(),
      timeframe,
      startDate,
      endDate,
    })

    if (marketData.length < 3) {
      return NextResponse.json(
        { error: 'Insufficient market data (need at least 3 bars)' },
        { status: 400 }
      )
    }

    // Convert to MarketBar format
    const bars = marketData.map(bar => ({
      timestamp: bar.timestamp,
      open: Number(bar.open),
      high: Number(bar.high),
      low: Number(bar.low),
      close: Number(bar.close),
      volume: Number(bar.volume),
    }))

    // Detect FVG patterns
    const fvgService = new FvgDetectionService({
      minGapSizePct: body.minGapSizePct || 0.1,
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
      patterns: patterns.map(p => ({
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
      })),
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
