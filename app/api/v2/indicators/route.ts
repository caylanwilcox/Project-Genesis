/**
 * Technical Indicators API Endpoint
 * GET /api/v2/indicators?ticker=SPY&timeframe=1h&limit=100
 *
 * Returns calculated technical indicators for ML features
 */

import { NextRequest, NextResponse } from 'next/server'
import { marketDataRepo } from '@/repositories'
import { technicalIndicatorsService } from '@/services/technicalIndicatorsService'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const ticker = searchParams.get('ticker')
    const timeframe = searchParams.get('timeframe')
    const limit = parseInt(searchParams.get('limit') || '500')

    if (!ticker || !timeframe) {
      return NextResponse.json(
        { error: 'Missing required params: ticker, timeframe' },
        { status: 400 }
      )
    }

    // Get market data
    const data = await marketDataRepo.getOHLCV(
      ticker.toUpperCase(),
      timeframe,
      undefined,
      undefined,
      limit + 50 // Extra bars for indicator warmup
    )

    if (data.length < 50) {
      return NextResponse.json(
        { error: 'Insufficient data for indicator calculation (need at least 50 bars)' },
        { status: 400 }
      )
    }

    // Calculate all indicators
    const indicators = technicalIndicatorsService.calculateAllIndicators(data)

    // Return only the requested limit (after warmup period)
    const result = indicators.slice(-limit)

    return NextResponse.json({
      success: true,
      ticker: ticker.toUpperCase(),
      timeframe,
      count: result.length,
      indicators: result,
    })
  } catch (error) {
    console.error('Indicators API error:', error)
    return NextResponse.json(
      {
        error: 'Failed to calculate indicators',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}

/**
 * POST /api/v2/indicators
 * Calculate indicators for specific date range and store in features table
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { ticker, timeframe, daysBack } = body

    if (!ticker || !timeframe) {
      return NextResponse.json(
        { error: 'Missing required fields: ticker, timeframe' },
        { status: 400 }
      )
    }

    const days = daysBack || 365

    // Calculate date range
    const endDate = new Date()
    const startDate = new Date()
    startDate.setDate(startDate.getDate() - days)

    // Get market data
    const data = await marketDataRepo.getOHLCV(
      ticker.toUpperCase(),
      timeframe,
      startDate,
      endDate,
      10000
    )

    if (data.length < 50) {
      return NextResponse.json(
        { error: 'Insufficient data for indicator calculation' },
        { status: 400 }
      )
    }

    // Calculate all indicators
    const indicators = technicalIndicatorsService.calculateAllIndicators(data)

    // Filter to only include complete indicator sets
    const completeIndicators = indicators.filter(
      i => i.rsi_14 !== undefined && i.macd !== undefined && i.atr_14 !== undefined
    )

    return NextResponse.json({
      success: true,
      ticker: ticker.toUpperCase(),
      timeframe,
      daysProcessed: days,
      barsProcessed: data.length,
      indicatorsCalculated: completeIndicators.length,
      sample: completeIndicators.slice(-5),
    })
  } catch (error) {
    console.error('Indicators POST error:', error)
    return NextResponse.json(
      {
        error: 'Failed to calculate indicators',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}
