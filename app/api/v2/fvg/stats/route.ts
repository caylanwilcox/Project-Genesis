/**
 * FVG Statistics API Endpoint
 * GET /api/v2/fvg/stats
 *
 * Returns win rate statistics for FVG patterns
 * Critical for ML training - shows which setups are profitable
 */

import { NextRequest, NextResponse } from 'next/server'
import { fvgDetectionRepo } from '@/repositories'

// Force dynamic rendering - don't pre-render at build time
export const dynamic = 'force-dynamic'
import type { TradingMode } from '@/services/fvgDetectionService'

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const ticker = searchParams.get('ticker')
    const tradingMode = searchParams.get('tradingMode') as TradingMode
    const fvgType = searchParams.get('fvgType') as 'bullish' | 'bearish' | null

    if (!ticker || !tradingMode) {
      return NextResponse.json(
        { error: 'Missing required fields: ticker, tradingMode' },
        { status: 400 }
      )
    }

    // Get win rate statistics
    const stats = await fvgDetectionRepo.getWinRateStats(
      ticker.toUpperCase(),
      tradingMode,
      fvgType || undefined
    )

    if (!stats) {
      return NextResponse.json({
        success: true,
        message: 'No completed FVG patterns found',
        ticker: ticker.toUpperCase(),
        tradingMode,
        fvgType: fvgType || 'all',
        stats: null,
      })
    }

    return NextResponse.json({
      success: true,
      ticker: ticker.toUpperCase(),
      tradingMode,
      fvgType: fvgType || 'all',
      stats: {
        totalDetections: stats.totalDetections,
        filledCount: stats.filledCount,
        tp1Count: stats.tp1Count,
        tp2Count: stats.tp2Count,
        tp3Count: stats.tp3Count,
        stopLossCount: stats.stopLossCount,
        tp1WinRate: stats.tp1WinRate.toFixed(2) + '%',
        tp2WinRate: stats.tp2WinRate.toFixed(2) + '%',
        tp3WinRate: stats.tp3WinRate.toFixed(2) + '%',
        avgHoldTimeMins: Math.round(stats.avgHoldTimeMins),
        avgHoldTimeFormatted: formatHoldTime(stats.avgHoldTimeMins),
      },
    })
  } catch (error) {
    console.error('FVG stats error:', error)
    return NextResponse.json(
      {
        error: 'Failed to retrieve FVG statistics',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}

function formatHoldTime(mins: number): string {
  if (mins < 60) {
    return `${Math.round(mins)} minutes`
  }
  if (mins < 1440) {
    return `${(mins / 60).toFixed(1)} hours`
  }
  return `${(mins / 1440).toFixed(1)} days`
}
