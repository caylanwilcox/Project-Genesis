/**
 * FVG Outcome Labeling API Endpoint
 * POST /api/v2/fvg/label
 *
 * Labels historical FVG patterns with outcomes (TP1, TP2, TP3, stop loss)
 * This is critical for ML training data generation
 */

import { NextRequest, NextResponse } from 'next/server'
import { fvgOutcomeLabelingService } from '@/services/fvgOutcomeLabelingService'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { ticker, timeframe } = body

    if (!ticker || !timeframe) {
      return NextResponse.json(
        { error: 'Missing required fields: ticker, timeframe' },
        { status: 400 }
      )
    }

    console.log(`[API /v2/fvg/label] Starting labeling for ${ticker} ${timeframe}`)

    const summary = await fvgOutcomeLabelingService.labelAllFvgs(
      ticker.toUpperCase(),
      timeframe
    )

    return NextResponse.json({
      success: true,
      message: `Labeled ${summary.totalProcessed} FVG patterns`,
      ticker: ticker.toUpperCase(),
      timeframe,
      summary,
    })
  } catch (error) {
    console.error('FVG labeling error:', error)
    return NextResponse.json(
      {
        error: 'Failed to label FVG patterns',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}
