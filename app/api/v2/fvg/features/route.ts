/**
 * FVG Features API Endpoint
 * POST /api/v2/fvg/features
 *
 * Generates ML training dataset by enriching FVG detections
 * with technical indicator features
 */

import { NextRequest, NextResponse } from 'next/server'
import { fvgFeatureService } from '@/services/fvgFeatureService'

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

    console.log(`[API /v2/fvg/features] Generating ML dataset for ${ticker} ${timeframe}`)

    const { features, summary } = await fvgFeatureService.generateMLDataset(
      ticker.toUpperCase(),
      timeframe
    )

    return NextResponse.json({
      success: true,
      message: `Generated ${features.length} feature sets`,
      ticker: ticker.toUpperCase(),
      timeframe,
      summary,
      features,
    })
  } catch (error) {
    console.error('FVG features error:', error)
    return NextResponse.json(
      {
        error: 'Failed to generate features',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}
