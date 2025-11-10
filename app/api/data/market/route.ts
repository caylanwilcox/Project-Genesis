import { NextRequest, NextResponse } from 'next/server'
import { dataIngestionService } from '@/services/dataIngestionService'
import { Timeframe } from '@/types/polygon'

// Force dynamic rendering - don't pre-render at build time
export const dynamic = 'force-dynamic'

/**
 * GET /api/data/market?ticker=SPY&timeframe=1h&limit=100
 * Fetch market data from database
 */
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const ticker = searchParams.get('ticker')
    const timeframe = searchParams.get('timeframe')
    const limit = parseInt(searchParams.get('limit') || '100')

    if (!ticker || !timeframe) {
      return NextResponse.json(
        { success: false, error: 'Missing ticker or timeframe parameter' },
        { status: 400 }
      )
    }

    const data = await dataIngestionService.getMarketData(
      ticker,
      timeframe as Timeframe,
      limit
    )

    return NextResponse.json({
      success: true,
      ticker,
      timeframe,
      count: data.length,
      data
    })

  } catch (error: any) {
    console.error('[API /market] Error:', error)
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    )
  }
}
