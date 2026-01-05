import { NextRequest, NextResponse } from 'next/server'
import { dataIngestionService } from '@/services/dataIngestionService'

// Force dynamic rendering - don't pre-render at build time
export const dynamic = 'force-dynamic'

/**
 * GET /api/data/ingest/status
 * Check ingestion status and data availability
 */
export async function GET(request: NextRequest) {
  try {
    const tickers = ['SPY', 'QQQ', 'IWM', 'UVXY']

    const status = []

    for (const ticker of tickers) {
      const summary = await dataIngestionService.getDataSummary(ticker)
      const hasHourly = await dataIngestionService.hasData(ticker, '1h')
      const hasDaily = await dataIngestionService.hasData(ticker, '1d')

      status.push({
        ticker,
        hasData: hasHourly || hasDaily,
        hasHourly,
        hasDaily,
        summary: summary || []
      })
    }

    return NextResponse.json({
      success: true,
      status
    })

  } catch (error: any) {
    console.error('[API /ingest/status] Error:', error)
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    )
  }
}
