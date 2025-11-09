import { NextRequest, NextResponse } from 'next/server'
import { dataIngestionServiceV2 } from '@/services/dataIngestionService.v2'

/**
 * GET /api/v2/data/ingest/status
 * Check ingestion status and data availability (Prisma version)
 */
export async function GET(request: NextRequest) {
  try {
    const tickers = ['SPY', 'QQQ', 'IWM', 'UVXY']

    const status = []

    for (const ticker of tickers) {
      const summary = await dataIngestionServiceV2.getDataSummary(ticker)
      const hasHourly = await dataIngestionServiceV2.hasData(ticker, '1h')
      const hasDaily = await dataIngestionServiceV2.hasData(ticker, '1d')

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
    console.error('[API /v2/ingest/status] Error:', error)
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    )
  }
}
