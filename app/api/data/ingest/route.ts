import { NextRequest, NextResponse } from 'next/server'
import { dataIngestionService } from '@/services/dataIngestionService'
import { Timeframe } from '@/types/polygon'

/**
 * POST /api/data/ingest
 * Trigger data ingestion for specified tickers
 *
 * Body: {
 *   ticker?: string,           // Single ticker (e.g., 'SPY')
 *   tickers?: string[],        // Multiple tickers
 *   timeframe?: string,        // Single timeframe (e.g., '1h')
 *   timeframes?: string[],     // Multiple timeframes
 *   daysBack?: number          // How many days to fetch (default: 30)
 * }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    const ticker = body.ticker
    const tickers = body.tickers || (ticker ? [ticker] : ['SPY', 'QQQ', 'IWM', 'UVXY'])
    const timeframe = body.timeframe
    const timeframes = body.timeframes || (timeframe ? [timeframe] : ['1h', '1d'])
    const daysBack = body.daysBack || 30

    console.log(`[API /ingest] Request: ${tickers.join(',')} ${timeframes.join(',')} ${daysBack} days`)

    // Ingest data for all combinations
    const results = []

    for (const t of tickers) {
      for (const tf of timeframes) {
        const result = await dataIngestionService.ingestHistoricalData(
          t,
          tf as Timeframe,
          daysBack
        )
        results.push({
          ticker: t,
          timeframe: tf,
          ...result
        })

        // Wait 200ms between requests to respect rate limits
        await new Promise(resolve => setTimeout(resolve, 200))
      }
    }

    const totalSuccess = results.filter(r => r.success).length
    const totalFailed = results.filter(r => !r.success).length
    const totalBarsInserted = results.reduce((sum, r) => sum + r.barsInserted, 0)

    return NextResponse.json({
      success: true,
      summary: {
        totalJobs: results.length,
        successful: totalSuccess,
        failed: totalFailed,
        totalBarsInserted
      },
      results
    })

  } catch (error: any) {
    console.error('[API /ingest] Error:', error)
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    )
  }
}

/**
 * GET /api/data/ingest/status
 * Check ingestion status and data availability
 */
export async function GET(request: NextRequest) {
  try {
    const tickers = ['SPY', 'QQQ', 'IWM', 'UVXY']
    const timeframes = ['1h', '1d']

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
