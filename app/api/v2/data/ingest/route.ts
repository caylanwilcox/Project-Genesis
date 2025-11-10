import { NextRequest, NextResponse } from 'next/server'
import { dataIngestionServiceV2 } from '@/services/dataIngestionService.v2'
import { Timeframe } from '@/types/polygon'

// Force dynamic rendering - don't pre-render at build time
export const dynamic = 'force-dynamic'

/**
 * POST /api/v2/data/ingest
 * Trigger data ingestion for specified tickers (Prisma version)
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

    console.log(`[API /v2/ingest] Request: ${tickers.join(',')} ${timeframes.join(',')} ${daysBack} days`)

    // Ingest data for all combinations
    const results = []

    for (const t of tickers) {
      for (const tf of timeframes) {
        const result = await dataIngestionServiceV2.ingestHistoricalData(
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
    console.error('[API /v2/ingest] Error:', error)
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    )
  }
}
