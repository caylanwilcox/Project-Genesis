import { NextRequest, NextResponse } from 'next/server'
import { dataIngestionServiceV2 } from '@/services/dataIngestionService.v2'
import { marketDataRepo } from '@/repositories'
import { Timeframe } from '@/types/polygon'

/**
 * POST /api/v2/data/backfill
 * Backfill historical data with train/test split
 *
 * Body: {
 *   yearsBack?: number (default: 2)
 *   trainTestSplit?: number (default: 0.7)
 * }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json().catch(() => ({}))
    const yearsBack = body.yearsBack || 2
    const trainTestSplit = body.trainTestSplit || 0.7

    const tickers = ['SPY', 'QQQ', 'IWM', 'UVXY']
    const timeframes: Timeframe[] = ['1h', '1d']

    const results: any[] = []

    for (const ticker of tickers) {
      for (const timeframe of timeframes) {
        const startTime = Date.now()

        try {
          // Calculate date range
          const endDate = new Date()
          const startDate = new Date()
          startDate.setFullYear(startDate.getFullYear() - yearsBack)

          // Calculate split point
          const totalDaysSpan = Math.floor((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24))
          const trainDaysSpan = Math.floor(totalDaysSpan * trainTestSplit)

          const trainEndDate = new Date(startDate)
          trainEndDate.setDate(trainEndDate.getDate() + trainDaysSpan)

          const testStartDate = new Date(trainEndDate)
          testStartDate.setDate(testStartDate.getDate() + 1)

          // Ingest data
          const daysToFetch = Math.ceil(totalDaysSpan)
          const result = await dataIngestionServiceV2.ingestHistoricalData(
            ticker,
            timeframe,
            daysToFetch
          )

          if (!result.success) {
            throw new Error(result.error || 'Unknown error')
          }

          // Get ingested data to calculate split
          const allData = await marketDataRepo.findMany(
            { ticker, timeframe, startDate, endDate },
            100000
          )

          allData.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime())

          const trainingData = allData.filter(bar => bar.timestamp <= trainEndDate)
          const testingData = allData.filter(bar => bar.timestamp >= testStartDate)

          results.push({
            ticker,
            timeframe,
            success: true,
            totalBars: allData.length,
            trainingBars: trainingData.length,
            testingBars: testingData.length,
            trainingDateRange: {
              start: trainingData[0]?.timestamp || startDate,
              end: trainingData[trainingData.length - 1]?.timestamp || trainEndDate,
            },
            testingDateRange: {
              start: testingData[0]?.timestamp || testStartDate,
              end: testingData[testingData.length - 1]?.timestamp || endDate,
            },
            durationMs: Date.now() - startTime
          })

        } catch (error: any) {
          results.push({
            ticker,
            timeframe,
            success: false,
            error: error.message,
            durationMs: Date.now() - startTime
          })
        }
      }
    }

    const successful = results.filter(r => r.success)
    const failed = results.filter(r => !r.success)

    return NextResponse.json({
      success: true,
      summary: {
        totalJobs: results.length,
        successful: successful.length,
        failed: failed.length,
        totalBars: successful.reduce((sum, r) => sum + r.totalBars, 0),
        trainingBars: successful.reduce((sum, r) => sum + r.trainingBars, 0),
        testingBars: successful.reduce((sum, r) => sum + r.testingBars, 0),
        trainTestSplit: `${trainTestSplit * 100}% / ${(1 - trainTestSplit) * 100}%`
      },
      results
    })

  } catch (error: any) {
    console.error('[API /v2/backfill] Error:', error)
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    )
  }
}
