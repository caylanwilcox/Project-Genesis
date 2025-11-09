/**
 * Data Ingestion Service (Prisma version)
 * Fetches market data from Polygon.io and stores in database using Prisma repositories
 */

import { polygonService } from './polygonService'
import { marketDataRepo, ingestionLogRepo } from '@/repositories'
import { Timeframe } from '@/types/polygon'

interface IngestionResult {
  success: boolean
  barsFetched: number
  barsInserted: number
  barsSkipped: number
  error?: string
  durationMs: number
}

export class DataIngestionServiceV2 {
  /**
   * Fetch and store historical market data for a ticker
   * @param ticker Stock symbol (e.g., 'SPY', 'QQQ')
   * @param timeframe Data interval (e.g., '1h', '1d')
   * @param daysBack How many days of history to fetch
   * @param displayTimeframe Display timeframe for special date handling
   */
  async ingestHistoricalData(
    ticker: string,
    timeframe: Timeframe,
    daysBack: number = 30,
    displayTimeframe?: string
  ): Promise<IngestionResult> {
    const startTime = Date.now()

    try {
      console.log(`[DataIngestion] Starting ingestion for ${ticker} ${timeframe} (${daysBack} days)`)

      // Calculate date range
      const endDate = new Date()
      const startDate = new Date()
      startDate.setDate(startDate.getDate() - daysBack)

      // Fetch data from Polygon
      const limit = this.calculateLimit(timeframe, daysBack)
      const polygonData = await polygonService.getAggregates(
        ticker,
        timeframe,
        limit,
        displayTimeframe
      )

      if (!polygonData || polygonData.length === 0) {
        throw new Error(`No data fetched from Polygon for ${ticker}`)
      }

      console.log(`[DataIngestion] Fetched ${polygonData.length} bars from Polygon`)

      // Transform data to database format
      const marketDataRecords = polygonData.map(bar => ({
        ticker: ticker.toUpperCase(),
        timeframe,
        timestamp: new Date(bar.time),
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
        volume: bar.volume, // Now Decimal, no conversion needed
        source: 'polygon' as const,
      }))

      // Insert into database using repository
      const inserted = await marketDataRepo.upsertMany(marketDataRecords)

      const barsInserted = inserted.length
      const barsSkipped = polygonData.length - barsInserted
      const durationMs = Date.now() - startTime

      // Log the ingestion
      await ingestionLogRepo.create({
        ticker: ticker.toUpperCase(),
        timeframe,
        startDate,
        endDate,
        barsFetched: polygonData.length,
        barsInserted,
        barsSkipped,
        status: 'success',
        durationMs,
      })

      console.log(`[DataIngestion] Success: ${barsInserted} bars inserted, ${barsSkipped} skipped, ${durationMs}ms`)

      return {
        success: true,
        barsFetched: polygonData.length,
        barsInserted,
        barsSkipped,
        durationMs
      }

    } catch (error: any) {
      const durationMs = Date.now() - startTime

      console.error('[DataIngestion] Error:', error)

      // Log the failure
      await ingestionLogRepo.create({
        ticker: ticker.toUpperCase(),
        timeframe,
        startDate: new Date(),
        endDate: new Date(),
        barsFetched: 0,
        barsInserted: 0,
        barsSkipped: 0,
        status: 'error',
        errorMessage: error.message,
        durationMs
      })

      return {
        success: false,
        barsFetched: 0,
        barsInserted: 0,
        barsSkipped: 0,
        error: error.message,
        durationMs
      }
    }
  }

  /**
   * Fetch historical data for all tickers
   */
  async ingestAllTickers(
    tickers: string[] = ['SPY', 'QQQ', 'IWM', 'UVXY'],
    timeframes: Timeframe[] = ['1h', '1d'],
    daysBack: number = 30
  ): Promise<IngestionResult[]> {
    const results: IngestionResult[] = []

    for (const ticker of tickers) {
      for (const timeframe of timeframes) {
        const result = await this.ingestHistoricalData(ticker, timeframe, daysBack)
        results.push(result)

        // Wait 200ms between requests to respect rate limits
        await new Promise(resolve => setTimeout(resolve, 200))
      }
    }

    return results
  }

  /**
   * Get market data from database
   */
  async getMarketData(
    ticker: string,
    timeframe: Timeframe,
    limit: number = 100
  ) {
    const data = await marketDataRepo.findMany(
      { ticker, timeframe },
      limit
    )

    // Convert Prisma Decimal to number for JSON serialization
    return data.map(bar => ({
      ...bar,
      open: Number(bar.open),
      high: Number(bar.high),
      low: Number(bar.low),
      close: Number(bar.close),
      volume: Number(bar.volume),
    }))
  }

  /**
   * Check if we have data for a ticker/timeframe
   */
  async hasData(ticker: string, timeframe: Timeframe): Promise<boolean> {
    return await marketDataRepo.exists(ticker, timeframe)
  }

  /**
   * Get data summary for a ticker
   */
  async getDataSummary(ticker: string) {
    return await marketDataRepo.getSummary(ticker)
  }

  /**
   * Calculate how many bars to fetch based on timeframe and days
   */
  private calculateLimit(timeframe: Timeframe, daysBack: number): number {
    const tradingHoursPerDay = 6.5
    const tradingMinutesPerDay = tradingHoursPerDay * 60

    switch (timeframe) {
      case '1m':
        return Math.ceil(daysBack * tradingMinutesPerDay)
      case '5m':
        return Math.ceil(daysBack * (tradingMinutesPerDay / 5))
      case '1h':
        return Math.ceil(daysBack * tradingHoursPerDay)
      case '4h':
        return Math.ceil(daysBack * (tradingHoursPerDay / 4))
      case '1d':
        return daysBack
      case '1w':
        return Math.ceil(daysBack / 7)
      case '1M':
        return Math.ceil(daysBack / 30)
      default:
        // Fallback for unknown timeframes
        return Math.ceil(daysBack * tradingHoursPerDay)
    }
  }
}

export const dataIngestionServiceV2 = new DataIngestionServiceV2()
