import { supabase, MarketData } from '@/lib/supabase'
import { polygonService } from './polygonService'
import { Timeframe } from '@/types/polygon'

interface IngestionResult {
  success: boolean
  barsFetched: number
  barsInserted: number
  barsSkipped: number
  error?: string
  durationMs: number
}

export class DataIngestionService {
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
        timestamp: new Date(bar.time).toISOString(),
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
        volume: bar.volume,
        source: 'polygon'
      }))

      // Insert into Supabase (using upsert to handle duplicates)
      const { data, error } = await supabase
        .from('market_data')
        .upsert(marketDataRecords, {
          onConflict: 'ticker,timeframe,timestamp',
          ignoreDuplicates: false
        })
        .select()

      if (error) {
        throw error
      }

      const barsInserted = data?.length || 0
      const barsSkipped = polygonData.length - barsInserted
      const durationMs = Date.now() - startTime

      // Log the ingestion
      await this.logIngestion({
        ticker: ticker.toUpperCase(),
        timeframe,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
        barsFetched: polygonData.length,
        barsInserted,
        barsSkipped,
        status: 'success',
        durationMs
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
      await this.logIngestion({
        ticker: ticker.toUpperCase(),
        timeframe,
        startDate: new Date().toISOString(),
        endDate: new Date().toISOString(),
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
  ): Promise<MarketData[]> {
    const { data, error } = await supabase
      .from('market_data')
      .select('*')
      .eq('ticker', ticker.toUpperCase())
      .eq('timeframe', timeframe)
      .order('timestamp', { ascending: false })
      .limit(limit)

    if (error) {
      console.error('[DataIngestion] Error fetching from DB:', error)
      throw error
    }

    return (data || []).reverse() // Return in chronological order
  }

  /**
   * Check if we have data for a ticker/timeframe
   */
  async hasData(ticker: string, timeframe: Timeframe): Promise<boolean> {
    const { count, error } = await supabase
      .from('market_data')
      .select('*', { count: 'exact', head: true })
      .eq('ticker', ticker.toUpperCase())
      .eq('timeframe', timeframe)

    if (error) {
      console.error('[DataIngestion] Error checking data:', error)
      return false
    }

    return (count || 0) > 0
  }

  /**
   * Get data summary for a ticker
   */
  async getDataSummary(ticker: string) {
    const { data, error } = await supabase
      .from('market_data_summary')
      .select('*')
      .eq('ticker', ticker.toUpperCase())

    if (error) {
      console.error('[DataIngestion] Error fetching summary:', error)
      return null
    }

    return data
  }

  /**
   * Calculate how many bars to fetch based on timeframe and days
   */
  private calculateLimit(timeframe: Timeframe, daysBack: number): number {
    const hoursPerDay = 24
    const tradingHoursPerDay = 6.5

    switch (timeframe) {
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
        return 100
    }
  }

  /**
   * Log ingestion activity
   */
  private async logIngestion(log: {
    ticker: string
    timeframe: string
    startDate: string
    endDate: string
    barsFetched: number
    barsInserted: number
    barsSkipped: number
    status: string
    errorMessage?: string
    durationMs: number
  }) {
    try {
      await supabase.from('ingestion_log').insert({
        ticker: log.ticker,
        timeframe: log.timeframe,
        start_date: log.startDate,
        end_date: log.endDate,
        bars_fetched: log.barsFetched,
        bars_inserted: log.barsInserted,
        bars_skipped: log.barsSkipped,
        status: log.status,
        error_message: log.errorMessage,
        duration_ms: log.durationMs
      })
    } catch (error) {
      console.error('[DataIngestion] Failed to log ingestion:', error)
    }
  }
}

export const dataIngestionService = new DataIngestionService()
