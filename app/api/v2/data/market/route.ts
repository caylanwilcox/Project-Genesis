import { NextRequest, NextResponse } from 'next/server'
import { dataIngestionServiceV2 } from '@/services/dataIngestionService.v2'
import { marketDataRepo } from '@/repositories/marketDataRepository'
import { Timeframe } from '@/types/polygon'

// Force dynamic rendering - don't pre-render at build time
export const dynamic = 'force-dynamic'

const POLYGON_API_KEY = process.env.POLYGON_API_KEY || process.env.NEXT_PUBLIC_POLYGON_API_KEY

/**
 * Fetch historical bars directly from Polygon API
 * Fallback when database is unavailable
 */
async function fetchFromPolygon(
  ticker: string,
  timeframe: string,
  fromDate: string,
  toDate: string,
  limit: number
) {
  // Map timeframe to Polygon multiplier/span
  const tfMap: Record<string, { multiplier: number; timespan: string }> = {
    '1m': { multiplier: 1, timespan: 'minute' },
    '5m': { multiplier: 5, timespan: 'minute' },
    '15m': { multiplier: 15, timespan: 'minute' },
    '30m': { multiplier: 30, timespan: 'minute' },
    '1h': { multiplier: 1, timespan: 'hour' },
    '4h': { multiplier: 4, timespan: 'hour' },
    '1d': { multiplier: 1, timespan: 'day' },
  }

  const config = tfMap[timeframe] || { multiplier: 5, timespan: 'minute' }

  // Use higher limit for Polygon API to avoid pagination issues
  // A full trading day of 1-minute bars = ~390 bars, 5-minute = ~78 bars
  const polygonLimit = Math.max(limit, 5000)
  const url = `https://api.polygon.io/v2/aggs/ticker/${ticker}/range/${config.multiplier}/${config.timespan}/${fromDate}/${toDate}?adjusted=true&sort=asc&limit=${polygonLimit}&apiKey=${POLYGON_API_KEY}`

  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Polygon API error: ${response.status}`)
  }

  const data = await response.json()

  if (!data.results || data.results.length === 0) {
    return []
  }

  // Transform to our format
  return data.results.map((bar: any) => ({
    time: bar.t,
    open: bar.o,
    high: bar.h,
    low: bar.l,
    close: bar.c,
    volume: bar.v,
  }))
}

/**
 * GET /api/v2/data/market?ticker=SPY&timeframe=1h&limit=100
 * GET /api/v2/data/market?ticker=SPY&timeframe=5m&date=2025-12-30&endTime=14:30
 * Fetch market data from database (Prisma version)
 *
 * Optional date/time filtering for replay mode:
 * - date: Filter to a specific date (YYYY-MM-DD)
 * - endTime: End time for the date (HH:MM) - filters bars up to this time
 */
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const ticker = searchParams.get('ticker')
    const timeframe = searchParams.get('timeframe')
    const limit = parseInt(searchParams.get('limit') || '100')
    const date = searchParams.get('date')      // End date (required for replay)
    const fromDate = searchParams.get('fromDate') // Start date (optional, for multi-day fetch)
    const endTime = searchParams.get('endTime')

    if (!ticker || !timeframe) {
      return NextResponse.json(
        { success: false, error: 'Missing ticker or timeframe parameter' },
        { status: 400 }
      )
    }

    // If date is provided, use date-filtered query (for replay mode)
    if (date) {
      // Parse dates - use fromDate if provided, otherwise use date for single-day
      const startDateStr = fromDate || date
      const startDate = new Date(`${startDateStr}T09:30:00-05:00`) // Market open ET
      let endDate: Date

      if (endTime) {
        // Parse HH:MM format
        const [hours, minutes] = endTime.split(':').map(Number)
        endDate = new Date(`${date}T${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:00-05:00`)
      } else {
        // Default to market close
        endDate = new Date(`${date}T16:00:00-05:00`)
      }

      try {
        // Try database first
        const data = await marketDataRepo.getOHLCV(
          ticker,
          timeframe,
          startDate,
          endDate,
          limit
        )

        // If database has data, return it
        if (data.length > 0) {
          return NextResponse.json({
            success: true,
            ticker,
            timeframe,
            fromDate: startDateStr,
            date,
            endTime: endTime || '16:00',
            count: data.length,
            source: 'database',
            data
          })
        }

        // Database returned empty, fall through to Polygon
        console.log('[API /v2/market] Database returned no data, falling back to Polygon')
      } catch (dbError) {
        // Database failed, fall through to Polygon
        console.warn('[API /v2/market] Database unavailable, falling back to Polygon:', dbError)
      }

      // Fallback to Polygon API
      if (!POLYGON_API_KEY) {
        return NextResponse.json({
          success: true,
          ticker,
          timeframe,
          date,
          endTime: endTime || '16:00',
          count: 0,
          source: 'none',
          data: [],
          warning: 'No data in database and no Polygon API key configured'
        })
      }

      const polygonData = await fetchFromPolygon(ticker, timeframe, startDateStr, date, limit)

      // Filter to endTime if specified
      const endTimeMs = endDate.getTime()
      const filteredData = polygonData.filter((bar: any) => bar.time <= endTimeMs)

      return NextResponse.json({
        success: true,
        ticker,
        timeframe,
        fromDate: startDateStr,
        date,
        endTime: endTime || '16:00',
        count: filteredData.length,
        source: 'polygon',
        data: filteredData
      })
    }

    // Default behavior: get most recent data
    try {
      const data = await dataIngestionServiceV2.getMarketData(
        ticker,
        timeframe as Timeframe,
        limit
      )

      return NextResponse.json({
        success: true,
        ticker,
        timeframe,
        count: data.length,
        source: 'database',
        data
      })
    } catch (dbError) {
      // Database failed, fallback to Polygon
      console.warn('[API /v2/market] Database unavailable, falling back to Polygon:', dbError)

      if (!POLYGON_API_KEY) {
        throw new Error('Database unavailable and no Polygon API key configured')
      }

      // Calculate date range for recent data
      const now = new Date()
      const from = new Date(now)
      from.setDate(from.getDate() - 7) // Last 7 days

      const toDate = now.toISOString().split('T')[0]
      const fromDate = from.toISOString().split('T')[0]

      const data = await fetchFromPolygon(ticker, timeframe, fromDate, toDate, limit)

      return NextResponse.json({
        success: true,
        ticker,
        timeframe,
        count: data.length,
        source: 'polygon',
        data
      })
    }

  } catch (error: any) {
    console.error('[API /v2/market] Error:', error)
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    )
  }
}
