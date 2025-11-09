/**
 * Market Data Repository
 * Handles all database operations for market_data table
 */

import { prisma } from '@/lib/prisma'
import { MarketData, Prisma } from '@prisma/client'

export interface MarketDataFilter {
  ticker?: string
  timeframe?: string
  startDate?: Date
  endDate?: Date
}

export class MarketDataRepository {
  /**
   * Insert or update market data (upsert)
   */
  async upsertMany(data: Omit<MarketData, 'id' | 'createdAt'>[]) {
    const operations = data.map(record =>
      prisma.marketData.upsert({
        where: {
          ticker_timeframe_timestamp: {
            ticker: record.ticker,
            timeframe: record.timeframe,
            timestamp: record.timestamp,
          },
        },
        update: {
          open: record.open,
          high: record.high,
          low: record.low,
          close: record.close,
          volume: record.volume,
          source: record.source,
        },
        create: record,
      })
    )

    return await prisma.$transaction(operations)
  }

  /**
   * Find market data with filters
   */
  async findMany(filter: MarketDataFilter = {}, limit: number = 100) {
    const where: Prisma.MarketDataWhereInput = {}

    if (filter.ticker) {
      where.ticker = filter.ticker.toUpperCase()
    }

    if (filter.timeframe) {
      where.timeframe = filter.timeframe
    }

    if (filter.startDate || filter.endDate) {
      where.timestamp = {}
      if (filter.startDate) {
        where.timestamp.gte = filter.startDate
      }
      if (filter.endDate) {
        where.timestamp.lte = filter.endDate
      }
    }

    return await prisma.marketData.findMany({
      where,
      orderBy: { timestamp: 'desc' },
      take: limit,
    })
  }

  /**
   * Find latest market data for a ticker
   */
  async findLatest(ticker: string, timeframe: string, limit: number = 1) {
    return await prisma.marketData.findMany({
      where: {
        ticker: ticker.toUpperCase(),
        timeframe,
      },
      orderBy: { timestamp: 'desc' },
      take: limit,
    })
  }

  /**
   * Check if data exists for ticker/timeframe
   */
  async exists(ticker: string, timeframe: string): Promise<boolean> {
    const count = await prisma.marketData.count({
      where: {
        ticker: ticker.toUpperCase(),
        timeframe,
      },
    })
    return count > 0
  }

  /**
   * Get data summary statistics
   */
  async getSummary(ticker: string) {
    const result = await prisma.$queryRaw<
      Array<{
        ticker: string
        timeframe: string
        bar_count: bigint
        earliest_date: Date
        latest_date: Date
      }>
    >`
      SELECT
        ticker,
        timeframe,
        COUNT(*) as bar_count,
        MIN(timestamp) as earliest_date,
        MAX(timestamp) as latest_date
      FROM market_data
      WHERE ticker = ${ticker.toUpperCase()}
      GROUP BY ticker, timeframe
      ORDER BY timeframe
    `

    return result.map(row => ({
      ticker: row.ticker,
      timeframe: row.timeframe,
      barCount: Number(row.bar_count),
      earliestDate: row.earliest_date,
      latestDate: row.latest_date,
    }))
  }

  /**
   * Get count of records
   */
  async count(filter: MarketDataFilter = {}): Promise<number> {
    const where: Prisma.MarketDataWhereInput = {}

    if (filter.ticker) {
      where.ticker = filter.ticker.toUpperCase()
    }

    if (filter.timeframe) {
      where.timeframe = filter.timeframe
    }

    if (filter.startDate || filter.endDate) {
      where.timestamp = {}
      if (filter.startDate) {
        where.timestamp.gte = filter.startDate
      }
      if (filter.endDate) {
        where.timestamp.lte = filter.endDate
      }
    }

    return await prisma.marketData.count({ where })
  }

  /**
   * Delete old data (for cleanup)
   */
  async deleteOlderThan(date: Date) {
    return await prisma.marketData.deleteMany({
      where: {
        timestamp: {
          lt: date,
        },
      },
    })
  }

  /**
   * Get OHLCV data for charting
   */
  async getOHLCV(
    ticker: string,
    timeframe: string,
    startDate?: Date,
    endDate?: Date,
    limit: number = 1000
  ) {
    const where: Prisma.MarketDataWhereInput = {
      ticker: ticker.toUpperCase(),
      timeframe,
    }

    if (startDate || endDate) {
      where.timestamp = {}
      if (startDate) {
        where.timestamp.gte = startDate
      }
      if (endDate) {
        where.timestamp.lte = endDate
      }
    }

    const data = await prisma.marketData.findMany({
      where,
      orderBy: { timestamp: 'asc' },
      take: limit,
      select: {
        timestamp: true,
        open: true,
        high: true,
        low: true,
        close: true,
        volume: true,
      },
    })

    return data.map(bar => ({
      time: bar.timestamp.getTime(),
      open: Number(bar.open),
      high: Number(bar.high),
      low: Number(bar.low),
      close: Number(bar.close),
      volume: Number(bar.volume),
    }))
  }
}

export const marketDataRepo = new MarketDataRepository()
