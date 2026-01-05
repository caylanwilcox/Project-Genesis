/**
 * Ingestion Log Repository
 * Handles all database operations for ingestion_log table
 */

import { prisma } from '@/lib/prisma'
import { IngestionLog, Prisma } from '@prisma/client'

export interface IngestionLogFilter {
  ticker?: string
  timeframe?: string
  status?: string
  startDate?: Date
  endDate?: Date
}

export class IngestionLogRepository {
  /**
   * Create a new ingestion log entry
   */
  async create(data: Omit<IngestionLog, 'id' | 'createdAt'>) {
    return await prisma.ingestionLog.create({
      data,
    })
  }

  /**
   * Find logs with filters
   */
  async findMany(filter: IngestionLogFilter = {}, limit: number = 100) {
    const where: Prisma.IngestionLogWhereInput = {}

    if (filter.ticker) {
      where.ticker = filter.ticker.toUpperCase()
    }

    if (filter.timeframe) {
      where.timeframe = filter.timeframe
    }

    if (filter.status) {
      where.status = filter.status
    }

    if (filter.startDate || filter.endDate) {
      where.createdAt = {}
      if (filter.startDate) {
        where.createdAt.gte = filter.startDate
      }
      if (filter.endDate) {
        where.createdAt.lte = filter.endDate
      }
    }

    return await prisma.ingestionLog.findMany({
      where,
      orderBy: { createdAt: 'desc' },
      take: limit,
    })
  }

  /**
   * Get latest log for a ticker/timeframe
   */
  async getLatest(ticker: string, timeframe: string) {
    return await prisma.ingestionLog.findFirst({
      where: {
        ticker: ticker.toUpperCase(),
        timeframe,
      },
      orderBy: { createdAt: 'desc' },
    })
  }

  /**
   * Get ingestion statistics
   */
  async getStats(ticker?: string, timeframe?: string) {
    const where: Prisma.IngestionLogWhereInput = {}

    if (ticker) {
      where.ticker = ticker.toUpperCase()
    }

    if (timeframe) {
      where.timeframe = timeframe
    }

    const result = await prisma.ingestionLog.aggregate({
      where,
      _count: { id: true },
      _sum: {
        barsFetched: true,
        barsInserted: true,
        barsSkipped: true,
        durationMs: true,
      },
      _avg: {
        durationMs: true,
      },
    })

    const successCount = await prisma.ingestionLog.count({
      where: {
        ...where,
        status: 'success',
      },
    })

    const errorCount = await prisma.ingestionLog.count({
      where: {
        ...where,
        status: 'error',
      },
    })

    return {
      totalRuns: result._count.id,
      successfulRuns: successCount,
      failedRuns: errorCount,
      totalBarsFetched: result._sum.barsFetched || 0,
      totalBarsInserted: result._sum.barsInserted || 0,
      totalBarsSkipped: result._sum.barsSkipped || 0,
      totalDurationMs: result._sum.durationMs || 0,
      avgDurationMs: result._avg.durationMs || 0,
    }
  }

  /**
   * Get recent errors
   */
  async getRecentErrors(limit: number = 10) {
    return await prisma.ingestionLog.findMany({
      where: {
        status: 'error',
        errorMessage: { not: null },
      },
      orderBy: { createdAt: 'desc' },
      take: limit,
    })
  }

  /**
   * Delete old logs (for cleanup)
   */
  async deleteOlderThan(date: Date) {
    return await prisma.ingestionLog.deleteMany({
      where: {
        createdAt: {
          lt: date,
        },
      },
    })
  }

  /**
   * Get ingestion history summary
   */
  async getHistorySummary() {
    const result = await prisma.$queryRaw<
      Array<{
        ticker: string
        timeframe: string
        total_runs: bigint
        successful_runs: bigint
        failed_runs: bigint
        total_bars_inserted: bigint
        last_run: Date
      }>
    >`
      SELECT
        ticker,
        timeframe,
        COUNT(*) as total_runs,
        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_runs,
        SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as failed_runs,
        SUM(bars_inserted) as total_bars_inserted,
        MAX(created_at) as last_run
      FROM ingestion_log
      GROUP BY ticker, timeframe
      ORDER BY ticker, timeframe
    `

    return result.map(row => ({
      ticker: row.ticker,
      timeframe: row.timeframe,
      totalRuns: Number(row.total_runs),
      successfulRuns: Number(row.successful_runs),
      failedRuns: Number(row.failed_runs),
      totalBarsInserted: Number(row.total_bars_inserted),
      lastRun: row.last_run,
    }))
  }
}

export const ingestionLogRepo = new IngestionLogRepository()
