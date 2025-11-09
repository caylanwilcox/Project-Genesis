/**
 * Features Repository
 * Handles all database operations for features table (technical indicators)
 */

import { prisma } from '@/lib/prisma'
import { Feature, Prisma } from '@prisma/client'

export interface FeatureFilter {
  ticker?: string
  timeframe?: string
  featureName?: string
  startDate?: Date
  endDate?: Date
}

export class FeaturesRepository {
  /**
   * Insert or update features (upsert)
   */
  async upsertMany(data: Omit<Feature, 'id' | 'createdAt'>[]) {
    const operations = data.map(record =>
      prisma.feature.upsert({
        where: {
          ticker_timeframe_timestamp_featureName: {
            ticker: record.ticker,
            timeframe: record.timeframe,
            timestamp: record.timestamp,
            featureName: record.featureName,
          },
        },
        update: {
          featureValue: record.featureValue,
        },
        create: record,
      })
    )

    return await prisma.$transaction(operations)
  }

  /**
   * Find features with filters
   */
  async findMany(filter: FeatureFilter = {}, limit: number = 100) {
    const where: Prisma.FeatureWhereInput = {}

    if (filter.ticker) {
      where.ticker = filter.ticker.toUpperCase()
    }

    if (filter.timeframe) {
      where.timeframe = filter.timeframe
    }

    if (filter.featureName) {
      where.featureName = filter.featureName
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

    return await prisma.feature.findMany({
      where,
      orderBy: { timestamp: 'desc' },
      take: limit,
    })
  }

  /**
   * Get latest feature values for a ticker
   */
  async getLatestFeatures(
    ticker: string,
    timeframe: string,
    featureNames?: string[]
  ) {
    const where: Prisma.FeatureWhereInput = {
      ticker: ticker.toUpperCase(),
      timeframe,
    }

    if (featureNames && featureNames.length > 0) {
      where.featureName = { in: featureNames }
    }

    // Get latest timestamp first
    const latest = await prisma.feature.findFirst({
      where: {
        ticker: ticker.toUpperCase(),
        timeframe,
      },
      orderBy: { timestamp: 'desc' },
      select: { timestamp: true },
    })

    if (!latest) {
      return []
    }

    // Get all features for latest timestamp
    return await prisma.feature.findMany({
      where: {
        ...where,
        timestamp: latest.timestamp,
      },
    })
  }

  /**
   * Get feature time series for charting
   */
  async getFeatureTimeSeries(
    ticker: string,
    timeframe: string,
    featureName: string,
    startDate?: Date,
    endDate?: Date,
    limit: number = 1000
  ) {
    const where: Prisma.FeatureWhereInput = {
      ticker: ticker.toUpperCase(),
      timeframe,
      featureName,
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

    const data = await prisma.feature.findMany({
      where,
      orderBy: { timestamp: 'asc' },
      take: limit,
      select: {
        timestamp: true,
        featureValue: true,
      },
    })

    return data.map(point => ({
      time: point.timestamp.getTime(),
      value: Number(point.featureValue),
    }))
  }

  /**
   * Get all available feature names for a ticker
   */
  async getFeatureNames(ticker: string, timeframe: string): Promise<string[]> {
    const result = await prisma.feature.groupBy({
      by: ['featureName'],
      where: {
        ticker: ticker.toUpperCase(),
        timeframe,
      },
    })

    return result.map(r => r.featureName)
  }

  /**
   * Check if features exist for ticker/timeframe
   */
  async exists(
    ticker: string,
    timeframe: string,
    featureName?: string
  ): Promise<boolean> {
    const where: Prisma.FeatureWhereInput = {
      ticker: ticker.toUpperCase(),
      timeframe,
    }

    if (featureName) {
      where.featureName = featureName
    }

    const count = await prisma.feature.count({ where })
    return count > 0
  }

  /**
   * Get feature statistics
   */
  async getStats(ticker: string, timeframe: string, featureName: string) {
    const result = await prisma.$queryRaw<
      Array<{
        min_value: number
        max_value: number
        avg_value: number
        count: bigint
      }>
    >`
      SELECT
        MIN(feature_value::float) as min_value,
        MAX(feature_value::float) as max_value,
        AVG(feature_value::float) as avg_value,
        COUNT(*) as count
      FROM features
      WHERE ticker = ${ticker.toUpperCase()}
        AND timeframe = ${timeframe}
        AND feature_name = ${featureName}
    `

    if (result.length === 0) {
      return null
    }

    return {
      min: result[0].min_value,
      max: result[0].max_value,
      avg: result[0].avg_value,
      count: Number(result[0].count),
    }
  }

  /**
   * Delete old features (for cleanup)
   */
  async deleteOlderThan(date: Date) {
    return await prisma.feature.deleteMany({
      where: {
        timestamp: {
          lt: date,
        },
      },
    })
  }

  /**
   * Bulk insert features (for feature engineering)
   */
  async createMany(data: Omit<Feature, 'id' | 'createdAt'>[]) {
    return await prisma.feature.createMany({
      data,
      skipDuplicates: true,
    })
  }
}

export const featuresRepo = new FeaturesRepository()
