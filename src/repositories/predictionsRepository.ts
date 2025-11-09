/**
 * Predictions Repository
 * Handles all database operations for predictions table
 */

import { prisma } from '@/lib/prisma'
import { Prediction, Prisma } from '@prisma/client'

export interface PredictionFilter {
  ticker?: string
  timeframe?: string
  modelName?: string
  startDate?: Date
  endDate?: Date
}

export interface PredictionAccuracy {
  modelName: string
  totalPredictions: number
  correctPredictions: number
  accuracy: number
  avgConfidence: number
}

export class PredictionsRepository {
  /**
   * Create a new prediction
   */
  async create(data: Omit<Prediction, 'id' | 'createdAt'>) {
    return await prisma.prediction.create({
      data,
    })
  }

  /**
   * Insert or update prediction (upsert)
   */
  async upsert(data: Omit<Prediction, 'id' | 'createdAt'>) {
    return await prisma.prediction.upsert({
      where: {
        ticker_timeframe_timestamp_modelName: {
          ticker: data.ticker,
          timeframe: data.timeframe,
          timestamp: data.timestamp,
          modelName: data.modelName,
        },
      },
      update: {
        predictedDirection: data.predictedDirection,
        predictedChange: data.predictedChange,
        confidence: data.confidence,
        actualDirection: data.actualDirection,
        actualChange: data.actualChange,
        accuracy: data.accuracy,
      },
      create: data,
    })
  }

  /**
   * Bulk insert predictions
   */
  async createMany(data: Omit<Prediction, 'id' | 'createdAt'>[]) {
    return await prisma.prediction.createMany({
      data,
      skipDuplicates: true,
    })
  }

  /**
   * Find predictions with filters
   */
  async findMany(filter: PredictionFilter = {}, limit: number = 100) {
    const where: Prisma.PredictionWhereInput = {}

    if (filter.ticker) {
      where.ticker = filter.ticker.toUpperCase()
    }

    if (filter.timeframe) {
      where.timeframe = filter.timeframe
    }

    if (filter.modelName) {
      where.modelName = filter.modelName
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

    return await prisma.prediction.findMany({
      where,
      orderBy: { timestamp: 'desc' },
      take: limit,
    })
  }

  /**
   * Get latest prediction for a ticker/model
   */
  async getLatest(ticker: string, timeframe: string, modelName: string) {
    return await prisma.prediction.findFirst({
      where: {
        ticker: ticker.toUpperCase(),
        timeframe,
        modelName,
      },
      orderBy: { timestamp: 'desc' },
    })
  }

  /**
   * Update prediction with actual values (for backtesting)
   */
  async updateActuals(
    id: string,
    actualDirection: string,
    actualChange: number
  ) {
    const prediction = await prisma.prediction.findUnique({
      where: { id },
    })

    if (!prediction) {
      throw new Error('Prediction not found')
    }

    // Calculate accuracy (1.0 if direction matches, 0.0 if not)
    const accuracy =
      prediction.predictedDirection === actualDirection ? 1.0 : 0.0

    return await prisma.prediction.update({
      where: { id },
      data: {
        actualDirection,
        actualChange,
        accuracy,
      },
    })
  }

  /**
   * Get model accuracy statistics
   */
  async getModelAccuracy(
    modelName: string,
    ticker?: string,
    timeframe?: string,
    startDate?: Date,
    endDate?: Date
  ): Promise<PredictionAccuracy | null> {
    const where: Prisma.PredictionWhereInput = {
      modelName,
      actualDirection: { not: null }, // Only include predictions with actuals
    }

    if (ticker) {
      where.ticker = ticker.toUpperCase()
    }

    if (timeframe) {
      where.timeframe = timeframe
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

    const result = await prisma.prediction.aggregate({
      where,
      _count: { id: true },
      _avg: {
        accuracy: true,
        confidence: true,
      },
    })

    if (result._count.id === 0) {
      return null
    }

    const correctPredictions = await prisma.prediction.count({
      where: {
        ...where,
        accuracy: 1.0,
      },
    })

    return {
      modelName,
      totalPredictions: result._count.id,
      correctPredictions,
      accuracy: result._avg.accuracy || 0,
      avgConfidence: Number(result._avg.confidence) || 0,
    }
  }

  /**
   * Get all model accuracies
   */
  async getAllModelAccuracies(
    ticker?: string,
    timeframe?: string
  ): Promise<PredictionAccuracy[]> {
    // Get unique model names
    const models = await prisma.prediction.groupBy({
      by: ['modelName'],
      where: {
        ticker: ticker ? ticker.toUpperCase() : undefined,
        timeframe,
        actualDirection: { not: null },
      },
    })

    const accuracies = await Promise.all(
      models.map(m =>
        this.getModelAccuracy(m.modelName, ticker, timeframe)
      )
    )

    return accuracies.filter((a): a is PredictionAccuracy => a !== null)
  }

  /**
   * Get prediction time series (for charting)
   */
  async getPredictionTimeSeries(
    ticker: string,
    timeframe: string,
    modelName: string,
    startDate?: Date,
    endDate?: Date,
    limit: number = 1000
  ) {
    const where: Prisma.PredictionWhereInput = {
      ticker: ticker.toUpperCase(),
      timeframe,
      modelName,
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

    const data = await prisma.prediction.findMany({
      where,
      orderBy: { timestamp: 'asc' },
      take: limit,
      select: {
        timestamp: true,
        predictedDirection: true,
        predictedChange: true,
        confidence: true,
        actualDirection: true,
        actualChange: true,
        accuracy: true,
      },
    })

    return data.map(point => ({
      time: point.timestamp.getTime(),
      predictedDirection: point.predictedDirection,
      predictedChange: point.predictedChange ? Number(point.predictedChange) : null,
      confidence: point.confidence ? Number(point.confidence) : null,
      actualDirection: point.actualDirection,
      actualChange: point.actualChange ? Number(point.actualChange) : null,
      accuracy: point.accuracy ? Number(point.accuracy) : null,
    }))
  }

  /**
   * Get count of predictions
   */
  async count(filter: PredictionFilter = {}): Promise<number> {
    const where: Prisma.PredictionWhereInput = {}

    if (filter.ticker) {
      where.ticker = filter.ticker.toUpperCase()
    }

    if (filter.timeframe) {
      where.timeframe = filter.timeframe
    }

    if (filter.modelName) {
      where.modelName = filter.modelName
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

    return await prisma.prediction.count({ where })
  }

  /**
   * Delete old predictions (for cleanup)
   */
  async deleteOlderThan(date: Date) {
    return await prisma.prediction.deleteMany({
      where: {
        timestamp: {
          lt: date,
        },
      },
    })
  }
}

export const predictionsRepo = new PredictionsRepository()
