/**
 * ML Prediction Service
 *
 * Integrates with the Python ML server to get win probability predictions
 * for FVG patterns. Provides caching and fallback behavior.
 */

import { FvgMLFeatures } from './fvgFeatureService'

const ML_SERVER_URL = process.env.NEXT_PUBLIC_ML_SERVER_URL || 'https://project-genesis-6roa.onrender.com'

export interface MLPrediction {
  prediction: 'win' | 'loss'
  winProbability: number
  confidence: number
  modelAccuracy: number
}

export interface MLServerStatus {
  online: boolean
  modelLoaded: boolean
  metrics?: {
    accuracy: number
    precision: number
    recall: number
    f1: number
  }
}

class MLPredictionService {
  private serverOnline: boolean = false
  private lastHealthCheck: number = 0
  private healthCheckInterval: number = 30000 // 30 seconds

  /**
   * Check if ML server is available
   */
  async checkHealth(): Promise<MLServerStatus> {
    try {
      const response = await fetch(`${ML_SERVER_URL}/health`, {
        signal: AbortSignal.timeout(5000),
      })

      if (!response.ok) {
        this.serverOnline = false
        return { online: false, modelLoaded: false }
      }

      const health = await response.json()
      this.serverOnline = health.status === 'ok' && health.model_loaded
      this.lastHealthCheck = Date.now()

      // Get model info
      let metrics
      try {
        const infoResponse = await fetch(`${ML_SERVER_URL}/model_info`)
        if (infoResponse.ok) {
          const info = await infoResponse.json()
          metrics = info.metrics
        }
      } catch {
        // Ignore model info errors
      }

      return {
        online: this.serverOnline,
        modelLoaded: health.model_loaded,
        metrics,
      }
    } catch {
      this.serverOnline = false
      return { online: false, modelLoaded: false }
    }
  }

  /**
   * Check if server is available (with caching)
   */
  async isAvailable(): Promise<boolean> {
    if (Date.now() - this.lastHealthCheck > this.healthCheckInterval) {
      await this.checkHealth()
    }
    return this.serverOnline
  }

  /**
   * Get win probability prediction for FVG features
   */
  async predict(features: FvgMLFeatures): Promise<MLPrediction | null> {
    if (!(await this.isAvailable())) {
      console.log('[ML Service] Server not available, skipping prediction')
      return null
    }

    try {
      const response = await fetch(`${ML_SERVER_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(features),
        signal: AbortSignal.timeout(10000),
      })

      if (!response.ok) {
        console.error('[ML Service] Prediction failed:', await response.text())
        return null
      }

      const result = await response.json()

      return {
        prediction: result.prediction,
        winProbability: result.win_probability,
        confidence: result.confidence,
        modelAccuracy: result.model_accuracy,
      }
    } catch (error) {
      console.error('[ML Service] Prediction error:', error)
      return null
    }
  }

  /**
   * Get batch predictions for multiple FVGs
   */
  async batchPredict(
    fvgList: FvgMLFeatures[]
  ): Promise<Map<string, MLPrediction>> {
    const predictions = new Map<string, MLPrediction>()

    if (!(await this.isAvailable())) {
      console.log('[ML Service] Server not available, skipping batch prediction')
      return predictions
    }

    try {
      const response = await fetch(`${ML_SERVER_URL}/batch_predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fvgs: fvgList }),
        signal: AbortSignal.timeout(30000),
      })

      if (!response.ok) {
        console.error('[ML Service] Batch prediction failed:', await response.text())
        return predictions
      }

      const result = await response.json()

      for (const pred of result.predictions) {
        predictions.set(pred.fvg_id, {
          prediction: pred.prediction,
          winProbability: pred.win_probability,
          confidence: pred.confidence,
          modelAccuracy: result.model_accuracy,
        })
      }

      return predictions
    } catch (error) {
      console.error('[ML Service] Batch prediction error:', error)
      return predictions
    }
  }

  /**
   * Calculate confidence tier for display
   */
  getConfidenceTier(winProbability: number): {
    tier: 'high' | 'medium' | 'low'
    label: string
    color: string
  } {
    if (winProbability >= 0.75) {
      return { tier: 'high', label: 'High Confidence', color: '#22c55e' }
    } else if (winProbability >= 0.55) {
      return { tier: 'medium', label: 'Medium Confidence', color: '#eab308' }
    } else {
      return { tier: 'low', label: 'Low Confidence', color: '#ef4444' }
    }
  }

  /**
   * Get recommendation based on prediction
   */
  getRecommendation(prediction: MLPrediction): {
    action: 'take' | 'skip' | 'caution'
    reason: string
  } {
    if (prediction.winProbability >= 0.75) {
      return {
        action: 'take',
        reason: `Strong ${Math.round(prediction.winProbability * 100)}% win probability`,
      }
    } else if (prediction.winProbability >= 0.55) {
      return {
        action: 'caution',
        reason: `Moderate ${Math.round(prediction.winProbability * 100)}% win probability - consider additional confirmation`,
      }
    } else {
      return {
        action: 'skip',
        reason: `Weak ${Math.round(prediction.winProbability * 100)}% win probability - consider passing`,
      }
    }
  }
}

export const mlPredictionService = new MLPredictionService()
