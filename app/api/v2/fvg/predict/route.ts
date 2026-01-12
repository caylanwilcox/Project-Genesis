/**
 * FVG Prediction API Endpoint
 * POST /api/v2/fvg/predict
 *
 * Uses the trained XGBoost model to predict win probability for FVG patterns.
 * Requires the Python prediction server running on port 5001.
 */

import { NextRequest, NextResponse } from 'next/server'
import { fvgFeatureService } from '@/services/fvgFeatureService'
import { fvgDetectionRepo } from '@/repositories'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

const ML_SERVER_URL = process.env.ML_SERVER_URL || 'https://project-genesis-6roa.onrender.com'

interface PredictionResult {
  fvg_id: string
  prediction: 'win' | 'loss'
  win_probability: number
  confidence: number
}

async function getPrediction(features: Record<string, unknown>): Promise<PredictionResult | null> {
  try {
    const response = await fetch(`${ML_SERVER_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(features),
    })

    if (!response.ok) {
      console.error('[FVG Predict] ML server error:', await response.text())
      return null
    }

    const result = await response.json()
    return {
      fvg_id: features.fvg_id as string,
      prediction: result.prediction,
      win_probability: result.win_probability,
      confidence: result.confidence,
    }
  } catch (error) {
    console.error('[FVG Predict] Failed to get prediction:', error)
    return null
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { fvg_id, ticker, timeframe, features } = body

    // Option 1: Predict for a specific FVG by ID
    if (fvg_id) {
      const fvg = await fvgDetectionRepo.findById(fvg_id)
      if (!fvg) {
        return NextResponse.json(
          { error: 'FVG not found' },
          { status: 404 }
        )
      }

      const fvgFeatures = await fvgFeatureService.getFeaturesForFvg(fvg)
      if (!fvgFeatures) {
        return NextResponse.json(
          { error: 'Could not generate features for FVG' },
          { status: 400 }
        )
      }

      const prediction = await getPrediction(fvgFeatures as unknown as Record<string, unknown>)
      if (!prediction) {
        return NextResponse.json(
          { error: 'ML server unavailable. Start with: cd ml && python3 predict_server.py' },
          { status: 503 }
        )
      }

      return NextResponse.json({
        success: true,
        prediction,
        features: fvgFeatures,
      })
    }

    // Option 2: Predict with provided features directly
    if (features) {
      const prediction = await getPrediction(features)
      if (!prediction) {
        return NextResponse.json(
          { error: 'ML server unavailable. Start with: cd ml && python3 predict_server.py' },
          { status: 503 }
        )
      }

      return NextResponse.json({
        success: true,
        prediction,
      })
    }

    // Option 3: Predict for all pending FVGs for a ticker
    if (ticker && timeframe) {
      const fvgs = await fvgDetectionRepo.findMany(
        { ticker: ticker.toUpperCase(), timeframe, finalOutcome: null },
        100
      )

      if (fvgs.length === 0) {
        return NextResponse.json({
          success: true,
          message: 'No pending FVGs found',
          predictions: [],
        })
      }

      const predictions: PredictionResult[] = []

      for (const fvg of fvgs) {
        const fvgFeatures = await fvgFeatureService.getFeaturesForFvg(fvg)
        if (fvgFeatures) {
          const prediction = await getPrediction(fvgFeatures as unknown as Record<string, unknown>)
          if (prediction) {
            predictions.push(prediction)
          }
        }
      }

      return NextResponse.json({
        success: true,
        message: `Generated ${predictions.length} predictions`,
        predictions,
      })
    }

    return NextResponse.json(
      { error: 'Missing required fields: fvg_id, features, or (ticker + timeframe)' },
      { status: 400 }
    )
  } catch (error) {
    console.error('FVG prediction error:', error)
    return NextResponse.json(
      {
        error: 'Failed to generate prediction',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}

// Health check for ML server
export async function GET() {
  try {
    const response = await fetch(`${ML_SERVER_URL}/health`)

    if (!response.ok) {
      return NextResponse.json({
        ml_server: 'error',
        message: 'ML server returned error',
      })
    }

    const health = await response.json()

    // Also get model info
    const infoResponse = await fetch(`${ML_SERVER_URL}/model_info`)
    const modelInfo = infoResponse.ok ? await infoResponse.json() : null

    return NextResponse.json({
      ml_server: 'online',
      model_loaded: health.model_loaded,
      model_metrics: modelInfo?.metrics || null,
      feature_count: modelInfo?.feature_cols?.length || 0,
    })
  } catch {
    return NextResponse.json({
      ml_server: 'offline',
      message: 'Start ML server with: cd ml && python3 predict_server.py',
    })
  }
}
