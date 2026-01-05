import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

interface MLPrediction {
  date: string
  symbol: string
  vwap: number
  unit: number
  levels: {
    long: { t1: number; t2: number; t3: number; stop_loss: number }
    short: { t1: number; t2: number; t3: number; stop_loss: number }
  }
  probabilities: {
    long: { touch_t1: number; touch_t2: number; touch_t3: number; touch_sl: number }
    short: { touch_t1: number; touch_t2: number; touch_t3: number; touch_sl: number }
  }
  first_touch_probs: Record<string, number>
  recommendation: {
    bias: 'LONG' | 'SHORT' | 'NEUTRAL'
    confidence: number
    rationale: string
  }
}

// Cache predictions for 5 minutes
let cachedPrediction: MLPrediction | null = null
let cacheTime: number = 0
const CACHE_TTL = 5 * 60 * 1000 // 5 minutes

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const symbol = searchParams.get('symbol') || 'SPY'
  const currentPrice = parseFloat(searchParams.get('price') || '0')

  // Check cache
  if (cachedPrediction && Date.now() - cacheTime < CACHE_TTL) {
    // Update levels based on current price if provided
    if (currentPrice > 0) {
      const updatedPrediction = updateLevelsWithPrice(cachedPrediction, currentPrice)
      return NextResponse.json(updatedPrediction)
    }
    return NextResponse.json(cachedPrediction)
  }

  try {
    // Run Python inference script
    const mlPipelinePath = path.join(process.cwd(), 'ml_pipeline')

    const prediction = await runPythonInference(mlPipelinePath, symbol, currentPrice)

    // Cache the result
    cachedPrediction = prediction
    cacheTime = Date.now()

    return NextResponse.json(prediction)
  } catch (error) {
    console.error('[ML-Prediction] Error:', error)

    // Return default prediction if ML fails
    const fallbackPrediction = generateFallbackPrediction(symbol, currentPrice)
    return NextResponse.json(fallbackPrediction)
  }
}

function runPythonInference(mlPath: string, symbol: string, price: number): Promise<MLPrediction> {
  return new Promise((resolve, reject) => {
    const pythonScript = `
import sys
sys.path.insert(0, '${mlPath}')
import json
import os
import pandas as pd
from datetime import datetime

# Set up paths
os.chdir('${mlPath}')

try:
    from inference import PlanGenerator
    from config import ALL_FEATURES
    import numpy as np

    # Load latest features if available
    features_path = 'data/features.parquet'
    if os.path.exists(features_path):
        features_df = pd.read_parquet(features_path)
        latest_features = features_df.iloc[-1].to_dict()
    else:
        # Use mock features
        np.random.seed(42)
        latest_features = {feat: np.random.random() * 2 - 1 for feat in ALL_FEATURES}
        latest_features['atr_14'] = ${price > 0 ? price * 0.01 : 3.5}
        latest_features['rsi_14'] = 55
        latest_features['day_of_week'] = datetime.now().weekday()
        latest_features['month'] = datetime.now().month

    # Load labels to get VWAP/unit if available
    labels_path = 'data/labels.parquet'
    if os.path.exists(labels_path):
        labels_df = pd.read_parquet(labels_path)
        latest_labels = labels_df.iloc[-1]
        vwap = float(latest_labels.get('vwap', ${price > 0 ? price : 475}))
        unit = float(latest_labels.get('unit', ${price > 0 ? price * 0.007 : 3.5}))
    else:
        vwap = ${price > 0 ? price : 475}
        unit = ${price > 0 ? price * 0.007 : 3.5}

    # Generate prediction
    generator = PlanGenerator(model_dir='models')
    plan = generator.generate_plan(
        features=latest_features,
        vwap=vwap,
        unit=unit,
        date=datetime.now().strftime('%Y-%m-%d'),
        symbol='${symbol}'
    )

    print(plan.to_json())
except Exception as e:
    import traceback
    # Return error as JSON
    error_result = {
        "error": str(e),
        "traceback": traceback.format_exc()
    }
    print(json.dumps(error_result))
`

    const python = spawn('python3', ['-c', pythonScript], {
      env: { ...process.env },
      cwd: mlPath
    })

    let stdout = ''
    let stderr = ''

    python.stdout.on('data', (data) => {
      stdout += data.toString()
    })

    python.stderr.on('data', (data) => {
      stderr += data.toString()
    })

    python.on('close', (code) => {
      if (code !== 0) {
        console.error('[ML-Prediction] Python error:', stderr)
        reject(new Error(`Python exited with code ${code}: ${stderr}`))
        return
      }

      try {
        const result = JSON.parse(stdout.trim())
        if (result.error) {
          console.error('[ML-Prediction] Python error:', result.error)
          reject(new Error(result.error))
          return
        }
        resolve(result as MLPrediction)
      } catch (e) {
        console.error('[ML-Prediction] Failed to parse output:', stdout)
        reject(new Error('Failed to parse ML output'))
      }
    })

    python.on('error', (err) => {
      reject(err)
    })

    // Timeout after 10 seconds
    setTimeout(() => {
      python.kill()
      reject(new Error('ML inference timeout'))
    }, 10000)
  })
}

function updateLevelsWithPrice(prediction: MLPrediction, currentPrice: number): MLPrediction {
  // Recalculate levels using current price as VWAP
  const unit = currentPrice * 0.007 // ~0.7% of price as unit

  return {
    ...prediction,
    vwap: currentPrice,
    unit,
    levels: {
      long: {
        t1: currentPrice + 0.5 * unit,
        t2: currentPrice + 1.0 * unit,
        t3: currentPrice + 1.5 * unit,
        stop_loss: currentPrice - 1.25 * unit,
      },
      short: {
        t1: currentPrice - 0.5 * unit,
        t2: currentPrice - 1.0 * unit,
        t3: currentPrice - 1.5 * unit,
        stop_loss: currentPrice + 1.25 * unit,
      }
    }
  }
}

function generateFallbackPrediction(symbol: string, price: number): MLPrediction {
  const vwap = price > 0 ? price : 475
  const unit = vwap * 0.007

  return {
    date: new Date().toISOString().split('T')[0],
    symbol,
    vwap,
    unit,
    levels: {
      long: {
        t1: vwap + 0.5 * unit,
        t2: vwap + 1.0 * unit,
        t3: vwap + 1.5 * unit,
        stop_loss: vwap - 1.25 * unit,
      },
      short: {
        t1: vwap - 0.5 * unit,
        t2: vwap - 1.0 * unit,
        t3: vwap - 1.5 * unit,
        stop_loss: vwap + 1.25 * unit,
      }
    },
    probabilities: {
      long: { touch_t1: 0.5, touch_t2: 0.3, touch_t3: 0.15, touch_sl: 0.2 },
      short: { touch_t1: 0.5, touch_t2: 0.3, touch_t3: 0.15, touch_sl: 0.2 }
    },
    first_touch_probs: {
      t1_long: 0.25,
      t1_short: 0.25,
      none: 0.5
    },
    recommendation: {
      bias: 'NEUTRAL',
      confidence: 0.5,
      rationale: 'Using fallback prediction - ML models not available'
    }
  }
}
