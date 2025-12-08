/**
 * Hook to fetch ML predictions for FVG patterns
 * Calls the Railway-hosted ML prediction server
 */

import { useState, useEffect, useRef } from 'react'
import { FvgPattern } from './fvgDrawing'
import { CandleData } from './types'

const ML_SERVER_URL = process.env.NEXT_PUBLIC_ML_SERVER_URL || 'https://genesis-production-c1e9.up.railway.app'

interface MLPredictionResponse {
  prediction: 'win' | 'loss'
  win_probability: number
  confidence: number
  confidence_tier: 'very_high' | 'high' | 'medium' | 'low'
  recommendation: 'TRADE' | 'CAUTIOUS' | 'SKIP'
  model_accuracy: number
  ticker: string
}

/**
 * Build ML features from FVG pattern and surrounding candle data
 */
function buildMLFeatures(
  pattern: FvgPattern,
  data: CandleData[],
  ticker: string
): Record<string, any> {
  const idx = pattern.startIndex

  // Get surrounding candles for indicator calculation
  const candle1 = data[idx] || data[0]
  const candle2 = data[idx + 1] || candle1
  const candle3 = data[idx + 2] || candle2

  // Calculate basic indicators from recent data
  const recentData = data.slice(Math.max(0, idx - 20), idx + 3)

  // Simple moving averages
  const closes = recentData.map(c => c.close)
  const sma20 = closes.length >= 20
    ? closes.slice(-20).reduce((a, b) => a + b, 0) / 20
    : closes.reduce((a, b) => a + b, 0) / closes.length

  const sma50 = data.slice(Math.max(0, idx - 50), idx + 1)
  const sma50Val = sma50.length >= 20
    ? sma50.map(c => c.close).reduce((a, b) => a + b, 0) / sma50.length
    : sma20

  // ATR calculation (simplified)
  const trueRanges = recentData.slice(-14).map((c, i, arr) => {
    if (i === 0) return c.high - c.low
    const prev = arr[i - 1]
    return Math.max(
      c.high - c.low,
      Math.abs(c.high - prev.close),
      Math.abs(c.low - prev.close)
    )
  })
  const atr14 = trueRanges.reduce((a, b) => a + b, 0) / trueRanges.length

  // RSI calculation (simplified)
  const changes = closes.slice(-15).map((c, i, arr) => i === 0 ? 0 : c - arr[i - 1]).slice(1)
  const gains = changes.filter(c => c > 0)
  const losses = changes.filter(c => c < 0).map(c => Math.abs(c))
  const avgGain = gains.length > 0 ? gains.reduce((a, b) => a + b, 0) / 14 : 0
  const avgLoss = losses.length > 0 ? losses.reduce((a, b) => a + b, 0) / 14 : 0.001
  const rs = avgGain / avgLoss
  const rsi14 = 100 - (100 / (1 + rs))

  // Volume ratio
  const volumes = recentData.map(c => c.volume)
  const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length
  const volumeRatio = candle2.volume / (avgVolume || 1)

  // MACD (simplified)
  const ema12 = closes.slice(-12).reduce((a, b) => a + b, 0) / Math.min(12, closes.length)
  const ema26 = closes.slice(-26).reduce((a, b) => a + b, 0) / Math.min(26, closes.length)
  const macd = ema12 - ema26
  const macdSignal = macd * 0.9 // Simplified
  const macdHistogram = macd - macdSignal

  // Bollinger Bands (simplified)
  const stdDev = Math.sqrt(
    closes.slice(-20).reduce((sum, c) => sum + Math.pow(c - sma20, 2), 0) / Math.min(20, closes.length)
  )
  const bbBandwidth = (stdDev * 4) / sma20 * 100

  // Price vs SMA
  const currentPrice = candle3.close
  const priceVsSma20 = ((currentPrice - sma20) / sma20) * 100
  const priceVsSma50 = ((currentPrice - sma50Val) / sma50Val) * 100

  // Time features
  const timestamp = candle3.time
  const date = new Date(timestamp)
  const hourOfDay = date.getUTCHours()
  const dayOfWeek = date.getUTCDay()

  // RSI zone
  const rsiZone = rsi14 < 30 ? 'oversold' : rsi14 > 70 ? 'overbought' : 'neutral'

  // MACD trend
  const macdTrend = macdHistogram > 0.1 ? 'bullish' : macdHistogram < -0.1 ? 'bearish' : 'neutral'

  // Volatility regime
  const atrPct = (atr14 / currentPrice) * 100
  const volatilityRegime = atrPct < 0.5 ? 'low' : atrPct > 1.5 ? 'high' : 'medium'

  return {
    ticker,
    fvg_type: pattern.type,
    gap_size_pct: pattern.gapSizePct,
    validation_score: pattern.validationScore,
    volume_profile: pattern.volumeProfile,
    market_structure: pattern.marketStructure,
    rsi_14: rsi14,
    macd: macd,
    macd_signal: macdSignal,
    macd_histogram: macdHistogram,
    atr_14: atr14,
    sma_20: sma20,
    sma_50: sma50Val,
    ema_12: ema12,
    ema_26: ema26,
    bb_bandwidth: bbBandwidth,
    volume_ratio: volumeRatio,
    price_vs_sma20: priceVsSma20,
    price_vs_sma50: priceVsSma50,
    hour_of_day: hourOfDay,
    day_of_week: dayOfWeek,
    rsi_zone: rsiZone,
    macd_trend: macdTrend,
    volatility_regime: volatilityRegime,
  }
}

/**
 * Hook to fetch and attach ML predictions to FVG patterns
 */
export function useFvgMLPredictions(
  patterns: FvgPattern[],
  data: CandleData[],
  ticker: string,
  enabled: boolean = true
): FvgPattern[] {
  const [predictedPatterns, setPredictedPatterns] = useState<FvgPattern[]>(patterns)
  const fetchedIdsRef = useRef<Set<string>>(new Set())
  const abortControllerRef = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!enabled || patterns.length === 0 || data.length === 0) {
      setPredictedPatterns(patterns)
      return
    }

    // Find patterns that need predictions
    const patternsNeedingPrediction = patterns.filter(
      p => p.id && !fetchedIdsRef.current.has(p.id) && !p.mlPrediction
    )

    if (patternsNeedingPrediction.length === 0) {
      // Just update with existing predictions
      setPredictedPatterns(patterns.map(p => {
        const existing = predictedPatterns.find(ep => ep.id === p.id)
        return existing?.mlPrediction ? { ...p, mlPrediction: existing.mlPrediction } : p
      }))
      return
    }

    // Cancel previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    abortControllerRef.current = new AbortController()

    const fetchPredictions = async () => {
      try {
        // Batch predict for all patterns needing predictions
        const fvgsWithFeatures = patternsNeedingPrediction.map(pattern => ({
          fvg_id: pattern.id,
          ...buildMLFeatures(pattern, data, ticker)
        }))

        const response = await fetch(`${ML_SERVER_URL}/batch_predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fvgs: fvgsWithFeatures, ticker }),
          signal: abortControllerRef.current?.signal,
        })

        if (!response.ok) {
          console.warn('[ML Predictions] Server returned error:', response.status)
          return
        }

        const result = await response.json()
        const predictions = result.predictions || []

        // Create a map of predictions by fvg_id
        const predictionMap = new Map<string, MLPredictionResponse>()
        for (const pred of predictions) {
          if (pred.fvg_id) {
            predictionMap.set(pred.fvg_id, pred)
            fetchedIdsRef.current.add(pred.fvg_id)
          }
        }

        // Merge predictions into patterns
        setPredictedPatterns(patterns.map(pattern => {
          const pred = pattern.id ? predictionMap.get(pattern.id) : null
          if (pred) {
            return {
              ...pattern,
              mlPrediction: {
                winProbability: pred.win_probability,
                recommendation: pred.recommendation,
                confidenceTier: pred.confidence_tier,
                modelAccuracy: pred.model_accuracy || 0.72,
              }
            }
          }
          // Keep existing prediction if available
          const existing = predictedPatterns.find(ep => ep.id === pattern.id)
          return existing?.mlPrediction ? { ...pattern, mlPrediction: existing.mlPrediction } : pattern
        }))

        console.log(`[ML Predictions] Fetched ${predictions.length} predictions for ${patternsNeedingPrediction.length} FVGs`)
      } catch (error: any) {
        if (error.name !== 'AbortError') {
          console.warn('[ML Predictions] Failed to fetch:', error.message)
        }
      }
    }

    // Debounce the fetch
    const timeoutId = setTimeout(fetchPredictions, 300)
    return () => clearTimeout(timeoutId)
  }, [patterns, data, ticker, enabled])

  return predictedPatterns
}
