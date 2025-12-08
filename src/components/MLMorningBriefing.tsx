'use client'

import { useState, useEffect } from 'react'

const ML_SERVER_URL = process.env.NEXT_PUBLIC_ML_SERVER_URL || 'https://genesis-production-c1e9.up.railway.app'

interface TickerPrediction {
  direction: string
  emoji: string
  bullish_probability: number
  confidence: number
  fvg_recommendation: string
  current_price: number
  predicted_range: {
    low: number
    high: number
  }
  model_accuracy: number
  error?: string
}

interface MorningBriefing {
  generated_at: string
  market_day: string
  tickers: Record<string, TickerPrediction>
  overall_bias: string
  overall_emoji: string
  best_opportunity: {
    ticker: string
    confidence: number
    direction: string
  } | null
}

export function MLMorningBriefing() {
  const [briefing, setBriefing] = useState<MorningBriefing | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchBriefing = async () => {
      try {
        setLoading(true)
        const response = await fetch(`${ML_SERVER_URL}/morning_briefing`)

        if (!response.ok) {
          throw new Error('Failed to fetch briefing')
        }

        const data = await response.json()
        setBriefing(data)
        setError(null)
      } catch (err: any) {
        setError(err.message || 'Failed to load ML briefing')
      } finally {
        setLoading(false)
      }
    }

    fetchBriefing()

    // Refresh every 5 minutes
    const interval = setInterval(fetchBriefing, 5 * 60 * 1000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-700 rounded w-48 mb-4"></div>
          <div className="space-y-3">
            <div className="h-4 bg-gray-700 rounded w-full"></div>
            <div className="h-4 bg-gray-700 rounded w-3/4"></div>
          </div>
        </div>
      </div>
    )
  }

  if (error || !briefing) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="text-red-400 text-sm">
          ⚠️ {error || 'Unable to load ML predictions'}
        </div>
      </div>
    )
  }

  const getBiasColor = (bias: string) => {
    if (bias === 'BULLISH') return 'text-green-400'
    if (bias === 'BEARISH') return 'text-red-400'
    return 'text-yellow-400'
  }

  const getBiasBg = (bias: string) => {
    if (bias === 'BULLISH') return 'from-green-900/30 to-transparent'
    if (bias === 'BEARISH') return 'from-red-900/30 to-transparent'
    return 'from-yellow-900/30 to-transparent'
  }

  return (
    <div className={`bg-gradient-to-br ${getBiasBg(briefing.overall_bias)} bg-gray-900 border border-gray-800 rounded-lg p-4`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-white font-bold text-lg flex items-center gap-2">
            🤖 ML Daily Outlook
          </h3>
          <p className="text-gray-500 text-xs">{briefing.market_day}</p>
        </div>
        <div className="text-right">
          <div className={`text-2xl font-bold ${getBiasColor(briefing.overall_bias)}`}>
            {briefing.overall_emoji} {briefing.overall_bias}
          </div>
          <p className="text-gray-500 text-xs">Market Bias</p>
        </div>
      </div>

      {/* Best Opportunity */}
      {briefing.best_opportunity && (
        <div className="bg-gray-800/50 rounded-lg p-3 mb-4 border border-gray-700">
          <div className="text-gray-400 text-xs mb-1">🎯 BEST OPPORTUNITY</div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-white font-bold text-lg">
                {briefing.best_opportunity.ticker}
              </span>
              <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                briefing.best_opportunity.direction === 'BULLISH'
                  ? 'bg-green-500/20 text-green-400'
                  : briefing.best_opportunity.direction === 'BEARISH'
                  ? 'bg-red-500/20 text-red-400'
                  : 'bg-yellow-500/20 text-yellow-400'
              }`}>
                {briefing.best_opportunity.direction}
              </span>
            </div>
            <div className="text-right">
              <div className="text-white font-bold">
                {(briefing.best_opportunity.confidence * 100).toFixed(0)}%
              </div>
              <div className="text-gray-500 text-xs">confidence</div>
            </div>
          </div>
        </div>
      )}

      {/* Ticker Predictions */}
      <div className="space-y-2">
        {Object.entries(briefing.tickers).map(([ticker, pred]) => {
          if (pred.error) {
            return (
              <div key={ticker} className="flex items-center justify-between py-2 border-b border-gray-800 last:border-0">
                <span className="text-gray-400">{ticker}</span>
                <span className="text-red-400 text-xs">{pred.error}</span>
              </div>
            )
          }

          return (
            <div key={ticker} className="flex items-center justify-between py-2 border-b border-gray-800 last:border-0">
              <div className="flex items-center gap-3">
                <span className="text-lg">{pred.emoji}</span>
                <div>
                  <div className="text-white font-medium">{ticker}</div>
                  <div className="text-gray-500 text-xs">
                    ${pred.current_price} | Range: ${pred.predicted_range.low}-${pred.predicted_range.high}
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className={`font-bold ${
                  pred.direction === 'BULLISH' ? 'text-green-400' :
                  pred.direction === 'BEARISH' ? 'text-red-400' :
                  'text-yellow-400'
                }`}>
                  {(pred.bullish_probability * 100).toFixed(0)}% Bull
                </div>
                <div className="text-gray-500 text-xs">
                  Trade: {pred.fvg_recommendation} FVGs
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Footer */}
      <div className="mt-4 pt-3 border-t border-gray-800 flex items-center justify-between">
        <div className="text-gray-500 text-xs">
          Model Accuracy: ~68% | High-Conf: ~74%
        </div>
        <div className="text-gray-600 text-xs">
          Updated: {new Date(briefing.generated_at).toLocaleTimeString()}
        </div>
      </div>
    </div>
  )
}
