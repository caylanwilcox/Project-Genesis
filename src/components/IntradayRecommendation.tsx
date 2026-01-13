'use client'

import { useState, useEffect } from 'react'

interface IntradayData {
  action: string
  probability_a: number
  probability_b: number
  session: string
  confidence: number
  current_price: number
  price_11am?: number
  reason?: string
}

interface IntradayRecommendationProps {
  symbol: string
}

export function IntradayRecommendation({ symbol }: IntradayRecommendationProps) {
  const [data, setData] = useState<IntradayData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [marketClosed, setMarketClosed] = useState(false)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const response = await fetch(`/api/v2/trading-directions`)
        if (!response.ok) throw new Error('Failed to fetch')

        const result = await response.json()

        // Check if market is closed
        if (result.market_open === false) {
          setMarketClosed(true)
          setData(null)
          setError(null)
          return
        }

        setMarketClosed(false)
        const tickerData = result.tickers?.[symbol]

        if (tickerData) {
          setData({
            action: tickerData.action || 'NO_TRADE',
            probability_a: tickerData.probability_a || 0.5,
            probability_b: tickerData.probability_b || 0.5,
            session: tickerData.session || 'early',
            confidence: tickerData.confidence || 0,
            current_price: tickerData.current_price || 0,
            price_11am: tickerData.price_11am,
            reason: tickerData.reason
          })
          setError(null)
        } else {
          setError('No data available')
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 60000)
    return () => clearInterval(interval)
  }, [symbol])

  if (loading) {
    return (
      <div className="rounded-xl border border-gray-700 bg-gray-900/50 p-4">
        <div className="flex items-center gap-2 text-gray-400">
          <div className="animate-spin h-4 w-4 border-2 border-cyan-400 border-t-transparent rounded-full" />
          <span className="text-sm">Loading {symbol}...</span>
        </div>
      </div>
    )
  }

  // Market closed state
  if (marketClosed) {
    return (
      <div className="rounded-xl border border-gray-700 bg-gray-900/50 p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="font-bold text-white">{symbol}</span>
          <span className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-400">
            MARKET CLOSED
          </span>
        </div>
        <div className="text-gray-500 text-sm">
          Signals available during market hours (9:30 AM - 4:00 PM ET)
        </div>
        <div className="mt-3 pt-3 border-t border-gray-700">
          <div className="text-xs text-gray-600">
            V6 Intraday predictions update every minute during trading
          </div>
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="rounded-xl border border-gray-700 bg-gray-900/50 p-4">
        <div className="flex items-center gap-2 text-gray-400">
          <span className="text-yellow-500">⚠</span>
          <span className="text-sm">Data unavailable</span>
        </div>
      </div>
    )
  }

  // Determine recommendation based on V6 model output
  const activeProb = data.session === 'late' ? data.probability_b : data.probability_a
  const targetLabel = data.session === 'late' ? 'Close > 11AM' : 'Close > Open'

  let recommendation: {
    action: string
    color: string
    bgColor: string
    borderColor: string
    description: string
  }

  if (activeProb >= 0.85) {
    recommendation = {
      action: 'STRONG BUY',
      color: 'text-green-400',
      bgColor: 'bg-green-900/30',
      borderColor: 'border-green-500/50',
      description: 'High conviction bullish signal'
    }
  } else if (activeProb >= 0.75) {
    recommendation = {
      action: 'BUY CALL',
      color: 'text-green-400',
      bgColor: 'bg-green-900/20',
      borderColor: 'border-green-500/40',
      description: 'Bullish signal - expect higher close'
    }
  } else if (activeProb <= 0.15) {
    recommendation = {
      action: 'STRONG SELL',
      color: 'text-red-400',
      bgColor: 'bg-red-900/30',
      borderColor: 'border-red-500/50',
      description: 'High conviction bearish signal'
    }
  } else if (activeProb <= 0.25) {
    recommendation = {
      action: 'BUY PUT',
      color: 'text-red-400',
      bgColor: 'bg-red-900/20',
      borderColor: 'border-red-500/40',
      description: 'Bearish signal - expect lower close'
    }
  } else {
    recommendation = {
      action: 'NO TRADE',
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-900/10',
      borderColor: 'border-yellow-500/30',
      description: 'Neutral zone - wait for clearer signal'
    }
  }

  // Historical win rates by session
  const winRates: Record<string, Record<string, number>> = {
    SPY: { early: 0.726, late_a: 0.899, late_b: 0.817 },
    QQQ: { early: 0.754, late_a: 0.921, late_b: 0.802 },
    IWM: { early: 0.750, late_a: 0.911, late_b: 0.810 }
  }

  const tickerWinRates = winRates[symbol] || winRates.SPY
  const currentWinRate = data.session === 'late' ? tickerWinRates.late_b : tickerWinRates.early

  return (
    <div className={`rounded-xl border p-4 transition-all ${recommendation.bgColor} ${recommendation.borderColor}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className={`px-3 py-1.5 rounded-lg font-bold text-sm ${
            recommendation.action.includes('BUY') && !recommendation.action.includes('PUT')
              ? 'bg-green-500 text-white'
              : recommendation.action.includes('PUT') || recommendation.action.includes('SELL')
              ? 'bg-red-500 text-white'
              : 'bg-yellow-500 text-black'
          }`}>
            {recommendation.action}
          </span>
          {(activeProb >= 0.85 || activeProb <= 0.15) && (
            <span className="text-xs px-2 py-0.5 rounded bg-white/10 text-white font-medium">
              HIGH CONVICTION
            </span>
          )}
        </div>
        <span className="text-gray-400 text-sm">${data.current_price.toFixed(2)}</span>
      </div>

      {/* Probability Display */}
      <div className="flex items-center gap-4 mb-3">
        <div className="flex-1">
          <div className={`text-3xl font-bold ${recommendation.color}`}>
            {Math.round(activeProb * 100)}%
          </div>
          <div className="text-gray-500 text-xs">{targetLabel}</div>
        </div>
        <div className="text-right">
          <div className="text-gray-400 text-sm">Win Rate</div>
          <div className="text-cyan-400 font-bold">{Math.round(currentWinRate * 100)}%</div>
        </div>
      </div>

      {/* Session Info */}
      <div className="bg-gray-800/50 rounded-lg p-2 mb-3">
        <div className="flex justify-between text-xs">
          <span className="text-gray-500">Session</span>
          <span className={data.session === 'late' ? 'text-emerald-400' : 'text-amber-400'}>
            {data.session === 'late' ? '⏰ Late (Peak Accuracy)' : '🌅 Early Session'}
          </span>
        </div>
        {data.session === 'late' && data.price_11am && (
          <div className="flex justify-between text-xs mt-1">
            <span className="text-gray-500">11 AM Price</span>
            <span className="text-gray-300">${data.price_11am.toFixed(2)}</span>
          </div>
        )}
      </div>

      {/* Reasoning */}
      <div className="text-xs text-gray-400">
        <span className={recommendation.color}>{recommendation.description}</span>
        {data.reason && (
          <span className="block mt-1 text-gray-500">{data.reason}</span>
        )}
      </div>

      {/* Both Probabilities */}
      <div className="grid grid-cols-2 gap-2 mt-3 pt-3 border-t border-gray-700">
        <div className="text-center">
          <div className="text-gray-500 text-xs">Target A</div>
          <div className={`font-bold ${
            data.probability_a >= 0.75 ? 'text-green-400' :
            data.probability_a <= 0.25 ? 'text-red-400' : 'text-yellow-400'
          }`}>
            {Math.round(data.probability_a * 100)}%
          </div>
          <div className="text-gray-600 text-xs">Close {'>'} Open</div>
        </div>
        <div className="text-center">
          <div className="text-gray-500 text-xs">Target B</div>
          <div className={`font-bold ${
            data.probability_b >= 0.75 ? 'text-green-400' :
            data.probability_b <= 0.25 ? 'text-red-400' : 'text-yellow-400'
          }`}>
            {Math.round(data.probability_b * 100)}%
          </div>
          <div className="text-gray-600 text-xs">Close {'>'} 11AM</div>
        </div>
      </div>
    </div>
  )
}
