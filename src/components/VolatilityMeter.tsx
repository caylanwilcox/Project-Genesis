'use client'

import { useState, useEffect } from 'react'

const ML_SERVER_URL = process.env.NEXT_PUBLIC_ML_SERVER_URL || 'http://localhost:5001'

interface TickerVolatility {
  regime: 'LOW' | 'NORMAL' | 'HIGH'
  regime_label: string
  regime_color: string
  volatility_score: number
  volatility_percentile: number
  current_atr_pct: number
  current_daily_vol: number
  expected_range: string
  regime_model_stats: {
    direction_accuracy: number
    high_conf_accuracy: number
    high_mae: number
    low_mae: number
  } | null
  error?: string
}

interface VolatilityData {
  date: string
  generated_at: string
  market_volatility: 'LOW' | 'NORMAL' | 'HIGH'
  market_volatility_score: number
  trading_guidance: string
  tickers: Record<string, TickerVolatility>
}

export function VolatilityMeter() {
  const [data, setData] = useState<VolatilityData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    const fetchVolatility = async () => {
      try {
        setLoading(true)
        const response = await fetch(`${ML_SERVER_URL}/volatility_meter`)

        if (!response.ok) {
          throw new Error('Failed to fetch volatility data')
        }

        const result = await response.json()
        setData(result)
        setError(null)
      } catch (err: any) {
        setError(err.message || 'Failed to load volatility meter')
      } finally {
        setLoading(false)
      }
    }

    fetchVolatility()

    // Refresh every 5 minutes
    const interval = setInterval(fetchVolatility, 5 * 60 * 1000)
    return () => clearInterval(interval)
  }, [])

  const getRegimeColor = (regime: string) => {
    if (regime === 'LOW') return 'text-green-400'
    if (regime === 'HIGH') return 'text-red-400'
    return 'text-yellow-400'
  }

  const getRegimeBg = (regime: string) => {
    if (regime === 'LOW') return 'bg-green-500/20 border-green-500/30'
    if (regime === 'HIGH') return 'bg-red-500/20 border-red-500/30'
    return 'bg-yellow-500/20 border-yellow-500/30'
  }

  const getRegimeGradient = (regime: string) => {
    if (regime === 'LOW') return 'from-green-500'
    if (regime === 'HIGH') return 'from-red-500'
    return 'from-yellow-500'
  }

  const getRegimeIcon = (regime: string) => {
    if (regime === 'LOW') return '🟢'
    if (regime === 'HIGH') return '🔴'
    return '🟡'
  }

  if (loading) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="animate-pulse">
          <div className="h-5 bg-gray-700 rounded w-40 mb-3"></div>
          <div className="h-8 bg-gray-700 rounded w-full"></div>
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="text-red-400 text-sm">
          {error || 'Unable to load volatility meter'}
        </div>
      </div>
    )
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      {/* Header */}
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          <h3 className="text-white font-bold">Volatility Meter</h3>
          <span className="text-gray-500 text-xs">{data.date}</span>
        </div>
        <div className="flex items-center gap-3">
          <div className={`px-3 py-1 rounded-full text-sm font-bold ${getRegimeBg(data.market_volatility)} ${getRegimeColor(data.market_volatility)}`}>
            {getRegimeIcon(data.market_volatility)} {data.market_volatility}
          </div>
          <span className="text-gray-500">{expanded ? '▲' : '▼'}</span>
        </div>
      </div>

      {/* Market Score Bar */}
      <div className="mt-3">
        <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
          <span>Low Vol</span>
          <span>Normal</span>
          <span>High Vol</span>
        </div>
        <div className="relative h-3 bg-gray-800 rounded-full overflow-hidden">
          {/* Gradient background */}
          <div className="absolute inset-0 bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 opacity-30"></div>
          {/* Marker */}
          <div
            className="absolute top-0 bottom-0 w-1 bg-white shadow-lg transition-all duration-500"
            style={{ left: `${(data.market_volatility_score || 0.5) * 100}%` }}
          />
          {/* Threshold markers */}
          <div className="absolute top-0 bottom-0 w-px bg-gray-600" style={{ left: '30%' }} />
          <div className="absolute top-0 bottom-0 w-px bg-gray-600" style={{ left: '70%' }} />
        </div>
        <div className="text-center mt-1">
          <span className="text-gray-400 text-xs">
            Market Score: {((data.market_volatility_score || 0.5) * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Expanded Details */}
      {expanded && (
        <div className="mt-4 space-y-4">
          {/* Trading Guidance */}
          <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
            <div className="text-gray-400 text-xs mb-1">TRADING GUIDANCE</div>
            <p className="text-gray-300 text-sm">{data.trading_guidance}</p>
          </div>

          {/* Per-Ticker Volatility */}
          <div className="space-y-2">
            {Object.entries(data.tickers).map(([ticker, vol]) => {
              if (vol.error) {
                return (
                  <div key={ticker} className="flex items-center justify-between py-2">
                    <span className="text-gray-400">{ticker}</span>
                    <span className="text-red-400 text-xs">{vol.error}</span>
                  </div>
                )
              }

              return (
                <div
                  key={ticker}
                  className={`rounded-lg p-3 border ${getRegimeBg(vol.regime)}`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-white font-bold text-lg">{ticker}</span>
                      <span className={`text-sm font-medium ${getRegimeColor(vol.regime)}`}>
                        {getRegimeIcon(vol.regime)} {vol.regime_label}
                      </span>
                    </div>
                    <div className="text-right">
                      <div className={`font-bold ${getRegimeColor(vol.regime)}`}>
                        {vol.volatility_percentile?.toFixed(1) || '50.0'}%
                      </div>
                      <div className="text-gray-500 text-xs">percentile</div>
                    </div>
                  </div>

                  {/* Volatility bar */}
                  <div className="h-2 bg-gray-700 rounded-full overflow-hidden mb-2">
                    <div
                      className={`h-full bg-gradient-to-r ${getRegimeGradient(vol.regime)} to-transparent transition-all duration-500`}
                      style={{ width: `${vol.volatility_percentile || 50}%` }}
                    />
                  </div>

                  {/* Stats grid */}
                  <div className="grid grid-cols-4 gap-2 text-xs">
                    <div className="bg-gray-900/50 rounded p-2 text-center">
                      <div className="text-gray-500">ATR</div>
                      <div className="text-white font-medium">{vol.current_atr_pct?.toFixed(2) || '0.00'}%</div>
                    </div>
                    <div className="bg-gray-900/50 rounded p-2 text-center">
                      <div className="text-gray-500">Daily Vol</div>
                      <div className="text-white font-medium">{vol.current_daily_vol?.toFixed(2) || '0.00'}%</div>
                    </div>
                    <div className="bg-gray-900/50 rounded p-2 text-center">
                      <div className="text-gray-500">Exp Range</div>
                      <div className="text-white font-medium">{vol.expected_range || 'N/A'}</div>
                    </div>
                    {vol.regime_model_stats && (
                      <div className="bg-gray-900/50 rounded p-2 text-center">
                        <div className="text-gray-500">Model Acc</div>
                        <div className="text-white font-medium">{vol.regime_model_stats.high_conf_accuracy?.toFixed(1) || '0.0'}%</div>
                      </div>
                    )}
                  </div>

                  {/* Regime-specific advice */}
                  <div className="mt-2 text-xs text-gray-400">
                    {vol.regime === 'LOW' && (
                      <span>Best for: <span className="text-green-400">Range predictions</span> • Use tighter stops</span>
                    )}
                    {vol.regime === 'NORMAL' && (
                      <span>Balanced conditions • <span className="text-yellow-400">Standard risk management</span></span>
                    )}
                    {vol.regime === 'HIGH' && (
                      <span>Best for: <span className="text-red-400">Direction predictions</span> • Use wider stops</span>
                    )}
                  </div>
                </div>
              )
            })}
          </div>

          {/* Timestamp */}
          <div className="text-right text-gray-600 text-xs">
            Updated: {new Date(data.generated_at).toLocaleTimeString()}
          </div>
        </div>
      )}
    </div>
  )
}
