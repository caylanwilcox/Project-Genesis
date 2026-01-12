'use client'

import { useState, useEffect } from 'react'

interface TickerDirection {
  action: 'LONG' | 'SHORT' | 'NO_TRADE'
  reason: string
  probability_a: number
  probability_b: number
  bucket: string
  position_pct: number
  confidence: number
  current_price: number
  today_open: number
  price_11am: number | null
  today_change_pct: number
  stop_loss: number | null
  take_profit: number | null
  session: 'early' | 'late'
  multipliers: {
    time: number
    agreement: number
  }
  model_accuracy: {
    early: number
    late_a: number
    late_b: number
  }
  error?: string
}

interface TradingDirectionsData {
  generated_at: string
  current_time_et: string
  current_hour: number
  market_open: boolean
  session: 'early' | 'late'
  time_multiplier: number
  tickers: Record<string, TickerDirection>
  best_ticker: string | null
  summary: {
    actionable_tickers: string[]
    best_opportunity: string | null
    recommendation: string
  }
  trading_rules: {
    entry: string[]
    sizing: Record<string, string>
    exit: Record<string, string>
  }
  model_accuracy: {
    late_target_a: string
    late_target_b: string
    peak_hours: string
  }
  message?: string
  error?: string
}

export function TradingDirections() {
  const [data, setData] = useState<TradingDirectionsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showRules, setShowRules] = useState(false)

  useEffect(() => {
    const fetchDirections = async () => {
      try {
        const response = await fetch('/api/v2/trading-directions')
        if (!response.ok) {
          throw new Error('Failed to fetch trading directions')
        }
        const result = await response.json()
        if (result.error) {
          setError(result.error)
        } else {
          setData(result)
          setError(null)
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    }

    fetchDirections()
    // Refresh every 2 minutes
    const interval = setInterval(fetchDirections, 120000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="flex items-center gap-2">
          <div className="animate-spin h-4 w-4 border-2 border-cyan-400 border-t-transparent rounded-full"></div>
          <span className="text-gray-400 text-sm">Loading trading directions...</span>
        </div>
      </div>
    )
  }

  if (error || !data) {
    const isProduction = typeof window !== 'undefined' && !window.location.hostname.includes('localhost')
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="text-yellow-400 text-sm flex items-center gap-2">
          <span>⚠</span>
          <span>ML Server Not Connected</span>
        </div>
        <div className="text-gray-500 text-xs mt-2">
          {isProduction ? (
            <span>ML predictions require running the server locally</span>
          ) : (
            <span>Start server: <code className="bg-gray-800 px-1 rounded">cd ml && python -m server.app</code></span>
          )}
        </div>
      </div>
    )
  }

  const getActionColor = (action: string) => {
    switch (action) {
      case 'LONG': return 'bg-green-500'
      case 'SHORT': return 'bg-red-500'
      default: return 'bg-gray-600'
    }
  }

  // Convert LONG/SHORT to user-friendly actions
  const getActionText = (action: string | undefined) => {
    switch (action) {
      case 'LONG': return 'BUY CALL'
      case 'SHORT': return 'BUY PUT'
      default: return 'WAIT'
    }
  }

  const getActionBorderColor = (action: string) => {
    switch (action) {
      case 'LONG': return 'border-green-500/50'
      case 'SHORT': return 'border-red-500/50'
      default: return 'border-gray-700'
    }
  }

  const getBucketColor = (bucket: string) => {
    if (bucket.includes('very_strong')) return 'text-yellow-400'
    if (bucket.includes('strong')) return 'text-green-400'
    if (bucket.includes('moderate')) return 'text-blue-400'
    return 'text-gray-400'
  }

  const formatProbability = (prob: number) => {
    return `${Math.round(prob * 100)}%`
  }

  // Defensive normalization: the ML server can return partial/older shapes.
  const safeTickers = (data.tickers ?? {}) as Record<string, Partial<TickerDirection>>
  const safeSummary = data.summary ?? {
    actionable_tickers: [],
    best_opportunity: null,
    recommendation: '',
  }
  const safeRules = data.trading_rules ?? { entry: [], sizing: {}, exit: {} }

  const getModelAccuracyPct = (direction: Partial<TickerDirection>): number | null => {
    const early = direction.model_accuracy?.early
    const lateB = direction.model_accuracy?.late_b
    const chosen = direction.session === 'late' ? lateB : early
    return typeof chosen === 'number' && Number.isFinite(chosen) ? Math.round(chosen * 100) : null
  }

  const formatMoney = (value: number | null | undefined) => {
    if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
    return `$${value.toFixed(2)}`
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-cyan-900/30 to-blue-900/30 px-4 py-3 border-b border-gray-800">
        <div className="flex justify-between items-center">
          <div>
            <h3 className="text-white font-semibold flex items-center gap-2">
              <span className="text-cyan-400">V6</span> Trading Directions
              <span className={`px-2 py-0.5 rounded text-xs ${
                data.session === 'late' ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'
              }`}>
                {data.session === 'late' ? 'PEAK ACCURACY' : 'EARLY SESSION'}
              </span>
            </h3>
            <p className="text-gray-400 text-xs mt-0.5">
              {data.current_time_et} | Time Multiplier: {data.time_multiplier}x
            </p>
          </div>
          <div className="text-right">
            {data.best_ticker && (
              <div className="text-sm">
                <span className="text-gray-400">Best: </span>
                <span className="text-cyan-400 font-bold">{data.best_ticker}</span>
              </div>
            )}
            <button
              onClick={() => setShowRules(!showRules)}
              className="text-xs text-gray-500 hover:text-gray-300 mt-1"
            >
              {showRules ? 'Hide Rules' : 'Show Rules'}
            </button>
          </div>
        </div>
      </div>

      {/* Trading Rules (collapsible) */}
      {showRules && (
        <div className="px-4 py-3 bg-gray-800/50 border-b border-gray-800 text-xs">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <div className="text-gray-400 font-medium mb-1">Entry Rules</div>
              {safeRules.entry.map((rule, i) => (
                <div key={i} className="text-gray-500">{rule}</div>
              ))}
            </div>
            <div>
              <div className="text-gray-400 font-medium mb-1">Position Sizing</div>
              {Object.entries(safeRules.sizing).map(([key, value]) => (
                <div key={key} className="text-gray-500">
                  <span className="text-gray-400">{key}:</span> {value}
                </div>
              ))}
            </div>
            <div>
              <div className="text-gray-400 font-medium mb-1">Exit Rules</div>
              {Object.entries(safeRules.exit).map(([key, value]) => (
                <div key={key} className="text-gray-500">
                  <span className="text-gray-400">{key}:</span> {value}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Summary Banner */}
      <div className="px-4 py-2 bg-gray-800/30 border-b border-gray-800">
        <div className="text-sm text-gray-300">
          {safeSummary.recommendation}
        </div>
        {safeSummary.actionable_tickers.length > 0 && (
          <div className="text-xs text-gray-500 mt-0.5">
            Actionable: {safeSummary.actionable_tickers.join(', ')}
          </div>
        )}
      </div>

      {/* Ticker Cards */}
      <div className="p-4 space-y-3">
        {Object.entries(safeTickers).map(([ticker, direction]) => (
          <div
            key={ticker}
            className={`border rounded-lg p-3 ${getActionBorderColor(direction.action)} ${
              ticker === data.best_ticker ? 'ring-1 ring-cyan-500/50' : ''
            }`}
          >
            <div className="flex justify-between items-start">
              {/* Left: Ticker & Action */}
              <div className="flex items-center gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-white font-bold text-lg">{ticker}</span>
                    <span className={`px-2 py-0.5 rounded text-xs font-bold text-white ${getActionColor(direction.action ?? 'NO_TRADE')}`}>
                      {getActionText(direction.action)}
                    </span>
                    {ticker === data.best_ticker && (
                      <span className="text-cyan-400 text-xs">BEST</span>
                    )}
                  </div>
                  <div className="text-gray-400 text-xs mt-0.5">
                    {direction.error ? direction.error : (direction.reason || '—')}
                  </div>
                </div>
              </div>

              {/* Right: Target B Dominance (primary) or Target A (early) */}
              <div className="text-right">
                {direction.session === 'late' && typeof direction.probability_b === 'number' ? (
                  /* Late Session: Show Target B Dominance */
                  /* SPEC: Neutral zone is 25-75%. Only show bullish/bearish at extremes */
                  (() => {
                    const isBullish = direction.probability_b > 0.75
                    const isBearish = direction.probability_b < 0.25
                    const isNeutral = !isBullish && !isBearish
                    const confidence = Math.round(direction.probability_b * 100)
                    const dominance = isBullish ? 'BULL' : isBearish ? 'BEAR' : 'WAIT'
                    const textColor = isBullish ? 'text-green-400' : isBearish ? 'text-red-400' : 'text-yellow-400'
                    const bgColor = isBullish ? 'bg-green-500/20' : isBearish ? 'bg-red-500/20' : 'bg-yellow-500/10'

                    return (
                      <div className={`px-2 py-1 rounded ${bgColor}`}>
                        <div className="flex items-center gap-1 justify-end">
                          <span className={`font-bold text-sm ${textColor}`}>
                            {dominance} {isBullish ? '▲' : isBearish ? '▼' : '◆'}
                          </span>
                          <span className={`font-mono font-bold text-lg ${textColor}`}>
                            {confidence}%
                          </span>
                        </div>
                        {direction.price_11am && (
                          <div className="text-cyan-400 text-xs text-right font-mono">
                            11AM: ${direction.price_11am.toFixed(2)}
                          </div>
                        )}
                      </div>
                    )
                  })()
                ) : (
                  /* Early Session: Show Target A */
                  /* SPEC: Neutral zone is 25-75% */
                  <div className="flex items-center gap-2 justify-end">
                    <span className="text-gray-500 text-xs">Target A:</span>
                    <span className={`font-mono font-bold ${
                      direction.probability_a > 0.75 ? 'text-green-400' :
                      direction.probability_a < 0.25 ? 'text-red-400' : 'text-yellow-400'
                    }`}>
                      {typeof direction.probability_a === 'number' ? formatProbability(direction.probability_a) : '—'}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Details Row */}
            {direction.action !== 'NO_TRADE' && (
              <div className="mt-3 pt-3 border-t border-gray-800 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                <div>
                  <div className="text-gray-500">Position Size</div>
                  <div className="text-white font-bold">
                    {typeof direction.position_pct === 'number' ? `${direction.position_pct}%` : '—'}
                  </div>
                </div>
                <div>
                  <div className="text-gray-500">Confidence</div>
                  <div className={`font-bold ${getBucketColor(direction.bucket)}`}>
                    {typeof direction.confidence === 'number' ? `${direction.confidence}%` : '—'}
                  </div>
                </div>
                <div>
                  <div className="text-gray-500">Stop Loss</div>
                  <div className="text-red-400 font-mono">
                    {formatMoney(direction.stop_loss)}
                  </div>
                </div>
                <div>
                  <div className="text-gray-500">Take Profit</div>
                  <div className="text-green-400 font-mono">
                    {formatMoney(direction.take_profit)}
                  </div>
                </div>
              </div>
            )}

            {/* Price Info */}
            <div className="mt-2 flex justify-between text-xs text-gray-500">
              <span>
                Current: {formatMoney(direction.current_price)}
                {typeof direction.today_change_pct === 'number' && Number.isFinite(direction.today_change_pct) && (
                  <span className={direction.today_change_pct >= 0 ? 'text-green-400' : 'text-red-400'}>
                    {' '}({direction.today_change_pct >= 0 ? '+' : ''}{direction.today_change_pct.toFixed(2)}%)
                  </span>
                )}
              </span>
              <span>
                Model: {(() => {
                  const pct = getModelAccuracyPct(direction)
                  return pct === null ? '—' : `${pct}% acc`
                })()}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="px-4 py-2 bg-gray-800/30 border-t border-gray-800 text-xs text-gray-500">
        <div className="flex justify-between">
          <span>Model: V6 Time-Split | Target B: {data.model_accuracy.late_target_b}</span>
          <span>Peak: {data.model_accuracy.peak_hours}</span>
        </div>
      </div>
    </div>
  )
}
