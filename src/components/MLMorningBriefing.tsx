'use client'

import { useState, useEffect } from 'react'

const ML_SERVER_URL = process.env.NEXT_PUBLIC_ML_SERVER_URL || 'https://genesis-production-c1e9.up.railway.app'

// Daily Signals Types
interface TickerSignal {
  signal: 'BUY' | 'SELL' | 'HOLD'
  strength: 'STRONG' | 'MODERATE' | 'WEAK' | 'NEUTRAL'
  action: string
  emoji: string
  probability: number
  confidence: number
  current_price: number
  entry_price: number
  target_price: number
  stop_loss: number
  risk_reward: number
  predicted_range: {
    high: number
    low: number
  }
  highlow_model?: {
    predicted_high: number
    predicted_low: number
    high_pct: number
    low_pct: number
    capture_rate: number
  }
  model_accuracy: number
  error?: string
}

interface DailySignals {
  date: string
  generated_at: string
  is_after_hours: boolean
  next_trading_day: string
  tickers: Record<string, TickerSignal>
  summary: {
    buys: Array<{ ticker: string; strength: string; probability: number }>
    sells: Array<{ ticker: string; strength: string; probability: number }>
    holds: Array<{ ticker: string; probability: number }>
  }
  market_signal: string
  market_emoji: string
  market_action: string
  best_trade: {
    ticker: string
    signal: string
    strength: string
    confidence: number
    risk_reward: number
  } | null
}

// Technical Breakdown Types
interface TechnicalIndicator {
  name: string
  value: string
  signal: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
  reason: string
}

interface TickerBreakdown {
  current_price: number
  indicators: TechnicalIndicator[]
  summary: {
    bullish: number
    bearish: number
    neutral: number
    total: number
  }
  error?: string
}

interface SignalBreakdown {
  date: string
  generated_at: string
  tickers: Record<string, TickerBreakdown>
}

function InfoIcon({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="ml-2 w-5 h-5 rounded-full bg-gray-700 hover:bg-gray-600 text-gray-400 hover:text-white flex items-center justify-center text-xs font-bold transition-colors"
      title="How this works"
    >
      i
    </button>
  )
}

function InfoModal({
  isOpen,
  onClose,
  breakdown,
  loading
}: {
  isOpen: boolean
  onClose: () => void
  breakdown: SignalBreakdown | null
  loading: boolean
}) {
  if (!isOpen) return null

  const getSignalColor = (signal: string) => {
    if (signal === 'BULLISH') return 'text-green-400'
    if (signal === 'BEARISH') return 'text-red-400'
    return 'text-yellow-400'
  }

  const getSignalBg = (signal: string) => {
    if (signal === 'BULLISH') return 'bg-green-500/20'
    if (signal === 'BEARISH') return 'bg-red-500/20'
    return 'bg-yellow-500/20'
  }

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div
        className="bg-gray-900 border border-gray-700 rounded-lg p-6 max-w-2xl w-full max-h-[85vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-white font-bold text-lg">Technical Signal Breakdown</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-xl">&times;</button>
        </div>

        {loading ? (
          <div className="text-center py-8">
            <div className="animate-spin h-8 w-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
            <p className="text-gray-400">Loading technical indicators...</p>
          </div>
        ) : breakdown ? (
          <div className="space-y-6">
            {Object.entries(breakdown.tickers).map(([ticker, data]) => {
              if (data.error) {
                return (
                  <div key={ticker} className="text-red-400 text-sm">
                    {ticker}: {data.error}
                  </div>
                )
              }

              const bullishIndicators = data.indicators.filter(i => i.signal === 'BULLISH')
              const bearishIndicators = data.indicators.filter(i => i.signal === 'BEARISH')
              const neutralIndicators = data.indicators.filter(i => i.signal === 'NEUTRAL')

              return (
                <div key={ticker} className="border border-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <span className="text-white font-bold text-xl">{ticker}</span>
                      <span className="text-gray-400">${data.current_price}</span>
                    </div>
                    <div className="flex gap-2 text-xs">
                      <span className="bg-green-500/20 text-green-400 px-2 py-1 rounded">
                        {data.summary.bullish} Bullish
                      </span>
                      <span className="bg-red-500/20 text-red-400 px-2 py-1 rounded">
                        {data.summary.bearish} Bearish
                      </span>
                      <span className="bg-yellow-500/20 text-yellow-400 px-2 py-1 rounded">
                        {data.summary.neutral} Neutral
                      </span>
                    </div>
                  </div>

                  {/* Bearish Indicators */}
                  {bearishIndicators.length > 0 && (
                    <div className="mb-3">
                      <div className="text-red-400 text-xs font-semibold mb-2 uppercase tracking-wide">
                        Bearish Signals ({bearishIndicators.length})
                      </div>
                      <div className="grid gap-1">
                        {bearishIndicators.map((ind, idx) => (
                          <div key={idx} className="flex items-center justify-between bg-red-500/10 rounded px-2 py-1 text-xs">
                            <span className="text-gray-300 font-medium">{ind.name}</span>
                            <span className="text-red-400 font-mono">{ind.value}</span>
                            <span className="text-red-300 text-xs">{ind.reason}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Bullish Indicators */}
                  {bullishIndicators.length > 0 && (
                    <div className="mb-3">
                      <div className="text-green-400 text-xs font-semibold mb-2 uppercase tracking-wide">
                        Bullish Signals ({bullishIndicators.length})
                      </div>
                      <div className="grid gap-1">
                        {bullishIndicators.map((ind, idx) => (
                          <div key={idx} className="flex items-center justify-between bg-green-500/10 rounded px-2 py-1 text-xs">
                            <span className="text-gray-300 font-medium">{ind.name}</span>
                            <span className="text-green-400 font-mono">{ind.value}</span>
                            <span className="text-green-300 text-xs">{ind.reason}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Neutral Indicators */}
                  {neutralIndicators.length > 0 && (
                    <div>
                      <div className="text-yellow-400 text-xs font-semibold mb-2 uppercase tracking-wide">
                        Neutral ({neutralIndicators.length})
                      </div>
                      <div className="grid gap-1">
                        {neutralIndicators.map((ind, idx) => (
                          <div key={idx} className="flex items-center justify-between bg-yellow-500/10 rounded px-2 py-1 text-xs">
                            <span className="text-gray-300 font-medium">{ind.name}</span>
                            <span className="text-yellow-400 font-mono">{ind.value}</span>
                            <span className="text-yellow-300 text-xs">{ind.reason}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )
            })}

            <div className="bg-gray-800 rounded p-3 mt-4 text-xs text-gray-400">
              <strong>How signals are determined:</strong> Each indicator is evaluated against historical patterns.
              The ML model weighs these signals plus 50+ additional features to predict tomorrow's direction.
              Model accuracy: ~68% overall, ~74% on high-confidence signals.
            </div>
          </div>
        ) : (
          <p className="text-gray-400 text-center py-8">Unable to load breakdown</p>
        )}
      </div>
    </div>
  )
}

export function MLMorningBriefing() {
  const [signals, setSignals] = useState<DailySignals | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showInfo, setShowInfo] = useState(false)
  const [breakdown, setBreakdown] = useState<SignalBreakdown | null>(null)
  const [breakdownLoading, setBreakdownLoading] = useState(false)

  const fetchBreakdown = async () => {
    setBreakdownLoading(true)
    try {
      const response = await fetch(`${ML_SERVER_URL}/signal_breakdown`)
      if (response.ok) {
        const data = await response.json()
        setBreakdown(data)
      }
    } catch (err) {
      console.error('Failed to fetch breakdown:', err)
    } finally {
      setBreakdownLoading(false)
    }
  }

  const handleInfoClick = () => {
    setShowInfo(true)
    if (!breakdown) {
      fetchBreakdown()
    }
  }

  useEffect(() => {
    const fetchSignals = async () => {
      try {
        setLoading(true)
        const response = await fetch(`${ML_SERVER_URL}/daily_signals`)

        if (!response.ok) {
          throw new Error('Failed to fetch signals')
        }

        const data = await response.json()
        setSignals(data)
        setError(null)
      } catch (err: any) {
        setError(err.message || 'Failed to load ML signals')
      } finally {
        setLoading(false)
      }
    }

    fetchSignals()

    // Refresh every 5 minutes
    const interval = setInterval(fetchSignals, 5 * 60 * 1000)
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

  if (error || !signals) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="text-red-400 text-sm">
          {error || 'Unable to load ML signals'}
        </div>
      </div>
    )
  }

  const getSignalColor = (signal: string) => {
    if (signal === 'BUY') return 'text-green-400'
    if (signal === 'SELL') return 'text-red-400'
    return 'text-yellow-400'
  }

  const getSignalBg = (signal: string) => {
    if (signal === 'BUY') return 'bg-green-500/20'
    if (signal === 'SELL') return 'bg-red-500/20'
    return 'bg-yellow-500/20'
  }

  const getMarketBg = (signal: string) => {
    if (signal === 'BULLISH') return 'from-green-900/30 to-transparent'
    if (signal === 'BEARISH') return 'from-red-900/30 to-transparent'
    return 'from-yellow-900/30 to-transparent'
  }

  return (
    <div className={`bg-gradient-to-br ${getMarketBg(signals.market_signal)} bg-gray-900 border border-gray-800 rounded-lg p-4`}>
      {/* Info Modal */}
      <InfoModal
        isOpen={showInfo}
        onClose={() => setShowInfo(false)}
        breakdown={breakdown}
        loading={breakdownLoading}
      />

      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-white font-bold text-lg flex items-center">
            {signals.is_after_hours ? "Tomorrow's Outlook" : 'ML Daily Signals'}
            <InfoIcon onClick={handleInfoClick} />
          </h3>
          <p className="text-gray-500 text-xs">
            {signals.is_after_hours ? `Next: ${signals.next_trading_day}` : signals.date}
          </p>
          {signals.is_after_hours && (
            <p className="text-blue-400 text-xs mt-1">After-hours analysis via High/Low Model</p>
          )}
        </div>
        <div className="text-right">
          <div className={`text-xl font-bold ${getSignalColor(signals.market_signal === 'BULLISH' ? 'BUY' : signals.market_signal === 'BEARISH' ? 'SELL' : 'HOLD')}`}>
            {signals.market_emoji} {signals.market_action}
          </div>
          <p className="text-gray-500 text-xs">Market Bias</p>
        </div>
      </div>

      {/* Best Trade */}
      {signals.best_trade && signals.tickers[signals.best_trade.ticker] && (
        <div className="bg-gray-800/50 rounded-lg p-3 mb-4 border border-gray-700">
          <div className="text-gray-400 text-xs mb-1">BEST TRADE {signals.is_after_hours ? 'TOMORROW' : 'TODAY'}</div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-white font-bold text-xl">
                {signals.best_trade.ticker}
              </span>
              <span className={`px-3 py-1 rounded text-sm font-bold ${getSignalBg(signals.best_trade.signal)} ${getSignalColor(signals.best_trade.signal)}`}>
                {signals.best_trade.strength} {signals.best_trade.signal}
              </span>
            </div>
            <div className="text-right">
              <div className={`font-bold ${getSignalColor(signals.best_trade.signal)}`}>
                {signals.best_trade.signal === 'SELL'
                  ? ((1 - signals.tickers[signals.best_trade.ticker].probability) * 100).toFixed(0)
                  : (signals.tickers[signals.best_trade.ticker].probability * 100).toFixed(0)}% {signals.best_trade.signal === 'SELL' ? 'Bearish' : 'Bullish'}
              </div>
              <div className="text-blue-400 text-xs">
                R:R {signals.best_trade.risk_reward.toFixed(1)}:1
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Ticker Signals */}
      <div className="space-y-3">
        {Object.entries(signals.tickers).map(([ticker, sig]) => {
          if (sig.error) {
            return (
              <div key={ticker} className="flex items-center justify-between py-2 border-b border-gray-800 last:border-0">
                <span className="text-gray-400">{ticker}</span>
                <span className="text-red-400 text-xs">{sig.error}</span>
              </div>
            )
          }

          return (
            <div key={ticker} className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
              {/* Top row: Ticker + Signal */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <span className="text-xl">{sig.emoji}</span>
                  <div>
                    <div className="text-white font-bold text-lg">{ticker}</div>
                    <div className="text-gray-500 text-xs">
                      ${sig.current_price}
                    </div>
                  </div>
                </div>
                <div className={`px-4 py-2 rounded-lg text-center ${getSignalBg(sig.signal)}`}>
                  <div className={`font-bold text-lg ${getSignalColor(sig.signal)}`}>
                    {sig.signal}
                  </div>
                  <div className="text-gray-400 text-xs">
                    {sig.strength}
                  </div>
                </div>
              </div>

              {/* High/Low Model Predicted Range - Prominent Display */}
              {sig.highlow_model && signals.is_after_hours && (
                <div className="bg-blue-900/30 border border-blue-700/50 rounded-lg p-3 mb-3">
                  <div className="text-blue-400 text-xs font-semibold mb-2">
                    Tomorrow's ML Predicted Range
                    {sig.highlow_model.capture_rate > 0 && (
                      <span className="text-gray-500 font-normal ml-2">
                        ({sig.highlow_model.capture_rate.toFixed(0)}% capture rate)
                      </span>
                    )}
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="text-center flex-1">
                      <div className="text-red-400 text-xs">Low</div>
                      <div className="text-white font-bold text-lg">${sig.highlow_model.predicted_low}</div>
                      <div className="text-red-400 text-xs">-{sig.highlow_model.low_pct.toFixed(2)}%</div>
                    </div>
                    <div className="text-gray-600 px-3">→</div>
                    <div className="text-center flex-1">
                      <div className="text-green-400 text-xs">High</div>
                      <div className="text-white font-bold text-lg">${sig.highlow_model.predicted_high}</div>
                      <div className="text-green-400 text-xs">+{sig.highlow_model.high_pct.toFixed(2)}%</div>
                    </div>
                  </div>
                </div>
              )}

              {/* Trade Details */}
              <div className="grid grid-cols-4 gap-2 text-center text-xs">
                <div className="bg-gray-900/50 rounded p-2">
                  <div className="text-gray-500">Entry</div>
                  <div className="text-white font-medium">${sig.entry_price}</div>
                </div>
                <div className="bg-gray-900/50 rounded p-2">
                  <div className={sig.signal === 'BUY' ? 'text-green-400' : sig.signal === 'SELL' ? 'text-red-400' : 'text-gray-500'}>Target</div>
                  <div className={`font-medium ${sig.signal === 'BUY' ? 'text-green-400' : sig.signal === 'SELL' ? 'text-red-400' : 'text-white'}`}>
                    ${sig.target_price}
                  </div>
                </div>
                <div className="bg-gray-900/50 rounded p-2">
                  <div className="text-red-400">Stop</div>
                  <div className="text-red-400 font-medium">${sig.stop_loss}</div>
                </div>
                <div className="bg-gray-900/50 rounded p-2">
                  <div className="text-blue-400">R:R</div>
                  <div className="text-blue-400 font-medium">{sig.risk_reward}</div>
                </div>
              </div>

              {/* Probability bar - show probability matching signal direction */}
              <div className="mt-2">
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className={sig.signal === 'BUY' ? 'text-green-400' : sig.signal === 'SELL' ? 'text-red-400' : 'text-yellow-400'}>
                    {sig.signal === 'BUY' ? 'Bullish' : sig.signal === 'SELL' ? 'Bearish' : 'Neutral'} Probability
                  </span>
                  <span className={sig.signal === 'BUY' ? 'text-green-400' : sig.signal === 'SELL' ? 'text-red-400' : 'text-yellow-400'}>
                    {sig.signal === 'SELL' ? ((1 - sig.probability) * 100).toFixed(0) : (sig.probability * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex-1 h-2 bg-gray-700 rounded overflow-hidden">
                  <div
                    className={`h-full ${sig.signal === 'BUY' ? 'bg-green-500' : sig.signal === 'SELL' ? 'bg-red-500' : 'bg-yellow-500'}`}
                    style={{ width: `${sig.signal === 'SELL' ? (1 - sig.probability) * 100 : sig.probability * 100}%` }}
                  />
                </div>
                <div className="text-gray-600 text-xs mt-1">
                  Model accuracy: {(sig.model_accuracy * 100).toFixed(0)}%
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Summary */}
      <div className="mt-4 pt-3 border-t border-gray-800">
        <div className="flex items-center justify-between text-xs">
          <div className="flex gap-4">
            {signals.summary.buys.length > 0 && (
              <span className="text-green-400">
                {signals.summary.buys.length} BUY{signals.summary.buys.length > 1 ? 'S' : ''}
              </span>
            )}
            {signals.summary.sells.length > 0 && (
              <span className="text-red-400">
                {signals.summary.sells.length} SELL{signals.summary.sells.length > 1 ? 'S' : ''}
              </span>
            )}
            {signals.summary.holds.length > 0 && (
              <span className="text-yellow-400">
                {signals.summary.holds.length} HOLD{signals.summary.holds.length > 1 ? 'S' : ''}
              </span>
            )}
          </div>
          <div className="text-gray-600">
            {new Date(signals.generated_at).toLocaleTimeString()}
          </div>
        </div>
      </div>
    </div>
  )
}
