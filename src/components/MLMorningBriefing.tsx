'use client'

import { useState, useEffect } from 'react'
import { PriceRanges } from './PriceRanges'

// Use same-origin Next.js API routes to avoid CORS / env mismatches in the browser.
const DAILY_SIGNALS_URL = '/api/v2/ml/daily-signals'
const SIGNAL_BREAKDOWN_URL = '/api/v2/ml/signal-breakdown'

// Daily Signals Types
interface IntradayModel {
  probability: number
  probability_close_above_current: number | null
  probability_close_above_open: number | null
  confidence: number
  time_pct: number
  session_label: string
  current_vs_open: number
  position_in_range: number
  model_accuracy: number
  prediction_target: string
}

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
  remaining_potential?: {
    upside_pct: number
    downside_pct: number
    upside_dollars: number
    downside_dollars: number
    target_extended: boolean
    original_target_low: number
    original_target_high: number
  }
  highlow_model?: {
    predicted_high: number
    predicted_low: number
    high_pct: number
    low_pct: number
    capture_rate: number
  }
  intraday_model?: IntradayModel
  prediction_source?: 'daily' | 'intraday'
  model_accuracy: number
  error?: string
}

interface VixData {
  current: number
  change_pct: number
  regime: 'LOW' | 'NORMAL' | 'ELEVATED' | 'HIGH'
  emoji: string
  note: string
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
  vix?: VixData | null
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
      const response = await fetch(SIGNAL_BREAKDOWN_URL)
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
        const response = await fetch(DAILY_SIGNALS_URL)

        if (!response.ok) {
          const text = await response.text()
          throw new Error(text || 'Failed to fetch signals')
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

    // Refresh every 1 minute for real-time updates
    const interval = setInterval(fetchSignals, 60 * 1000)
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

      {/* VIX Display */}
      {signals.vix && !signals.vix.error && (
        <div className="bg-gray-800/50 rounded-lg p-3 mb-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-gray-400 text-sm font-medium">VIX</span>
              <span className="text-white font-bold text-lg">{signals.vix.current}</span>
              <span className={`text-xs ${signals.vix.change_pct >= 0 ? 'text-red-400' : 'text-green-400'}`}>
                {signals.vix.change_pct >= 0 ? '+' : ''}{signals.vix.change_pct}%
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                signals.vix.regime === 'HIGH' ? 'bg-red-500/20 text-red-400' :
                signals.vix.regime === 'ELEVATED' ? 'bg-orange-500/20 text-orange-400' :
                signals.vix.regime === 'NORMAL' ? 'bg-green-500/20 text-green-400' :
                'bg-blue-500/20 text-blue-400'
              }`}>
                {signals.vix.emoji} {signals.vix.regime}
              </span>
            </div>
          </div>
          <p className="text-gray-500 text-xs mt-1">{signals.vix.note}</p>
        </div>
      )}

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

          // Derive signal from intraday model when available
          const useIntraday = sig.intraday_model && sig.prediction_source === 'intraday'
          const displayProb = useIntraday ? sig.intraday_model!.probability : sig.probability

          // Calculate signal based on probability
          let derivedSignal: 'BUY' | 'SELL' | 'HOLD' = 'HOLD'
          let derivedStrength: 'STRONG' | 'MODERATE' | 'WEAK' | 'NEUTRAL' = 'NEUTRAL'
          let derivedEmoji = '⏸️'

          if (displayProb >= 0.7) {
            derivedSignal = 'BUY'
            derivedStrength = 'STRONG'
            derivedEmoji = '🟢'
          } else if (displayProb >= 0.6) {
            derivedSignal = 'BUY'
            derivedStrength = 'MODERATE'
            derivedEmoji = '🟢'
          } else if (displayProb >= 0.55) {
            derivedSignal = 'BUY'
            derivedStrength = 'WEAK'
            derivedEmoji = '🟡'
          } else if (displayProb <= 0.3) {
            derivedSignal = 'SELL'
            derivedStrength = 'STRONG'
            derivedEmoji = '🔴'
          } else if (displayProb <= 0.4) {
            derivedSignal = 'SELL'
            derivedStrength = 'MODERATE'
            derivedEmoji = '🔴'
          } else if (displayProb <= 0.45) {
            derivedSignal = 'SELL'
            derivedStrength = 'WEAK'
            derivedEmoji = '🟡'
          } else {
            derivedSignal = 'HOLD'
            derivedStrength = 'NEUTRAL'
            derivedEmoji = '⏸️'
          }

          // Use derived signal if intraday, otherwise use API signal
          const finalSignal = useIntraday ? derivedSignal : sig.signal
          const finalStrength = useIntraday ? derivedStrength : sig.strength
          const finalEmoji = useIntraday ? derivedEmoji : sig.emoji

          return (
            <div key={ticker} className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
              {/* Top row: Ticker + Signal */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <span className="text-xl">{finalEmoji}</span>
                  <div>
                    <div className="text-white font-bold text-lg">{ticker}</div>
                    <div className="text-gray-500 text-xs">
                      ${sig.current_price}
                    </div>
                  </div>
                </div>
                <div className={`px-4 py-2 rounded-lg text-center ${getSignalBg(finalSignal)}`}>
                  <div className={`font-bold text-lg ${getSignalColor(finalSignal)}`}>
                    {finalSignal}
                  </div>
                  <div className="text-gray-400 text-xs">
                    {finalStrength}
                  </div>
                </div>
              </div>

              {/* INTRADAY MODEL - Both Targets */}
              {sig.intraday_model && sig.prediction_source === 'intraday' && (
                <div className="bg-purple-900/30 border border-purple-700/50 rounded-lg p-3 mb-3">
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-purple-400 text-xs font-semibold">
                      📊 INTRADAY PREDICTION
                      <span className="text-gray-500 font-normal ml-2">
                        ({sig.intraday_model.session_label} • {(sig.intraday_model.time_pct * 100).toFixed(0)}% through day)
                      </span>
                    </div>
                    <div className="text-gray-500 text-xs">
                      {(sig.intraday_model.model_accuracy * 100).toFixed(0)}% accuracy
                    </div>
                  </div>

                  {/* Two predictions side by side */}
                  <div className="grid grid-cols-2 gap-3">
                    {/* Target A: Close > Open (Bullish Day) */}
                    <div className="bg-gray-900/50 rounded-lg p-3 text-center">
                      <div className="text-gray-400 text-xs mb-1">Bullish Day?</div>
                      <div className="text-gray-500 text-xs mb-2">(close {'>'} open)</div>
                      {sig.intraday_model.probability_close_above_open !== null ? (
                        <>
                          <div className={`text-2xl font-bold ${
                            sig.intraday_model.probability_close_above_open >= 0.6 ? 'text-green-400' :
                            sig.intraday_model.probability_close_above_open <= 0.4 ? 'text-red-400' :
                            'text-yellow-400'
                          }`}>
                            {(sig.intraday_model.probability_close_above_open * 100).toFixed(0)}%
                          </div>
                          <div className={`text-xs mt-1 ${
                            sig.intraday_model.probability_close_above_open >= 0.6 ? 'text-green-400' :
                            sig.intraday_model.probability_close_above_open <= 0.4 ? 'text-red-400' :
                            'text-yellow-400'
                          }`}>
                            {sig.intraday_model.probability_close_above_open >= 0.6 ? '🟢 YES' :
                             sig.intraday_model.probability_close_above_open <= 0.4 ? '🔴 NO' :
                             '🟡 UNCERTAIN'}
                          </div>
                        </>
                      ) : (
                        <div className="text-gray-500 text-sm">N/A</div>
                      )}
                    </div>

                    {/* Target B: Close > Current (Price Going Up) */}
                    <div className="bg-gray-900/50 rounded-lg p-3 text-center">
                      <div className="text-gray-400 text-xs mb-1">Price Going Up?</div>
                      <div className="text-gray-500 text-xs mb-2">(close {'>'} ${sig.current_price})</div>
                      <div className={`text-2xl font-bold ${
                        sig.intraday_model.probability >= 0.6 ? 'text-green-400' :
                        sig.intraday_model.probability <= 0.4 ? 'text-red-400' :
                        'text-yellow-400'
                      }`}>
                        {(sig.intraday_model.probability * 100).toFixed(0)}%
                      </div>
                      <div className={`text-xs mt-1 ${
                        sig.intraday_model.probability >= 0.6 ? 'text-green-400' :
                        sig.intraday_model.probability <= 0.4 ? 'text-red-400' :
                        'text-yellow-400'
                      }`}>
                        {sig.intraday_model.probability >= 0.6 ? '🟢 YES' :
                         sig.intraday_model.probability <= 0.4 ? '🔴 NO' :
                         '🟡 UNCERTAIN'}
                      </div>
                    </div>
                  </div>

                  {/* Confidence */}
                  <div className="text-center mt-2">
                    <div className="text-gray-500 text-xs">
                      Confidence: {(sig.intraday_model.confidence * 100).toFixed(0)}%
                      {sig.intraday_model.confidence < 0.2 && (
                        <span className="text-yellow-400 ml-2">(low - wait for confirmation)</span>
                      )}
                    </div>
                  </div>

                  {/* Session details */}
                  <div className="grid grid-cols-3 gap-2 mt-2 text-center text-xs">
                    <div className="bg-gray-900/50 rounded p-2">
                      <div className="text-gray-500">vs Open</div>
                      <div className={`font-medium ${sig.intraday_model.current_vs_open >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {sig.intraday_model.current_vs_open >= 0 ? '+' : ''}{sig.intraday_model.current_vs_open.toFixed(2)}%
                      </div>
                    </div>
                    <div className="bg-gray-900/50 rounded p-2">
                      <div className="text-gray-500">In Range</div>
                      <div className="text-white font-medium">
                        {sig.intraday_model.position_in_range.toFixed(0)}%
                      </div>
                    </div>
                    <div className="bg-gray-900/50 rounded p-2">
                      <div className="text-gray-500">Session</div>
                      <div className="text-cyan-400 font-medium">
                        {(sig.intraday_model.time_pct * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>

                  {/* Timing warning */}
                  {sig.intraday_model.time_pct < 0.1 && (
                    <div className="mt-2 text-center text-xs text-yellow-400 bg-yellow-500/10 rounded p-2">
                      ⚠️ Early session - wait 30-40 min for reliable signals
                    </div>
                  )}
                </div>
              )}

              {/* High/Low Model Predicted Range - Show when after hours OR alongside intraday */}
              {sig.highlow_model && (signals.is_after_hours || !sig.intraday_model) && (
                <div className="bg-blue-900/30 border border-blue-700/50 rounded-lg p-3 mb-3">
                  <div className="text-blue-400 text-xs font-semibold mb-2">
                    {signals.is_after_hours ? "Tomorrow's" : "Today's"} ML Predicted Range
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

              {/* Trade Details with Remaining Potential */}
              <div className="grid grid-cols-4 gap-2 text-center text-xs mb-2">
                <div className="bg-gray-900/50 rounded p-2">
                  <div className="text-gray-500">Now</div>
                  <div className="text-white font-medium">${sig.current_price}</div>
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

              {/* Remaining Potential */}
              {sig.remaining_potential && (
                <div className="flex items-center justify-between bg-gray-900/30 rounded px-3 py-2 mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-green-400 text-xs">
                      ↑ ${sig.remaining_potential.upside_dollars} ({sig.remaining_potential.upside_pct}%)
                    </span>
                    <span className="text-gray-600">|</span>
                    <span className="text-red-400 text-xs">
                      ↓ ${sig.remaining_potential.downside_dollars} ({sig.remaining_potential.downside_pct}%)
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    {sig.remaining_potential.target_extended && (
                      <span className="text-yellow-400 text-xs bg-yellow-500/10 px-1.5 py-0.5 rounded">
                        Extended
                      </span>
                    )}
                    <span className="text-gray-500 text-xs">Remaining potential</span>
                  </div>
                </div>
              )}

              {/* Multi-Timeframe Price Ranges */}
              <div className="mb-2">
                <PriceRanges ticker={ticker} currentPrice={sig.current_price} />
              </div>

              {/* Probability display - use intraday model when available */}
              <div className="mt-2">
                {(() => {
                  // Use intraday probability if available, otherwise daily
                  const displayProb = sig.intraday_model && sig.prediction_source === 'intraday'
                    ? sig.intraday_model.probability
                    : sig.probability
                  const displayAccuracy = sig.intraday_model && sig.prediction_source === 'intraday'
                    ? sig.intraday_model.model_accuracy
                    : sig.model_accuracy
                  const source = sig.prediction_source === 'intraday' ? 'Intraday' : 'Daily'

                  return (
                    <>
                      {/* Bull vs Bear bar */}
                      <div className="flex items-center gap-2 text-xs mb-1">
                        <span className="text-green-400 w-16 text-right">{(displayProb * 100).toFixed(0)}% Bull</span>
                        <div className="flex-1 h-2 bg-gray-700 rounded overflow-hidden flex">
                          <div className="h-full bg-green-500" style={{ width: `${displayProb * 100}%` }} />
                          <div className="h-full bg-red-500" style={{ width: `${(1 - displayProb) * 100}%` }} />
                        </div>
                        <span className="text-red-400 w-16">{((1 - displayProb) * 100).toFixed(0)}% Bear</span>
                      </div>
                      {/* Edge indicator */}
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-gray-500">
                          Edge: {Math.abs(displayProb - 0.5) * 100 < 5 ? (
                            <span className="text-yellow-400">~Coin flip</span>
                          ) : (
                            <span className={displayProb > 0.5 ? 'text-green-400' : 'text-red-400'}>
                              +{(Math.abs(displayProb - 0.5) * 100).toFixed(0)}% {displayProb > 0.5 ? 'bullish' : 'bearish'}
                            </span>
                          )}
                        </span>
                        <span className="text-gray-600">
                          {source} model: {(displayAccuracy * 100).toFixed(0)}%
                        </span>
                      </div>
                    </>
                  )
                })()}
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
