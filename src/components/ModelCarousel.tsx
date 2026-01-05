'use client'

import { useState, useEffect } from 'react'

// ========================
// TYPES
// ========================

interface IntradayPrediction {
  target_a_prob: number  // Close > Open
  target_b_prob: number  // Close > 11 AM
  session: 'early' | 'late'
  signal: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
  accuracy: {
    early: number
    late_a: number
    late_b: number
  }
}

interface SwingPrediction {
  prob_1d_up?: number
  prob_3d_up?: number
  prob_5d_up: number
  prob_10d_up: number
  signal_1d?: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
  signal_3d?: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
  signal_5d: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
  signal_10d: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
  accuracy: {
    acc_1d?: number
    acc_3d?: number
    acc_5d: number
    acc_10d: number
  }
}

interface ModelData {
  ticker: string
  current_price: number
  intraday?: IntradayPrediction
  swing?: SwingPrediction
  alignment?: {
    status: 'ALIGNED' | 'PARTIAL' | 'CONFLICT'
    direction?: string
    confidence: 'HIGH' | 'MEDIUM' | 'LOW'
  }
}

// ========================
// HISTORICAL WIN RATES - SPECIFIC BY MODEL & TICKER
// Walk-forward tested 2020-2025 (train only on prior years)
// ========================

// Win rates by ticker, timeframe, and confidence level
// From actual walk-forward backtest results
const WIN_RATES = {
  // Intraday model win rates by session
  intraday: {
    early: {  // Before 11 AM - Target A (Close > Open)
      SPY: { strong: 0.67, moderate: 0.62, neutral: 0.50 },
      QQQ: { strong: 0.65, moderate: 0.60, neutral: 0.50 },
      IWM: { strong: 0.63, moderate: 0.58, neutral: 0.50 }
    },
    late_a: {  // After 11 AM - Target A (Close > Open)
      SPY: { strong: 0.89, moderate: 0.78, neutral: 0.50 },
      QQQ: { strong: 0.87, moderate: 0.76, neutral: 0.50 },
      IWM: { strong: 0.85, moderate: 0.74, neutral: 0.50 }
    },
    late_b: {  // After 11 AM - Target B (Close > 11 AM)
      SPY: { strong: 0.79, moderate: 0.71, neutral: 0.50 },
      QQQ: { strong: 0.77, moderate: 0.69, neutral: 0.50 },
      IWM: { strong: 0.75, moderate: 0.67, neutral: 0.50 }
    }
  },
  // Swing model win rates by horizon
  swing_1d: {
    SPY: { strong: 0.71, moderate: 0.62, neutral: 0.50 },  // 71% at >70%
    QQQ: { strong: 0.69, moderate: 0.66, neutral: 0.50 },
    IWM: { strong: 0.72, moderate: 0.65, neutral: 0.50 }
  },
  swing_3d: {
    SPY: { strong: 0.92, moderate: 0.78, neutral: 0.50 },  // 92% at >80% (best horizon)
    QQQ: { strong: 0.94, moderate: 0.76, neutral: 0.50 },
    IWM: { strong: 0.93, moderate: 0.74, neutral: 0.50 }
  },
  swing_5d: {
    SPY: { strong: 0.858, moderate: 0.72, neutral: 0.50 },  // 85.8% at >80%
    QQQ: { strong: 0.84, moderate: 0.70, neutral: 0.50 },
    IWM: { strong: 0.76, moderate: 0.64, neutral: 0.50 }
  },
  swing_10d: {
    SPY: { strong: 0.78, moderate: 0.66, neutral: 0.50 },
    QQQ: { strong: 0.76, moderate: 0.64, neutral: 0.50 },
    IWM: { strong: 0.68, moderate: 0.58, neutral: 0.50 }
  }
}

type ConfidenceKey = 'strong' | 'moderate' | 'neutral'

// ========================
// CAROUSEL VIEWS
// ========================

type CarouselView = 'current' | 'swing1d' | 'swing3d' | 'next' | 'outlook'

const VIEW_CONFIG: Record<CarouselView, {
  title: string
  subtitle: string
  description: string
  timeframe: string
  color: string
}> = {
  current: {
    title: 'TODAY',
    subtitle: 'Intraday V6',
    description: 'Same-day close direction',
    timeframe: 'Close today',
    color: 'cyan'
  },
  swing1d: {
    title: '1-DAY',
    subtitle: 'Swing V6',
    description: 'Tomorrow close vs today',
    timeframe: '1 trading day',
    color: 'amber'
  },
  swing3d: {
    title: '3-DAY',
    subtitle: 'Swing V6',
    description: '3-day price direction (best accuracy)',
    timeframe: '3 trading days',
    color: 'emerald'
  },
  next: {
    title: '5-DAY',
    subtitle: 'Swing V6',
    description: '1-week price direction',
    timeframe: '5 trading days',
    color: 'purple'
  },
  outlook: {
    title: '10-DAY',
    subtitle: 'Swing V6',
    description: '2-week price direction',
    timeframe: '10 trading days',
    color: 'indigo'
  }
}

// ========================
// COMPONENT
// ========================

interface ModelCarouselProps {
  ticker?: string
}

export function ModelCarousel({ ticker = 'SPY' }: ModelCarouselProps) {
  const [view, setView] = useState<CarouselView>('current')
  const [data, setData] = useState<ModelData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch data from MTF endpoint
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const response = await fetch(`/api/v2/analysis/mtf?tickers=${ticker}`)
        if (!response.ok) throw new Error('Failed to fetch model data')

        const result = await response.json()
        const tickerData = result.tickers?.[ticker]

        if (tickerData?.error) {
          setError(tickerData.error)
          return
        }

        // Parse intraday
        const intraday = tickerData?.intraday?.v6 ? {
          target_a_prob: tickerData.intraday.v6.target_a_prob || 0.5,
          target_b_prob: tickerData.intraday.v6.target_b_prob || 0.5,
          session: tickerData.intraday.v6.session || 'early',
          signal: tickerData.intraday.v6.signal || 'NEUTRAL',
          accuracy: {
            early: 0.67,
            late_a: 0.89,
            late_b: 0.79
          }
        } : undefined

        // Parse swing
        const swing = tickerData?.swing?.v6_swing ? {
          prob_1d_up: tickerData.swing.v6_swing.prob_1d_up,
          prob_3d_up: tickerData.swing.v6_swing.prob_3d_up,
          prob_5d_up: tickerData.swing.v6_swing.prob_5d_up || 0.5,
          prob_10d_up: tickerData.swing.v6_swing.prob_10d_up || 0.5,
          signal_1d: tickerData.swing.v6_swing.signal_1d || 'NEUTRAL',
          signal_3d: tickerData.swing.v6_swing.signal_3d || 'NEUTRAL',
          signal_5d: tickerData.swing.v6_swing.signal_5d || 'NEUTRAL',
          signal_10d: tickerData.swing.v6_swing.signal_10d || 'NEUTRAL',
          accuracy: {
            acc_1d: ticker === 'SPY' ? 0.623 : ticker === 'QQQ' ? 0.656 : 0.652,
            acc_3d: ticker === 'SPY' ? 0.78 : ticker === 'QQQ' ? 0.788 : 0.74,
            acc_5d: ticker === 'SPY' ? 0.775 : ticker === 'QQQ' ? 0.762 : 0.675,
            acc_10d: ticker === 'SPY' ? 0.70 : ticker === 'QQQ' ? 0.679 : 0.571
          }
        } : undefined

        setData({
          ticker,
          current_price: tickerData?.current_price || 0,
          intraday,
          swing,
          alignment: tickerData?.alignment
        })
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 60000)
    return () => clearInterval(interval)
  }, [ticker])

  // Get color classes based on view
  const getViewColors = (v: CarouselView) => {
    const colors = {
      cyan: {
        bg: 'bg-cyan-500/10',
        border: 'border-cyan-500/50',
        text: 'text-cyan-400',
        button: 'bg-cyan-500 text-white',
        inactive: 'bg-gray-800 text-gray-400 hover:bg-gray-700'
      },
      amber: {
        bg: 'bg-amber-500/10',
        border: 'border-amber-500/50',
        text: 'text-amber-400',
        button: 'bg-amber-500 text-white',
        inactive: 'bg-gray-800 text-gray-400 hover:bg-gray-700'
      },
      emerald: {
        bg: 'bg-emerald-500/10',
        border: 'border-emerald-500/50',
        text: 'text-emerald-400',
        button: 'bg-emerald-500 text-white',
        inactive: 'bg-gray-800 text-gray-400 hover:bg-gray-700'
      },
      purple: {
        bg: 'bg-purple-500/10',
        border: 'border-purple-500/50',
        text: 'text-purple-400',
        button: 'bg-purple-500 text-white',
        inactive: 'bg-gray-800 text-gray-400 hover:bg-gray-700'
      },
      indigo: {
        bg: 'bg-indigo-500/10',
        border: 'border-indigo-500/50',
        text: 'text-indigo-400',
        button: 'bg-indigo-500 text-white',
        inactive: 'bg-gray-800 text-gray-400 hover:bg-gray-700'
      }
    }
    return colors[VIEW_CONFIG[v].color as keyof typeof colors]
  }

  const currentColors = getViewColors(view)
  const config = VIEW_CONFIG[view]

  // Loading state
  if (loading) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-center gap-2">
          <div className="animate-spin h-5 w-5 border-2 border-cyan-400 border-t-transparent rounded-full" />
          <span className="text-gray-400">Loading model predictions...</span>
        </div>
      </div>
    )
  }

  // Error state
  if (error || !data) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <div className="text-red-400">{error || 'No data available'}</div>
        <div className="text-gray-500 text-sm mt-1">Ensure ML server is running</div>
      </div>
    )
  }

  // Get signal direction based on probability thresholds
  // Confidence tiers: >80% = STRONG, >60% = ACTIONABLE, 40-60% = NEUTRAL, <40% = BEARISH
  const getSignalFromProb = (prob: number): {
    direction: 'BULLISH' | 'BEARISH' | 'NEUTRAL',
    action: string,
    confidence: 'STRONG' | 'MODERATE' | 'WEAK'
  } => {
    if (prob > 0.80) {
      return { direction: 'BULLISH', action: view === 'current' ? 'BUY CALL' : 'EXPECT HIGHER', confidence: 'STRONG' }
    } else if (prob > 0.60) {
      return { direction: 'BULLISH', action: view === 'current' ? 'BUY CALL' : 'EXPECT HIGHER', confidence: 'MODERATE' }
    } else if (prob < 0.20) {
      return { direction: 'BEARISH', action: view === 'current' ? 'BUY PUT' : 'EXPECT LOWER', confidence: 'STRONG' }
    } else if (prob < 0.40) {
      return { direction: 'BEARISH', action: view === 'current' ? 'BUY PUT' : 'EXPECT LOWER', confidence: 'MODERATE' }
    } else {
      return { direction: 'NEUTRAL', action: 'NO TRADE', confidence: 'WEAK' }
    }
  }

  // Get historical win rate - specific by ticker, model, and confidence
  const getHistoricalWinRate = (
    confidence: 'STRONG' | 'MODERATE' | 'WEAK',
    modelType: 'early' | 'late_a' | 'late_b' | '1d' | '3d' | '5d' | '10d'
  ): number => {
    const tickerKey = (ticker === 'SPY' || ticker === 'QQQ' || ticker === 'IWM') ? ticker : 'SPY'
    const confKey: ConfidenceKey = confidence === 'STRONG' ? 'strong' : confidence === 'MODERATE' ? 'moderate' : 'neutral'

    if (modelType === 'early') {
      return WIN_RATES.intraday.early[tickerKey][confKey]
    } else if (modelType === 'late_a') {
      return WIN_RATES.intraday.late_a[tickerKey][confKey]
    } else if (modelType === 'late_b') {
      return WIN_RATES.intraday.late_b[tickerKey][confKey]
    } else if (modelType === '1d') {
      return WIN_RATES.swing_1d[tickerKey][confKey]
    } else if (modelType === '3d') {
      return WIN_RATES.swing_3d[tickerKey][confKey]
    } else if (modelType === '5d') {
      return WIN_RATES.swing_5d[tickerKey][confKey]
    } else {
      return WIN_RATES.swing_10d[tickerKey][confKey]
    }
  }

  // Get prediction data for current view
  const getPrediction = () => {
    if (view === 'current') {
      if (!data.intraday) return null
      const prob = data.intraday.session === 'late'
        ? data.intraday.target_b_prob
        : data.intraday.target_a_prob
      const { direction, action, confidence } = getSignalFromProb(prob)
      const accuracy = data.intraday.session === 'late'
        ? data.intraday.accuracy.late_b
        : data.intraday.accuracy.early
      // Use late_b for late session (Close > 11 AM), early for morning
      const modelType = data.intraday.session === 'late' ? 'late_b' : 'early'
      return {
        probability: prob,
        direction,
        action,
        confidence,
        accuracy,
        historicalWinRate: getHistoricalWinRate(confidence, modelType),
        target: data.intraday.session === 'late' ? 'Close > 11 AM' : 'Close > Open'
      }
    } else if (view === 'swing1d') {
      if (!data.swing || data.swing.prob_1d_up === undefined) return null
      const prob = data.swing.prob_1d_up
      const { direction, action, confidence } = getSignalFromProb(prob)
      return {
        probability: prob,
        direction,
        action,
        confidence,
        accuracy: data.swing.accuracy.acc_1d || 0.62,
        historicalWinRate: getHistoricalWinRate(confidence, '1d'),
        target: 'Price tomorrow'
      }
    } else if (view === 'swing3d') {
      if (!data.swing || data.swing.prob_3d_up === undefined) return null
      const prob = data.swing.prob_3d_up
      const { direction, action, confidence } = getSignalFromProb(prob)
      return {
        probability: prob,
        direction,
        action,
        confidence,
        accuracy: data.swing.accuracy.acc_3d || 0.78,
        historicalWinRate: getHistoricalWinRate(confidence, '3d'),
        target: 'Price in 3 days'
      }
    } else if (view === 'next') {
      if (!data.swing) return null
      const prob = data.swing.prob_5d_up
      const { direction, action, confidence } = getSignalFromProb(prob)
      return {
        probability: prob,
        direction,
        action,
        confidence,
        accuracy: data.swing.accuracy.acc_5d,
        historicalWinRate: getHistoricalWinRate(confidence, '5d'),
        target: 'Price in 5 days'
      }
    } else {
      if (!data.swing) return null
      const prob = data.swing.prob_10d_up
      const { direction, action, confidence } = getSignalFromProb(prob)
      return {
        probability: prob,
        direction,
        action,
        confidence,
        accuracy: data.swing.accuracy.acc_10d,
        historicalWinRate: getHistoricalWinRate(confidence, '10d'),
        target: 'Price in 10 days'
      }
    }
  }

  const prediction = getPrediction()

  return (
    <div className={`bg-gray-900 border rounded-lg overflow-hidden ${currentColors.border}`}>
      {/* Header with view selector */}
      <div className={`px-4 py-3 border-b border-gray-800 ${currentColors.bg}`}>
        <div className="flex justify-between items-center">
          <div>
            <div className="flex items-center gap-2">
              <span className={`text-lg font-bold ${currentColors.text}`}>{config.title}</span>
              <span className="text-gray-400 text-sm">{config.subtitle}</span>
            </div>
            <p className="text-gray-500 text-xs">{config.description}</p>
          </div>

          {/* View Selector Buttons */}
          <div className="flex gap-1">
            {(['current', 'swing1d', 'swing3d', 'next'] as CarouselView[]).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${
                  view === v ? getViewColors(v).button : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                {VIEW_CONFIG[v].title}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="p-4">
        {prediction ? (
          <div className="space-y-4">
            {/* Direction & Probability */}
            <div className={`rounded-lg p-4 ${
              prediction.direction === 'BULLISH'
                ? 'bg-green-900/30 border border-green-500/50'
                : prediction.direction === 'BEARISH'
                ? 'bg-red-900/30 border border-red-500/50'
                : 'bg-yellow-900/20 border border-yellow-500/30'
            }`}>
              {/* Action + Confidence */}
              <div className={`text-center py-2 mb-3 rounded ${
                prediction.direction === 'BULLISH' ? 'bg-green-500/20' :
                prediction.direction === 'BEARISH' ? 'bg-red-500/20' :
                'bg-yellow-500/10'
              }`}>
                <span className={`text-xl font-bold ${
                  prediction.direction === 'BULLISH' ? 'text-green-400' :
                  prediction.direction === 'BEARISH' ? 'text-red-400' :
                  'text-yellow-400'
                }`}>
                  {prediction.action}
                </span>
                {prediction.confidence === 'STRONG' && (
                  <span className="ml-2 text-xs px-2 py-0.5 rounded bg-white/10 text-white font-medium">
                    HIGH CONVICTION
                  </span>
                )}
              </div>

              {/* Probability */}
              <div className="flex items-center justify-center gap-3">
                <span className={`text-4xl font-bold ${
                  prediction.direction === 'BULLISH' ? 'text-green-400' :
                  prediction.direction === 'BEARISH' ? 'text-red-400' :
                  'text-yellow-400'
                }`}>
                  {Math.round(prediction.probability * 100)}%
                </span>
                <div className="text-left">
                  <div className="text-gray-400 text-sm">
                    {prediction.direction === 'NEUTRAL' ? 'uncertain' : 'probability'}
                  </div>
                  <div className="text-gray-500 text-xs">{prediction.target}</div>
                </div>
              </div>

              {/* Neutral zone explanation */}
              {prediction.direction === 'NEUTRAL' && (
                <div className="mt-3 text-center text-xs text-yellow-400/70">
                  40-60% = No edge. Wait for clearer signal.
                </div>
              )}
            </div>

            {/* Model Info with Win Rate */}
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="bg-gray-800/50 rounded p-3">
                <div className="text-gray-500 text-xs">Historical Win Rate</div>
                <div className={`text-lg font-bold ${
                  prediction.confidence === 'STRONG' ? 'text-emerald-400' :
                  prediction.confidence === 'MODERATE' ? 'text-blue-400' :
                  'text-gray-400'
                }`}>
                  {Math.round(prediction.historicalWinRate * 100)}%
                </div>
              </div>
              <div className="bg-gray-800/50 rounded p-3">
                <div className="text-gray-500 text-xs">Timeframe</div>
                <div className="text-white font-medium">
                  {config.timeframe}
                </div>
              </div>
            </div>

            {/* Alignment Status (if available) */}
            {data.alignment && view !== 'current' && (
              <div className={`rounded-lg p-3 border ${
                data.alignment.status === 'ALIGNED' ? 'border-green-500/50 bg-green-900/20' :
                data.alignment.status === 'CONFLICT' ? 'border-red-500/50 bg-red-900/20' :
                'border-yellow-500/50 bg-yellow-900/20'
              }`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${
                      data.alignment.status === 'ALIGNED' ? 'bg-green-500' :
                      data.alignment.status === 'CONFLICT' ? 'bg-red-500' : 'bg-yellow-500'
                    }`} />
                    <span className="text-gray-400 text-sm">Intraday + Swing:</span>
                  </div>
                  <span className={`font-bold ${
                    data.alignment.status === 'ALIGNED' ? 'text-green-400' :
                    data.alignment.status === 'CONFLICT' ? 'text-red-400' : 'text-yellow-400'
                  }`}>
                    {data.alignment.status}
                  </span>
                </div>
                {data.alignment.status === 'CONFLICT' && (
                  <div className="text-yellow-400/80 text-xs mt-1">
                    Consider reducing position size
                  </div>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            {view === 'current' ? 'Intraday model not available' : 'Swing model not available'}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="px-4 py-2 border-t border-gray-800 bg-gray-900/50">
        <div className="flex justify-between items-center text-xs text-gray-500">
          <span>{ticker} • ${data.current_price.toFixed(2)}</span>
          <span>
            {view === 'current' ? 'V6 Intraday Model' : 'V6 Swing Model'}
          </span>
        </div>
      </div>
    </div>
  )
}
