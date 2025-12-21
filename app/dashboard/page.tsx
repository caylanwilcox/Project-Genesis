'use client'

import { useRouter } from 'next/navigation'
import { useState, useEffect, useMemo } from 'react'
import { useMultiTickerData } from '@/hooks/useMultiTickerData'
import { MLMorningBriefing } from '@/components/MLMorningBriefing'
import { VolatilityMeter } from '@/components/VolatilityMeter'
import { TradingDirections } from '@/components/TradingDirections'

// ML Signals types
interface IntradayModel {
  probability: number
  confidence: number
  time_pct: number
  session_label: string
  current_vs_open: number
  position_in_range: number
  model_accuracy: number
}

interface MLTickerSignal {
  signal: 'BUY' | 'SELL' | 'HOLD'
  strength: 'STRONG' | 'MODERATE' | 'WEAK' | 'NEUTRAL'
  probability: number
  confidence: number
  current_price: number
  target_price: number
  stop_loss: number
  predicted_range: { high: number; low: number }
  highlow_model?: {
    predicted_high: number
    predicted_low: number
    high_pct: number
    low_pct: number
  }
  intraday_model?: IntradayModel
  prediction_source?: 'daily' | 'intraday'
  model_accuracy: number
}

interface MLSignalsData {
  tickers: Record<string, MLTickerSignal>
  is_after_hours: boolean
}

interface TickerData {
  symbol: string
  displaySymbol: string  // What to show (e.g., "VIX" for VIXY)
  signal: 'strong_buy' | 'buy' | 'neutral' | 'sell' | 'strong_sell'
  action: 'BUY NOW' | 'WAIT' | 'SELL NOW' | 'HOLD' | 'EXIT'
  urgency: 'IMMEDIATE' | 'SOON' | 'WATCH' | 'LOW'
  timeToAction: string
  confidence: number
  recommendation: string
  price: number
  change: number
  changePercent: number
  volume: string
  rsi: number
  macd: string
  momentum: number
  supportLevel: number
  resistanceLevel: number
  volatility: 'Low' | 'Medium' | 'High' | 'Extreme'
  trend: 'Bullish' | 'Bearish' | 'Neutral'
  institutionalFlow: 'Accumulation' | 'Distribution' | 'Mixed'
  optionFlow: 'Bullish' | 'Bearish' | 'Neutral'
  entryWindow: string
  exitTarget: number
  stopLoss: number
  techSignal: string
  backtestedAccuracy: number
  sellPrice: number
  estimatedTimeInTrade: string
  timeDecay: number  // Time decay factor (theta-like) - hourly decay rate
  timeDecayLabel: 'LOW' | 'MODERATE' | 'HIGH' | 'EXTREME'
  // ML Model fields
  mlProbability?: number
  mlModelAccuracy?: number
  mlSessionLabel?: string
  mlTimePct?: number
  mlPredictionSource?: 'daily' | 'intraday'
}

// Map internal symbols to display names
const SYMBOL_DISPLAY_MAP: Record<string, string> = {
  'VIXY': 'VIX',  // VIXY ETF tracks VIX futures
  'SPY': 'SPY',
  'QQQ': 'QQQ',
  'IWM': 'IWM',
}

const SYMBOLS = ['SPY', 'VIXY', 'QQQ', 'IWM']

export default function Dashboard() {
  const router = useRouter()
  const [currentTime, setCurrentTime] = useState(new Date())
  const [marketStatus, setMarketStatus] = useState<'Pre-Market' | 'Open' | 'Closed' | 'After-Hours'>('Open')
  const [timeRemaining, setTimeRemaining] = useState<Record<string, number>>({})
  const [mlSignals, setMlSignals] = useState<MLSignalsData | null>(null)

  // Fetch real ticker data from Polygon.io
  // Use faster refresh when on Starter+ plan (snapshots + relaxed rate limits)
  const planEnv = process.env.NEXT_PUBLIC_POLYGON_PLAN?.toLowerCase()
  const refreshMs = planEnv === 'starter' || planEnv === 'developer' ? 5000 : 60000
  const { tickers: polygonTickers, isLoading, error } = useMultiTickerData(SYMBOLS, true, refreshMs)

  // Fetch ML signals
  useEffect(() => {
    const fetchMLSignals = async () => {
      try {
        const response = await fetch('/api/v2/ml/daily-signals')
        if (response.ok) {
          const data = await response.json()
          setMlSignals(data)
        }
      } catch (err) {
        console.error('Failed to fetch ML signals:', err)
      }
    }

    fetchMLSignals()
    // Refresh ML signals every minute
    const interval = setInterval(fetchMLSignals, 60000)
    return () => clearInterval(interval)
  }, [])

  // Calculate signals and metrics based on real data + ML signals
  const tickers = useMemo(() => {
    const result: TickerData[] = []

    SYMBOLS.forEach(symbol => {
      const polygonData = polygonTickers.get(symbol)

      if (!polygonData) return

      const { price, change, changePercent, volume, high, low, prevClose } = polygonData

      // Get ML signal for this ticker (skip VIXY as ML doesn't cover it)
      const mlSignal = mlSignals?.tickers?.[symbol]
      const useIntraday = mlSignal?.intraday_model && mlSignal?.prediction_source === 'intraday'
      const mlProb = useIntraday
        ? mlSignal.intraday_model!.probability
        : (mlSignal?.probability ?? 0.5)
      const mlAccuracy = useIntraday
        ? mlSignal.intraday_model!.model_accuracy
        : (mlSignal?.model_accuracy ?? 0.5)

      // Calculate technical indicators (simplified)
      const priceRange = high - low
      const volatilityPercent = (priceRange / price) * 100

      // Determine volatility
      let volatility: TickerData['volatility'] = 'Low'
      if (volatilityPercent > 3) volatility = 'Extreme'
      else if (volatilityPercent > 2) volatility = 'High'
      else if (volatilityPercent > 1) volatility = 'Medium'

      // Calculate momentum based on price change
      const momentum = Math.min(100, Math.max(0, 50 + (changePercent * 10)))

      // Simple RSI approximation based on momentum
      const rsi = Math.round(30 + (momentum * 0.6))

      // Determine trend
      let trend: TickerData['trend'] = 'Neutral'
      if (changePercent > 0.5) trend = 'Bullish'
      else if (changePercent < -0.5) trend = 'Bearish'

      // MACD status based on trend
      let macd = 'Converging'
      if (trend === 'Bullish') macd = changePercent > 1 ? 'Bullish Cross' : 'Bullish'
      else if (trend === 'Bearish') macd = changePercent < -1 ? 'Bearish Cross' : 'Bearish'

      // Use ML probability for confidence (convert to percentage)
      const confidence = mlSignal
        ? Math.round(mlProb * 100)
        : Math.round(Math.min(95, Math.max(20, 50 + Math.abs(changePercent) * 10)))

      // Derive signal from ML probability
      let signal: TickerData['signal'] = 'neutral'
      let recommendation = 'NEUTRAL'
      let action: TickerData['action'] = 'WAIT'
      let urgency: TickerData['urgency'] = 'WATCH'
      let timeToAction = 'MONITORING'
      let institutionalFlow: TickerData['institutionalFlow'] = 'Mixed'
      let optionFlow: TickerData['optionFlow'] = 'Neutral'
      let techSignal = 'ML Intraday Prediction'
      let backtestedAccuracy = mlAccuracy * 100

      // Signal based on ML probability (will price go UP from current?)
      if (mlProb >= 0.7) {
        signal = 'strong_buy'
        recommendation = 'STRONG BUY'
        action = 'BUY NOW'
        urgency = 'IMMEDIATE'
        timeToAction = mlSignal?.intraday_model?.session_label || '< 5 MIN'
        institutionalFlow = 'Accumulation'
        optionFlow = 'Bullish'
        techSignal = `${Math.round(mlProb * 100)}% chance price goes UP`
      } else if (mlProb >= 0.6) {
        signal = 'buy'
        recommendation = 'BUY'
        action = 'BUY NOW'
        urgency = 'SOON'
        timeToAction = mlSignal?.intraday_model?.session_label || '< 15 MIN'
        institutionalFlow = 'Accumulation'
        optionFlow = 'Bullish'
        techSignal = `${Math.round(mlProb * 100)}% chance price goes UP`
      } else if (mlProb >= 0.55) {
        signal = 'buy'
        recommendation = 'LEAN BUY'
        action = 'WAIT'
        urgency = 'WATCH'
        timeToAction = 'MONITORING'
        optionFlow = 'Bullish'
        techSignal = `${Math.round(mlProb * 100)}% chance price goes UP`
      } else if (mlProb <= 0.3) {
        signal = 'strong_sell'
        recommendation = 'EXIT NOW'
        action = 'EXIT'
        urgency = 'IMMEDIATE'
        timeToAction = mlSignal?.intraday_model?.session_label || 'IMMEDIATELY'
        institutionalFlow = 'Distribution'
        optionFlow = 'Bearish'
        techSignal = `${Math.round((1 - mlProb) * 100)}% chance price goes DOWN`
      } else if (mlProb <= 0.4) {
        signal = 'sell'
        recommendation = 'SELL'
        action = 'SELL NOW'
        urgency = 'SOON'
        timeToAction = mlSignal?.intraday_model?.session_label || '< 30 MIN'
        institutionalFlow = 'Distribution'
        optionFlow = 'Bearish'
        techSignal = `${Math.round((1 - mlProb) * 100)}% chance price goes DOWN`
      } else if (mlProb <= 0.45) {
        signal = 'sell'
        recommendation = 'LEAN SELL'
        action = 'WAIT'
        urgency = 'WATCH'
        timeToAction = 'MONITORING'
        optionFlow = 'Bearish'
        techSignal = `${Math.round((1 - mlProb) * 100)}% chance price goes DOWN`
      }

      // Use ML predicted range for price levels if available
      const mlHigh = mlSignal?.highlow_model?.predicted_high ?? mlSignal?.predicted_range?.high
      const mlLow = mlSignal?.highlow_model?.predicted_low ?? mlSignal?.predicted_range?.low

      const supportLevel = mlLow ?? price * 0.99
      const resistanceLevel = mlHigh ?? price * 1.01
      const exitTarget = mlHigh ?? price * 1.01
      const stopLoss = mlLow ?? price * 0.99
      const sellPrice = mlHigh ?? price * 1.01

      // Entry window
      const entryWindow = action === 'BUY NOW'
        ? `NOW - $${(price * 0.999).toFixed(2)}`
        : action === 'SELL NOW'
        ? `SHORT - $${(price * 1.001).toFixed(2)}`
        : `WAIT - $${supportLevel.toFixed(2)}`

      // Estimated time in trade
      let estimatedTimeInTrade = 'Wait for signal'
      if (action === 'BUY NOW' || action === 'SELL NOW') {
        estimatedTimeInTrade = urgency === 'IMMEDIATE' ? '30-60 min' : '1-3 hours'
      }

      // Calculate time decay factor (theta-like for options)
      // Based on: time remaining in day + volatility + price movement
      const now = new Date()
      const etTime = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }))
      const hours = etTime.getHours()
      const minutes = etTime.getMinutes()

      // Calculate time remaining in trading day (9:30 AM - 4:00 PM ET)
      let hoursRemaining = 0
      if (hours >= 9 && hours < 16) {
        hoursRemaining = 16 - hours - (minutes / 60)
        if (hours === 9 && minutes < 30) hoursRemaining = 6.5
      }

      // Time decay accelerates as end of day approaches
      // Formula: decay = baseDecay * (1 + (1 - hoursRemaining/6.5) * volatilityMultiplier)
      const baseDecay = volatilityPercent * 0.1  // Base hourly decay as % of price
      const timeMultiplier = hoursRemaining > 0 ? Math.pow((6.5 - hoursRemaining) / 6.5, 1.5) : 1
      const volatilityMultiplier = volatility === 'Extreme' ? 3 : volatility === 'High' ? 2 : volatility === 'Medium' ? 1.5 : 1
      const timeDecay = Math.round(baseDecay * (1 + timeMultiplier) * volatilityMultiplier * 100) / 100

      // Classify time decay
      let timeDecayLabel: TickerData['timeDecayLabel'] = 'LOW'
      if (timeDecay >= 0.5) timeDecayLabel = 'EXTREME'
      else if (timeDecay >= 0.3) timeDecayLabel = 'HIGH'
      else if (timeDecay >= 0.15) timeDecayLabel = 'MODERATE'

      result.push({
        symbol,
        displaySymbol: SYMBOL_DISPLAY_MAP[symbol] || symbol,
        signal,
        action,
        urgency,
        timeToAction,
        confidence,
        recommendation,
        price,
        change,
        changePercent,
        volume,
        rsi,
        macd,
        momentum: Math.round(momentum),
        supportLevel,
        resistanceLevel,
        volatility,
        trend,
        institutionalFlow,
        optionFlow,
        entryWindow,
        exitTarget,
        stopLoss,
        techSignal,
        backtestedAccuracy: Math.round(backtestedAccuracy * 10) / 10,
        sellPrice,
        estimatedTimeInTrade,
        timeDecay,
        timeDecayLabel,
        // ML Model fields
        mlProbability: mlProb,
        mlModelAccuracy: mlAccuracy,
        mlSessionLabel: mlSignal?.intraday_model?.session_label,
        mlTimePct: mlSignal?.intraday_model?.time_pct,
        mlPredictionSource: mlSignal?.prediction_source,
      })
    })

    return result
  }, [polygonTickers, mlSignals])

  // Initialize countdown timers
  useEffect(() => {
    const initialTimers: Record<string, number> = {}
    tickers.forEach(ticker => {
      if (ticker.urgency === 'IMMEDIATE') {
        initialTimers[ticker.symbol] = 300 // 5 minutes
      } else if (ticker.urgency === 'SOON') {
        initialTimers[ticker.symbol] = 900 // 15 minutes
      }
    })
    setTimeRemaining(initialTimers)
  }, [tickers])

  // Update clock and countdown timers
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(new Date())

      // Update countdown timers
      setTimeRemaining(prev => {
        const updated = { ...prev }
        Object.keys(updated).forEach(symbol => {
          if (updated[symbol] > 0) {
            updated[symbol] -= 1
          }
        })
        return updated
      })
    }, 1000)

    return () => clearInterval(interval)
  }, [])

  // Determine market status based on time
  useEffect(() => {
    const updateMarketStatus = () => {
      // Convert to Eastern Time
      const now = new Date()
      const etTime = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }))
      const hours = etTime.getHours()
      const minutes = etTime.getMinutes()
      const day = etTime.getDay()

      // Weekend
      if (day === 0 || day === 6) {
        setMarketStatus('Closed')
        return
      }

      // Weekday - check against ET market hours
      if (hours < 9 || (hours === 9 && minutes < 30)) {
        setMarketStatus('Pre-Market')
      } else if (hours < 16) {
        setMarketStatus('Open')
      } else if (hours < 20) {
        setMarketStatus('After-Hours')
      } else {
        setMarketStatus('Closed')
      }
    }

    updateMarketStatus()
    const interval = setInterval(updateMarketStatus, 60000)
    return () => clearInterval(interval)
  }, [])

  const formatTimer = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const getActionColor = (action: TickerData['action']) => {
    switch (action) {
      case 'BUY NOW': return 'bg-green-500 animate-pulse'
      case 'SELL NOW': return 'bg-red-500 animate-pulse'
      case 'EXIT': return 'bg-red-600 animate-pulse'
      case 'HOLD': return 'bg-blue-500'
      case 'WAIT': return 'bg-gray-600'
    }
  }

  const getSignalColor = (signal: TickerData['signal']) => {
    switch (signal) {
      case 'strong_buy': return 'from-green-600 to-green-400'
      case 'buy': return 'from-green-500 to-green-300'
      case 'neutral': return 'from-yellow-600 to-yellow-400'
      case 'sell': return 'from-red-500 to-red-300'
      case 'strong_sell': return 'from-red-600 to-red-400'
    }
  }

  const getUrgencyColor = (urgency: TickerData['urgency']) => {
    switch (urgency) {
      case 'IMMEDIATE': return 'text-red-400 animate-pulse'
      case 'SOON': return 'text-yellow-400'
      case 'WATCH': return 'text-blue-400'
      case 'LOW': return 'text-gray-400'
    }
  }

  if (isLoading && tickers.length === 0) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-green-400 mb-4"></div>
          <div className="text-white text-xl font-medium">Loading Live Market Data...</div>
          <div className="text-gray-400 text-sm mt-2">Fetching real-time data from Polygon.io</div>
          <div className="text-gray-500 text-xs mt-2">This may take up to 15 seconds...</div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center max-w-md">
          <div className="text-red-400 text-6xl mb-4">⚠️</div>
          <div className="text-white text-xl font-medium mb-2">Unable to Load Market Data</div>
          <div className="text-gray-400 text-sm mb-4">{error.message}</div>
          <div className="text-yellow-400 text-xs">
            Please ensure your Polygon.io API key is configured in .env.local
          </div>
        </div>
      </div>
    )
  }

  if (tickers.length === 0 && !isLoading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center max-w-md">
          <div className="text-yellow-400 text-6xl mb-4">📊</div>
          <div className="text-white text-xl font-medium">No Market Data Available</div>
          <div className="text-gray-400 text-sm mt-2">
            {error ? error.message : 'Please check your internet connection and try again'}
          </div>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-black relative overflow-x-hidden">
      {/* Market Status Bar */}
      <div className="absolute top-0 left-0 right-0 bg-gray-900 border-b border-gray-800 px-2 md:px-4 py-2 z-10">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2 md:gap-6">
            <div className="flex items-center gap-1 md:gap-2">
              <div className={`w-2 h-2 rounded-full animate-pulse ${
                marketStatus === 'Open' ? 'bg-green-400' :
                marketStatus === 'Closed' ? 'bg-red-400' :
                'bg-yellow-400'
              }`}></div>
              <span className={`text-xs md:text-sm font-medium ${
                marketStatus === 'Open' ? 'text-green-400' :
                marketStatus === 'Closed' ? 'text-red-400' :
                'text-yellow-400'
              }`}>MARKET {marketStatus.toUpperCase()}</span>
            </div>
            <span className="text-gray-400 text-xs md:text-sm hidden sm:block">{currentTime.toLocaleTimeString()}</span>
          </div>
          <div className="flex items-center gap-2 md:gap-4 text-xs md:text-sm">
            <div className="text-gray-400 hidden md:block">
              <span className="text-gray-500">Data Source:</span>
              <span className="text-cyan-400 ml-2">Polygon.io</span>
            </div>
            <div className="text-gray-400">
              <span className="text-gray-500 hidden md:inline">Live Data</span>
              {isLoading ? (
                <span className="text-yellow-400 ml-1 md:ml-2 animate-pulse">⟳</span>
              ) : (
                <span className="text-green-400 ml-1 md:ml-2">✓</span>
              )}
            </div>
            <div className="text-gray-400">
              <span className="text-gray-500 hidden md:inline">Active Signals:</span>
              <span className="text-yellow-400 ml-1 md:ml-2 font-bold">
                {tickers.filter(t => t.action === 'BUY NOW' || t.action === 'SELL NOW').length}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ML Morning Briefing & Volatility Meter */}
      <div className="px-2 pt-14 md:pt-12 md:px-4 space-y-2">
        <TradingDirections />
        <VolatilityMeter />
        <MLMorningBriefing />
      </div>

      {/* Responsive Grid Layout */}
      <div className="min-h-screen pt-4 md:pt-4">
        <div className="flex flex-col gap-2 p-2 md:grid md:grid-cols-2 md:grid-rows-2 md:gap-0 md:h-[calc(100vh-12rem)] md:p-0">
            {tickers.map((ticker) => (
              <div
                key={ticker.symbol}
                onClick={() => {
                  router.push(`/ticker/${ticker.symbol}`)
                }}
                className="relative cursor-pointer border border-gray-800 transition-all duration-300 hover:border-gray-600 bg-gray-900 overflow-hidden rounded-lg flex-1 md:rounded-none md:min-h-0"
              >
                {/* Animated Background */}
                <div className={`absolute inset-0 opacity-10 bg-gradient-to-br ${getSignalColor(ticker.signal)}`}></div>
                <div className="relative h-full flex flex-col p-2 md:p-3 overflow-hidden">
                  {/* Top Section - Header */}
                  <div className="flex-shrink-0">
                    <div className="flex justify-between items-start mb-1 md:mb-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1 md:gap-2 mb-1">
                          <h2 className="text-lg md:text-2xl font-bold text-white truncate">{ticker.displaySymbol}</h2>
                          <div className={`px-1 md:px-2 py-0.5 rounded text-[10px] md:text-xs font-bold text-white ${getActionColor(ticker.action)}`}>
                            {ticker.action}
                          </div>
                        </div>
                        <div className="text-sm md:text-xl font-light text-gray-300">
                          ${ticker.price.toFixed(2)}
                        </div>
                        <div className={`text-xs md:text-sm font-medium ${ticker.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {ticker.change >= 0 ? '↑' : '↓'} {Math.abs(ticker.change).toFixed(2)} ({ticker.changePercent >= 0 ? '+' : ''}{ticker.changePercent.toFixed(2)}%)
                        </div>
                      </div>
                      <div className="text-right flex-shrink-0">
                        <div className={`text-sm md:text-base font-bold ${getUrgencyColor(ticker.urgency)}`}>
                          {ticker.urgency}
                        </div>
                        {timeRemaining[ticker.symbol] && timeRemaining[ticker.symbol] > 0 && (
                          <div className="mt-1">
                            <div className="text-[10px] md:text-xs text-gray-500">TIME LEFT</div>
                            <div className="text-xs md:text-sm font-mono text-yellow-400">
                              {formatTimer(timeRemaining[ticker.symbol])}
                            </div>
                          </div>
                        )}
                        <div className="mt-1 text-[10px] md:text-xs text-gray-500 truncate">
                          {ticker.timeToAction}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Key Levels Section */}
                  <div className="flex-shrink-0 mb-1 md:mb-2">
                    <div className="flex justify-between">
                      <div className="space-y-0.5 md:space-y-1 text-xs">
                        <div>
                          <div className="text-gray-500 text-[10px] md:text-xs">SELL</div>
                          <div className="text-yellow-400 font-bold text-xs md:text-sm">${ticker.sellPrice.toFixed(2)}</div>
                        </div>
                        <div>
                          <div className="text-gray-500 text-[10px] md:text-xs">TARGET</div>
                          <div className="text-green-400 font-bold text-xs md:text-sm">${ticker.exitTarget.toFixed(2)}</div>
                        </div>
                        <div>
                          <div className="text-gray-500 text-[10px] md:text-xs">STOP</div>
                          <div className="text-red-400 font-bold text-xs md:text-sm">${ticker.stopLoss.toFixed(2)}</div>
                        </div>
                      </div>
                      <div className="text-right space-y-1">
                        <div>
                          <div className="text-gray-500 text-[10px] md:text-xs">TIME</div>
                          <div className="text-cyan-400 font-bold text-[10px] md:text-xs">{ticker.estimatedTimeInTrade}</div>
                        </div>
                        <div>
                          <div className="text-gray-500 text-[10px] md:text-xs">DECAY</div>
                          <div className={`font-bold text-[10px] md:text-xs ${
                            ticker.timeDecayLabel === 'EXTREME' ? 'text-red-400' :
                            ticker.timeDecayLabel === 'HIGH' ? 'text-orange-400' :
                            ticker.timeDecayLabel === 'MODERATE' ? 'text-yellow-400' :
                            'text-green-400'
                          }`}>
                            {ticker.timeDecayLabel} ({ticker.timeDecay}%/hr)
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Center Section - Signal */}
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <div className="text-center">
                      <div className={`inline-block px-3 md:px-6 py-2 md:py-4 rounded md:rounded-lg border md:border-2 ${
                        ticker.signal === 'strong_buy' ? 'border-green-400 md:shadow-green-400/50' :
                        ticker.signal === 'buy' ? 'border-green-300 md:shadow-green-300/30' :
                        ticker.signal === 'neutral' ? 'border-yellow-400 md:shadow-yellow-400/30' :
                        ticker.signal === 'sell' ? 'border-red-300 md:shadow-red-300/30' :
                        'border-red-400 md:shadow-red-400/50'
                      } md:shadow-lg bg-gray-900/90 backdrop-blur-sm`}>
                        <div className={`text-lg md:text-2xl font-bold md:tracking-wider bg-gradient-to-r ${getSignalColor(ticker.signal)} bg-clip-text text-transparent`}>
                          {ticker.recommendation}
                        </div>
                        <div className="text-[10px] md:text-xs md:mt-1 text-gray-400">
                          {ticker.signal === 'strong_buy' ? '🔥 Max Opportunity' :
                           ticker.signal === 'buy' ? '✓ Good Entry' :
                           ticker.signal === 'neutral' ? '⏸ Wait' :
                           ticker.signal === 'sell' ? '⚠️ Consider Exit' :
                           '🚨 Exit Now'}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Bottom Right Corner - AI Model Accuracy */}
                  <div className="absolute bottom-2 md:bottom-3 right-2 md:right-3 text-right">
                    <div className="text-white font-bold text-sm md:text-lg">{ticker.confidence}%</div>
                    <div className="text-gray-500 text-[10px] md:text-xs">ML Probability</div>
                    {ticker.mlModelAccuracy && (
                      <div className="text-cyan-400 text-[10px] md:text-xs mt-0.5">
                        {Math.round(ticker.mlModelAccuracy * 100)}% Model Accuracy
                      </div>
                    )}
                  </div>

                  {/* Desktop Hover Overlay */}
                  <div className="hidden md:block absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent opacity-0 hover:opacity-100 transition-opacity duration-300 pointer-events-none md:flex items-end">
                    <div className="p-4 w-full">
                      <div className="text-white text-base font-medium">View Full Analysis →</div>
                      <div className="text-gray-400 text-xs mt-1">Charts, execution, risk management</div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  )
}
