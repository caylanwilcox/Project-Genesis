'use client'

import { useRouter } from 'next/navigation'
import { useState, useEffect, useMemo } from 'react'
import { useMultiTickerData } from '@/hooks/useMultiTickerData'
import { NorthstarPanel } from '@/components/NorthstarPanel'
import { ModelCarousel } from '@/components/ModelCarousel'
import { SwingRecommendation, type SwingData } from '@/components/SwingRecommendation'
import { IntradayRecommendation } from '@/components/IntradayRecommendation'
import { polygonWebSocketService, Trade } from '@/services/polygonWebSocket'

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
  model_accuracy: {
    early: number
    late_a: number
    late_b: number
  }
}

// Convert LONG/SHORT to user-friendly actions
const getActionLabel = (action: string, isBullish: boolean): string => {
  if (action === 'NO_TRADE') return 'NO TRADE'
  return isBullish ? 'BUY CALL' : 'BUY PUT'
}

const getActionDescription = (action: string, session: 'early' | 'late', price11am: number | null, todayOpen: number | null): string => {
  if (action === 'NO_TRADE') return 'Signal not strong enough - wait for clearer direction'
  if (session === 'late' && price11am) {
    const target = action === 'LONG' ? 'above' : 'below'
    return `Model predicts close will be ${target} $${price11am.toFixed(2)} (11 AM price)`
  }
  if (todayOpen) {
    const target = action === 'LONG' ? 'above' : 'below'
    return `Model predicts close will be ${target} $${todayOpen.toFixed(2)} (open price)`
  }
  return action === 'LONG' ? 'Bullish signal - expecting price to rise' : 'Bearish signal - expecting price to fall'
}

interface TradingData {
  current_time_et: string
  session: 'early' | 'late'
  market_open: boolean
  tickers: Record<string, TickerDirection>
  best_ticker: string | null
  summary: {
    recommendation: string
  }
}

const SYMBOLS = ['SPY', 'QQQ', 'IWM', 'UVXY']

// Track locked signals to prevent flip-flopping
interface LockedSignal {
  direction: 'BULLISH' | 'BEARISH'
  confidence: number
  lockedAt: string // hour when locked, e.g. "12:00"
}
type LockedSignals = Record<string, LockedSignal>

// Countdown component for Target B (activates at 12 PM ET - late session)
function TargetBCountdown() {
  const [timeLeft, setTimeLeft] = useState('')

  useEffect(() => {
    const updateCountdown = () => {
      const now = new Date()
      const etTime = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }))
      const hours = etTime.getHours()
      const minutes = etTime.getMinutes()
      const seconds = etTime.getSeconds()

      // Target B activates at 12:00 PM ET (late session starts)
      if (hours >= 12) {
        setTimeLeft('Active')
        return
      }

      // Calculate time until 12 PM
      const targetHour = 12
      const hoursLeft = targetHour - hours - 1
      const minutesLeft = 59 - minutes
      const secondsLeft = 60 - seconds

      if (hoursLeft > 0) {
        setTimeLeft(`${hoursLeft}h ${minutesLeft}m`)
      } else {
        setTimeLeft(`${minutesLeft}m ${secondsLeft}s`)
      }
    }

    updateCountdown()
    const interval = setInterval(updateCountdown, 1000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="flex justify-between items-center">
      <span className="text-gray-500 text-xs">Target B <span className="text-gray-600">(12 PM ET)</span></span>
      <span className="text-yellow-400 text-sm font-mono">{timeLeft}</span>
    </div>
  )
}

// Live price data from WebSocket
interface LivePriceData {
  price: number
  lastUpdate: number
  flash: 'up' | 'down' | null
}

export default function Dashboard() {
  const router = useRouter()
  const [currentTime, setCurrentTime] = useState(new Date())
  const [marketStatus, setMarketStatus] = useState<'Pre-Market' | 'Open' | 'Closed' | 'After-Hours'>('Open')
  const [tradingData, setTradingData] = useState<TradingData | null>(null)
  const [loading, setLoading] = useState(true)
  const [lockedSignals, setLockedSignals] = useState<LockedSignals>({})
  const [livePrices, setLivePrices] = useState<Map<string, LivePriceData>>(new Map())
  const [wsConnected, setWsConnected] = useState(false)
  const [swingDataMap, setSwingDataMap] = useState<Map<string, SwingData>>(new Map())
  const [intradayDataMap, setIntradayDataMap] = useState<Map<string, any>>(new Map())

  // Get current hour in ET for signal locking
  const getCurrentHourET = () => {
    const now = new Date()
    const etTime = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }))
    return `${etTime.getHours()}:00`
  }

  // Lock a signal for a ticker
  const lockSignal = (symbol: string, direction: 'BULLISH' | 'BEARISH', confidence: number) => {
    const hourKey = `${symbol}_${getCurrentHourET()}`
    if (!lockedSignals[hourKey]) {
      setLockedSignals(prev => ({
        ...prev,
        [hourKey]: { direction, confidence, lockedAt: getCurrentHourET() }
      }))
    }
  }

  // Check if signal has flipped from locked
  const hasSignalFlipped = (symbol: string, currentDirection: 'BULLISH' | 'BEARISH'): LockedSignal | null => {
    const hourKey = `${symbol}_${getCurrentHourET()}`
    const locked = lockedSignals[hourKey]
    if (locked && locked.direction !== currentDirection) {
      return locked
    }
    return null
  }

  // Fetch real ticker data from Polygon.io
  const { tickers: polygonTickers, isLoading, error } = useMultiTickerData(SYMBOLS, true, 30000)

  // Fetch V6 trading directions
  useEffect(() => {
    const fetchDirections = async () => {
      try {
        const response = await fetch('/api/v2/trading-directions')
        if (response.ok) {
          const data = await response.json()
          setTradingData(data)
        }
      } catch (err) {
        console.error('Failed to fetch trading directions:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchDirections()
    const interval = setInterval(fetchDirections, 60000)
    return () => clearInterval(interval)
  }, [])

  // Fetch swing and intraday data for AI recommendations
  useEffect(() => {
    const fetchSwingData = async () => {
      try {
        // Fetch for each symbol
        for (const symbol of SYMBOLS) {
          const response = await fetch(`/api/v2/northstar?ticker=${symbol}`)
          if (response.ok) {
            const data = await response.json()
            const tickerData = data.tickers?.[symbol]
            if (tickerData?.swing) {
              setSwingDataMap(prev => new Map(prev).set(symbol, tickerData.swing))
            }
            if (tickerData?.northstar) {
              setIntradayDataMap(prev => new Map(prev).set(symbol, tickerData.northstar))
            }
          }
        }
      } catch (err) {
        console.error('Failed to fetch swing data:', err)
      }
    }

    fetchSwingData()
    const interval = setInterval(fetchSwingData, 60000)
    return () => clearInterval(interval)
  }, [])

  // Update clock
  useEffect(() => {
    const interval = setInterval(() => setCurrentTime(new Date()), 1000)
    return () => clearInterval(interval)
  }, [])

  // Determine market status
  useEffect(() => {
    const updateMarketStatus = () => {
      const now = new Date()
      const etTime = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }))
      const hours = etTime.getHours()
      const minutes = etTime.getMinutes()
      const day = etTime.getDay()

      if (day === 0 || day === 6) {
        setMarketStatus('Closed')
        return
      }

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

  // WebSocket subscription for instant price updates
  useEffect(() => {
    // Connect to WebSocket
    polygonWebSocketService.connect()
      .then(() => setWsConnected(true))
      .catch(() => setWsConnected(false))

    // Subscribe to all symbols
    const unsubscribes = SYMBOLS.map(symbol => {
      return polygonWebSocketService.onTrade(symbol, (trade: Trade) => {
        setLivePrices(prev => {
          const newMap = new Map(prev)
          const current = prev.get(symbol)
          const flash: 'up' | 'down' | null = current
            ? (trade.price > current.price ? 'up' : trade.price < current.price ? 'down' : null)
            : null

          newMap.set(symbol, {
            price: trade.price,
            lastUpdate: Date.now(),
            flash
          })

          // Clear flash after animation
          if (flash) {
            setTimeout(() => {
              setLivePrices(p => {
                const updated = new Map(p)
                const data = updated.get(symbol)
                if (data) {
                  updated.set(symbol, { ...data, flash: null })
                }
                return updated
              })
            }, 300)
          }

          return newMap
        })
      })
    })

    return () => {
      unsubscribes.forEach(unsub => unsub())
    }
  }, [])

  // Combine polygon data with ML signals and live WebSocket prices
  const tickers = useMemo(() => {
    return SYMBOLS.map(symbol => {
      const polygonData = polygonTickers.get(symbol)
      const mlData = tradingData?.tickers?.[symbol]
      const liveData = livePrices.get(symbol)

      // Use live WebSocket price if available and recent (within 60s), otherwise fallback
      const useWebSocketPrice = liveData && (Date.now() - liveData.lastUpdate < 60000)
      const price = useWebSocketPrice ? liveData.price : (polygonData?.price ?? mlData?.current_price ?? 0)

      return {
        symbol,
        price,
        priceFlash: liveData?.flash ?? null,
        lastUpdate: liveData?.lastUpdate ?? null,
        change: polygonData?.change ?? 0,
        changePercent: polygonData?.changePercent ?? mlData?.today_change_pct ?? 0,
        action: mlData?.action ?? 'NO_TRADE',
        probability: mlData?.probability_a ?? 0.5,
        probabilityB: mlData?.probability_b ?? 0.5,
        bucket: mlData?.bucket ?? 'neutral',
        positionSize: mlData?.position_pct ?? 0,
        stopLoss: mlData?.stop_loss,
        takeProfit: mlData?.take_profit,
        reason: mlData?.reason ?? '',
        session: mlData?.session ?? 'early',
        todayOpen: mlData?.today_open ?? null,
        price11am: mlData?.price_11am ?? null,
        modelAccuracy: mlData?.session === 'late'
          ? mlData?.model_accuracy?.late_a
          : mlData?.model_accuracy?.early,
        isBest: symbol === tradingData?.best_ticker,
      }
    }).filter(t => t.price > 0)
  }, [polygonTickers, tradingData, livePrices])

  const getActionStyle = (action: string, isBest: boolean) => {
    const base = isBest ? 'ring-2 ring-cyan-400' : ''
    switch (action) {
      case 'LONG': return `bg-green-900/40 border-green-500/50 ${base}`
      case 'SHORT': return `bg-red-900/40 border-red-500/50 ${base}`
      default: return `bg-gray-900/40 border-gray-700 ${base}`
    }
  }

  const getActionBadge = (action: string) => {
    switch (action) {
      case 'LONG': return 'bg-green-500 text-white'
      case 'SHORT': return 'bg-red-500 text-white'
      default: return 'bg-gray-600 text-gray-300'
    }
  }

  // Get user-friendly action text
  const getActionText = (action: string) => {
    switch (action) {
      case 'LONG': return 'BUY CALL'
      case 'SHORT': return 'BUY PUT'
      default: return 'WAIT'
    }
  }

  const getProbabilityColor = (prob: number, action: string) => {
    if (action === 'NO_TRADE') return 'text-gray-500'
    if (prob >= 0.8) return 'text-green-400'
    if (prob >= 0.6) return 'text-yellow-400'
    return 'text-gray-400'
  }

  if (isLoading && tickers.length === 0) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-400 mb-4"></div>
          <div className="text-white text-xl">Loading...</div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-400 text-xl mb-2">Unable to Load Data</div>
          <div className="text-gray-500">{error.message}</div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-black p-4">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center gap-4">
          <div className={`w-3 h-3 rounded-full ${
            marketStatus === 'Open' ? 'bg-green-400 animate-pulse' :
            marketStatus === 'Closed' ? 'bg-red-400' : 'bg-yellow-400'
          }`} />
          <span className={`font-medium ${
            marketStatus === 'Open' ? 'text-green-400' :
            marketStatus === 'Closed' ? 'text-red-400' : 'text-yellow-400'
          }`}>
            {marketStatus.toUpperCase()}
          </span>
          <span className="text-gray-500">{currentTime.toLocaleTimeString()}</span>
          {wsConnected && (
            <span className="flex items-center gap-1 text-xs text-green-500">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
              LIVE
            </span>
          )}
          <button
            onClick={() => router.push('/replay')}
            className="ml-4 px-3 py-1 bg-purple-500/20 text-purple-400 hover:bg-purple-500/30 rounded-lg text-sm font-medium transition-colors"
          >
            Replay Mode
          </button>
        </div>
        <div className="text-right">
          {tradingData?.session && (
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
              tradingData.session === 'late'
                ? 'bg-green-500/20 text-green-400'
                : 'bg-yellow-500/20 text-yellow-400'
            }`}>
              {tradingData.session === 'late' ? '🎯 Peak Accuracy' : '⏳ Early Session'}
            </span>
          )}
        </div>
      </div>

      {/* Summary */}
      {tradingData?.summary?.recommendation && (
        <div className="mb-6 p-4 bg-gray-900 border border-gray-800 rounded-lg">
          <p className="text-gray-300">{tradingData.summary.recommendation}</p>
        </div>
      )}

      {/* Ticker Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {tickers.map(ticker => (
          <div
            key={ticker.symbol}
            onClick={() => router.push(`/ticker/${ticker.symbol}`)}
            className={`cursor-pointer border rounded-xl p-5 transition-all hover:scale-[1.02] ${getActionStyle(ticker.action, ticker.isBest)}`}
          >
            {/* Header */}
            <div className="flex justify-between items-start mb-4">
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-2xl font-bold text-white">{ticker.symbol}</h2>
                  {ticker.isBest && <span className="text-cyan-400 text-xs">BEST</span>}
                </div>
                <div
                  className={`text-3xl font-light mt-1 transition-all duration-200 ${
                    ticker.priceFlash === 'up' ? 'text-green-400 scale-105' :
                    ticker.priceFlash === 'down' ? 'text-red-400 scale-105' :
                    'text-white'
                  }`}
                  style={{
                    textShadow: ticker.priceFlash ? '0 0 8px currentColor' : 'none'
                  }}
                >
                  ${ticker.price.toFixed(2)}
                </div>
                <div className={`text-sm ${ticker.changePercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {ticker.changePercent >= 0 ? '▲' : '▼'} {Math.abs(ticker.changePercent).toFixed(2)}%
                </div>
                {/* Faint timestamp showing last update */}
                {ticker.lastUpdate && (
                  <div className="text-[10px] text-gray-600 mt-0.5 font-mono">
                    {new Date(ticker.lastUpdate).toLocaleTimeString('en-US', {
                      hour: '2-digit',
                      minute: '2-digit',
                      second: '2-digit',
                      hour12: false
                    })}
                  </div>
                )}
              </div>
              <div className={`px-4 py-2 rounded-lg font-bold text-lg ${getActionBadge(ticker.action)}`}>
                {getActionText(ticker.action)}
              </div>
            </div>

            {/* Target B Dominance - Primary Signal for Late Session */}
            {ticker.session === 'late' ? (
              <div className="mb-4 space-y-2">
                {/* Target B Dominance Box - Primary */}
                {(() => {
                  const isBullish = ticker.probabilityB >= 0.5
                  const confidence = Math.round(Math.max(ticker.probabilityB, 1 - ticker.probabilityB) * 100)
                  const direction: 'BULLISH' | 'BEARISH' = isBullish ? 'BULLISH' : 'BEARISH'
                  const actionLabel = isBullish ? 'BUY CALL' : 'BUY PUT'
                  const bgColor = isBullish ? 'bg-green-900/40 border-green-500/50' : 'bg-red-900/40 border-red-500/50'
                  const textColor = isBullish ? 'text-green-400' : 'text-red-400'

                  // Lock signal and check for flip
                  lockSignal(ticker.symbol, direction, confidence)
                  const flipped = hasSignalFlipped(ticker.symbol, direction)

                  return (
                    <div className={`rounded-lg p-3 border ${bgColor}`}>
                      {/* Flip Warning */}
                      {flipped && (
                        <div className="mb-2 p-2 bg-yellow-500/20 border border-yellow-500/50 rounded text-xs">
                          <div className="text-yellow-400 font-bold">Signal Updated</div>
                          <div className="text-yellow-300">
                            Model recalculated with new price data
                          </div>
                        </div>
                      )}
                      {/* Clear action at top */}
                      <div className={`text-center py-2 mb-2 rounded ${isBullish ? 'bg-green-500/20' : 'bg-red-500/20'}`}>
                        <span className={`text-lg font-bold ${textColor}`}>
                          {actionLabel}
                        </span>
                      </div>
                      {/* What the model is predicting */}
                      <div className="text-center text-gray-300 text-sm mb-2">
                        {ticker.price11am ? (
                          <>
                            Close will be <span className={textColor}>{isBullish ? 'above' : 'below'}</span> ${ticker.price11am.toFixed(2)}
                            <div className="text-gray-500 text-xs">(11 AM price)</div>
                          </>
                        ) : (
                          <>Price expected to {isBullish ? 'rise' : 'fall'}</>
                        )}
                      </div>
                      <div className="flex items-center justify-center gap-2">
                        <span className={`text-2xl font-bold ${textColor}`}>
                          {confidence}%
                        </span>
                        <span className="text-gray-500 text-xs">probability</span>
                      </div>
                    </div>
                  )
                })()}

                {/* Target A Box - Secondary reference */}
                {(() => {
                  const isBullish = ticker.probability >= 0.5
                  const confidence = Math.round(Math.max(ticker.probability, 1 - ticker.probability) * 100)
                  const actionLabel = isBullish ? 'CALL' : 'PUT'
                  const bgColor = isBullish ? 'bg-green-900/20 border-green-500/30' : 'bg-red-900/20 border-red-500/30'
                  const textColor = isBullish ? 'text-green-400/80' : 'text-red-400/80'

                  return (
                    <div className={`rounded-lg p-2 border ${bgColor}`}>
                      <div className="flex justify-between items-center">
                        <div className="text-gray-500 text-xs">
                          vs Open ${ticker.todayOpen?.toFixed(2) ?? '—'}
                        </div>
                        <div className={`text-sm font-bold ${textColor}`}>
                          {actionLabel} {confidence}%
                        </div>
                      </div>
                    </div>
                  )
                })()}
              </div>
            ) : (
              /* Early Session - Show Target A with warning */
              <div className="mb-4">
                {/* Early session warning banner */}
                <div className="mb-2 p-2 bg-yellow-500/10 border border-yellow-500/30 rounded text-xs text-center">
                  <span className="text-yellow-400">Early Session - Signal Accuracy Lower</span>
                  <div className="text-yellow-300/70">Wait for 11 AM for higher confidence signals</div>
                </div>
                {/* Target A Box */}
                {(() => {
                  const isBullish = ticker.probability >= 0.5
                  const confidence = Math.round(Math.max(ticker.probability, 1 - ticker.probability) * 100)
                  const actionLabel = isBullish ? 'BUY CALL' : 'BUY PUT'
                  const bgColor = isBullish ? 'bg-green-900/40 border-green-500/50' : 'bg-red-900/40 border-red-500/50'
                  const textColor = isBullish ? 'text-green-400' : 'text-red-400'

                  return (
                    <div className={`rounded-lg p-3 border ${bgColor}`}>
                      {/* Clear action at top */}
                      <div className={`text-center py-2 mb-2 rounded ${isBullish ? 'bg-green-500/20' : 'bg-red-500/20'}`}>
                        <span className={`text-lg font-bold ${textColor}`}>
                          {actionLabel}
                        </span>
                      </div>
                      {/* What the model is predicting */}
                      <div className="text-center text-gray-300 text-sm mb-2">
                        {ticker.todayOpen ? (
                          <>
                            Close will be <span className={textColor}>{isBullish ? 'above' : 'below'}</span> ${ticker.todayOpen.toFixed(2)}
                            <div className="text-gray-500 text-xs">(open price)</div>
                          </>
                        ) : (
                          <>Price expected to {isBullish ? 'rise' : 'fall'}</>
                        )}
                      </div>
                      <div className="flex items-center justify-center gap-2">
                        <span className={`text-2xl font-bold ${textColor}`}>
                          {confidence}%
                        </span>
                        <span className="text-gray-500 text-xs">probability</span>
                      </div>
                    </div>
                  )
                })()}
                {/* Target B Countdown */}
                <div className="mt-2 p-2 bg-gray-800/50 rounded border border-gray-700">
                  <TargetBCountdown />
                </div>
              </div>
            )}

            {/* Action Details */}
            {ticker.action !== 'NO_TRADE' && (
              <div className="space-y-3 pt-3 border-t border-gray-800">
                {/* Position Size */}
                <div className="flex justify-between">
                  <span className="text-gray-500">Position</span>
                  <span className="text-white font-medium">{ticker.positionSize}%</span>
                </div>

                {/* Stop & Target */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-gray-500 text-xs">STOP</div>
                    <div className="text-red-400 font-mono">
                      ${ticker.stopLoss?.toFixed(2) ?? '—'}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500 text-xs">TARGET</div>
                    <div className="text-green-400 font-mono">
                      ${ticker.takeProfit?.toFixed(2) ?? '—'}
                    </div>
                  </div>
                </div>

                {/* Reason */}
                {ticker.reason && (
                  <div className="text-gray-400 text-sm">
                    {ticker.reason}
                  </div>
                )}
              </div>
            )}

          </div>
        ))}
      </div>

      {/* Model Carousel - Intraday/Swing Multi-Timeframe View */}
      <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-4">
        <ModelCarousel ticker="SPY" />
        <ModelCarousel ticker="QQQ" />
        <ModelCarousel ticker="IWM" />
      </div>

      {/* AI Trading Recommendations */}
      <div className="mt-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">AI Trading Recommendations</h3>
          <span className="text-xs px-2 py-1 rounded bg-cyan-500/20 text-cyan-400">
            V6 Intraday Model
          </span>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <IntradayRecommendation symbol="SPY" />
          <IntradayRecommendation symbol="QQQ" />
          <IntradayRecommendation symbol="IWM" />
        </div>
      </div>

      {/* Northstar Phase Pipeline */}
      <div className="mt-6">
        <NorthstarPanel />
      </div>

      {/* Footer */}
      <div className="mt-6 text-center text-gray-600 text-sm">
        V6 Intraday + Swing Models • Northstar Pipeline • Click ticker for full analysis
      </div>
    </div>
  )
}
