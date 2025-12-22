'use client'

import { useRouter } from 'next/navigation'
import { useState, useEffect, useMemo } from 'react'
import { useMultiTickerData } from '@/hooks/useMultiTickerData'

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

export default function Dashboard() {
  const router = useRouter()
  const [currentTime, setCurrentTime] = useState(new Date())
  const [marketStatus, setMarketStatus] = useState<'Pre-Market' | 'Open' | 'Closed' | 'After-Hours'>('Open')
  const [tradingData, setTradingData] = useState<TradingData | null>(null)
  const [loading, setLoading] = useState(true)

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

  // Combine polygon data with ML signals
  const tickers = useMemo(() => {
    return SYMBOLS.map(symbol => {
      const polygonData = polygonTickers.get(symbol)
      const mlData = tradingData?.tickers?.[symbol]

      return {
        symbol,
        price: polygonData?.price ?? mlData?.current_price ?? 0,
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
        modelAccuracy: mlData?.session === 'late'
          ? mlData?.model_accuracy?.late_a
          : mlData?.model_accuracy?.early,
        isBest: symbol === tradingData?.best_ticker,
      }
    }).filter(t => t.price > 0)
  }, [polygonTickers, tradingData])

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
                <div className="text-3xl font-light text-white mt-1">
                  ${ticker.price.toFixed(2)}
                </div>
                <div className={`text-sm ${ticker.changePercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {ticker.changePercent >= 0 ? '▲' : '▼'} {Math.abs(ticker.changePercent).toFixed(2)}%
                </div>
              </div>
              <div className={`px-4 py-2 rounded-lg font-bold text-lg ${getActionBadge(ticker.action)}`}>
                {ticker.action === 'NO_TRADE' ? 'WAIT' : ticker.action}
              </div>
            </div>

            {/* Probabilities - Target A & B */}
            <div className="mb-4 space-y-2">
              {/* Target A: Close > Open */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-gray-500 text-xs">Target A <span className="text-gray-600">(Close &gt; Open)</span></span>
                  <span className={`text-xl font-bold ${getProbabilityColor(ticker.probability, ticker.action)}`}>
                    {Math.round(ticker.probability * 100)}%
                  </span>
                </div>
                <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all ${
                      ticker.probability >= 0.5 ? 'bg-green-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${ticker.probability * 100}%` }}
                  />
                </div>
              </div>

              {/* Target B: Close > 11 AM (or countdown if early) */}
              {ticker.session === 'late' ? (
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-gray-500 text-xs">Target B <span className="text-gray-600">(Close &gt; 11AM)</span></span>
                    <span className={`text-lg font-bold ${getProbabilityColor(ticker.probabilityB, ticker.action)}`}>
                      {Math.round(ticker.probabilityB * 100)}%
                    </span>
                  </div>
                  <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all ${
                        ticker.probabilityB >= 0.5 ? 'bg-green-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${ticker.probabilityB * 100}%` }}
                    />
                  </div>
                </div>
              ) : (
                <TargetBCountdown />
              )}
            </div>

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

            {/* Model Accuracy */}
            {ticker.modelAccuracy && (
              <div className="mt-3 pt-3 border-t border-gray-800 flex justify-between">
                <span className="text-gray-600 text-xs">Model Accuracy</span>
                <span className="text-cyan-400 text-xs">{Math.round(ticker.modelAccuracy * 100)}%</span>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="mt-6 text-center text-gray-600 text-sm">
        V6 Model • Click ticker for full analysis
      </div>
    </div>
  )
}
