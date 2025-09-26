'use client'

import { useRouter } from 'next/navigation'
import { useState, useEffect } from 'react'

interface TickerData {
  symbol: string
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
}

export default function Dashboard() {
  const router = useRouter()
  const [isMobile, setIsMobile] = useState(false)
  const [tickers, setTickers] = useState<TickerData[]>([
    {
      symbol: 'SPY',
      signal: 'strong_buy',
      action: 'BUY NOW',
      urgency: 'IMMEDIATE',
      timeToAction: '< 5 MIN',
      confidence: 92,
      recommendation: 'STRONG BUY',
      price: 445.20,
      change: 2.35,
      changePercent: 0.53,
      volume: '85.2M',
      rsi: 58,
      macd: 'Bullish Cross',
      momentum: 72,
      supportLevel: 443.50,
      resistanceLevel: 448.00,
      volatility: 'Low',
      trend: 'Bullish',
      institutionalFlow: 'Accumulation',
      optionFlow: 'Bullish',
      entryWindow: 'NOW - 444.50',
      exitTarget: 447.50,
      stopLoss: 443.00,
      techSignal: 'RSI Oversold + MACD Bullish Cross',
      backtestedAccuracy: 87.3,
      sellPrice: 449.25,
      estimatedTimeInTrade: '2-4 hours'
    },
    {
      symbol: 'QQQ',
      signal: 'neutral',
      action: 'WAIT',
      urgency: 'WATCH',
      timeToAction: 'WAIT FOR SIGNAL',
      confidence: 48,
      recommendation: 'NEUTRAL',
      price: 385.50,
      change: -1.20,
      changePercent: -0.31,
      volume: '42.7M',
      rsi: 45,
      macd: 'Converging',
      momentum: 35,
      supportLevel: 383.00,
      resistanceLevel: 388.50,
      volatility: 'Medium',
      trend: 'Neutral',
      institutionalFlow: 'Mixed',
      optionFlow: 'Neutral',
      entryWindow: 'WAIT - 383.00',
      exitTarget: 388.00,
      stopLoss: 381.00,
      techSignal: 'Range-bound Consolidation',
      backtestedAccuracy: 52.1,
      sellPrice: 387.75,
      estimatedTimeInTrade: 'Wait for signal'
    },
    {
      symbol: 'IWM',
      signal: 'buy',
      action: 'BUY NOW',
      urgency: 'SOON',
      timeToAction: '< 15 MIN',
      confidence: 78,
      recommendation: 'BUY',
      price: 218.75,
      change: 1.85,
      changePercent: 0.85,
      volume: '31.5M',
      rsi: 62,
      macd: 'Bullish',
      momentum: 68,
      supportLevel: 217.00,
      resistanceLevel: 221.50,
      volatility: 'Medium',
      trend: 'Bullish',
      institutionalFlow: 'Accumulation',
      optionFlow: 'Bullish',
      entryWindow: 'NOW - 218.00',
      exitTarget: 220.50,
      stopLoss: 216.50,
      techSignal: 'Volume Breakout + RSI Momentum',
      backtestedAccuracy: 76.8,
      sellPrice: 222.15,
      estimatedTimeInTrade: '1-3 hours'
    },
    {
      symbol: 'VIX',
      signal: 'sell',
      action: 'SELL NOW',
      urgency: 'IMMEDIATE',
      timeToAction: '< 10 MIN',
      confidence: 71,
      recommendation: 'SHORT',
      price: 14.25,
      change: -0.35,
      changePercent: -2.40,
      volume: 'N/A',
      rsi: 38,
      macd: 'Bearish',
      momentum: 25,
      supportLevel: 13.50,
      resistanceLevel: 15.00,
      volatility: 'High',
      trend: 'Bearish',
      institutionalFlow: 'Distribution',
      optionFlow: 'Bearish',
      entryWindow: 'SHORT - 14.50',
      exitTarget: 13.00,
      stopLoss: 15.00,
      techSignal: 'Mean Reversion + Low Volatility',
      backtestedAccuracy: 68.9,
      sellPrice: 12.85,
      estimatedTimeInTrade: '30-60 min'
    },
  ])

  const [currentTime, setCurrentTime] = useState(new Date())
  const [marketStatus, setMarketStatus] = useState<'Pre-Market' | 'Open' | 'Closed' | 'After-Hours'>('Open')
  const [timeRemaining, setTimeRemaining] = useState<Record<string, number>>({})

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768)
    }

    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  useEffect(() => {
    // Initialize countdown timers
    const initialTimers: Record<string, number> = {}
    tickers.forEach(ticker => {
      if (ticker.urgency === 'IMMEDIATE') {
        initialTimers[ticker.symbol] = 300 // 5 minutes
      } else if (ticker.urgency === 'SOON') {
        initialTimers[ticker.symbol] = 900 // 15 minutes
      }
    })
    setTimeRemaining(initialTimers)
  }, [])

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

      // Simulate live price updates
      setTickers(prevTickers =>
        prevTickers.map(ticker => {
          const priceChange = (Math.random() - 0.5) * 0.5
          const newPrice = ticker.price + priceChange
          const newChange = ticker.change + priceChange
          const newChangePercent = (newChange / (newPrice - newChange)) * 100

          // Update signal based on new metrics
          const newRSI = Math.max(20, Math.min(80, ticker.rsi + (Math.random() - 0.5) * 5))
          const newMomentum = Math.max(0, Math.min(100, ticker.momentum + (Math.random() - 0.5) * 10))
          const newConfidence = Math.round((newRSI / 100 * 40) + (newMomentum / 100 * 60) + (Math.random() * 10))

          let signal: TickerData['signal'] = 'neutral'
          let recommendation = 'NEUTRAL'
          let action: TickerData['action'] = 'WAIT'
          let urgency: TickerData['urgency'] = 'WATCH'
          let timeToAction = 'MONITORING'

          // Align all indicators based on signal strength
          let newMACDStatus = 'Converging'
          let newInstitutionalFlow: typeof ticker.institutionalFlow = 'Mixed'
          let adjustedRSI = newRSI
          let newTechSignal = 'Range-bound Consolidation'
          let newBacktestedAccuracy = 52.1

          if (newConfidence > 85) {
            signal = 'strong_buy'
            recommendation = 'STRONG BUY'
            action = 'BUY NOW'
            urgency = 'IMMEDIATE'
            timeToAction = '< 5 MIN'
            // Align bullish indicators
            newMACDStatus = 'Bullish Cross'
            newInstitutionalFlow = 'Accumulation'
            adjustedRSI = Math.max(55, Math.min(75, newRSI)) // Bullish RSI range
            newTechSignal = 'RSI Oversold + MACD Bullish Cross'
            newBacktestedAccuracy = 85.0 + Math.random() * 10
          } else if (newConfidence > 70) {
            signal = 'buy'
            recommendation = 'BUY'
            action = 'BUY NOW'
            urgency = 'SOON'
            timeToAction = '< 15 MIN'
            // Align bullish indicators
            newMACDStatus = 'Bullish'
            newInstitutionalFlow = 'Accumulation'
            adjustedRSI = Math.max(50, Math.min(70, newRSI)) // Moderately bullish RSI
            newTechSignal = 'Volume Breakout + RSI Momentum'
            newBacktestedAccuracy = 75.0 + Math.random() * 10
          } else if (newConfidence > 35) {
            signal = 'neutral'
            recommendation = 'NEUTRAL'
            action = 'WAIT'
            urgency = 'WATCH'
            timeToAction = 'WAIT FOR SIGNAL'
            // Neutral indicators
            newMACDStatus = 'Converging'
            newInstitutionalFlow = 'Mixed'
            adjustedRSI = Math.max(40, Math.min(60, newRSI)) // Neutral RSI range
            newTechSignal = 'Range-bound Consolidation'
            newBacktestedAccuracy = 50.0 + Math.random() * 10
          } else if (newConfidence > 20) {
            signal = 'sell'
            recommendation = 'SELL'
            action = 'SELL NOW'
            urgency = 'SOON'
            timeToAction = '< 30 MIN'
            // Align bearish indicators
            newMACDStatus = 'Bearish'
            newInstitutionalFlow = 'Distribution'
            adjustedRSI = Math.max(30, Math.min(50, newRSI)) // Moderately bearish RSI
            newTechSignal = 'Resistance Rejection + Volume Decline'
            newBacktestedAccuracy = 65.0 + Math.random() * 10
          } else {
            signal = 'strong_sell'
            recommendation = 'EXIT NOW'
            action = 'EXIT'
            urgency = 'IMMEDIATE'
            timeToAction = 'IMMEDIATELY'
            // Align strongly bearish indicators
            newMACDStatus = 'Bearish Cross'
            newInstitutionalFlow = 'Distribution'
            adjustedRSI = Math.max(20, Math.min(45, newRSI)) // Bearish RSI range
            newTechSignal = 'Bearish Divergence + Support Break'
            newBacktestedAccuracy = 70.0 + Math.random() * 15
          }

          // Update entry/exit windows based on price
          const entryWindow = action === 'BUY NOW'
            ? `NOW - $${(newPrice - 0.5).toFixed(2)}`
            : action === 'SELL NOW'
            ? `SHORT - $${(newPrice + 0.5).toFixed(2)}`
            : `WAIT - $${ticker.supportLevel.toFixed(2)}`

          return {
            ...ticker,
            price: newPrice,
            change: newChange,
            changePercent: newChangePercent,
            rsi: Math.round(adjustedRSI),
            macd: newMACDStatus,
            institutionalFlow: newInstitutionalFlow,
            momentum: Math.round(newMomentum),
            confidence: newConfidence,
            signal,
            recommendation,
            action,
            urgency,
            timeToAction,
            entryWindow,
            techSignal: newTechSignal,
            backtestedAccuracy: Math.round(newBacktestedAccuracy * 10) / 10,
          }
        })
      )
    }, 1000)

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

  return (
    <div className="min-h-screen bg-black relative overflow-x-hidden">
      {/* Market Status Bar */}
      <div className="absolute top-0 left-0 right-0 bg-gray-900 border-b border-gray-800 px-2 md:px-4 py-2 z-10">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2 md:gap-6">
            <div className="flex items-center gap-1 md:gap-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-green-400 text-xs md:text-sm font-medium">MARKET {marketStatus.toUpperCase()}</span>
            </div>
            <span className="text-gray-400 text-xs md:text-sm hidden sm:block">{currentTime.toLocaleTimeString()}</span>
          </div>
          <div className="flex items-center gap-2 md:gap-4 text-xs md:text-sm">
            <div className="text-gray-400 hidden md:block">
              <span className="text-gray-500">AI Model:</span>
              <span className="text-cyan-400 ml-2">Neural-V3.2</span>
            </div>
            <div className="text-gray-400">
              <span className="text-gray-500 hidden md:inline">Accuracy:</span>
              <span className="text-green-400 ml-1 md:ml-2">94.7%</span>
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

      {/* Conditional Grid Layout Based on Screen Size */}
      <div className={`min-h-screen ${isMobile ? 'pt-20' : 'pt-12'}`}>
        {isMobile ? (
          // Mobile: 1x4 grid (vertical stack) with proper spacing
          <div className="flex flex-col gap-2 p-2 h-[calc(100vh-5rem)]">
            {tickers.map((ticker) => (
              <div
                key={ticker.symbol}
                onClick={() => router.push(`/ticker/${ticker.symbol}`)}
                className="relative cursor-pointer border border-gray-800 transition-all duration-300 hover:border-gray-600 bg-gray-900 overflow-hidden rounded-lg flex-1 min-h-[150px]"
              >
                {/* Animated Background */}
                <div className={`absolute inset-0 opacity-10 bg-gradient-to-br ${getSignalColor(ticker.signal)}`}></div>
                <div className="relative h-full flex flex-col p-2 space-y-2">
                  {/* Top Row: Symbol and Action */}
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <h2 className="text-lg font-bold text-white">{ticker.symbol}</h2>
                        <div className={`px-1 py-0.5 rounded text-[10px] font-bold text-white ${getActionColor(ticker.action)}`}>
                          {ticker.action}
                        </div>
                      </div>
                      <div className="text-sm font-light text-gray-300">
                        ${ticker.price.toFixed(2)}
                      </div>
                      <div className={`text-xs font-medium ${ticker.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {ticker.change >= 0 ? '↑' : '↓'} {Math.abs(ticker.change).toFixed(2)} ({ticker.changePercent >= 0 ? '+' : ''}{ticker.changePercent.toFixed(2)}%)
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-sm font-bold ${getUrgencyColor(ticker.urgency)}`}>
                        {ticker.urgency}
                      </div>
                      {timeRemaining[ticker.symbol] && timeRemaining[ticker.symbol] > 0 && (
                        <div className="mt-2">
                          <div className="text-[10px] text-gray-500">TIME LEFT</div>
                          <div className="text-sm font-mono text-yellow-400">
                            {formatTimer(timeRemaining[ticker.symbol])}
                          </div>
                        </div>
                      )}
                      <div className="mt-1 text-[10px] text-gray-500">
                        {ticker.timeToAction}
                      </div>
                    </div>
                  </div>

                  {/* Key Levels */}
                  <div className="grid grid-cols-2 gap-1 text-xs mb-1">
                    <div>
                      <div className="text-gray-500 text-[10px]">TARGET</div>
                      <div className="text-green-400 font-bold text-sm">${ticker.exitTarget.toFixed(2)}</div>
                    </div>
                    <div>
                      <div className="text-gray-500 text-[10px]">SELL</div>
                      <div className="text-yellow-400 font-bold text-sm">${ticker.sellPrice.toFixed(2)}</div>
                    </div>
                    <div>
                      <div className="text-gray-500 text-[10px]">STOP</div>
                      <div className="text-red-400 font-bold text-sm">${ticker.stopLoss.toFixed(2)}</div>
                    </div>
                    <div>
                      <div className="text-gray-500 text-[10px]">TIME</div>
                      <div className="text-cyan-400 font-bold text-xs">{ticker.estimatedTimeInTrade}</div>
                    </div>
                  </div>

                  {/* Center Section - Main Signal */}
                  <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                      <div className={`inline-block px-3 py-2 rounded-lg border-2 ${
                        ticker.signal === 'strong_buy' ? 'border-green-400 shadow-green-400/50' :
                        ticker.signal === 'buy' ? 'border-green-300 shadow-green-300/30' :
                        ticker.signal === 'neutral' ? 'border-yellow-400 shadow-yellow-400/30' :
                        ticker.signal === 'sell' ? 'border-red-300 shadow-red-300/30' :
                        'border-red-400 shadow-red-400/50'
                      } shadow-lg`}>
                        <div className={`text-lg font-bold tracking-wider bg-gradient-to-r ${getSignalColor(ticker.signal)} bg-clip-text text-transparent`}>
                          {ticker.recommendation}
                        </div>
                        <div className="mt-1 text-xs text-gray-400">
                          {ticker.signal === 'strong_buy' ? '🔥 Maximum Opportunity' :
                           ticker.signal === 'buy' ? '✓ Good Entry Point' :
                           ticker.signal === 'neutral' ? '⏸ Wait for Confirmation' :
                           ticker.signal === 'sell' ? '⚠️ Consider Exit' :
                           '🚨 Exit Immediately'}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Bottom Section - AI Confidence */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500 text-xs">AI Confidence</span>
                      <span className="text-white font-bold text-xs">{ticker.confidence}%</span>
                    </div>
                    <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full transition-all duration-1000 bg-gradient-to-r ${getSignalColor(ticker.signal)}`}
                        style={{ width: `${ticker.confidence}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          // Desktop: 2x2 grid
          <div className="grid grid-cols-2 grid-rows-2 gap-0 h-screen">
            {tickers.map((ticker) => (
              <div
                key={ticker.symbol}
                onClick={() => router.push(`/ticker/${ticker.symbol}`)}
                className="relative cursor-pointer border border-gray-800 transition-all duration-300 hover:border-gray-600 bg-gray-900 overflow-hidden"
              >
                {/* Animated Background */}
                <div className={`absolute inset-0 opacity-10 bg-gradient-to-br ${getSignalColor(ticker.signal)}`}></div>
                <div className="relative h-full flex flex-col justify-between p-6 lg:p-8">
                  {/* Desktop content - same as mobile but with larger text sizes */}
                  <div>
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <div className="flex items-center gap-3 mb-2">
                          <h2 className="text-4xl font-bold text-white">{ticker.symbol}</h2>
                          <div className={`px-3 py-1 rounded-lg text-xs font-bold text-white ${getActionColor(ticker.action)}`}>
                            {ticker.action}
                          </div>
                        </div>
                        <div className="text-3xl font-light text-gray-300">
                          ${ticker.price.toFixed(2)}
                        </div>
                        <div className={`text-lg font-medium mt-1 ${ticker.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {ticker.change >= 0 ? '↑' : '↓'} {Math.abs(ticker.change).toFixed(2)} ({ticker.changePercent >= 0 ? '+' : ''}{ticker.changePercent.toFixed(2)}%)
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`text-2xl font-bold ${getUrgencyColor(ticker.urgency)}`}>
                          {ticker.urgency}
                        </div>
                        {timeRemaining[ticker.symbol] && timeRemaining[ticker.symbol] > 0 && (
                          <div className="mt-2">
                            <div className="text-xs text-gray-500">TIME LEFT</div>
                            <div className="text-xl font-mono text-yellow-400">
                              {formatTimer(timeRemaining[ticker.symbol])}
                            </div>
                          </div>
                        )}
                        <div className="mt-2 text-xs text-gray-500">
                          {ticker.timeToAction}
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm mb-4">
                      <div>
                        <div className="text-gray-500 text-xs">TARGET</div>
                        <div className="text-green-400 font-bold text-lg">${ticker.exitTarget.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-gray-500 text-xs">SELL</div>
                        <div className="text-yellow-400 font-bold text-lg">${ticker.sellPrice.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-gray-500 text-xs">STOP</div>
                        <div className="text-red-400 font-bold">${ticker.stopLoss.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-gray-500 text-xs">TIME</div>
                        <div className="text-cyan-400 font-bold">{ticker.estimatedTimeInTrade}</div>
                      </div>
                    </div>
                  </div>

                  <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                      <div className={`inline-block px-8 py-6 rounded-xl border-2 ${
                        ticker.signal === 'strong_buy' ? 'border-green-400 shadow-green-400/50' :
                        ticker.signal === 'buy' ? 'border-green-300 shadow-green-300/30' :
                        ticker.signal === 'neutral' ? 'border-yellow-400 shadow-yellow-400/30' :
                        ticker.signal === 'sell' ? 'border-red-300 shadow-red-300/30' :
                        'border-red-400 shadow-red-400/50'
                      } shadow-lg`}>
                        <div className={`text-3xl font-bold tracking-wider bg-gradient-to-r ${getSignalColor(ticker.signal)} bg-clip-text text-transparent`}>
                          {ticker.recommendation}
                        </div>
                        <div className="mt-2 text-sm text-gray-400">
                          {ticker.signal === 'strong_buy' ? '🔥 Maximum Opportunity' :
                           ticker.signal === 'buy' ? '✓ Good Entry Point' :
                           ticker.signal === 'neutral' ? '⏸ Wait for Confirmation' :
                           ticker.signal === 'sell' ? '⚠️ Consider Exit' :
                           '🚨 Exit Immediately'}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-500 text-sm">AI Confidence</span>
                        <span className="text-white font-bold">{ticker.confidence}%</span>
                      </div>
                      <div className="w-full h-3 bg-gray-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all duration-1000 bg-gradient-to-r ${getSignalColor(ticker.signal)}`}
                          style={{ width: `${ticker.confidence}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 hover:opacity-100 transition-opacity duration-300 pointer-events-none flex items-end">
                    <div className="p-8 w-full">
                      <div className="text-white text-lg font-medium">View Full Analysis & Execute Trade →</div>
                      <div className="text-gray-400 text-sm mt-1">Detailed charts, order execution, and risk management</div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}