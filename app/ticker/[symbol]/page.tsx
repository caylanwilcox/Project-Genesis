'use client'

import { useParams, useRouter } from 'next/navigation'
import { useState, useEffect } from 'react'
import dynamic from 'next/dynamic'

const ProfessionalChart = dynamic(() => import('@/components/ProfessionalChart').then(m => m.ProfessionalChart), {
  ssr: false,
})

interface TimeframeAnalysis {
  timeframe: string
  signal: 'Strong Buy' | 'Buy' | 'Neutral' | 'Sell' | 'Strong Sell'
  strength: number
}

interface TickerDetails {
  symbol: string
  price: number
  change: number
  changePercent: number
  signal: 'strong_buy' | 'buy' | 'neutral' | 'sell' | 'strong_sell'
  confidence: number
  recommendation: string
  entryPoints: {
    support: number
    recommendedEntry: string
    optimalEntry: number
  }
  targets: {
    target1: { price: number; percent: number }
    target2: { price: number; percent: number }
    target3: { price: number; percent: number }
  }
  risk: {
    stopLoss: number
    riskRewardRatio: string
    maxDrawdown: number
  }
  marketConditions: {
    trend: string
    volume: string
    rsi: number
    macd: string
    bollingerBands: string
  }
  sentiment: {
    overall: 'Extremely Bullish' | 'Bullish' | 'Neutral' | 'Bearish' | 'Extremely Bearish'
    retail: number
    institutional: number
    smartMoney: number
  }
  optionsFlow: {
    callVolume: string
    putVolume: string
    putCallRatio: number
    unusualActivity: boolean
  }
  timeframes: TimeframeAnalysis[]
}

const tickerData: Record<string, TickerDetails> = {
  SPY: {
    symbol: 'SPY',
    price: 445.20,
    change: 2.35,
    changePercent: 0.53,
    signal: 'strong_buy',
    confidence: 92,
    recommendation: 'STRONG BUY',
    entryPoints: {
      support: 443.50,
      recommendedEntry: '$444.50 - $445.00',
      optimalEntry: 444.75,
    },
    targets: {
      target1: { price: 447.50, percent: 0.5 },
      target2: { price: 450.00, percent: 1.1 },
      target3: { price: 453.50, percent: 1.9 },
    },
    risk: {
      stopLoss: 443.00,
      riskRewardRatio: '1:3.2',
      maxDrawdown: 0.5,
    },
    marketConditions: {
      trend: 'Strong Bullish',
      volume: 'Above Average (85.2M)',
      rsi: 58,
      macd: 'Bullish Cross',
      bollingerBands: 'Trading Above Middle',
    },
    sentiment: {
      overall: 'Bullish',
      retail: 65,
      institutional: 78,
      smartMoney: 82,
    },
    optionsFlow: {
      callVolume: '125.3K',
      putVolume: '67.2K',
      putCallRatio: 0.54,
      unusualActivity: true,
    },
    timeframes: [
      { timeframe: '1 Min', signal: 'Buy', strength: 75 },
      { timeframe: '5 Min', signal: 'Strong Buy', strength: 85 },
      { timeframe: '15 Min', signal: 'Strong Buy', strength: 90 },
      { timeframe: '1 Hour', signal: 'Buy', strength: 78 },
      { timeframe: 'Daily', signal: 'Strong Buy', strength: 92 },
      { timeframe: 'Weekly', signal: 'Buy', strength: 80 },
    ],
  },
  QQQ: {
    symbol: 'QQQ',
    price: 385.50,
    change: -1.20,
    changePercent: -0.31,
    signal: 'neutral',
    confidence: 48,
    recommendation: 'NEUTRAL',
    entryPoints: {
      support: 383.00,
      recommendedEntry: '$382.50 - $383.50',
      optimalEntry: 383.25,
    },
    targets: {
      target1: { price: 388.00, percent: 0.6 },
      target2: { price: 391.00, percent: 1.4 },
      target3: { price: 394.50, percent: 2.3 },
    },
    risk: {
      stopLoss: 381.00,
      riskRewardRatio: '1:2.0',
      maxDrawdown: 0.8,
    },
    marketConditions: {
      trend: 'Neutral',
      volume: 'Average (42.7M)',
      rsi: 45,
      macd: 'Converging',
      bollingerBands: 'Near Middle Band',
    },
    sentiment: {
      overall: 'Neutral',
      retail: 52,
      institutional: 48,
      smartMoney: 45,
    },
    optionsFlow: {
      callVolume: '78.5K',
      putVolume: '72.1K',
      putCallRatio: 0.92,
      unusualActivity: false,
    },
    timeframes: [
      { timeframe: '1 Min', signal: 'Neutral', strength: 50 },
      { timeframe: '5 Min', signal: 'Sell', strength: 42 },
      { timeframe: '15 Min', signal: 'Neutral', strength: 48 },
      { timeframe: '1 Hour', signal: 'Neutral', strength: 52 },
      { timeframe: 'Daily', signal: 'Neutral', strength: 48 },
      { timeframe: 'Weekly', signal: 'Buy', strength: 60 },
    ],
  },
  IWM: {
    symbol: 'IWM',
    price: 218.75,
    change: 1.85,
    changePercent: 0.85,
    signal: 'buy',
    confidence: 78,
    recommendation: 'BUY',
    entryPoints: {
      support: 217.00,
      recommendedEntry: '$217.75 - $218.25',
      optimalEntry: 218.00,
    },
    targets: {
      target1: { price: 220.50, percent: 0.8 },
      target2: { price: 222.00, percent: 1.5 },
      target3: { price: 224.50, percent: 2.6 },
    },
    risk: {
      stopLoss: 216.50,
      riskRewardRatio: '1:2.8',
      maxDrawdown: 0.7,
    },
    marketConditions: {
      trend: 'Bullish',
      volume: 'High (31.5M)',
      rsi: 62,
      macd: 'Bullish',
      bollingerBands: 'Breaking Upper Band',
    },
    sentiment: {
      overall: 'Bullish',
      retail: 70,
      institutional: 68,
      smartMoney: 75,
    },
    optionsFlow: {
      callVolume: '45.2K',
      putVolume: '28.7K',
      putCallRatio: 0.63,
      unusualActivity: true,
    },
    timeframes: [
      { timeframe: '1 Min', signal: 'Buy', strength: 72 },
      { timeframe: '5 Min', signal: 'Buy', strength: 75 },
      { timeframe: '15 Min', signal: 'Strong Buy', strength: 82 },
      { timeframe: '1 Hour', signal: 'Buy', strength: 78 },
      { timeframe: 'Daily', signal: 'Buy', strength: 78 },
      { timeframe: 'Weekly', signal: 'Neutral', strength: 55 },
    ],
  },
  VIX: {
    symbol: 'VIX',
    price: 14.25,
    change: -0.35,
    changePercent: -2.40,
    signal: 'sell',
    confidence: 71,
    recommendation: 'CAUTION',
    entryPoints: {
      support: 13.50,
      recommendedEntry: '$13.25 - $13.75',
      optimalEntry: 13.50,
    },
    targets: {
      target1: { price: 15.00, percent: 5.3 },
      target2: { price: 16.00, percent: 12.3 },
      target3: { price: 17.50, percent: 22.8 },
    },
    risk: {
      stopLoss: 13.00,
      riskRewardRatio: '1:1.5',
      maxDrawdown: 1.2,
    },
    marketConditions: {
      trend: 'Bearish (Low Volatility)',
      volume: 'N/A',
      rsi: 38,
      macd: 'Bearish',
      bollingerBands: 'Near Lower Band',
    },
    sentiment: {
      overall: 'Bearish',
      retail: 35,
      institutional: 40,
      smartMoney: 30,
    },
    optionsFlow: {
      callVolume: 'N/A',
      putVolume: 'N/A',
      putCallRatio: 0,
      unusualActivity: false,
    },
    timeframes: [
      { timeframe: '1 Min', signal: 'Sell', strength: 65 },
      { timeframe: '5 Min', signal: 'Strong Sell', strength: 72 },
      { timeframe: '15 Min', signal: 'Sell', strength: 68 },
      { timeframe: '1 Hour', signal: 'Sell', strength: 70 },
      { timeframe: 'Daily', signal: 'Sell', strength: 71 },
      { timeframe: 'Weekly', signal: 'Neutral', strength: 50 },
    ],
  },
}

export default function TickerPage() {
  const params = useParams()
  const router = useRouter()
  const symbol = params?.symbol as string
  const [ticker, setTicker] = useState<TickerDetails | null>(null)
  const [livePrice, setLivePrice] = useState(0)

  useEffect(() => {
    if (symbol && tickerData[symbol.toUpperCase()]) {
      const data = tickerData[symbol.toUpperCase()]
      setTicker(data)
      setLivePrice(data.price)
    } else if (symbol) {
      router.push('/')
    }
  }, [symbol, router])

  useEffect(() => {
    if (!ticker) return

    const interval = setInterval(() => {
      setLivePrice(prev => {
        const change = (Math.random() - 0.5) * 0.5
        return prev + change
      })
    }, 2000)

    return () => clearInterval(interval)
  }, [ticker])

  if (!ticker) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-white mb-4"></div>
          <div className="text-white text-lg font-medium">Loading ticker data...</div>
          <div className="text-gray-400 text-sm mt-2">Please wait while we fetch the latest information</div>
        </div>
      </div>
    )
  }

  const getSignalColor = (signal: string) => {
    if (signal.includes('Strong Buy')) return 'text-green-400'
    if (signal.includes('Buy')) return 'text-green-300'
    if (signal.includes('Strong Sell')) return 'text-red-400'
    if (signal.includes('Sell')) return 'text-red-300'
    return 'text-yellow-400'
  }

  return (
    <div className="min-h-screen bg-black text-white animate-in fade-in duration-300">
      {/* Ticker Header Section */}
      <header className="bg-gray-900 border-b border-gray-800 py-4 px-4">
        <div className="max-w-7xl mx-auto">
          <button
            onClick={() => router.push('/')}
            className="mb-4 text-sm text-gray-500 hover:text-white transition-colors flex items-center gap-2"
          >
            ← Back to Dashboard
          </button>
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            <div className="flex flex-col sm:flex-row sm:items-baseline gap-4 sm:gap-6">
              <h1 className="text-2xl sm:text-3xl font-bold">{ticker.symbol}</h1>
              <div className="flex items-baseline gap-3">
                <span className="text-xl sm:text-2xl font-semibold transition-all duration-200 hover:text-green-300">
                  ${livePrice.toFixed(2)}
                </span>
                <span className={`text-base sm:text-lg font-medium transition-colors ${ticker.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {ticker.change >= 0 ? '+' : ''}{ticker.change.toFixed(2)} ({ticker.changePercent.toFixed(2)}%)
                </span>
              </div>
            </div>
            <div className={`px-4 sm:px-6 py-2 sm:py-3 rounded-lg text-sm sm:text-lg font-bold text-center ${
              ticker.signal === 'strong_buy' ? 'bg-green-500/20 text-green-400 border border-green-500' :
              ticker.signal === 'buy' ? 'bg-green-400/20 text-green-300 border border-green-400' :
              ticker.signal === 'neutral' ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500' :
              ticker.signal === 'sell' ? 'bg-red-400/20 text-red-300 border border-red-400' :
              'bg-red-500/20 text-red-400 border border-red-500'
            }`}>
              {ticker.recommendation}
            </div>
          </div>
        </div>
      </header>

      {/* Price Chart Section */}
      <section className="bg-gray-900 border-b border-gray-800 py-4">
        <div className="max-w-7xl mx-auto px-4">
          <div className="max-w-[85%] mx-auto h-[320px] sm:h-[380px] lg:h-[420px] relative">
            <ProfessionalChart
              symbol={ticker.symbol}
              currentPrice={livePrice}
              stopLoss={ticker.risk.stopLoss}
              targets={[
                ticker.targets.target1.price,
                ticker.targets.target2.price,
                ticker.targets.target3.price
              ]}
              entryPoint={ticker.entryPoints.optimalEntry}
            />
          </div>
        </div>
      </section>

      {/* Timeframe Analysis Section */}
      <section className="bg-gray-900 border-b border-gray-800 p-4">
        <div className="max-w-7xl mx-auto">
          <h3 className="text-sm text-gray-400 mb-4 font-semibold tracking-wider">MULTI-TIMEFRAME ANALYSIS</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 sm:gap-4">
            {ticker.timeframes.map((tf) => (
              <div key={tf.timeframe} className="text-center bg-gray-800/50 rounded-lg p-3 border border-gray-700/50">
                <div className="text-xs text-gray-500 mb-2 font-medium">{tf.timeframe}</div>
                <div className={`text-xs sm:text-sm font-bold mb-2 ${getSignalColor(tf.signal)}`}>
                  {tf.signal}
                </div>
                <div className="w-full h-1.5 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${
                      tf.signal.includes('Buy') ? 'bg-green-400' :
                      tf.signal.includes('Sell') ? 'bg-red-400' :
                      'bg-yellow-400'
                    }`}
                    style={{ width: `${tf.strength}%` }}
                  />
                </div>
                <div className="text-xs text-gray-500 mt-1">{tf.strength}%</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Main Trading Analysis Section */}
      <main className="max-w-7xl mx-auto px-4 py-4 sm:py-6">
        {/* Primary Trading Metrics */}
        <section className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3 sm:gap-4 mb-4 sm:mb-6">
          {/* Entry Points Card */}
          <article className="bg-gray-900 rounded-lg p-3 sm:p-4 border border-gray-800 hover:border-gray-700 transition-colors">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
              <h3 className="text-blue-400 font-semibold uppercase text-sm tracking-wider">Entry Points</h3>
            </div>
            <div className="space-y-4">
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Current Support</div>
                <div className="text-lg sm:text-xl font-bold">${ticker.entryPoints.support.toFixed(2)}</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Recommended Entry</div>
                <div className="text-sm sm:text-base font-semibold">{ticker.entryPoints.recommendedEntry}</div>
              </div>
              <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Optimal Entry</div>
                <div className="text-lg sm:text-xl font-bold text-cyan-400">${ticker.entryPoints.optimalEntry.toFixed(2)}</div>
              </div>
            </div>
          </article>

          {/* Target Levels Card */}
          <article className="bg-gray-900 rounded-lg p-3 sm:p-4 border border-gray-800 hover:border-gray-700 transition-colors">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <h3 className="text-green-400 font-semibold uppercase text-sm tracking-wider">Target Levels</h3>
            </div>
            <div className="space-y-4">
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Target 1</div>
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1">
                  <div className="text-lg sm:text-xl font-bold">${ticker.targets.target1.price.toFixed(2)}</div>
                  <span className="text-green-400 text-xs sm:text-sm font-semibold">+{ticker.targets.target1.percent}%</span>
                </div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Target 2</div>
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1">
                  <div className="text-lg sm:text-xl font-bold">${ticker.targets.target2.price.toFixed(2)}</div>
                  <span className="text-green-400 text-xs sm:text-sm font-semibold">+{ticker.targets.target2.percent}%</span>
                </div>
              </div>
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Target 3</div>
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1">
                  <div className="text-lg sm:text-xl font-bold text-green-400">${ticker.targets.target3.price.toFixed(2)}</div>
                  <span className="text-green-400 text-xs sm:text-sm font-semibold">+{ticker.targets.target3.percent}%</span>
                </div>
              </div>
            </div>
          </article>

          {/* Risk Management Card */}
          <article className="bg-gray-900 rounded-lg p-3 sm:p-4 border border-gray-800 hover:border-gray-700 transition-colors">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-2 h-2 bg-red-400 rounded-full"></div>
              <h3 className="text-red-400 font-semibold uppercase text-sm tracking-wider">Risk Management</h3>
            </div>
            <div className="space-y-4">
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Stop Loss</div>
                <div className="text-lg sm:text-xl font-bold text-red-400">${ticker.risk.stopLoss.toFixed(2)}</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Risk/Reward</div>
                <div className="text-lg sm:text-xl font-bold">{ticker.risk.riskRewardRatio}</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Max Drawdown</div>
                <div className="text-lg sm:text-xl font-bold">{ticker.risk.maxDrawdown}%</div>
              </div>
            </div>
          </article>

          {/* Market Conditions Card */}
          <article className="bg-gray-900 rounded-lg p-3 sm:p-4 border border-gray-800 hover:border-gray-700 transition-colors">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
              <h3 className="text-purple-400 font-semibold uppercase text-sm tracking-wider">Market Conditions</h3>
            </div>
            <div className="space-y-4">
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Trend</div>
                <div className="text-sm sm:text-base font-semibold">{ticker.marketConditions.trend}</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Volume</div>
                <div className="text-sm sm:text-base font-semibold">{ticker.marketConditions.volume}</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">RSI</div>
                <div className="flex items-center justify-between">
                  <div className={`text-lg sm:text-xl font-bold ${
                    ticker.marketConditions.rsi > 70 ? 'text-red-400' :
                    ticker.marketConditions.rsi < 30 ? 'text-green-400' :
                    'text-yellow-400'
                  }`}>{ticker.marketConditions.rsi}</div>
                  <div className={`text-xs font-medium px-2 py-1 rounded ${
                    ticker.marketConditions.rsi > 70 ? 'bg-red-500/20 text-red-400' :
                    ticker.marketConditions.rsi < 30 ? 'bg-green-500/20 text-green-400' :
                    'bg-yellow-500/20 text-yellow-400'
                  }`}>
                    {ticker.marketConditions.rsi > 70 ? 'Overbought' :
                     ticker.marketConditions.rsi < 30 ? 'Oversold' :
                     'Normal'}
                  </div>
                </div>
              </div>
            </div>
          </article>
        </section>

        {/* Key Exit Points Summary */}
        <section className="flex justify-center mb-4 sm:mb-6">
          <div className="bg-gray-900 rounded-lg p-3 border border-gray-800 hover:border-gray-700 transition-colors">
            <div className="flex items-center gap-6 sm:gap-8">
              <div className="text-center">
                <div className="text-gray-400 text-xs font-medium mb-1">Sell Target</div>
                <div className="text-green-400 font-bold text-base">${ticker.targets.target3.price.toFixed(2)}</div>
                <div className="text-green-400 text-xs">+{ticker.targets.target3.percent}%</div>
              </div>
              <div className="w-px h-6 bg-gray-700"></div>
              <div className="text-center">
                <div className="text-gray-400 text-xs font-medium mb-1">Stop Loss</div>
                <div className="text-red-400 font-bold text-base">${ticker.risk.stopLoss.toFixed(2)}</div>
                <div className="text-red-400 text-xs">-{((ticker.price - ticker.risk.stopLoss) / ticker.price * 100).toFixed(1)}%</div>
              </div>
            </div>
          </div>
        </section>

        {/* Advanced Market Analysis */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-3 sm:gap-4">
          {/* Market Sentiment Card */}
          <article className="bg-gray-900 rounded-lg p-3 sm:p-4 border border-gray-800 hover:border-gray-700 transition-colors">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-2 h-2 bg-cyan-400 rounded-full"></div>
              <h3 className="text-cyan-400 font-semibold uppercase text-sm tracking-wider">Market Sentiment</h3>
            </div>
            <div className="space-y-4">
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex justify-between mb-2">
                  <span className="text-gray-400 text-xs font-medium">Retail Sentiment</span>
                  <span className="text-white font-bold text-sm">{ticker.sentiment.retail}%</span>
                </div>
                <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-400 rounded-full transition-all duration-300" style={{ width: `${ticker.sentiment.retail}%` }} />
                </div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex justify-between mb-2">
                  <span className="text-gray-400 text-xs font-medium">Institutional</span>
                  <span className="text-white font-bold text-sm">{ticker.sentiment.institutional}%</span>
                </div>
                <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div className="h-full bg-purple-400 rounded-full transition-all duration-300" style={{ width: `${ticker.sentiment.institutional}%` }} />
                </div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex justify-between mb-2">
                  <span className="text-gray-400 text-xs font-medium">Smart Money</span>
                  <span className="text-white font-bold text-sm">{ticker.sentiment.smartMoney}%</span>
                </div>
                <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div className="h-full bg-yellow-400 rounded-full transition-all duration-300" style={{ width: `${ticker.sentiment.smartMoney}%` }} />
                </div>
              </div>
              <div className="text-center mt-4 pt-4 border-t border-gray-700/50">
                <div className="text-gray-400 text-xs font-medium mb-2">Overall Sentiment</div>
                <div className={`text-lg sm:text-xl font-bold ${
                  ticker.sentiment.overall.includes('Bullish') ? 'text-green-400' :
                  ticker.sentiment.overall.includes('Bearish') ? 'text-red-400' :
                  'text-yellow-400'
                }`}>{ticker.sentiment.overall}</div>
              </div>
            </div>
          </article>

          {/* Options Flow Card */}
          <article className="bg-gray-900 rounded-lg p-3 sm:p-4 border border-gray-800 hover:border-gray-700 transition-colors">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-2 h-2 bg-orange-400 rounded-full"></div>
              <h3 className="text-orange-400 font-semibold uppercase text-sm tracking-wider">Options Flow</h3>
            </div>
            <div className="space-y-4">
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex justify-between">
                  <span className="text-gray-400 text-xs font-medium">Call Volume</span>
                  <span className="text-green-400 font-bold text-sm">{ticker.optionsFlow.callVolume}</span>
                </div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex justify-between">
                  <span className="text-gray-400 text-xs font-medium">Put Volume</span>
                  <span className="text-red-400 font-bold text-sm">{ticker.optionsFlow.putVolume}</span>
                </div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex justify-between">
                  <span className="text-gray-400 text-xs font-medium">Put/Call Ratio</span>
                  <span className={`font-bold text-sm ${
                    ticker.optionsFlow.putCallRatio > 1 ? 'text-red-400' :
                    ticker.optionsFlow.putCallRatio < 0.7 ? 'text-green-400' :
                    'text-yellow-400'
                  }`}>{ticker.optionsFlow.putCallRatio.toFixed(2)}</span>
                </div>
              </div>
              <div className="pt-3 mt-3 border-t border-gray-700/50">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 text-xs font-medium">Unusual Activity</span>
                  <div className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                    ticker.optionsFlow.unusualActivity
                      ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 animate-pulse'
                      : 'bg-gray-800 text-gray-500 border border-gray-700'
                  }`}>
                    {ticker.optionsFlow.unusualActivity ? 'DETECTED' : 'NONE'}
                  </div>
                </div>
              </div>
            </div>
          </article>

          {/* Technical Indicators Card */}
          <article className="bg-gray-900 rounded-lg p-3 sm:p-4 border border-gray-800 hover:border-gray-700 transition-colors">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-2 h-2 bg-indigo-400 rounded-full"></div>
              <h3 className="text-indigo-400 font-semibold uppercase text-sm tracking-wider">Technical Indicators</h3>
            </div>
            <div className="space-y-4">
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex justify-between">
                  <span className="text-gray-400 text-xs font-medium">MACD</span>
                  <span className={`font-bold text-sm ${
                    ticker.marketConditions.macd.includes('Bullish') ? 'text-green-400' :
                    ticker.marketConditions.macd.includes('Bearish') ? 'text-red-400' :
                    'text-yellow-400'
                  }`}>{ticker.marketConditions.macd}</span>
                </div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex justify-between">
                  <span className="text-gray-400 text-xs font-medium">Bollinger Bands</span>
                  <span className="font-bold text-white text-sm">{ticker.marketConditions.bollingerBands}</span>
                </div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 text-xs font-medium">RSI Status</span>
                  <div className={`px-2 py-1 rounded text-xs font-bold ${
                    ticker.marketConditions.rsi > 70 ? 'bg-red-500/20 text-red-400' :
                    ticker.marketConditions.rsi < 30 ? 'bg-green-500/20 text-green-400' :
                    'bg-yellow-500/20 text-yellow-400'
                  }`}>
                    {ticker.marketConditions.rsi > 70 ? 'Overbought' :
                     ticker.marketConditions.rsi < 30 ? 'Oversold' :
                     'Normal'}
                  </div>
                </div>
              </div>
            </div>
          </article>
        </section>
      </main>
    </div>
  )
}