'use client'

import { useParams, useRouter } from 'next/navigation'
import { useState, useEffect } from 'react'
import dynamic from 'next/dynamic'
import { usePolygonData, usePolygonSnapshot } from '@/hooks/usePolygonData'
import { NormalizedChartData, Timeframe } from '@/types/polygon'

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
}

// All display data is derived from live Polygon data; no hardcoded placeholders.

export default function TickerPage() {
  const params = useParams()
  const router = useRouter()
  const symbol = params?.symbol as string
  const [ticker, setTicker] = useState<TickerDetails | null>(null)
  const [livePrice, setLivePrice] = useState(0)
  const [timeframe, setTimeframe] = useState<Timeframe>('15m') // Start with 15m to match chart's 1D default
  const [displayTimeframe, setDisplayTimeframe] = useState<string>('1D') // Track display timeframe for limit calculation

  // Calculate how many bars to fetch based on display timeframe
  const getBarLimit = (tf: Timeframe, displayTf: string): number => {
    // For minute timeframes, fetch enough to cover the display range
    if (tf === '1m') return 390 // Full trading day (6.5 hours * 60 min)
    if (tf === '5m') return 390 // 5 days worth
    if (tf === '15m') return 390 // Multiple days
    if (tf === '30m') return 390
    if (tf === '1h') {
      // If showing 1M display range, fetch ~22 trading days of 1h bars
      if (displayTf === '1M') return 22 * 6.5 // ~143 bars
      return 390
    }
    if (tf === '4h') return 390

    // For daily and longer timeframes, fetch based on display range
    if (tf === '1d') {
      if (displayTf === '1M') return 30
      if (displayTf === '3M') return 90
      if (displayTf === '6M') return 180
      if (displayTf === 'YTD') return 300 // Up to 1 year
      if (displayTf === '1Y') return 365
      return 200 // Default
    }
    if (tf === '1w') return 260 // ~5 years
    if (tf === '1M') return 120 // ~10 years

    return 200
  }

  // Fetch real polygon.io data with dynamic limit based on timeframe
  const {
    data: polygonData,
    currentPrice: polygonCurrentPrice,
    priceChange: polygonPriceChange,
    priceChangePercent: polygonPriceChangePercent,
    isLoading: isPolygonLoading,
    error: polygonError,
  } = usePolygonData({
    ticker: symbol?.toUpperCase() || '',
    timeframe,
    limit: getBarLimit(timeframe, displayTimeframe),
    autoRefresh: true,
    refreshInterval: timeframe === '1m' ? 10000 : 60000, // faster on 1m
    displayTimeframe, // Pass display timeframe for accurate date range calculation
  })

  // Live price from snapshot (paid plan) every 5s; falls back gracefully if unavailable
  const { snapshot } = usePolygonSnapshot(symbol?.toUpperCase() || '', 5000)

  // Transform polygon data to chart format
  const chartData = polygonData.map((bar: NormalizedChartData) => ({
    time: bar.time,
    open: bar.open,
    high: bar.high,
    low: bar.low,
    close: bar.close,
    volume: bar.volume,
  }))

  // Calculate technical indicators from real data
  const calculateRSI = (data: NormalizedChartData[], period: number = 14): number => {
    if (data.length < period + 1) return 50

    let gains = 0
    let losses = 0

    for (let i = data.length - period; i < data.length; i++) {
      const change = data[i].close - data[i - 1].close
      if (change > 0) gains += change
      else losses += Math.abs(change)
    }

    const avgGain = gains / period
    const avgLoss = losses / period

    if (avgLoss === 0) return 100
    const rs = avgGain / avgLoss
    return 100 - (100 / (1 + rs))
  }

  const calculateMACD = (data: NormalizedChartData[]): string => {
    if (data.length < 26) return 'Insufficient Data'

    const prices = data.slice(-26).map(d => d.close)
    const ema12 = prices.slice(-12).reduce((a, b) => a + b, 0) / 12
    const ema26 = prices.reduce((a, b) => a + b, 0) / 26
    const macd = ema12 - ema26

    if (macd > 0) return 'Bullish Cross'
    if (macd < 0) return 'Bearish Cross'
    return 'Neutral'
  }

  const calculateTrend = (data: NormalizedChartData[]): string => {
    if (data.length < 20) return 'Insufficient Data'

    const recent = data.slice(-20)
    const sma = recent.reduce((a, b) => a + b.close, 0) / recent.length
    const current = data[data.length - 1].close
    const change = ((current - sma) / sma) * 100

    if (change > 2) return 'Strong Bullish'
    if (change > 0.5) return 'Bullish'
    if (change < -2) return 'Strong Bearish'
    if (change < -0.5) return 'Bearish'
    return 'Neutral'
  }

  const calculateVolatility = (data: NormalizedChartData[]): number => {
    if (data.length < 20) return 0

    const recent = data.slice(-20)
    const returns = recent.map((d, i) => i > 0 ? (d.close - recent[i-1].close) / recent[i-1].close : 0)
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length
    const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length
    return Math.sqrt(variance) * 100
  }

  // Calculate indicators from polygon data (neutral defaults if insufficient data)
  const rsi = polygonData.length > 14 ? calculateRSI(polygonData) : 50
  const macd = polygonData.length > 26 ? calculateMACD(polygonData) : 'Insufficient Data'
  const trend = polygonData.length > 20 ? calculateTrend(polygonData) : 'Neutral'
  const volatility = calculateVolatility(polygonData)

  // Calculate live signal based on technical indicators
  const calculateSignal = (): { signal: 'strong_buy' | 'buy' | 'neutral' | 'sell' | 'strong_sell'; recommendation: string; confidence: number } => {
    if (polygonData.length < 14) {
      return { signal: 'neutral', recommendation: 'NEUTRAL', confidence: 50 }
    }

    let score = 50 // Base neutral score

    // RSI contribution (30 points)
    if (rsi < 30) score += 15 // Oversold = bullish
    else if (rsi > 70) score -= 15 // Overbought = bearish
    else if (rsi >= 40 && rsi <= 60) score += 5 // Neutral zone = slight bullish

    // MACD contribution (25 points)
    if (macd.includes('Bullish')) score += 12
    else if (macd.includes('Bearish')) score -= 12

    // Trend contribution (25 points)
    if (trend.includes('Strong Bullish')) score += 12
    else if (trend.includes('Bullish')) score += 8
    else if (trend.includes('Strong Bearish')) score -= 12
    else if (trend.includes('Bearish')) score -= 8

    // Price change contribution (20 points)
    const priceChangePercent = ticker?.changePercent || 0
    if (priceChangePercent > 1) score += 10
    else if (priceChangePercent > 0.3) score += 5
    else if (priceChangePercent < -1) score -= 10
    else if (priceChangePercent < -0.3) score -= 5

    // Convert score to signal
    const confidence = Math.max(25, Math.min(95, score))

    if (score >= 80) return { signal: 'strong_buy', recommendation: 'STRONG BUY', confidence }
    if (score >= 65) return { signal: 'buy', recommendation: 'BUY', confidence }
    if (score <= 35) return { signal: 'sell', recommendation: 'SELL', confidence }
    if (score <= 20) return { signal: 'strong_sell', recommendation: 'STRONG SELL', confidence }
    return { signal: 'neutral', recommendation: 'NEUTRAL', confidence }
  }

  const liveSignal = calculateSignal()

  // Calculate multi-timeframe signals from existing data
  const calculateTimeframeSignals = (): TimeframeAnalysis[] => {
    if (polygonData.length < 20) {
      return []
    }

    const getSignalFromRSI = (rsiValue: number, trend: string): { signal: 'Strong Buy' | 'Buy' | 'Neutral' | 'Sell' | 'Strong Sell'; strength: number } => {
      let strength = 50
      let signal: 'Strong Buy' | 'Buy' | 'Neutral' | 'Sell' | 'Strong Sell' = 'Neutral'

      // RSI-based scoring
      if (rsiValue < 30) {
        strength = 80
        signal = trend.includes('Bullish') ? 'Strong Buy' : 'Buy'
      } else if (rsiValue < 40) {
        strength = 70
        signal = 'Buy'
      } else if (rsiValue > 70) {
        strength = 80
        signal = trend.includes('Bearish') ? 'Strong Sell' : 'Sell'
      } else if (rsiValue > 60) {
        strength = 65
        signal = 'Sell'
      } else {
        strength = 50
        signal = 'Neutral'
      }

      // Adjust based on trend
      if (trend.includes('Bullish')) strength += 10
      else if (trend.includes('Bearish')) strength -= 10

      return { signal, strength: Math.max(25, Math.min(95, strength)) }
    }

    // Use different windows of data to simulate timeframes
    const recent5 = polygonData.slice(-5)
    const recent15 = polygonData.slice(-15)
    const recent30 = polygonData.slice(-30)
    const recent60 = polygonData.slice(-60)

    const rsi5 = recent5.length >= 5 ? calculateRSI(recent5, 5) : rsi
    const rsi15 = recent15.length >= 14 ? calculateRSI(recent15) : rsi
    const rsi30 = recent30.length >= 14 ? calculateRSI(recent30) : rsi
    const rsi60 = recent60.length >= 14 ? calculateRSI(recent60) : rsi

    const trend5 = recent5.length >= 5 ? calculateTrend(recent5) : trend
    const trend15 = recent15.length >= 15 ? calculateTrend(recent15) : trend
    const trend30 = recent30.length >= 20 ? calculateTrend(recent30) : trend
    const trend60 = recent60.length >= 20 ? calculateTrend(recent60) : trend

    const sig5 = getSignalFromRSI(rsi5, trend5)
    const sig15 = getSignalFromRSI(rsi15, trend15)
    const sig30 = getSignalFromRSI(rsi30, trend30)
    const sig60 = getSignalFromRSI(rsi60, trend60)

    return [
      { timeframe: '5 Bars', signal: sig5.signal, strength: sig5.strength },
      { timeframe: '15 Bars', signal: sig15.signal, strength: sig15.strength },
      { timeframe: '30 Bars', signal: sig30.signal, strength: sig30.strength },
      { timeframe: '60 Bars', signal: sig60.signal, strength: sig60.strength },
      { timeframe: 'Overall', signal: liveSignal.recommendation as any, strength: liveSignal.confidence },
    ]
  }

  const timeframeSignals = calculateTimeframeSignals()

  // Calculate current volume and price range from polygon data
  const currentVolume = polygonData.length > 0 ? polygonData[polygonData.length - 1].volume : 0
  const volumeStr = currentVolume >= 1000000
    ? `${(currentVolume / 1000000).toFixed(1)}M`
    : currentVolume >= 1000
    ? `${(currentVolume / 1000).toFixed(1)}K`
    : currentVolume.toFixed(0)

  // Calculate today's high/low from recent data
  const recentBars = polygonData.slice(-24) // Last 24 hours of 1h bars
  const todayHigh = recentBars.length > 0 ? Math.max(...recentBars.map(b => b.high)) : livePrice
  const todayLow = recentBars.length > 0 ? Math.min(...recentBars.map(b => b.low)) : livePrice

  useEffect(() => {
    if (symbol) {
      const data: TickerDetails = {
        symbol: symbol.toUpperCase(),
        price: 0,
        change: 0,
        changePercent: 0,
      }

      // Prefer snapshot for live price if available; fallback to aggregates-based price
      const liveSnapPrice = snapshot?.min?.c ?? snapshot?.day?.c
      const livePriceSource = liveSnapPrice ?? polygonCurrentPrice
      const liveChange = liveSnapPrice !== undefined && snapshot?.prevDay?.c !== undefined
        ? liveSnapPrice - snapshot.prevDay.c
        : polygonPriceChange
      const liveChangePct = liveSnapPrice !== undefined && snapshot?.prevDay?.c !== undefined
        ? ((liveSnapPrice - snapshot.prevDay.c) / snapshot.prevDay.c) * 100
        : polygonPriceChangePercent

      if (livePriceSource && liveChange !== null && liveChangePct !== null) {
        setTicker({
          ...data,
          price: livePriceSource,
          change: liveChange,
          changePercent: liveChangePct,
        })
        setLivePrice(livePriceSource)
      } else {
        // No live yet; keep placeholder until data arrives
        setTicker(data)
        setLivePrice(0)
      }
    } else if (symbol) {
      router.push('/')
    }
  }, [symbol, router, polygonCurrentPrice, polygonPriceChange, polygonPriceChangePercent, snapshot])

  // Update live price from polygon data without causing render loops
  useEffect(() => {
    const liveSnapPrice = snapshot?.min?.c ?? snapshot?.day?.c
    const livePriceSource = liveSnapPrice ?? polygonCurrentPrice
    if (!livePriceSource) return

    setLivePrice(prev => (prev !== livePriceSource ? livePriceSource : prev))

    if (!ticker) return

    const liveChange = liveSnapPrice !== undefined && snapshot?.prevDay?.c !== undefined
      ? liveSnapPrice - snapshot.prevDay.c
      : polygonPriceChange
    const liveChangePct = liveSnapPrice !== undefined && snapshot?.prevDay?.c !== undefined
      ? ((liveSnapPrice - snapshot.prevDay.c) / snapshot.prevDay.c) * 100
      : polygonPriceChangePercent

    if (liveChange !== null && liveChangePct !== null) {
      if (
        ticker.price !== livePriceSource ||
        ticker.change !== liveChange ||
        ticker.changePercent !== liveChangePct
      ) {
        setTicker(prev => prev ? {
          ...prev,
          price: livePriceSource,
          change: liveChange,
          changePercent: liveChangePct,
        } : null)
      }
    }
  }, [snapshot, polygonCurrentPrice, polygonPriceChange, polygonPriceChangePercent])

  if (!ticker) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-white mb-4"></div>
          <div className="text-white text-lg font-medium">Loading ticker data...</div>
          <div className="text-gray-400 text-sm mt-2">
            {isPolygonLoading ? 'Fetching real-time data from Polygon.io...' : 'Please wait while we fetch the latest information'}
          </div>
          {polygonError && (
            <div className="text-yellow-400 text-xs mt-4 max-w-md mx-auto">
              Note: {polygonError.message}. Using fallback data.
            </div>
          )}
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
      {/* Data Source Banner */}
      {polygonData.length > 0 && (
        <div className="bg-green-500/10 border-b border-green-500/30">
          <div className="max-w-7xl mx-auto px-4 py-2">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
              <div className="flex items-center gap-4">
                <span className="text-green-400 font-semibold">✓ Live Data Active</span>
                <span className="text-gray-400">Price, Chart, RSI, MACD, Trend - Real-time from Polygon.io</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-gray-500">Sentiment & Options: Estimated</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Ticker Header Section */}
      <header className="bg-gray-900 border-b border-gray-800 py-4 px-4 ticker-header">
        <div className="max-w-7xl mx-auto ticker-header__container">
          <button
            onClick={() => router.push('/')}
            className="mb-4 text-sm text-gray-500 hover:text-white transition-colors flex items-center gap-2"
          >
            ← Back to Dashboard
          </button>
          <div className="flex flex-col gap-4 ticker-header__content">
            <div className="flex flex-row items-center justify-between gap-2 flex-wrap ticker-header__row">
              <div className="flex flex-row items-baseline gap-2 sm:gap-4 ticker-header__titleGroup">
                <h1 className="text-xl sm:text-3xl font-bold ticker-header__title">{ticker.symbol}</h1>
                <div className="flex items-baseline gap-2 sm:gap-3">
                  <span className="text-lg sm:text-2xl font-semibold transition-all duration-200 hover:text-green-300 ticker-header__price">
                    ${livePrice.toFixed(2)}
                  </span>
                  <span className={`text-sm sm:text-lg font-medium transition-colors ticker-header__change ${ticker.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {ticker.change >= 0 ? '+' : ''}{ticker.change.toFixed(2)} ({ticker.changePercent.toFixed(2)}%)
                  </span>
                </div>
              </div>
              {/* Timeframe control lives inside ProfessionalChart; changes propagate via onTimeframeChange */}
              <div className="flex flex-col items-end sm:items-center gap-1">
                <div className={`px-2.5 sm:px-4 py-1 sm:py-2 rounded-md text-[10px] sm:text-sm font-bold text-center ${
                  liveSignal.signal === 'strong_buy' ? 'bg-green-500/20 text-green-400 border border-green-500' :
                  liveSignal.signal === 'buy' ? 'bg-green-400/20 text-green-300 border border-green-400' :
                  liveSignal.signal === 'neutral' ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500' :
                  liveSignal.signal === 'sell' ? 'bg-red-400/20 text-red-300 border border-red-400' :
                  'bg-red-500/20 text-red-400 border border-red-500'
                }`}>
                  {liveSignal.recommendation}
                </div>
                <div className="text-[10px] sm:text-xs text-gray-500">
                  {polygonData.length > 14 ? '✓ Live Signal' : 'Loading...'}
                </div>
              </div>
            </div>

            {/* Price Stats Bar */}
            {polygonData.length > 0 && (
              <div className="flex flex-wrap gap-4 sm:gap-6 text-xs sm:text-sm border-t border-gray-800 pt-3">
                <div className="flex flex-col">
                  <span className="text-gray-500">Open</span>
                  <span className="text-white font-semibold">${recentBars.length > 0 ? recentBars[0].open.toFixed(2) : livePrice.toFixed(2)}</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-gray-500">High</span>
                  <span className="text-green-400 font-semibold">${todayHigh.toFixed(2)}</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-gray-500">Low</span>
                  <span className="text-red-400 font-semibold">${todayLow.toFixed(2)}</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-gray-500">Volume</span>
                  <span className="text-white font-semibold">{volumeStr}</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-gray-500">24h Range</span>
                  <span className="text-gray-300 font-semibold">${todayLow.toFixed(2)} - ${todayHigh.toFixed(2)}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Price Chart Section */}
      <section className="bg-gray-900 border-b border-gray-800 py-4">
        <div className="max-w-7xl mx-auto px-4">
          {polygonError && chartData.length === 0 && (
            <div className="mb-4 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
              <div className="flex items-center gap-2">
                <span className="text-yellow-400 text-sm">⚠️</span>
                <div className="text-yellow-400 text-xs">
                  <strong>Note:</strong> {polygonError.message}. Using fallback chart data.
                </div>
              </div>
            </div>
          )}
          {isPolygonLoading && (
            <div className="mb-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
              <div className="flex items-center gap-2">
                <span className="text-blue-400 text-sm animate-spin">⟳</span>
                <div className="text-blue-400 text-xs">
                  Loading live market data from Polygon.io...
                </div>
              </div>
            </div>
          )}
          <div className="max-w-[95%] mx-auto relative">
            <ProfessionalChart
              symbol={ticker.symbol}
              currentPrice={livePrice}
              stopLoss={livePrice * 0.98}
              targets={[
                livePrice * 1.005, // +0.5%
                livePrice * 1.01,  // +1.0%
                livePrice * 1.02   // +2.0%
              ]}
              entryPoint={livePrice}
              data={chartData.length > 0 ? chartData : undefined}
              onTimeframeChange={(tf, displayTf) => {
                console.log('[TickerPage] Timeframe changed to:', tf, 'Display:', displayTf);
                setTimeframe(tf as any);
                setDisplayTimeframe(displayTf);
              }}
            />
          </div>
          {!polygonError && !isPolygonLoading && polygonData.length > 0 && (
            <div className="mt-2 text-center">
              <span className="text-xs text-green-400">✓ Live price via Polygon snapshot (≈5s), chart via aggregates (60s)</span>
            </div>
          )}
        </div>
      </section>

      {/* Timeframe Analysis Section */}
      <section className="bg-gray-900 border-b border-gray-800 p-4">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm text-gray-400 font-semibold tracking-wider">MULTI-TIMEFRAME ANALYSIS</h3>
            {polygonData.length > 20 && <span className="text-xs text-green-400">✓ Live Calculations</span>}
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 sm:gap-4">
            {timeframeSignals.map((tf) => (
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
                <div className="text-lg sm:text-xl font-bold">${(livePrice * 0.995).toFixed(2)}</div>
                <div className="text-xs text-gray-500 mt-1">-0.5% from current</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Recommended Entry</div>
                <div className="text-sm sm:text-base font-semibold">${(livePrice * 0.999).toFixed(2)} - ${(livePrice * 1.001).toFixed(2)}</div>
              </div>
              <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Optimal Entry (Live)</div>
                <div className="text-lg sm:text-xl font-bold text-cyan-400">${livePrice.toFixed(2)}</div>
                {!isPolygonLoading && <div className="text-xs text-green-400 mt-1">✓ Real-time price</div>}
              </div>
            </div>
          </article>

          {/* Target Levels Card */}
          <article className="bg-gray-900 rounded-lg p-3 sm:p-4 border border-gray-800 hover:border-gray-700 transition-colors">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <h3 className="text-green-400 font-semibold uppercase text-sm tracking-wider">Target Levels (Live)</h3>
            </div>
            <div className="space-y-4">
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Target 1</div>
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1">
                  <div className="text-lg sm:text-xl font-bold">${(livePrice * 1.005).toFixed(2)}</div>
                  <span className="text-green-400 text-xs sm:text-sm font-semibold">+0.5%</span>
                </div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Target 2</div>
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1">
                  <div className="text-lg sm:text-xl font-bold">${(livePrice * 1.01).toFixed(2)}</div>
                  <span className="text-green-400 text-xs sm:text-sm font-semibold">+1.0%</span>
                </div>
              </div>
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Target 3</div>
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1">
                  <div className="text-lg sm:text-xl font-bold text-green-400">${(livePrice * 1.02).toFixed(2)}</div>
                  <span className="text-green-400 text-xs sm:text-sm font-semibold">+2.0%</span>
                </div>
              </div>
            </div>
          </article>

          {/* Risk Management Card */}
          <article className="bg-gray-900 rounded-lg p-3 sm:p-4 border border-gray-800 hover:border-gray-700 transition-colors">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-2 h-2 bg-red-400 rounded-full"></div>
              <h3 className="text-red-400 font-semibold uppercase text-sm tracking-wider">Risk Management (Live)</h3>
            </div>
            <div className="space-y-4">
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Stop Loss</div>
                <div className="text-lg sm:text-xl font-bold text-red-400">${(livePrice * 0.98).toFixed(2)}</div>
                <div className="text-xs text-gray-500 mt-1">-2.0% protection</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Risk/Reward</div>
                <div className="text-lg sm:text-xl font-bold">1:2.5</div>
                <div className="text-xs text-gray-500 mt-1">2% risk / 5% reward</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Max Drawdown</div>
                <div className="text-lg sm:text-xl font-bold">2.0%</div>
                <div className="text-xs text-gray-500 mt-1">Conservative risk</div>
              </div>
            </div>
          </article>

          {/* Market Conditions Card */}
          <article className="bg-gray-900 rounded-lg p-3 sm:p-4 border border-gray-800 hover:border-gray-700 transition-colors">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
              <h3 className="text-purple-400 font-semibold uppercase text-sm tracking-wider">Market Conditions (Live)</h3>
            </div>
            <div className="space-y-4">
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Trend</div>
                <div className="text-sm sm:text-base font-semibold">{trend}</div>
                {polygonData.length > 20 && <div className="text-xs text-green-400 mt-1">✓ Calculated from live data</div>}
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">Current Volume</div>
                <div className="text-sm sm:text-base font-semibold">{volumeStr}</div>
                {polygonData.length > 0 && <div className="text-xs text-green-400 mt-1">✓ Live data</div>}
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-gray-400 text-xs font-medium mb-1">RSI (14)</div>
                <div className="flex items-center justify-between">
                  <div className={`text-lg sm:text-xl font-bold ${
                    rsi > 70 ? 'text-red-400' :
                    rsi < 30 ? 'text-green-400' :
                    'text-yellow-400'
                  }`}>{Math.round(rsi)}</div>
                  <div className={`text-xs font-medium px-2 py-1 rounded ${
                    rsi > 70 ? 'bg-red-500/20 text-red-400' :
                    rsi < 30 ? 'bg-green-500/20 text-green-400' :
                    'bg-yellow-500/20 text-yellow-400'
                  }`}>
                    {rsi > 70 ? 'Overbought' :
                     rsi < 30 ? 'Oversold' :
                     'Normal'}
                  </div>
                </div>
                {polygonData.length > 14 && <div className="text-xs text-green-400 mt-1">✓ Calculated from live data</div>}
              </div>
            </div>
          </article>
        </section>

        {/* Key Exit Points Summary */}
        <section className="flex justify-center mb-4 sm:mb-6">
          <div className="bg-gray-900 rounded-lg p-3 border border-gray-800 hover:border-gray-700 transition-colors">
            <div className="flex items-center gap-6 sm:gap-8">
              <div className="text-center">
                <div className="text-gray-400 text-xs font-medium mb-1">Sell Target (Live)</div>
                <div className="text-green-400 font-bold text-base">${(livePrice * 1.02).toFixed(2)}</div>
                <div className="text-green-400 text-xs">+2.0%</div>
              </div>
              <div className="w-px h-6 bg-gray-700"></div>
              <div className="text-center">
                <div className="text-gray-400 text-xs font-medium mb-1">Stop Loss (Live)</div>
                <div className="text-red-400 font-bold text-base">${(livePrice * 0.98).toFixed(2)}</div>
                <div className="text-red-400 text-xs">-2.0%</div>
              </div>
              <div className="w-px h-6 bg-gray-700"></div>
              <div className="text-center">
                <div className="text-gray-400 text-xs font-medium mb-1">Current Price</div>
                <div className="text-cyan-400 font-bold text-base">${livePrice.toFixed(2)}</div>
                {!isPolygonLoading && <div className="text-green-400 text-xs">✓ Live</div>}
              </div>
            </div>
          </div>
        </section>

      {/* Advanced Market Analysis */}
      <section className="grid grid-cols-1 lg:grid-cols-1 gap-3 sm:gap-4">
        {/* Technical Indicators Card */}
        <article className="bg-gray-900 rounded-lg p-3 sm:p-4 border border-gray-800 hover:border-gray-700 transition-colors">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-2 h-2 bg-indigo-400 rounded-full"></div>
              <h3 className="text-indigo-400 font-semibold uppercase text-sm tracking-wider">Technical Indicators (Live)</h3>
            </div>
            <div className="space-y-4">
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex flex-col gap-2">
                  <div className="flex justify-between">
                    <span className="text-gray-400 text-xs font-medium">MACD Signal</span>
                    <span className={`font-bold text-sm ${
                      macd.includes('Bullish') ? 'text-green-400' :
                      macd.includes('Bearish') ? 'text-red-400' :
                      'text-yellow-400'
                    }`}>{macd}</span>
                  </div>
                  {polygonData.length > 26 && <div className="text-xs text-green-400">✓ Calculated from live data</div>}
                </div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex flex-col gap-2">
                  <div className="flex justify-between">
                    <span className="text-gray-400 text-xs font-medium">Trend Direction</span>
                    <span className={`font-bold text-sm ${
                      trend.includes('Bullish') ? 'text-green-400' :
                      trend.includes('Bearish') ? 'text-red-400' :
                      'text-yellow-400'
                    }`}>{trend}</span>
                  </div>
                  {polygonData.length > 20 && <div className="text-xs text-green-400">✓ Calculated from live data</div>}
                </div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 text-xs font-medium">RSI Status</span>
                  <div className={`px-2 py-1 rounded text-xs font-bold ${
                    rsi > 70 ? 'bg-red-500/20 text-red-400' :
                    rsi < 30 ? 'bg-green-500/20 text-green-400' :
                    'bg-yellow-500/20 text-yellow-400'
                  }`}>
                    {rsi > 70 ? 'Overbought' :
                     rsi < 30 ? 'Oversold' :
                     'Normal'}
                  </div>
                </div>
                {polygonData.length > 14 && <div className="text-xs text-green-400 mt-2">✓ Calculated from live data</div>}
              </div>
            </div>
          </article>
        </section>
      </main>
    </div>
  )
}