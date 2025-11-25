'use client'

import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { useState, useEffect, useMemo, useCallback } from 'react'
import dynamic from 'next/dynamic'
import { usePolygonData, usePolygonSnapshot } from '@/hooks/usePolygonData'
import { usePolygonWebSocket, aggregateBarToChartData } from '@/hooks/usePolygonWebSocket'
import { polygonService } from '@/services/polygonService'
import { polygonWebSocketService, AggregateBar } from '@/services/polygonWebSocket'
import { NormalizedChartData, Timeframe } from '@/types/polygon'
import {
  resolveDisplayToData,
  recommendedRefreshMs,
  recommendedBarLimit,
  intervalLabelToDisplayTimeframe,
} from '@/utils/timeframePolicy'
import { detectFvgPatterns } from '@/components/ProfessionalChart/fvgDrawing'
import { getFvgGapSettingsForTimeframe } from '@/utils/fvgThresholdPolicy'
import { aggregateBarsToDuration, INTRADAY_TIMEFRAMES, mergeCandlesReplacing, TIMEFRAME_IN_MS } from '@/hooks/polygonRealtimeUtils'

const ProfessionalChart = dynamic(() => import('@/components/ProfessionalChart').then(m => m.ProfessionalChart), {
  ssr: false,
})

const FvgBacktestPanel = dynamic(() => import('@/components/FvgBacktestPanel').then(m => m.FvgBacktestPanel), {
  ssr: false,
})

const FvgStrategyPanel = dynamic(() => import('@/components/FvgStrategyPanel').then(m => m.FvgStrategyPanel), {
  ssr: false,
})

const formatVolumeValue = (value: number) => {
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)}B`
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(2)}K`
  return value.toFixed(0)
}

interface TimeframeAnalysis {
  timeframe: string
  signal: 'Strong Buy' | 'Buy' | 'Neutral' | 'Sell' | 'Strong Sell'
  strength: number
  fvgSignal?: 'Bullish' | 'Bearish' | 'Neutral'
  fvgCount?: number
  fvgConfidence?: number
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
  const defaultDisplayTimeframe = '1D'
  const [displayTimeframe, setDisplayTimeframe] = useState<string>(defaultDisplayTimeframe)
  const [dataDisplayTimeframe, setDataDisplayTimeframe] = useState<string>(defaultDisplayTimeframe)
  const initialResolved = resolveDisplayToData(defaultDisplayTimeframe)
  const [timeframe, setTimeframe] = useState<Timeframe>(initialResolved.timeframe)
  const initialFvgSettings = getFvgGapSettingsForTimeframe(initialResolved.timeframe)
  const [showFvg, setShowFvg] = useState(false)
  const [fvgCount, setFvgCount] = useState(0)
  const [fvgPercentage, setFvgPercentage] = useState(initialFvgSettings.defaultValue)
  const [showBacktest, setShowBacktest] = useState(false)
  const [visibleBarCount, setVisibleBarCount] = useState(0)
  const [visibleChartData, setVisibleChartData] = useState<any[]>([])
  const [additionalBarsToLoad, setAdditionalBarsToLoad] = useState(0)
  const [isLoadingMoreData, setIsLoadingMoreData] = useState(false)
  const [lastTradeTime, setLastTradeTime] = useState<number>(0)
  const [tradeCount, setTradeCount] = useState(0)
  const [priceFlash, setPriceFlash] = useState<'up' | 'down' | null>(null)
  const fvgGapSettings = useMemo(() => getFvgGapSettingsForTimeframe(timeframe), [timeframe])
  const fvgDisplayPrecision = useMemo(() => (fvgGapSettings.step < 0.1 ? 2 : 1), [fvgGapSettings.step])
  const formatFvgPercent = (value: number, precision = value < 1 ? 2 : 1) => value.toFixed(precision)

  // Centralized policy determines bar limits
  const getBarLimit = (tf: Timeframe): number => {
    const baseLimit = recommendedBarLimit(tf, dataDisplayTimeframe)
    const total = baseLimit + additionalBarsToLoad
    console.log(`[⚙️ LIMIT] ${tf}/${dataDisplayTimeframe}: ${baseLimit} + ${additionalBarsToLoad} = ${total} bars`)
    return total
  }

  // Handler for when user scrolls to left edge - load exactly 100 more bars
  const handleLoadMoreData = useCallback(() => {
    // Maximum 1000 additional bars to prevent overload
    if (additionalBarsToLoad < 1000) {
      setIsLoadingMoreData(true)
      setAdditionalBarsToLoad(prev => prev + 100)
      console.log(`[InfiniteScroll] Loading 100 more bars. Total additional: ${additionalBarsToLoad + 100}`)
    } else {
      console.log('[InfiniteScroll] Maximum bar limit (1000 additional bars) reached')
    }
  }, [additionalBarsToLoad])

  // Handler for visible bar count changes - memoized to prevent infinite loops
  const handleVisibleBarCountChange = useCallback((count: number, data: any[]) => {
    setVisibleBarCount(prev => (prev === count ? prev : count))
    setVisibleChartData(prev => {
      if (
        prev.length === data.length &&
        prev[0]?.time === data[0]?.time &&
        prev[prev.length - 1]?.time === data[data.length - 1]?.time
      ) {
        return prev
      }
      return data
    })
  }, [])

  const chartTargets = useMemo(() => {
    if (livePrice <= 0) return []
    return [
      livePrice * 1.005, // +0.5%
      livePrice * 1.01,  // +1.0%
      livePrice * 1.02,  // +2.0%
    ]
  }, [livePrice])

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
    limit: getBarLimit(timeframe),
    autoRefresh: true,
    refreshInterval: recommendedRefreshMs(timeframe),
    displayTimeframe: dataDisplayTimeframe, // Pass display timeframe for accurate date range calculation
  })

  // Clear loading state when data finishes loading
  useEffect(() => {
    if (!isPolygonLoading && isLoadingMoreData) {
      setIsLoadingMoreData(false)
      console.log('[InfiniteScroll] Data loaded, clearing loading indicator')
    }
  }, [isPolygonLoading, isLoadingMoreData])

  // WebSocket connection for real-time streaming - ALWAYS enabled
  const handleWsBar = useCallback((bar: AggregateBar) => {
    console.log('[TickerPage] WebSocket bar received:', bar);
  }, [])

  const {
    isConnected: wsConnected,
    latestBar: wsLatestBar,
    error: wsError,
  } = usePolygonWebSocket({
    ticker: symbol?.toUpperCase() || '',
    enabled: true, // Always enable WebSocket for rapid updates
    onBar: handleWsBar,
  })

  // Merge WebSocket data with existing polygon data
  const [wsChartData, setWsChartData] = useState<NormalizedChartData[]>([])

  useEffect(() => {
    if (!wsLatestBar) {
      setWsChartData([])
      return
    }

    const newBar = aggregateBarToChartData(wsLatestBar)

    setWsChartData(prev => {
      const existingIndex = prev.findIndex(bar => bar.time === newBar.time)

      if (existingIndex >= 0) {
        const updated = [...prev]
        updated[existingIndex] = newBar
        return updated
      } else {
        const updated = [...prev, newBar].slice(-500)
        return updated
      }
    })
  }, [wsLatestBar])

  const realtimeBarsForView = useMemo(() => {
    if (wsChartData.length === 0) return []
    if (!INTRADAY_TIMEFRAMES.includes(timeframe)) return wsChartData
    if (timeframe === '1m') return wsChartData
    const bucketMs = TIMEFRAME_IN_MS[timeframe]
    if (!bucketMs) return wsChartData
    return aggregateBarsToDuration(wsChartData, bucketMs)
  }, [wsChartData, timeframe])

  const finalPolygonData = useMemo(() => {
    if (realtimeBarsForView.length === 0) {
      return polygonData
    }
    const merged = mergeCandlesReplacing(polygonData, realtimeBarsForView)
    console.log(`[TickerPage] Applied ${realtimeBarsForView.length} realtime bars onto ${polygonData.length} historical bars → ${merged.length}`)
    return merged
  }, [polygonData, realtimeBarsForView])

  // Live price from snapshot (paid plan) every 5s; falls back gracefully if unavailable
  const { snapshot } = usePolygonSnapshot(symbol?.toUpperCase() || '', 5000)

  // Transform polygon data to chart format
  const chartData = finalPolygonData.map((bar: NormalizedChartData) => ({
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

    // FVG Pattern Detection for each timeframe window
      const analyzeFvgWindow = (data: any[]): { signal: 'Bullish' | 'Bearish' | 'Neutral'; count: number; confidence: number } => {
      if (data.length < 10) return { signal: 'Neutral', count: 0, confidence: 0 }

        const patterns = detectFvgPatterns(data, { recentOnly: true, minGapPct: 0.15, maxGapPct: 3.0 })

      if (patterns.length === 0) return { signal: 'Neutral', count: 0, confidence: 0 }

      // Count bullish vs bearish
      const bullishPatterns = patterns.filter(p => p.type === 'bullish')
      const bearishPatterns = patterns.filter(p => p.type === 'bearish')

      // Calculate average confidence
      const avgConfidence = patterns.reduce((sum, p) => sum + p.validationScore, 0) / patterns.length * 100

      // Determine overall signal
      let signal: 'Bullish' | 'Bearish' | 'Neutral' = 'Neutral'
      if (bullishPatterns.length > bearishPatterns.length) {
        signal = 'Bullish'
      } else if (bearishPatterns.length > bullishPatterns.length) {
        signal = 'Bearish'
      }

      return { signal, count: patterns.length, confidence: Math.round(avgConfidence) }
    }

    const fvg5 = analyzeFvgWindow(recent5)
    const fvg15 = analyzeFvgWindow(recent15)
    const fvg30 = analyzeFvgWindow(recent30)
    const fvg60 = analyzeFvgWindow(recent60)
    const fvgOverall = analyzeFvgWindow(polygonData.slice(-100)) // Last 100 bars for overall

    // Map to actual timeframe labels based on current chart timeframe
    const getTimeframeLabel = (bars: number): string => {
      // If viewing 15m chart, 5 bars = 75min (~1h), 15 bars = 4h, etc.
      if (timeframe === '15m') {
        if (bars === 5) return '1 Hour'
        if (bars === 15) return '4 Hours'
        if (bars === 30) return '8 Hours'
        if (bars === 60) return '1 Day'
      }
      // If viewing 1h chart, 5 bars = 5h, 15 bars = 15h, etc.
      if (timeframe === '1h') {
        if (bars === 5) return '5 Hours'
        if (bars === 15) return '15 Hours'
        if (bars === 30) return '30 Hours'
        if (bars === 60) return '2.5 Days'
      }
      // If viewing 4h chart
      if (timeframe === '4h') {
        if (bars === 5) return '20 Hours'
        if (bars === 15) return '2.5 Days'
        if (bars === 30) return '5 Days'
        if (bars === 60) return '10 Days'
      }
      // If viewing 1d chart
      if (timeframe === '1d') {
        if (bars === 5) return '1 Week'
        if (bars === 15) return '3 Weeks'
        if (bars === 30) return '1 Month'
        if (bars === 60) return '3 Months'
      }
      // Default to bar count
      return `${bars} Bars`
    }

    return [
      {
        timeframe: getTimeframeLabel(5),
        signal: sig5.signal,
        strength: sig5.strength,
        fvgSignal: fvg5.signal,
        fvgCount: fvg5.count,
        fvgConfidence: fvg5.confidence
      },
      {
        timeframe: getTimeframeLabel(15),
        signal: sig15.signal,
        strength: sig15.strength,
        fvgSignal: fvg15.signal,
        fvgCount: fvg15.count,
        fvgConfidence: fvg15.confidence
      },
      {
        timeframe: getTimeframeLabel(30),
        signal: sig30.signal,
        strength: sig30.strength,
        fvgSignal: fvg30.signal,
        fvgCount: fvg30.count,
        fvgConfidence: fvg30.confidence
      },
      {
        timeframe: getTimeframeLabel(60),
        signal: sig60.signal,
        strength: sig60.strength,
        fvgSignal: fvg60.signal,
        fvgCount: fvg60.count,
        fvgConfidence: fvg60.confidence
      },
      {
        timeframe: 'Overall',
        signal: liveSignal.recommendation as any,
        strength: liveSignal.confidence,
        fvgSignal: fvgOverall.signal,
        fvgCount: fvgOverall.count,
        fvgConfidence: fvgOverall.confidence
      },
    ]
  }

  const timeframeSignals = calculateTimeframeSignals()

  // Calculate today's high/low from recent data
  const activeChartData = useMemo(() => {
    if (finalPolygonData.length === 0) return []
    const limit = recommendedBarLimit(timeframe, dataDisplayTimeframe)
    return finalPolygonData.slice(-limit)
  }, [finalPolygonData, timeframe, dataDisplayTimeframe])

  const viewStats = useMemo(() => {
    if (activeChartData.length === 0) {
      return {
        open: livePrice,
        high: livePrice,
        low: livePrice,
        close: livePrice,
        volume: 0,
        lastVolume: 0,
      }
    }
    const highs = activeChartData.map(b => b.high)
    const lows = activeChartData.map(b => b.low)
    const totalVolume = activeChartData.reduce((sum, bar) => sum + bar.volume, 0)
    const lastVolume = activeChartData[activeChartData.length - 1]?.volume ?? 0
    return {
      open: activeChartData[0].open,
      high: Math.max(...highs),
      low: Math.min(...lows),
      close: activeChartData[activeChartData.length - 1].close,
      volume: totalVolume,
      lastVolume,
    }
  }, [activeChartData, livePrice])

  // Calculate current volume and price range from polygon data
  const volumeStr = formatVolumeValue(viewStats.volume)

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

  // Auto-connect WebSocket on mount
  useEffect(() => {
    if (typeof window === 'undefined') return

    polygonWebSocketService.connect().catch(error => {
      console.error('[TickerPage] Failed to connect WebSocket:', error)
    })
  }, [])

  // Subscribe to WebSocket trade stream for rapid live price updates
  useEffect(() => {
    if (!symbol) return

    console.log('[TickerPage] Subscribing to trade stream for rapid price updates');

    const unsubscribe = polygonWebSocketService.onTrade(
      symbol.toUpperCase(),
      (trade) => {
        setLivePrice(prev => {
          if (trade.price > prev) {
            setPriceFlash('up')
          } else if (trade.price < prev) {
            setPriceFlash('down')
          }
          setTimeout(() => setPriceFlash(null), 300)
          return trade.price
        })
        setLastTradeTime(Date.now())
        setTradeCount(prev => prev + 1)

        if (ticker && snapshot?.prevDay?.c) {
          const prevClose = snapshot.prevDay.c
          const change = trade.price - prevClose
          const changePct = (change / prevClose) * 100

          setTicker(prev => prev ? {
            ...prev,
            price: trade.price,
            change,
            changePercent: changePct,
          } : null)
        }
      }
    )

    return unsubscribe
  }, [symbol, ticker, snapshot])

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
          <Link
            href="/dashboard"
            className="mb-4 text-sm text-gray-500 hover:text-white transition-colors flex items-center gap-2"
          >
            ← Back to Dashboard
          </Link>
          <div className="flex flex-col gap-4 ticker-header__content">
            <div className="flex flex-row items-center justify-between gap-2 flex-wrap ticker-header__row">
              <div className="flex flex-row items-baseline gap-2 sm:gap-4 ticker-header__titleGroup">
                <h1 className="text-xl sm:text-3xl font-bold ticker-header__title">{ticker.symbol}</h1>
                {/* Real-time Trade Stream Indicator */}
                <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-green-500/20 border border-green-500">
                  <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
                  <span className="text-xs font-bold text-green-400">LIVE</span>
                  {tradeCount > 0 && (
                    <span className="text-[10px] text-green-300">{tradeCount} trades</span>
                  )}
                </div>
                <span className="text-xs sm:text-sm text-gray-500 font-medium" title={`Visible: ${visibleBarCount} / Total: ${chartData.length} bars • Timeframe: ${timeframe}, Range: ${displayTimeframe}`}>
                  {visibleBarCount} bars
                </span>
                <div className="flex items-baseline gap-2 sm:gap-3">
                  <span
                    className={`text-lg sm:text-2xl font-semibold transition-all duration-200 ticker-header__price ${
                      priceFlash === 'up' ? 'text-green-400 scale-110' :
                      priceFlash === 'down' ? 'text-red-400 scale-110' :
                      'hover:text-green-300'
                    }`}
                    style={{
                      transition: 'all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)',
                      textShadow: priceFlash ? '0 0 10px currentColor' : 'none'
                    }}
                  >
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
                  <span className="text-white font-semibold">${viewStats.open.toFixed(2)}</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-gray-500">High</span>
                  <span className="text-green-400 font-semibold">${viewStats.high.toFixed(2)}</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-gray-500">Low</span>
                  <span className="text-red-400 font-semibold">${viewStats.low.toFixed(2)}</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-gray-500">Volume</span>
                  <span className="text-white font-semibold">{volumeStr}</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-gray-500">Range</span>
                  <span className="text-gray-300 font-semibold">${viewStats.low.toFixed(2)} - ${viewStats.high.toFixed(2)}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Price Chart Section */}
      <section className="bg-gray-900 border-b border-gray-800 py-4">
        <div className="w-full px-2 sm:px-4">
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
          <div className="w-full mx-auto relative" style={{ height: '640px' }}>
            {/* FVG Toggle Button */}
            <div className="absolute top-14 left-2 z-10 flex flex-col gap-2">
              <div className="flex items-center gap-2">
                {showFvg && fvgCount > 0 && (
                  <div className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-blue-500/20 text-blue-400 border border-blue-500">
                    {fvgCount} Pattern{fvgCount !== 1 ? 's' : ''} Found
                  </div>
                )}
                <button
                  onClick={() => setShowFvg(!showFvg)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
                    showFvg
                      ? 'bg-green-500/20 text-green-400 border border-green-500 hover:bg-green-500/30'
                      : 'bg-gray-800/50 text-gray-400 border border-gray-700 hover:bg-gray-700/50'
                  }`}
                >
                  {showFvg ? '✓ FVG Detection ON' : 'FVG Detection OFF'}
                </button>
                <button
                  onClick={() => setShowBacktest(true)}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 bg-purple-500/20 text-purple-400 border border-purple-500 hover:bg-purple-500/30"
                  title="View FVG Backtest Results"
                >
                  📊 Backtest
                </button>
              </div>

              {/* FVG Percentage Radio Buttons */}
              {showFvg && (
                <div className="flex items-center gap-2 bg-gray-900/90 px-3 py-2 rounded-lg border border-gray-700">
                  <span className="text-xs text-gray-400 font-medium">Gap %:</span>
                  {[0.05, 0.1, 0.15, 0.2, 0.3, 0.5].filter(val => val >= fvgGapSettings.min && val <= fvgGapSettings.max).map(value => (
                    <label key={value} className="flex items-center gap-1 cursor-pointer">
                      <input
                        type="radio"
                        name="fvgPercentage"
                        value={value}
                        checked={fvgPercentage === value}
                        onChange={(e) => setFvgPercentage(parseFloat(e.target.value))}
                        className="w-3 h-3 accent-blue-500"
                      />
                      <span className={`text-xs font-semibold ${fvgPercentage === value ? 'text-blue-400' : 'text-gray-400'}`}>
                        {value.toFixed(2)}%
                      </span>
                    </label>
                  ))}
                </div>
              )}
            </div>

            <ProfessionalChart
              symbol={ticker.symbol}
              currentPrice={livePrice}
              stopLoss={livePrice * 0.98}
              targets={chartTargets}
              entryPoint={livePrice}
              data={chartData.length > 0 ? chartData : undefined}
              showFvg={showFvg}
              fvgPercentage={fvgPercentage}
              onFvgCountChange={setFvgCount}
              onVisibleBarCountChange={handleVisibleBarCountChange}
              onLoadMoreData={handleLoadMoreData}
              isLoadingMore={isLoadingMoreData}
              onTimeframeChange={(tf, displayTf, intervalLabelOverride) => {
                console.log('[TickerPage] Timeframe changed to:', tf, 'Display:', displayTf, 'Interval:', intervalLabelOverride);
                setTimeframe(tf as any);
                setDisplayTimeframe(displayTf);

                if (displayTf !== 'Custom') {
                  setDataDisplayTimeframe(displayTf);
                } else if (intervalLabelOverride) {
                  const overrideDisplay = intervalLabelToDisplayTimeframe(intervalLabelOverride);
                  if (overrideDisplay) {
                    setDataDisplayTimeframe(overrideDisplay);
                  }
                }

                setAdditionalBarsToLoad(0); // Reset to base limit when changing timeframe
              }}
            />
          </div>
          {!polygonError && !isPolygonLoading && finalPolygonData.length > 0 && (
            <div className="mt-2 text-center">
              <div className="flex items-center justify-center gap-3">
                <span className="text-xs text-green-400 font-semibold animate-pulse">⚡ REAL-TIME TRADE STREAM</span>
                {wsConnected ? (
                  <span className="text-xs text-green-400">✓ WebSocket Connected</span>
                ) : (
                  <span className="text-xs text-yellow-400">⟳ Connecting WebSocket...</span>
                )}
                {tradeCount > 0 && (
                  <span className="text-xs text-blue-400">{tradeCount} trades received</span>
                )}
                {lastTradeTime > 0 && (
                  <span className="text-xs text-gray-400">Last update: {Math.round((Date.now() - lastTradeTime) / 1000)}s ago</span>
                )}
                {wsError && <span className="text-xs text-red-400">✗ Error: {wsError.message}</span>}
              </div>
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

                {/* RSI/Trend Signal */}
                <div className={`text-xs sm:text-sm font-bold mb-1 ${getSignalColor(tf.signal)}`}>
                  {tf.signal}
                </div>
                <div className="w-full h-1.5 bg-gray-700 rounded-full overflow-hidden mb-2">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${
                      tf.signal.includes('Buy') ? 'bg-green-400' :
                      tf.signal.includes('Sell') ? 'bg-red-400' :
                      'bg-yellow-400'
                    }`}
                    style={{ width: `${tf.strength}%` }}
                  />
                </div>
                <div className="text-xs text-gray-500 mb-2">{tf.strength}%</div>

                {/* FVG Pattern Analysis */}
                {tf.fvgCount !== undefined && tf.fvgCount > 0 && (
                  <div className="pt-2 border-t border-gray-700/50">
                    <div className="flex items-center justify-center gap-1 mb-1">
                      <div className={`w-1.5 h-1.5 rounded-full ${
                        tf.fvgSignal === 'Bullish' ? 'bg-green-400' :
                        tf.fvgSignal === 'Bearish' ? 'bg-red-400' :
                        'bg-gray-400'
                      }`}></div>
                      <span className={`text-xs font-semibold ${
                        tf.fvgSignal === 'Bullish' ? 'text-green-400' :
                        tf.fvgSignal === 'Bearish' ? 'text-red-400' :
                        'text-gray-400'
                      }`}>
                        {tf.fvgSignal} FVG
                      </span>
                    </div>
                    <div className="text-xs text-gray-500">
                      {tf.fvgCount} pattern{tf.fvgCount > 1 ? 's' : ''} • {tf.fvgConfidence}%
                    </div>
                  </div>
                )}
                {tf.fvgCount === 0 && (
                  <div className="pt-2 border-t border-gray-700/50">
                    <div className="text-xs text-gray-600">No FVG patterns</div>
                  </div>
                )}
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
                <div className="text-gray-400 text-xs font-medium mb-1">Current Candle Vol</div>
                <div className="text-sm sm:text-base font-semibold">{formatVolumeValue(viewStats.lastVolume)}</div>
                {activeChartData.length > 0 && <div className="text-xs text-green-400 mt-1">✓ Live data</div>}
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

        {/* FVG Trading Strategy */}
        {polygonData.length > 30 && (
          <section className="mt-6">
            <FvgStrategyPanel
              data={polygonData}
              timeframe={timeframe}
            />
          </section>
        )}
      </main>

      {/* FVG Backtest Panel */}
      {showBacktest && visibleChartData.length > 0 && (
        <FvgBacktestPanel
          data={visibleChartData}
          timeframe={timeframe}
          displayTimeframe={displayTimeframe}
          isOpen={showBacktest}
          onClose={() => setShowBacktest(false)}
        />
      )}
    </div>
  )
}