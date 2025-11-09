import { useCallback, useState, useEffect } from 'react'
import { CandleData } from './types'
import { Timeframe } from '@/types/polygon'

export function useFullscreen(containerRef: React.RefObject<HTMLDivElement>) {
  const [isFullscreen, setIsFullscreen] = useState(false)

  const toggleFullscreen = useCallback(async () => {
    if (!containerRef.current) return
    try {
      if (!document.fullscreenElement) {
        await containerRef.current.requestFullscreen()
        setIsFullscreen(true)
      } else {
        await document.exitFullscreen()
        setIsFullscreen(false)
      }
    } catch (error) {
      console.error('Error toggling fullscreen:', error)
    }
  }, [containerRef])

  useEffect(() => {
    const handleFullscreenChange = () => setIsFullscreen(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  return { isFullscreen, toggleFullscreen }
}

export function useChartData(externalData: CandleData[] | undefined, currentPrice: number, onDataUpdate?: (data: CandleData[]) => void) {
  const [data, setData] = useState<CandleData[]>([])
  const [useExternalData, setUseExternalData] = useState(false)

  const generateCandleData = useCallback(() => {
    const candles: CandleData[] = []
    let price = currentPrice || 100
    const now = Date.now()
    for (let i = 99; i >= 0; i--) {
      const change = (Math.random() - 0.5) * price * 0.02
      const open = price
      price += change
      const close = price
      const highLow = Math.random() * price * 0.01
      const high = Math.max(open, close) + highLow
      const low = Math.min(open, close) - highLow
      candles.push({ time: now - i * 60000, open, high, low, close, volume: Math.floor(Math.random() * 1000000) + 500000 })
    }
    return candles
  }, [currentPrice])

  useEffect(() => {
    if (externalData && externalData.length > 0) {
      console.log(`[useChartData] Setting external data: ${externalData.length} bars`)
      // Only update if data is actually different (prevents flash of old data)
      setData(prevData => {
        // Check if data has actually changed
        if (prevData.length === externalData.length &&
            prevData[0]?.time === externalData[0]?.time &&
            prevData[prevData.length - 1]?.time === externalData[externalData.length - 1]?.time) {
          console.log('[useChartData] Data unchanged, skipping update')
          return prevData
        }
        return externalData
      })
      setUseExternalData(true)
      onDataUpdate?.(externalData)
    }
    // Don't show mock data - wait for real data from Polygon
    // If externalData is empty but we have old data, keep showing old data (prevents flash)
  }, [externalData, onDataUpdate])

  return { data, useExternalData }
}

export function useVisibleRange(
  data: CandleData[],
  panOffset: number,
  timeScale: number,
  displayTimeframe?: string,
  dataTimeframe?: string
) {
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 100 })

  useEffect(() => {
    if (data.length === 0) return

    // POLICY: Auto-fit mode when timeScale=1.0 and panOffset=0
    // Shows last N bars based on displayTimeframe to align with UX expectations
    const shouldAutoFit = timeScale === 1.0 && panOffset === 0

    // Determine default bars per display timeframe
    const getDefaultBarsForView = (display?: string, dataTf?: string): number => {
      if (!display || !dataTf) return 100

      // 1-minute chart configurations - show all available data
      if (dataTf === '1m') {
        if (display === '1D') return Math.min(data.length, 390) // Full trading day (6.5 hours)
        if (display === '5D') return Math.min(data.length, 1950) // 5 trading days
        if (display === '1M') return Math.min(data.length, 8190) // ~21 trading days
        return Math.min(data.length, 390) // Default to 1 day for other timeframes
      }

      // 5-minute chart configurations
      if (dataTf === '5m') {
        if (display === '1D') return Math.min(data.length, 78) // 6.5 hours
        if (display === '5D') return Math.min(data.length, 390) // 5 days
        return 30
      }

      // Other timeframe configurations
      if (display === '1D' && dataTf === '15m') return 30 // ~trading day
      if (display === '5D' && dataTf === '1h') return 40
      if (display === '1M' && dataTf === '4h') return 44
      if (display === '3M' && dataTf === '1d') return 63
      if (display === '6M' && dataTf === '1d') return 126
      if (display === 'YTD' && dataTf === '1d') return 200
      if (display === '1Y' && dataTf === '1d') return 252
      if (display === '5Y' && dataTf === '1w') return 260
      if (display === 'All' && dataTf === '1M') return data.length
      return 100
    }

    let effectiveCandlesInView: number
    let scrollBack: number

    const baseCandlesInView = getDefaultBarsForView(displayTimeframe, dataTimeframe)

    if (shouldAutoFit) {
      // Auto-fit mode: show last N bars based on timeframe defaults
      effectiveCandlesInView = Math.min(baseCandlesInView, data.length)
      scrollBack = 0
    } else {
      // Manual zoom/pan mode: scale relative to default view
      const candlesInView = Math.round(baseCandlesInView / timeScale)

      // panOffset = how far back in time we've scrolled
      effectiveCandlesInView = candlesInView
      scrollBack = Math.max(0, panOffset)
    }

    // Calculate visible range - end at latest data minus scroll back
    const end = Math.min(data.length, data.length - scrollBack)
    const start = Math.max(0, end - effectiveCandlesInView)

    setVisibleRange({ start, end })
  }, [panOffset, data.length, timeScale, displayTimeframe, dataTimeframe])

  return visibleRange
}

export function useCurrentTime() {
  const [currentTime, setCurrentTime] = useState(new Date())
  useEffect(() => {
    const updateTime = () => setCurrentTime(new Date())
    const timeInterval = window.setInterval(updateTime, 2000)
    return () => window.clearInterval(timeInterval)
  }, [])
  return currentTime
}
