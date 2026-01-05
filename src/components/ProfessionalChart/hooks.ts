import { useCallback, useState, useEffect, useMemo } from 'react'
import { CandleData } from './types'

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
      // Only update if data is actually different (prevents flash of old data)
      setData(prevData => {
        // Check if data has actually changed
        if (prevData.length === externalData.length &&
            prevData[0]?.time === externalData[0]?.time &&
            prevData[prevData.length - 1]?.time === externalData[externalData.length - 1]?.time) {
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

export function getDefaultBarsForView(
  display?: string,
  dataTf?: string,
  dataLength: number = 0,
) {
  // TradingView-like default: show ~50 bars at 1x zoom, allowing zoom out to see all
  if (!display || !dataTf) return Math.min(50, dataLength)

  if (display === '1D' && dataTf === '1m') return 390
  if (display === '1D' && dataTf === '5m') return 78
  if (display === '5D' && dataTf === '5m') return 390
  if (display === '1M' && dataTf === '1h') return 140
  if (display === '3M' && dataTf === '4h') return 100
  if (display === '6M' && dataTf === '1d') return 126
  if (display === 'YTD' && dataTf === '1d') return 200
  if (display === '1Y' && dataTf === '1d') return 252
  if (display === '5Y' && dataTf === '1w') return 260
  if (display === 'All' && dataTf === '1M') return dataLength || 100

  if (display === '1D') {
    if (dataTf === '15m') return 26
    if (dataTf === '30m') return 13
    if (dataTf === '1h') return 7
    if (dataTf === '2h') return 3
    if (dataTf === '4h') return 2
    if (dataTf === '1d') return 1
  }

  if (display === '5D') {
    if (dataTf === '1m') return 1950
    if (dataTf === '5m') return 390
    if (dataTf === '15m') return 130
    if (dataTf === '1h') return 33
    if (dataTf === '2h') return 16
    if (dataTf === '4h') return 8
    if (dataTf === '1d') return 5
  }

  if (display === '1M') {
    if (dataTf === '1m') return 8190
    if (dataTf === '5m') return 1638
    if (dataTf === '15m') return 546
    if (dataTf === '30m') return 273
    if (dataTf === '2h') return 68
    if (dataTf === '4h') return 34
    if (dataTf === '1d') return 21
  }

  if (display === '3M') {
    if (dataTf === '1m') return 24570
    if (dataTf === '5m') return 4914
    if (dataTf === '15m') return 1638
    if (dataTf === '30m') return 819
    if (dataTf === '1h') return 410
    if (dataTf === '2h') return 205
    if (dataTf === '4h') return 100
    if (dataTf === '1d') return 63
    if (dataTf === '1w') return 13
  }

  if (display === '6M') {
    if (dataTf === '4h') return 315
    if (dataTf === '2h') return 630
    if (dataTf === '1d') return 126
    if (dataTf === '1w') return 26
  }

  if (display === 'YTD') {
    if (dataTf === '4h') return 350
    if (dataTf === '1d') return 200
    if (dataTf === '1w') return 40
  }

  if (display === '1Y') {
    if (dataTf === '4h') return 410
    if (dataTf === '1d') return 252
    if (dataTf === '1w') return 52
  }

  if (display === '5Y') {
    if (dataTf === '1d') return 1260
    if (dataTf === '1w') return 260
    if (dataTf === '1M') return 60
  }

  if (display === 'All') {
    return dataLength || 100
  }

  // Default: show all data or 100 bars minimum
  return Math.max(100, dataLength)
}

export function useVisibleRange(
  data: CandleData[],
  panOffset: number,
  timeScale: number,
  displayTimeframe?: string,
  dataTimeframe?: string
) {
  // Use useMemo for immediate synchronous updates - no delay on zoom/pan
  const visibleRange = useMemo(() => {
    if (data.length === 0) return { start: 0, end: 100 }

    const baseCandlesInView = getDefaultBarsForView(displayTimeframe, dataTimeframe, data.length)
    const zoomedCandlesInView = Math.round(baseCandlesInView / Math.max(timeScale, 0.05)) // timeScale > 1 = zoom in (fewer candles), < 1 = zoom out (more candles)
    // Clamp to data.length max when zooming out, minimum 10 candles when zoomed in
    const effectiveCandlesInView = Math.max(10, Math.min(data.length, zoomedCandlesInView))

    // panOffset = how many candles we've scrolled back from the latest data
    // Negative panOffset = scrolling right past the last candle (into empty space)
    const scrollBack = Math.floor(panOffset)

    // Calculate visible range - the window size can change based on zoom (timeScale),
    // and slides through the entire dataset as user pans left/right
    //
    // When panOffset = 0: end = data.length (show most recent)
    // When panOffset > 0: end < data.length (scroll back in time / left)
    // When panOffset < 0: end > data.length (scroll into future / right - empty space)
    const end = data.length - scrollBack
    const start = end - effectiveCandlesInView

    // Clamp start to 0 minimum - don't allow scrolling past oldest data
    // This prevents candles from "disappearing" when dragging too far left
    const actualStart = Math.max(0, start)
    // Adjust end based on clamped start to maintain window size
    const actualEnd = Math.max(actualStart + effectiveCandlesInView, end)

    return { start: actualStart, end: actualEnd }
  }, [panOffset, data.length, displayTimeframe, dataTimeframe, timeScale])

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
