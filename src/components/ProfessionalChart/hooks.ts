import { useCallback, useState, useEffect } from 'react'
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
      setData(externalData)
      setUseExternalData(true)
      onDataUpdate?.(externalData)
    } else {
      setData(generateCandleData())
      setUseExternalData(false)
    }
  }, [externalData, generateCandleData, onDataUpdate])

  return { data, useExternalData }
}

export function useVisibleRange(data: CandleData[], panOffset: number, timeScale: number) {
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 100 })

  useEffect(() => {
    if (data.length === 0) return
    const baseCandlesInView = 100
    const candlesInView = Math.round(baseCandlesInView / timeScale)

    // Calculate how many candles to actually show
    // Negative panOffset = show fewer candles (creates white space on right)
    // Positive panOffset = scroll back in time
    // Whitespace should be a fixed percentage, not scale with zoom
    const whiteSpacePadding = panOffset < 0 ? Math.abs(panOffset) : 0
    const effectiveCandlesInView = Math.max(1, Math.round(candlesInView - whiteSpacePadding))
    const scrollBack = Math.max(0, panOffset)

    // Calculate visible range - end at latest data minus scroll back
    const end = Math.min(data.length, data.length - scrollBack)
    const start = Math.max(0, end - effectiveCandlesInView)

    setVisibleRange({ start, end })
  }, [panOffset, data.length, timeScale])

  return visibleRange
}

export function useCurrentTime() {
  const [currentTime, setCurrentTime] = useState(new Date())
  useEffect(() => {
    const updateTime = () => setCurrentTime(new Date())
    const timeInterval = window.setInterval(updateTime, 1000)
    return () => window.clearInterval(timeInterval)
  }, [])
  return currentTime
}
