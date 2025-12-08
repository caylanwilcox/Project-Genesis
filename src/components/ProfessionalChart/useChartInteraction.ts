import { useCallback, useEffect, useRef, useState } from 'react'
import { CandleData } from './types'
import { Timeframe } from '@/types/polygon'
import { getDefaultBarsForView } from './hooks'

export function useChartInteraction(
  data: CandleData[],
  panOffset: number,
  setPanOffset: (offset: number) => void,
  timeScale: number,
  setTimeScale: (scale: number | ((prev: number) => number)) => void,
  priceOffset: number,
  setPriceOffset: (offset: number) => void,
  displayTimeframe?: string,
  dataTimeframe?: Timeframe,
  onReachLeftEdge?: () => void,
  onZoomTransition?: (timeScale: number) => string | undefined
) {
  const [isPanning, setIsPanning] = useState(false)
  const [panStart, setPanStart] = useState<{ x: number; y: number; offsetX: number; offsetY: number } | null>(null)
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null)
  const [pinchStart, setPinchStart] = useState<{ distance: number; scale: number } | null>(null)
  const [lastLoadTrigger, setLastLoadTrigger] = useState(0)
  const panOffsetRef = useRef(panOffset)
  const timeScaleRef = useRef(timeScale)

  useEffect(() => {
    panOffsetRef.current = panOffset
  }, [panOffset])

  useEffect(() => {
    timeScaleRef.current = timeScale
  }, [timeScale])

  const clampRatio = useCallback((value: number) => Math.min(1, Math.max(0, value)), [])
  // Scale: >1 = zoom in (fewer candles, wider), <1 = zoom out (more candles, thinner)
  // Allow zooming out to 0.05 (20x more candles) and in to 5 (5x fewer candles)
  const clampScale = useCallback((value: number) => Math.max(0.05, Math.min(5, value)), [])

  const calcEffectiveCandles = useCallback((scale: number) => {
    if (data.length === 0) return 0
    const base = getDefaultBarsForView(displayTimeframe, dataTimeframe, data.length)
    const zoomed = Math.round(base / Math.max(scale, 0.05))
    // Clamp to data.length max when zooming out, minimum 10 candles when zoomed in
    return Math.max(10, Math.min(data.length, zoomed))
  }, [data.length, displayTimeframe, dataTimeframe])

  const adjustPanForZoom = useCallback((prevScale: number, newScale: number, pointerRatio: number) => {
    if (data.length === 0 || prevScale === newScale) return
    const ratio = Number.isFinite(pointerRatio) ? clampRatio(pointerRatio) : 0.5
    const effOld = calcEffectiveCandles(prevScale)
    const effNew = calcEffectiveCandles(newScale)
    if (effOld === 0 || effNew === 0) return

    const endOld = data.length - panOffsetRef.current
    const startOld = endOld - effOld
    const pointerIndex = startOld + ratio * effOld

    const startNew = pointerIndex - ratio * effNew
    const endNew = startNew + effNew

    // No clamping - allow free navigation
    const scrollBack = data.length - endNew
    setPanOffset(scrollBack)
  }, [calcEffectiveCandles, clampRatio, data.length, setPanOffset])

  const applyZoomToScale = useCallback((targetScale: number, pointerRatio: number) => {
    const prevScale = timeScaleRef.current
    const nextScale = clampScale(targetScale)
    adjustPanForZoom(prevScale, nextScale, pointerRatio)
    setTimeScale(nextScale)

    // Check if we should transition to a different timeframe
    if (onZoomTransition) {
      const newTimeframe = onZoomTransition(nextScale)
      if (newTimeframe) {
        // Timeframe changed - reset scale to 1 for smooth continuation
        // This creates a seamless zoom experience where zooming out
        // eventually transitions to a coarser timeframe with scale reset
        setTimeScale(1)
      }
    }
  }, [adjustPanForZoom, clampScale, setTimeScale, onZoomTransition])

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    console.log('[ChartInteraction] Mouse down - starting pan', { panOffset, priceOffset })
    setIsPanning(true)
    setPanStart({ x: e.clientX, y: e.clientY, offsetX: panOffset, offsetY: priceOffset })
  }, [panOffset, priceOffset])

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>, rect: DOMRect) => {
    if (isPanning && panStart) {
      // Horizontal panning (time axis)
      const deltaX = e.clientX - panStart.x
      // Base panning sensitivity on visible candles, not total data length
      // This gives consistent drag-to-scroll behavior regardless of how much data is loaded
      const visibleCandles = Math.min(120, data.length) // Match the visible window size from hooks.ts
      const candlesPerPixel = visibleCandles / rect.width
      const panSensitivity = 2.0 // Increase sensitivity to make panning easier
      const candleDelta = deltaX * candlesPerPixel * panSensitivity // Positive: drag left = scroll back in time
      const newOffsetX = panStart.offsetX + candleDelta

      // Allow free scrolling with soft limits
      // Positive = scroll left into history, Negative = scroll right into empty future space
      const maxLeftScroll = data.length // Can scroll back through all history
      const maxRightScroll = -50 // Can scroll 50 candle-widths into empty space on right
      const clampedOffset = Math.max(maxRightScroll, Math.min(maxLeftScroll, newOffsetX))
      setPanOffset(clampedOffset)

      // If user is panning left (increasing panOffset) and has reached near the left edge of data
      // trigger callback to load more historical data
      // Debounce to prevent multiple rapid triggers (2 second cooldown)
      const leftEdgeThreshold = data.length - 20 // Within 20 bars of the oldest data
      if (clampedOffset > leftEdgeThreshold && onReachLeftEdge) {
        const now = Date.now()
        if (now - lastLoadTrigger > 2000) {
          console.log('[ChartInteraction] Reached left edge, triggering onReachLeftEdge callback')
          setLastLoadTrigger(now)
          onReachLeftEdge()
        }
      }

      // Vertical panning (price axis)
      const deltaY = e.clientY - panStart.y
      // Drag down (deltaY > 0) = scroll prices up (show higher prices)
      // Drag up (deltaY < 0) = scroll prices down (show lower prices)
      const pricePerPixel = 0.03 // Smooth vertical panning
      const priceDelta = deltaY * pricePerPixel // Pull down = go up
      const newOffsetY = panStart.offsetY + priceDelta

      // Allow unlimited vertical panning
      setPriceOffset(newOffsetY)
    } else {
      setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top })
    }
  }, [isPanning, panStart, data.length, setPanOffset, timeScale, setPriceOffset, onReachLeftEdge, lastLoadTrigger])

  const handleMouseUp = useCallback(() => {
    setIsPanning(false)
    setPanStart(null)
  }, [])

  const handleMouseLeave = useCallback(() => {
    setMousePos(null)
    setIsPanning(false)
    setPanStart(null)
  }, [])

  const handleTouchStart = useCallback((e: React.TouchEvent<HTMLDivElement>) => {
    if (e.touches.length === 0) return

    // Handle pinch zoom
    if (e.touches.length === 2) {
      e.preventDefault()
      const touch1 = e.touches[0]
      const touch2 = e.touches[1]
      const distance = Math.hypot(
        touch2.clientX - touch1.clientX,
        touch2.clientY - touch1.clientY
      )
      setPinchStart({ distance, scale: timeScaleRef.current })
      return
    }

    // Handle single touch pan
    e.preventDefault()
    const t = e.touches[0]
    setIsPanning(true)
    setPanStart({ x: t.clientX, y: t.clientY, offsetX: panOffset, offsetY: priceOffset })
  }, [panOffset, priceOffset])

  const handleTouchMove = useCallback((e: React.TouchEvent<HTMLDivElement>, rect: DOMRect) => {
    if (e.touches.length === 0) return

    // Handle pinch zoom
    if (e.touches.length === 2) {
      e.preventDefault()

      const touch1 = e.touches[0]
      const touch2 = e.touches[1]
      const distance = Math.hypot(
        touch2.clientX - touch1.clientX,
        touch2.clientY - touch1.clientY
      )

      const ratio = clampRatio((((touch1.clientX + touch2.clientX) / 2) - rect.left) / rect.width)

      if (!pinchStart) {
        setPinchStart({ distance, scale: timeScaleRef.current })
      } else {
        const scaleFactor = distance / pinchStart.distance
        const newScale = pinchStart.scale * scaleFactor
        applyZoomToScale(newScale, ratio)
      }
      return
    }

    e.preventDefault()
    const t = e.touches[0]
    if (isPanning && panStart) {
      // Horizontal panning
      const deltaX = t.clientX - panStart.x
      // Base panning sensitivity on visible candles, not total data length
      const visibleCandles = Math.min(120, data.length) // Match the visible window size from hooks.ts
      const candlesPerPixel = visibleCandles / rect.width
      const panSensitivity = 2.0 // Increase sensitivity to make panning easier
      const candleDelta = deltaX * candlesPerPixel * panSensitivity // Positive: drag left = scroll back in time
      const newOffsetX = panStart.offsetX + candleDelta

      // Allow free scrolling with soft limits
      // Positive = scroll left into history, Negative = scroll right into empty future space
      const maxLeftScroll = data.length // Can scroll back through all history
      const maxRightScroll = -50 // Can scroll 50 candle-widths into empty space on right
      const clampedOffset = Math.max(maxRightScroll, Math.min(maxLeftScroll, newOffsetX))
      setPanOffset(clampedOffset)

      // If user is panning left (increasing panOffset) and has reached near the left edge of data
      // trigger callback to load more historical data
      // Debounce to prevent multiple rapid triggers (2 second cooldown)
      const leftEdgeThreshold = data.length - 20 // Within 20 bars of the oldest data
      if (clampedOffset > leftEdgeThreshold && onReachLeftEdge) {
        const now = Date.now()
        if (now - lastLoadTrigger > 2000) {
          console.log('[ChartInteraction] Reached left edge (touch), triggering onReachLeftEdge callback')
          setLastLoadTrigger(now)
          onReachLeftEdge()
        }
      }

      // Vertical panning
      const deltaY = t.clientY - panStart.y
      const pricePerPixel = 0.03 // Smooth vertical panning
      const priceDelta = deltaY * pricePerPixel // Pull down = go up
      const newOffsetY = panStart.offsetY + priceDelta
      setPriceOffset(newOffsetY)
    }
    setMousePos({ x: t.clientX - rect.left, y: t.clientY - rect.top })
  }, [isPanning, panStart, data.length, setPanOffset, pinchStart, timeScale, setTimeScale, setPriceOffset, onReachLeftEdge, lastLoadTrigger])

  const handleTouchEnd = useCallback(() => {
    setIsPanning(false)
    setPanStart(null)
    setMousePos(null)
    setPinchStart(null)
  }, [])

  const handleWheel = useCallback((e: React.WheelEvent<HTMLDivElement>, rect: DOMRect) => {
    e.preventDefault()
    e.stopPropagation()
    if (!rect) return
    const direction = e.deltaY < 0 ? 1 : -1
    const zoomStep = (e.ctrlKey ? 0.05 : 0.1) * direction
    const ratio = clampRatio((e.clientX - rect.left) / rect.width)
    applyZoomToScale(timeScaleRef.current + zoomStep, ratio)
  }, [applyZoomToScale, clampRatio])

  return {
    isPanning,
    mousePos,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    handleMouseLeave,
    handleTouchStart,
    handleTouchMove,
    handleTouchEnd,
    handleWheel,
  }
}
