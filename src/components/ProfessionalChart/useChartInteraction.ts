import { useCallback, useEffect, useState } from 'react'
import { CandleData } from './types'

export function useChartInteraction(
  data: CandleData[],
  panOffset: number,
  setPanOffset: (offset: number) => void,
  timeScale: number,
  setTimeScale: (scale: number | ((prev: number) => number)) => void,
  priceOffset: number,
  setPriceOffset: (offset: number) => void,
  onReachLeftEdge?: () => void
) {
  const [isPanning, setIsPanning] = useState(false)
  const [panStart, setPanStart] = useState<{ x: number; y: number; offsetX: number; offsetY: number } | null>(null)
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null)
  const [pinchStart, setPinchStart] = useState<{ distance: number; scale: number } | null>(null)
  const [lastLoadTrigger, setLastLoadTrigger] = useState(0)

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    setIsPanning(true)
    setPanStart({ x: e.clientX, y: e.clientY, offsetX: panOffset, offsetY: priceOffset })
  }, [panOffset, priceOffset])

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>, rect: DOMRect) => {
    if (isPanning && panStart) {
      // Horizontal panning (time axis)
      const deltaX = e.clientX - panStart.x
      const baseCandlesInView = 100
      const actualCandlesInView = baseCandlesInView / timeScale
      const candlesPerPixel = actualCandlesInView / rect.width
      const candleDelta = deltaX * candlesPerPixel // Direct 1:1 pixel mapping
      const newOffsetX = panStart.offsetX + candleDelta

      const maxOffset = Math.max(0, data.length - actualCandlesInView)
      const minOffset = 0
      const clampedOffset = Math.max(minOffset, Math.min(maxOffset, newOffsetX))
      setPanOffset(clampedOffset)

      // If user is trying to pan left beyond beginning (negative offset attempt)
      // and we're already at the left edge, trigger callback to load more data
      // Debounce to prevent multiple rapid triggers (2 second cooldown)
      if (newOffsetX < -10 && clampedOffset === 0 && onReachLeftEdge) {
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
      setPinchStart({ distance, scale: timeScale })
      return
    }

    // Handle single touch pan
    e.preventDefault()
    const t = e.touches[0]
    setIsPanning(true)
    setPanStart({ x: t.clientX, y: t.clientY, offsetX: panOffset, offsetY: priceOffset })
  }, [panOffset, priceOffset, timeScale])

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

      if (!pinchStart) {
        setPinchStart({ distance, scale: timeScale })
      } else {
        const scaleFactor = distance / pinchStart.distance
        const newScale = pinchStart.scale * scaleFactor
        setTimeScale(Math.max(0.2, Math.min(5, newScale)))
      }
      return
    }

    e.preventDefault()
    const t = e.touches[0]
    if (isPanning && panStart) {
      // Horizontal panning
      const deltaX = t.clientX - panStart.x
      const baseCandlesInView = 100
      const actualCandlesInView = baseCandlesInView / timeScale
      const candlesPerPixel = actualCandlesInView / rect.width
      const candleDelta = deltaX * candlesPerPixel // Direct 1:1 pixel mapping
      const newOffsetX = panStart.offsetX + candleDelta

      const maxOffset = Math.max(0, data.length - actualCandlesInView)
      const minOffset = 0
      const clampedOffset = Math.max(minOffset, Math.min(maxOffset, newOffsetX))
      setPanOffset(clampedOffset)

      // If user is trying to pan left beyond beginning, trigger callback to load more data
      // Debounce to prevent multiple rapid triggers (2 second cooldown)
      if (newOffsetX < -10 && clampedOffset === 0 && onReachLeftEdge) {
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
  }
}
