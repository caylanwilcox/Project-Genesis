import { useCallback, useEffect, useState } from 'react'
import { CandleData } from './types'

export function useChartInteraction(
  data: CandleData[],
  panOffset: number,
  setPanOffset: (offset: number) => void,
  timeScale: number,
  setTimeScale: (scale: number | ((prev: number) => number)) => void
) {
  const [isPanning, setIsPanning] = useState(false)
  const [panStart, setPanStart] = useState<{ x: number; offset: number } | null>(null)
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null)
  const [pinchStart, setPinchStart] = useState<{ distance: number; scale: number } | null>(null)

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    setIsPanning(true)
    setPanStart({ x: e.clientX, offset: panOffset })
  }, [panOffset])

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>, rect: DOMRect) => {
    if (isPanning && panStart) {
      const deltaX = e.clientX - panStart.x
      // Account for current zoom level when calculating pan
      const baseCandlesInView = 100
      const actualCandlesInView = baseCandlesInView / timeScale
      const candlesPerPixel = actualCandlesInView / rect.width
      // Drag right (deltaX > 0) = scroll back in time (offset increases)
      // Drag left (deltaX < 0) = scroll forward (offset decreases)
      const candleDelta = deltaX * candlesPerPixel * 2
      const newOffset = panStart.offset + candleDelta

      // POLICY: Allow panning back in time (positive offset)
      // Prevent future whitespace (no negative offset)
      const maxOffset = Math.max(0, data.length - actualCandlesInView)
      const minOffset = 0 // No future whitespace
      setPanOffset(Math.max(minOffset, Math.min(maxOffset, newOffset)))
    } else {
      setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top })
    }
  }, [isPanning, panStart, data.length, setPanOffset, timeScale])

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
    setPanStart({ x: t.clientX, offset: panOffset })
  }, [panOffset, timeScale])

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
      const deltaX = t.clientX - panStart.x
      // Account for current zoom level when calculating pan
      const baseCandlesInView = 100
      const actualCandlesInView = baseCandlesInView / timeScale
      const candlesPerPixel = actualCandlesInView / rect.width
      // Drag right (deltaX > 0) = scroll back in time (offset increases)
      // Drag left (deltaX < 0) = scroll forward (offset decreases)
      const candleDelta = deltaX * candlesPerPixel * 2
      const newOffset = panStart.offset + candleDelta

      // POLICY: Allow panning back in time (positive offset)
      // Prevent future whitespace (no negative offset)
      const maxOffset = Math.max(0, data.length - actualCandlesInView)
      const minOffset = 0 // No future whitespace
      setPanOffset(Math.max(minOffset, Math.min(maxOffset, newOffset)))
    }
    setMousePos({ x: t.clientX - rect.left, y: t.clientY - rect.top })
  }, [isPanning, panStart, data.length, setPanOffset, pinchStart, timeScale, setTimeScale])

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
