import { useCallback, useEffect, useState } from 'react'

export function useChartScaling() {
  const [priceScale, setPriceScale] = useState(1.0)
  const [isScaling, setIsScaling] = useState(false)
  const [scaleStart, setScaleStart] = useState<{ y: number; scale: number } | null>(null)
  const [timeScale, setTimeScale] = useState(1.0)
  const [isTimeScaling, setIsTimeScaling] = useState(false)
  const [timeScaleStart, setTimeScaleStart] = useState<{ x: number; scale: number } | null>(null)

  const handleScaleMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    e.stopPropagation()
    setIsScaling(true)
    setScaleStart({ y: e.clientY, scale: priceScale })
  }, [priceScale])

  const handleScaleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (isScaling && scaleStart) {
      const deltaY = e.clientY - scaleStart.y
      const scaleFactor = 1 - (deltaY / 300)
      const newScale = Math.max(0.1, Math.min(10, scaleStart.scale * scaleFactor))
      setPriceScale(newScale)
    }
  }, [isScaling, scaleStart])

  const handleScaleDoubleClick = useCallback(() => {
    setPriceScale(1.0)
  }, [])

  const handleTimeScaleMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    e.stopPropagation()
    setIsTimeScaling(true)
    setTimeScaleStart({ x: e.clientX, scale: timeScale })
  }, [timeScale])

  const handleTimeScaleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (isTimeScaling && timeScaleStart) {
      const deltaX = e.clientX - timeScaleStart.x
      const scaleFactor = 1 + (deltaX / 300)
      const newScale = Math.max(0.2, Math.min(5, timeScaleStart.scale * scaleFactor))
      setTimeScale(newScale)
    }
  }, [isTimeScaling, timeScaleStart])

  const handleTimeScaleDoubleClick = useCallback(() => {
    setTimeScale(1.0)
  }, [])

  useEffect(() => {
    const handleGlobalMouseUp = () => {
      if (isScaling) {
        setIsScaling(false)
        setScaleStart(null)
      }
      if (isTimeScaling) {
        setIsTimeScaling(false)
        setTimeScaleStart(null)
      }
    }
    document.addEventListener('mouseup', handleGlobalMouseUp)
    return () => document.removeEventListener('mouseup', handleGlobalMouseUp)
  }, [isScaling, isTimeScaling])

  return {
    priceScale,
    setPriceScale,
    timeScale,
    setTimeScale,
    isScaling,
    isTimeScaling,
    handleScaleMouseDown,
    handleScaleMouseMove,
    handleScaleDoubleClick,
    handleTimeScaleMouseDown,
    handleTimeScaleMouseMove,
    handleTimeScaleDoubleClick,
  }
}
