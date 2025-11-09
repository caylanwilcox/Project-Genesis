'use client'

import React, { useRef, useState, useCallback, useEffect } from 'react'
import styles from './ProfessionalChart.module.css'
import tagStyles from './ChartYAxisTags.module.css'
import { ProfessionalChartProps } from './ProfessionalChart/types'
import { resolveDisplayToData, intervalLabelToTimeframe } from '@/utils/timeframePolicy'
import { useChartData, useVisibleRange, useFullscreen, useCurrentTime } from './ProfessionalChart/hooks'
import { useChartInteraction } from './ProfessionalChart/useChartInteraction'
import { useChartScaling } from './ProfessionalChart/useChartScaling'
import { ChartHeader } from './ProfessionalChart/ChartHeader'
import { MainChart } from './ProfessionalChart/MainChart'
import { PriceTag } from './ProfessionalChart/priceLines'
import { YAxisGutter } from './ProfessionalChart/YAxisGutter'
import { XAxisGutter } from './ProfessionalChart/XAxisGutter'

export const ProfessionalChart: React.FC<ProfessionalChartProps> = ({
  symbol,
  currentPrice = 445.20,
  stopLoss,
  targets = [],
  entryPoint,
  data: externalData,
  onDataUpdate,
  onTimeframeChange,
  showFvg = false,
  onFvgCountChange,
  onVisibleBarCountChange,
  onLoadMoreData
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartMainAreaRef = useRef<HTMLDivElement>(null)
  const [timeframe, setTimeframe] = useState('1D')
  const [interval, setInterval] = useState('1 hour')
  const [showIntervalDropdown, setShowIntervalDropdown] = useState(false)
  const [panOffset, setPanOffset] = useState(0) // Start in auto-fit mode (no offset)
  const [priceOffset, setPriceOffset] = useState(0) // Vertical panning offset
  const [dataTimeframe, setDataTimeframe] = useState('1h')
  const [overlayTags, setOverlayTags] = useState<PriceTag[]>([])

  const { data } = useChartData(externalData, currentPrice, onDataUpdate)
  const { priceScale, setPriceScale, timeScale, setTimeScale } = useChartScaling()
  const visibleRange = useVisibleRange(data, panOffset, timeScale, timeframe, dataTimeframe)
  const { isPanning, mousePos, handleMouseDown, handleMouseMove, handleMouseUp, handleMouseLeave, handleTouchStart, handleTouchMove, handleTouchEnd } = useChartInteraction(data, panOffset, setPanOffset, timeScale, setTimeScale, priceOffset, setPriceOffset, onLoadMoreData)
  const { isFullscreen, toggleFullscreen } = useFullscreen(chartContainerRef)
  const currentTime = useCurrentTime()

  // Add wheel event listener with passive: false to allow preventDefault
  useEffect(() => {
    const chartArea = chartMainAreaRef.current
    if (!chartArea) return

    let rafId: number | null = null
    let pendingZoom = 1
    let mouseX: number | null = null

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault()
      e.stopPropagation()

      // Capture mouse position for zoom focus
      const rect = chartArea.getBoundingClientRect()
      mouseX = e.clientX - rect.left

      // Accumulate zoom changes for smooth animation
      const sensitivity = 0.0015 // Adjust this for zoom speed (higher = faster)
      const zoomDelta = -e.deltaY * sensitivity
      pendingZoom *= Math.exp(zoomDelta)

      // Cancel previous animation frame
      if (rafId !== null) {
        cancelAnimationFrame(rafId)
      }

      // Apply zoom on next frame for smooth rendering
      rafId = requestAnimationFrame(() => {
        const oldScale = timeScale
        const newScale = Math.max(0.2, Math.min(5, oldScale * pendingZoom))

        // Calculate cursor position as percentage of chart width
        if (mouseX !== null && rect) {
          const chartWidth = rect.width
          const cursorRatio = mouseX / chartWidth // 0 = left edge, 1 = right edge

          // Calculate how many candles are visible before/after zoom
          const baseCandlesInView = 100
          const oldVisibleCandles = baseCandlesInView / oldScale
          const newVisibleCandles = baseCandlesInView / newScale

          // Adjust pan offset to keep cursor position stable
          // The cursor should point to the same candle before and after zoom
          const candleDifference = oldVisibleCandles - newVisibleCandles
          const offsetAdjustment = candleDifference * cursorRatio

          setPanOffset(prev => Math.max(0, prev + offsetAdjustment))
        }

        setTimeScale(newScale)
        pendingZoom = 1 // Reset accumulator
        rafId = null
      })
    }

    chartArea.addEventListener('wheel', handleWheel, { passive: false })

    return () => {
      if (rafId !== null) {
        cancelAnimationFrame(rafId)
      }
      chartArea.removeEventListener('wheel', handleWheel)
    }
  }, [setTimeScale, setPanOffset, timeScale])

  const handleTimeframeClick = useCallback((tf: string) => {
    setTimeframe(tf); setPriceScale(1.0); setTimeScale(1.0); setPanOffset(0); setPriceOffset(0)
    const resolved = resolveDisplayToData(tf)
    setInterval(resolved.intervalLabel)
    setDataTimeframe(resolved.timeframe)
    onTimeframeChange?.(resolved.timeframe, tf)
  }, [onTimeframeChange, setPriceScale, setTimeScale])

  const handleIntervalChange = useCallback((newInterval: string) => {
    setInterval(newInterval); setShowIntervalDropdown(false); setTimeScale(1.0); setPanOffset(0); setPriceOffset(0)
    const mapped = intervalLabelToTimeframe(newInterval)
    if (mapped) { setDataTimeframe(mapped); onTimeframeChange?.(mapped, 'Custom') }
  }, [onTimeframeChange, setTimeScale])

  return (
    <div ref={chartContainerRef} className={styles.chartContainer}>
      <ChartHeader symbol={symbol} timeframe={timeframe} interval={interval} showIntervalDropdown={showIntervalDropdown}
        isFullscreen={isFullscreen} onTimeframeClick={handleTimeframeClick}
        onIntervalClick={() => setShowIntervalDropdown(!showIntervalDropdown)}
        onIntervalChange={handleIntervalChange} onFullscreenToggle={toggleFullscreen} />

      <div ref={chartMainAreaRef} className={`${styles.chartMainArea} ${isPanning ? styles.panning : styles.idle}`}
        onMouseDown={handleMouseDown} onMouseMove={(e) => handleMouseMove(e, e.currentTarget.getBoundingClientRect())}
        onMouseUp={handleMouseUp} onMouseLeave={handleMouseLeave} onTouchStart={handleTouchStart}
        onTouchMove={(e) => handleTouchMove(e, e.currentTarget.getBoundingClientRect())} onTouchEnd={handleTouchEnd}>
        <MainChart data={data} visibleRange={visibleRange} priceScale={priceScale} timeScale={timeScale}
          stopLoss={stopLoss} entryPoint={entryPoint} targets={targets} dataTimeframe={dataTimeframe}
          displayTimeframe={timeframe} onOverlayTagsUpdate={setOverlayTags} mousePos={mousePos} isPanning={isPanning} showFvg={showFvg} onFvgCountChange={onFvgCountChange} onVisibleBarCountChange={onVisibleBarCountChange} priceOffset={priceOffset} />
        <div className={tagStyles.yAxisTagsContainer}>
          {overlayTags.map((tag, idx) => (
            <div key={idx} className={`${tagStyles.yAxisTag} ${tagStyles[`yAxisTag--${tag.kind}`]}`} style={{ top: `${tag.y - 11}px` }}>
              <span className={tagStyles.yAxisTagLabel}>{tag.label}</span>
            </div>
          ))}
        </div>
        <YAxisGutter priceScale={priceScale} onPriceScaleChange={setPriceScale} />
        <XAxisGutter timeScale={timeScale} onTimeScaleChange={setTimeScale} />
      </div>

      <div className={styles.bottomInfoBar}>
        <span>{currentTime.toLocaleTimeString('en-US', { hour12: false, timeZone: 'America/New_York' })} (ET)</span>
        <span className={styles.timeSeparator}>|</span>
        <span>{currentTime.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'America/New_York' })}</span>
      </div>
    </div>
  )
}
