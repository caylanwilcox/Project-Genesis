'use client'

import React, { useRef, useState, useCallback, useEffect } from 'react'
import styles from './ProfessionalChart.module.css'
import tagStyles from './ChartYAxisTags.module.css'
import { ProfessionalChartProps, TIMEFRAME_CONFIGS } from './ProfessionalChart/types'
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
  onTimeframeChange
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartMainAreaRef = useRef<HTMLDivElement>(null)
  const [timeframe, setTimeframe] = useState('1D')
  const [interval, setInterval] = useState('15 min')
  const [showIntervalDropdown, setShowIntervalDropdown] = useState(false)
  const [panOffset, setPanOffset] = useState(-15) // Start with 15% white space on right
  const [dataTimeframe, setDataTimeframe] = useState('15m')
  const [overlayTags, setOverlayTags] = useState<PriceTag[]>([])

  const { data } = useChartData(externalData, currentPrice, onDataUpdate)
  const { priceScale, setPriceScale, timeScale, setTimeScale } = useChartScaling()
  const visibleRange = useVisibleRange(data, panOffset, timeScale)
  const { isPanning, mousePos, handleMouseDown, handleMouseMove, handleMouseUp, handleMouseLeave, handleTouchStart, handleTouchMove, handleTouchEnd } = useChartInteraction(data, panOffset, setPanOffset, timeScale, setTimeScale)
  const { isFullscreen, toggleFullscreen } = useFullscreen(chartContainerRef)
  const currentTime = useCurrentTime()

  // Add wheel event listener with passive: false to allow preventDefault
  useEffect(() => {
    const chartArea = chartMainAreaRef.current
    if (!chartArea) return

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault()
      e.stopPropagation()

      // Use multiplicative zoom for smooth, natural scaling
      // Negative deltaY = zoom in (increase scale)
      // Positive deltaY = zoom out (decrease scale)
      const zoomFactor = Math.pow(1.001, -e.deltaY)

      setTimeScale((prev) => {
        const newScale = prev * zoomFactor
        return Math.max(0.2, Math.min(5, newScale))
      })
    }

    chartArea.addEventListener('wheel', handleWheel, { passive: false })

    return () => {
      chartArea.removeEventListener('wheel', handleWheel)
    }
  }, [setTimeScale])

  const handleTimeframeClick = useCallback((tf: string) => {
    setTimeframe(tf); setPriceScale(1.0); setTimeScale(1.0); setPanOffset(-15) // Reset zoom and whitespace
    const mapped = TIMEFRAME_CONFIGS.dataTimeframeMap[tf]
    const intervalMapped = TIMEFRAME_CONFIGS.intervalDisplayMap[tf]
    if (intervalMapped) setInterval(intervalMapped)
    if (mapped) { setDataTimeframe(mapped); onTimeframeChange?.(mapped, tf) }
  }, [onTimeframeChange, setPriceScale, setTimeScale])

  const handleIntervalChange = useCallback((newInterval: string) => {
    setInterval(newInterval); setShowIntervalDropdown(false); setPanOffset(-15) // Reset whitespace
    const mapped = TIMEFRAME_CONFIGS.intervalToTimeframeMap[newInterval]
    if (mapped) { setDataTimeframe(mapped); onTimeframeChange?.(mapped, 'Custom') }
  }, [onTimeframeChange])

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
          onOverlayTagsUpdate={setOverlayTags} mousePos={mousePos} isPanning={isPanning} />
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
