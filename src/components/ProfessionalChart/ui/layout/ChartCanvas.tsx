import React, { useEffect, useState, useRef } from 'react'
import interactionStyles from '../../../ChartInteraction.module.css'
import { MainChart } from '../../MainChart'
import { ZoomIndicators } from '../../ZoomIndicators'
import { PriceTagsOverlay } from '../overlays/PriceTagsOverlay'
import { LoadingOverlay } from '../overlays/LoadingOverlay'
import { ZoomControls } from '../../interactions/zoom/ZoomControls'
import { MLOverlay } from '../../MLOverlay'
import { useHeightResize } from '../../interactions/gestures/useHeightResize'
import { useChartInteractionHandlers } from './ChartInteractionHandlers'
import type { CandleData, V6Prediction } from '../../types'
import type { PriceTag } from '../../priceLines'
import type { Timeframe } from '@/types/polygon'

interface ChartCanvasProps {
  chartAreaRef: React.RefObject<HTMLDivElement>
  containerRef: React.RefObject<HTMLDivElement>
  chartData: CandleData[]
  overlayTags: PriceTag[]
  visibleRange: { start: number; end: number }
  priceScale: number
  timeScale: number
  stopLoss?: number
  entryPoint?: number
  targets?: number[]
  dataTimeframe: Timeframe
  displayTimeframe: string
  onOverlayTagsUpdate: (tags: PriceTag[]) => void
  mousePos: { x: number; y: number } | null
  isPanning: boolean
  showFvg: boolean
  fvgPercentage?: number
  onFvgCountChange?: (count: number) => void
  onVisibleBarCountChange?: (count: number, visibleData: CandleData[]) => void
  priceOffset: number
  isLoadingMore: boolean
  customHeight: number | null
  onHeightChange: (height: number) => void
  className: string
  ticker?: string  // For ML predictions
  interaction: {
    handleMouseDown: (e: React.MouseEvent<HTMLDivElement>) => void
    handleMouseMove: (e: React.MouseEvent<HTMLDivElement>, rect: DOMRect) => void
    handleMouseUp: () => void
    handleMouseLeave: () => void
    handleTouchStart: (e: React.TouchEvent<HTMLDivElement>) => void
    handleTouchMove: (e: React.TouchEvent<HTMLDivElement>, rect: DOMRect) => void
    handleTouchEnd: () => void
    handleWheel: (e: React.WheelEvent<HTMLDivElement>, rect: DOMRect) => void
  }
  scaling: {
    priceScale: number
    timeScale: number
    setPriceScale: (scale: number | ((prev: number) => number)) => void
    setTimeScale: (scale: number | ((prev: number) => number)) => void
    handleScaleMouseDown: (e: React.MouseEvent) => void
    handleScaleMouseMove: (e: React.MouseEvent) => void
    handleScaleDoubleClick: () => void
    handleTimeScaleMouseDown: (e: React.MouseEvent) => void
    handleTimeScaleMouseMove: (e: React.MouseEvent) => void
    handleTimeScaleDoubleClick: () => void
  }
  // ML Overlay props
  v6Prediction?: V6Prediction
  showMLOverlay?: boolean
  showSessionLines?: boolean
  price11am?: number
  // Range levels for balanced/choppy market structure
  rangeHigh?: number
  rangeLow?: number
  rangeMid?: number
  // Retest levels - where price will retest on range failure
  retestHigh?: number
  retestLow?: number
}

/**
 * Main chart canvas area with all overlays and interactions
 */
export const ChartCanvas: React.FC<ChartCanvasProps> = ({
  chartAreaRef,
  containerRef,
  chartData,
  overlayTags,
  visibleRange,
  priceScale,
  timeScale,
  stopLoss,
  entryPoint,
  targets,
  dataTimeframe,
  displayTimeframe,
  onOverlayTagsUpdate,
  mousePos,
  isPanning,
  showFvg,
  fvgPercentage,
  onFvgCountChange,
  onVisibleBarCountChange,
  priceOffset,
  isLoadingMore,
  customHeight,
  onHeightChange,
  className,
  ticker,
  interaction,
  scaling,
  // ML Overlay props
  v6Prediction,
  showMLOverlay = false,
  showSessionLines = false,
  price11am,
  // Range levels
  rangeHigh,
  rangeLow,
  rangeMid,
  // Retest levels
  retestHigh,
  retestLow,
}) => {
  const { handleResizeStart } = useHeightResize({ containerRef, onHeightChange })
  const handlers = useChartInteractionHandlers({ chartAreaRef, interaction })
  const [chartSize, setChartSize] = useState({ width: 0, height: 0 })
  const resizeTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    const node = chartAreaRef.current
    if (!node || typeof ResizeObserver === 'undefined') {
      if (node) {
        const rect = node.getBoundingClientRect()
        setChartSize({ width: rect.width, height: rect.height })
      }
      return
    }

    const observer = new ResizeObserver(entries => {
      const entry = entries[0]
      if (!entry) return
      const { width, height } = entry.contentRect

      // Debounce resize updates to prevent render thrashing during window resize
      if (resizeTimeoutRef.current) {
        clearTimeout(resizeTimeoutRef.current)
      }

      resizeTimeoutRef.current = setTimeout(() => {
        setChartSize(prev => {
          if (Math.abs(prev.width - width) < 0.5 && Math.abs(prev.height - height) < 0.5) {
            return prev
          }
          return { width, height }
        })
      }, 100) // 100ms debounce
    })

    observer.observe(node)
    return () => {
      observer.disconnect()
      if (resizeTimeoutRef.current) {
        clearTimeout(resizeTimeoutRef.current)
      }
    }
  }, [chartAreaRef])

  return (
    <div
      ref={chartAreaRef}
      className={className}
      style={customHeight ? { minHeight: customHeight } : undefined}
      onMouseDown={handlers.handleMouseDown}
      onMouseMove={handlers.handleMouseMove}
      onMouseUp={handlers.handleMouseUp}
      onMouseLeave={handlers.handleMouseLeave}
      onTouchStart={handlers.handleTouchStart}
      onTouchMove={handlers.handleTouchMove}
      onTouchEnd={handlers.handleTouchEnd}
      onTouchCancel={handlers.handleTouchEnd}
      onWheel={handlers.handleWheel}
    >
      <MainChart
        data={chartData}
        visibleRange={visibleRange}
        priceScale={priceScale}
        timeScale={timeScale}
        stopLoss={stopLoss}
        entryPoint={entryPoint}
        targets={targets || []}
        dataTimeframe={dataTimeframe}
        displayTimeframe={displayTimeframe}
        onOverlayTagsUpdate={onOverlayTagsUpdate}
        mousePos={mousePos}
        isPanning={isPanning}
        showFvg={showFvg}
        fvgPercentage={fvgPercentage}
        onFvgCountChange={onFvgCountChange}
        onVisibleBarCountChange={onVisibleBarCountChange}
        priceOffset={priceOffset}
        chartAreaSize={chartSize}
        ticker={ticker}
        enableMLPredictions={showFvg}
        rangeHigh={rangeHigh}
        rangeLow={rangeLow}
        rangeMid={rangeMid}
        retestHigh={retestHigh}
        retestLow={retestLow}
      />

      <PriceTagsOverlay tags={overlayTags} />

      {/* ML Overlay - V6 prediction banner */}
      <MLOverlay
        v6Prediction={v6Prediction}
        showMLOverlay={showMLOverlay}
        showSessionLines={showSessionLines}
        price11am={price11am}
        chartWidth={chartSize.width}
        chartHeight={chartSize.height}
        padding={{ left: 10, top: 10, right: 80, bottom: 20 }}
      />

      <ZoomControls
        priceScale={priceScale}
        timeScale={timeScale}
        onPriceScaleMouseDown={scaling.handleScaleMouseDown}
        onPriceScaleMouseMove={scaling.handleScaleMouseMove}
        onPriceScaleDoubleClick={scaling.handleScaleDoubleClick}
        onTimeScaleMouseDown={scaling.handleTimeScaleMouseDown}
        onTimeScaleMouseMove={scaling.handleTimeScaleMouseMove}
        onTimeScaleDoubleClick={scaling.handleTimeScaleDoubleClick}
      />

      <div
        className={interactionStyles.heightResizeHandle}
        onMouseDown={handleResizeStart}
        title="Drag to resize chart height"
      />

      <ZoomIndicators
        timeScale={timeScale}
        onTimeZoomIn={() => scaling.setTimeScale((p: number) => Math.min(5, p * 1.2))}
        onTimeZoomOut={() => scaling.setTimeScale((p: number) => Math.max(0.05, p / 1.2))}
        onTimeReset={() => scaling.setTimeScale(1)}
      />

      <LoadingOverlay show={isLoadingMore && chartData.length > 0} />
    </div>
  )
}
