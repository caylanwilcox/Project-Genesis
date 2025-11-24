import React, { useEffect, useState } from 'react'
import interactionStyles from '../../../ChartInteraction.module.css'
import { MainChart } from '../../MainChart'
import { ZoomIndicators } from '../../ZoomIndicators'
import { PriceTagsOverlay } from '../overlays/PriceTagsOverlay'
import { LoadingOverlay } from '../overlays/LoadingOverlay'
import { ZoomControls } from '../../interactions/zoom/ZoomControls'
import { useHeightResize } from '../../interactions/gestures/useHeightResize'
import { useChartInteractionHandlers } from './ChartInteractionHandlers'
import type { CandleData } from '../../types'
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
  onFvgCountChange?: (count: number) => void
  onVisibleBarCountChange?: (count: number, visibleData: CandleData[]) => void
  priceOffset: number
  isLoadingMore: boolean
  customHeight: number | null
  onHeightChange: (height: number) => void
  className: string
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
  onFvgCountChange,
  onVisibleBarCountChange,
  priceOffset,
  isLoadingMore,
  customHeight,
  onHeightChange,
  className,
  interaction,
  scaling,
}) => {
  const { handleResizeStart } = useHeightResize({ containerRef, onHeightChange })
  const handlers = useChartInteractionHandlers({ chartAreaRef, interaction })
  const [chartSize, setChartSize] = useState({ width: 0, height: 0 })

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
      setChartSize(prev => {
        if (Math.abs(prev.width - width) < 0.5 && Math.abs(prev.height - height) < 0.5) {
          return prev
        }
        return { width, height }
      })
    })

    observer.observe(node)
    return () => observer.disconnect()
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
        onFvgCountChange={onFvgCountChange}
        onVisibleBarCountChange={onVisibleBarCountChange}
        priceOffset={priceOffset}
        chartAreaSize={chartSize}
      />

      <PriceTagsOverlay tags={overlayTags} />

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
        onTimeZoomOut={() => scaling.setTimeScale((p: number) => Math.max(0.2, p / 1.2))}
        onTimeReset={() => scaling.setTimeScale(1)}
      />

      <LoadingOverlay show={isLoadingMore && chartData.length > 0} />
    </div>
  )
}
