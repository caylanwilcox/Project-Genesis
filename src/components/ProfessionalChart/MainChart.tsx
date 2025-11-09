'use client'

import React, { useEffect, useRef, useState } from 'react'
import { CandleData } from './types'
import { setupCanvas, clearCanvas, calculatePadding, calculatePriceRange, formatVolume } from './canvasRendering'
import { drawPriceGrid, drawTimeGrid } from './gridDrawing'
import { drawCandles, drawVolumeBars } from './candleDrawing'
import { drawPriceLines, drawCurrentPriceLine, PriceTag } from './priceLines'
import { Crosshair } from './Crosshair'
import { detectFvgPatterns, drawFvgPatterns } from './fvgDrawing'
import { drawMarketHoursBackground } from './marketHoursDrawing'

interface MainChartProps {
  data: CandleData[]
  visibleRange: { start: number; end: number }
  priceScale: number
  timeScale: number
  stopLoss?: number
  entryPoint?: number
  targets?: number[]
  dataTimeframe: string
  displayTimeframe?: string
  onOverlayTagsUpdate: (tags: PriceTag[]) => void
  mousePos: { x: number; y: number } | null
  isPanning: boolean
  showFvg?: boolean
  onFvgCountChange?: (count: number) => void
  onVisibleBarCountChange?: (count: number, visibleData: CandleData[]) => void
  priceOffset?: number
}

export const MainChart: React.FC<MainChartProps> = ({
  data,
  visibleRange,
  priceScale,
  timeScale,
  stopLoss,
  entryPoint,
  targets = [],
  dataTimeframe,
  displayTimeframe,
  onOverlayTagsUpdate,
  mousePos,
  isPanning,
  showFvg = false,
  onFvgCountChange,
  onVisibleBarCountChange,
  priceOffset = 0,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const volumeCanvasRef = useRef<HTMLCanvasElement>(null)
  const [chartDimensions, setChartDimensions] = useState({
    chartWidth: 0,
    chartHeight: 0,
    padding: { left: 0, top: 0, right: 0, bottom: 0 },
    minPrice: 0,
    maxPrice: 0,
    priceRange: 0,
    baseWidth: 100,
    visibleData: [] as CandleData[]
  })

  useEffect(() => {
    const canvas = canvasRef.current
    const volumeCanvas = volumeCanvasRef.current
    if (!canvas || !volumeCanvas || data.length === 0) return

    const ctx = setupCanvas(canvas)
    const volCtx = setupCanvas(volumeCanvas)
    if (!ctx || !volCtx) return

    const parentWidth = canvas.parentElement?.clientWidth || 800
    const rect = canvas.getBoundingClientRect()
    const volRect = volumeCanvas.getBoundingClientRect()

    clearCanvas(ctx, rect.width, rect.height)
    clearCanvas(volCtx, volRect.width, volRect.height)

    const { padding, gutter, isNarrow } = calculatePadding(parentWidth)
    const chartWidth = rect.width - padding.left - padding.right
    const chartHeight = rect.height - padding.top - padding.bottom
    const volChartHeight = volRect.height - 5

    const visibleData = data.slice(visibleRange.start, visibleRange.end)
    if (visibleData.length === 0) return

    // Notify parent of visible bar count and data
    if (onVisibleBarCountChange) {
      onVisibleBarCountChange(visibleData.length, visibleData)
    }

    const { minPrice, maxPrice, priceRange } = calculatePriceRange(visibleData, priceScale, priceOffset)

    // POLICY: Detect auto-fit mode (timeScale=1.0) and use actual data length
    // Manual mode: use fixed base of 100 candles with zoom
    const isAutoFit = timeScale === 1.0 && visibleRange.start === 0 && visibleRange.end === data.length
    const baseCandlesInView = 100
    const baseWidth = isAutoFit ? visibleData.length : baseCandlesInView / timeScale

    setChartDimensions({
      chartWidth,
      chartHeight,
      padding,
      minPrice,
      maxPrice,
      priceRange,
      baseWidth,
      visibleData
    })

    drawPriceGrid({ ctx, rect, padding, chartWidth, chartHeight, minPrice, maxPrice, priceRange, isNarrow, gutter })
    drawTimeGrid(ctx, volCtx, rect, volRect, padding, chartWidth, chartHeight, volChartHeight, visibleData, dataTimeframe, displayTimeframe, isNarrow)

    // Draw market hours background (6am-1pm) for minute/hour timeframes
    if (dataTimeframe === '1m' || dataTimeframe === '5m' || dataTimeframe === '15m' || dataTimeframe === '30m' || dataTimeframe === '1h') {
      drawMarketHoursBackground(ctx, visibleData, padding, chartWidth, chartHeight, baseWidth)
    }

    drawCandles(ctx, visibleData, padding, chartWidth, chartHeight, minPrice, maxPrice, priceRange)

    const volumes = visibleData.map(d => d.volume).filter(v => v > 0)
    const maxVolume = volumes.length > 0 ? Math.max(...volumes) : 0
    const candleWidth = chartWidth / visibleData.length

    drawVolumeBars(volCtx, visibleData, candleWidth, volChartHeight, maxVolume, padding, chartWidth)

    // Draw FVG patterns if enabled
    if (showFvg && visibleData.length >= 3) {
      // Adjust gap thresholds based on timeframe
      const options = dataTimeframe === '1m' || dataTimeframe === '5m'
        ? { minGapPct: 0.01, maxGapPct: 1.0, recentOnly: false } // Lower thresholds for minute charts
        : { minGapPct: 0.1, maxGapPct: 5.0, recentOnly: false }  // Standard thresholds for hourly+ charts

      const fvgPatterns = detectFvgPatterns(visibleData, options)
      drawFvgPatterns(ctx, fvgPatterns, visibleData, padding, chartWidth, chartHeight, minPrice, maxPrice, priceRange, baseWidth, visibleRange.start)

      // Notify parent of FVG count
      if (onFvgCountChange) {
        onFvgCountChange(fvgPatterns.length)
      }
    } else if (!showFvg && onFvgCountChange) {
      onFvgCountChange(0)
    }

    const tags = drawPriceLines(ctx, rect, padding, chartHeight, minPrice, maxPrice, priceRange, stopLoss, entryPoint, targets)

    const isCurrentPriceVisible = visibleRange.end >= data.length
    const lastCandle = data[data.length - 1]
    const currentTag = drawCurrentPriceLine(ctx, rect, padding, chartHeight, minPrice, maxPrice, priceRange, lastCandle.close, isCurrentPriceVisible)

    if (currentTag) tags.push(currentTag)

    volCtx.fillStyle = '#6b7280'
    volCtx.font = isNarrow ? '7px monospace' : '9px monospace'
    volCtx.textAlign = 'center'
    const volLabelX = volRect.width - (gutter / 2)
    volCtx.fillText('Vol', volLabelX, volChartHeight - 18)
    volCtx.fillText(formatVolume(maxVolume), volLabelX, volChartHeight - 6)

    onOverlayTagsUpdate(tags)
  }, [data, visibleRange, priceScale, timeScale, stopLoss, entryPoint, targets, dataTimeframe, onOverlayTagsUpdate, showFvg, onVisibleBarCountChange, priceOffset])

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <canvas ref={canvasRef} style={{ width: '100%', height: 'calc(100% - 80px)' }} />
      <canvas ref={volumeCanvasRef} style={{ width: '100%', height: '80px' }} />
      <Crosshair
        mousePos={mousePos}
        visibleData={chartDimensions.visibleData}
        chartWidth={chartDimensions.chartWidth}
        chartHeight={chartDimensions.chartHeight}
        padding={chartDimensions.padding}
        minPrice={chartDimensions.minPrice}
        maxPrice={chartDimensions.maxPrice}
        priceRange={chartDimensions.priceRange}
        baseWidth={chartDimensions.baseWidth}
        isPanning={isPanning}
      />
    </div>
  )
}
