'use client'

import React, { useEffect, useRef, useState } from 'react'
import { CandleData } from './types'
import { setupCanvas, clearCanvas, calculatePadding, calculatePriceRange, formatVolume } from './canvasRendering'
import { drawPriceGrid, drawTimeGrid } from './gridDrawing'
import { drawCandles, drawVolumeBars } from './candleDrawing'
import { drawPriceLines, drawCurrentPriceLine, drawLowPriceMarker, drawHighPriceMarker, PriceTag } from './priceLines'
import { Crosshair } from './Crosshair'
import { detectFvgPatterns, drawFvgPatterns, findClickedFvg, FvgPattern } from './fvgDrawing'
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
  chartAreaSize?: { width: number; height: number }
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
  chartAreaSize,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const volumeCanvasRef = useRef<HTMLCanvasElement>(null)
  const [fvgPatterns, setFvgPatterns] = useState<FvgPattern[]>([])
  const drawnPatternsRef = useRef<FvgPattern[]>([]) // Store the patterns with dot positions after drawing
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

    const parentWidth = chartAreaSize?.width || canvas.parentElement?.clientWidth || 800
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

    // POLICY: Always use visibleData.length as baseWidth for proper alignment
    // This ensures candles fill the chart width and don't leave empty space
    const baseWidth = visibleData.length

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

    drawCandles(ctx, visibleData, padding, chartWidth, chartHeight, minPrice, maxPrice, priceRange, baseWidth)

    const volumes = visibleData.map(d => d.volume).filter(v => v > 0)
    const maxVolume = volumes.length > 0 ? Math.max(...volumes) : 0

    drawVolumeBars(volCtx, visibleData, volChartHeight, maxVolume, padding, chartWidth, baseWidth)

    // Draw FVG patterns if enabled
    if (showFvg && data.length >= 3) {
      // Adjust gap thresholds based on timeframe
      const isMinute = dataTimeframe === '1m' || dataTimeframe === '5m' || dataTimeframe === '15m'
      const options = isMinute
        ? { minGapPct: 0.2, maxGapPct: 5.0, recentOnly: false }
        : { minGapPct: 0.3, maxGapPct: 5.0, recentOnly: false }

      // Detect patterns on FULL dataset so overlays persist across zoom/pan
      const detectedPatterns = detectFvgPatterns(data, options)

      // Merge with existing state to preserve expanded status
      const mergedPatterns = detectedPatterns.map(detected => {
        const existing = fvgPatterns.find(p => p.id === detected.id)
        return existing ? { ...detected, expanded: existing.expanded } : detected
      })

      // Update state if patterns changed
      if (JSON.stringify(mergedPatterns.map(p => p.id)) !== JSON.stringify(fvgPatterns.map(p => p.id))) {
        setFvgPatterns(mergedPatterns)
      }

      drawFvgPatterns(ctx, mergedPatterns, visibleData, padding, chartWidth, chartHeight, minPrice, maxPrice, priceRange, baseWidth, visibleRange.start)

      // Store the drawn patterns with their dot positions for click detection
      drawnPatternsRef.current = mergedPatterns

      // Notify parent of FVG count
      if (onFvgCountChange) {
        // Count patterns that intersect the visible window (allowing forward extension)
        const gapExtendCandles = 30
        const start = visibleRange.start - gapExtendCandles
        const end = visibleRange.end
        const visibleCount = mergedPatterns.filter(p => p.startIndex >= start && p.startIndex < end).length
        onFvgCountChange(visibleCount)
      }
    } else if (!showFvg && onFvgCountChange) {
      onFvgCountChange(0)
    }

    const tags = drawPriceLines(ctx, rect, padding, chartHeight, minPrice, maxPrice, priceRange, stopLoss, entryPoint, targets)

    const isCurrentPriceVisible = visibleRange.end >= data.length
    const lastCandle = data[data.length - 1]
    const currentTag = drawCurrentPriceLine(ctx, rect, padding, chartHeight, minPrice, maxPrice, priceRange, lastCandle.close, isCurrentPriceVisible)

    if (currentTag) tags.push(currentTag)

    // Draw low price marker (like Webull) - find the candle with lowest low
    let lowestCandleIndex = 0
    let lowestPrice = visibleData[0].low
    visibleData.forEach((candle, index) => {
      if (candle.low < lowestPrice) {
        lowestPrice = candle.low
        lowestCandleIndex = index
      }
    })
    drawLowPriceMarker(ctx, rect, padding, chartHeight, minPrice, maxPrice, priceRange, lowestPrice, chartWidth, baseWidth, lowestCandleIndex)

    // Draw high price marker (like Webull) - find the candle with highest high
    let highestCandleIndex = 0
    let highestPrice = visibleData[0].high
    visibleData.forEach((candle, index) => {
      if (candle.high > highestPrice) {
        highestPrice = candle.high
        highestCandleIndex = index
      }
    })
    drawHighPriceMarker(ctx, rect, padding, chartHeight, minPrice, maxPrice, priceRange, highestPrice, chartWidth, baseWidth, highestCandleIndex)

    volCtx.fillStyle = '#6b7280'
    volCtx.font = isNarrow ? '7px monospace' : '9px monospace'
    volCtx.textAlign = 'center'
    const volLabelX = volRect.width - (gutter / 2)
    volCtx.fillText('Vol', volLabelX, volChartHeight - 18)
    volCtx.fillText(formatVolume(maxVolume), volLabelX, volChartHeight - 6)

    onOverlayTagsUpdate(tags)
  }, [
    data,
    visibleRange,
    priceScale,
    timeScale,
    stopLoss,
    entryPoint,
    targets,
    dataTimeframe,
    onOverlayTagsUpdate,
    showFvg,
    priceOffset,
    displayTimeframe,
    fvgPatterns,
    chartAreaSize?.width,
    chartAreaSize?.height,
  ])

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!showFvg || isPanning) return

    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const clickY = e.clientY - rect.top

    console.log('[FVG Click] Click at:', { clickX, clickY })
    console.log('[FVG Click] Drawn patterns:', drawnPatternsRef.current.map(p => ({
      id: p.id,
      expanded: p.expanded,
      dotX: (p as any).dotX,
      dotY: (p as any).dotY,
      dotRadius: (p as any).dotRadius
    })))

    // Find if any FVG dot was clicked (use drawn patterns which have dot positions)
    const clickedFvg = findClickedFvg(drawnPatternsRef.current, clickX, clickY)
    console.log('[FVG Click] Clicked FVG:', clickedFvg?.id)

    if (clickedFvg) {
      // Toggle the expanded state
      setFvgPatterns(patterns =>
        patterns.map(p =>
          p.id === clickedFvg.id ? { ...p, expanded: !p.expanded } : p
        )
      )
      console.log('[FVG Click] Toggled expanded state for:', clickedFvg.id)
    }
  }

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: 'calc(100% - 80px)', cursor: showFvg ? 'pointer' : 'default' }}
        onClick={handleCanvasClick}
      />
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
