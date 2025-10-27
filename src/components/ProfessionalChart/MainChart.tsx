'use client'

import React, { useEffect, useRef, useState } from 'react'
import { CandleData } from './types'
import { setupCanvas, clearCanvas, calculatePadding, calculatePriceRange, formatVolume } from './canvasRendering'
import { drawPriceGrid, drawTimeGrid } from './gridDrawing'
import { drawCandles, drawVolumeBars } from './candleDrawing'
import { drawPriceLines, drawCurrentPriceLine, PriceTag } from './priceLines'
import { Crosshair } from './Crosshair'

interface MainChartProps {
  data: CandleData[]
  visibleRange: { start: number; end: number }
  priceScale: number
  timeScale: number
  stopLoss?: number
  entryPoint?: number
  targets?: number[]
  dataTimeframe: string
  onOverlayTagsUpdate: (tags: PriceTag[]) => void
  mousePos: { x: number; y: number } | null
  isPanning: boolean
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
  onOverlayTagsUpdate,
  mousePos,
  isPanning,
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

    const { minPrice, maxPrice, priceRange } = calculatePriceRange(visibleData, priceScale)

    // baseWidth should always be based on zoom level, not data availability
    // This allows the chart to show empty space when zoomed out beyond available data
    const baseCandlesInView = 100
    const baseWidth = baseCandlesInView / timeScale

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
    drawTimeGrid(ctx, volCtx, rect, volRect, padding, chartWidth, chartHeight, volChartHeight, visibleData, dataTimeframe, isNarrow)

    drawCandles(ctx, visibleData, padding, chartWidth, chartHeight, minPrice, maxPrice, priceRange, baseWidth)

    const volumes = visibleData.map(d => d.volume).filter(v => v > 0)
    const maxVolume = volumes.length > 0 ? Math.max(...volumes) : 0
    const candleWidth = chartWidth / baseWidth

    drawVolumeBars(volCtx, visibleData, candleWidth, volChartHeight, maxVolume, padding)

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
  }, [data, visibleRange, priceScale, timeScale, stopLoss, entryPoint, targets, dataTimeframe, onOverlayTagsUpdate])

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
