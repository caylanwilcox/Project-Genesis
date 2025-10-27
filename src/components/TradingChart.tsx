'use client'

import React, { useEffect, useRef, useState } from 'react'
import { TradingChartHeader } from './TradingChart/TradingChartHeader'
import { TradingChartFooter } from './TradingChart/TradingChartFooter'
import { useChartData } from './TradingChart/useChartData'
import { drawGrid, drawPriceLabels, drawAreaChart } from './TradingChart/canvasDrawing'
import { drawPriceLine, drawCurrentPriceTag } from './TradingChart/priceLineDrawing'

interface TradingChartProps {
  symbol: string
  currentPrice?: number
  stopLoss?: number
  targets?: number[]
  entryPoint?: number
}

export const TradingChart: React.FC<TradingChartProps> = ({
  symbol,
  currentPrice = 445.20,
  stopLoss,
  targets = [],
  entryPoint
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [timeframe, setTimeframe] = useState('2h')
  const [chartType, setChartType] = useState<'price' | 'depth'>('price')
  const data = useChartData(currentPrice)
  const timeframes = ['6M', '3M', '1M', '5D', '1D', '4H', '1H', '2h']

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || data.length === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    canvas.style.width = rect.width + 'px'
    canvas.style.height = rect.height + 'px'
    ctx.scale(dpr, dpr)

    ctx.fillStyle = '#0B0E11'
    ctx.fillRect(0, 0, rect.width, rect.height)

    const padding = { top: 20, right: 60, bottom: 30, left: 10 }
    const chartWidth = rect.width - padding.left - padding.right
    const chartHeight = rect.height - padding.top - padding.bottom

    const minPrice = Math.min(...data.map(d => d.value), stopLoss || Infinity) * 0.998
    const maxPrice = Math.max(...data.map(d => d.value), ...(targets || [])) * 1.002
    const priceRange = maxPrice - minPrice

    drawGrid(ctx, rect, padding, chartWidth, chartHeight)
    drawPriceLabels(ctx, rect, padding, chartHeight, minPrice, maxPrice, priceRange)
    drawAreaChart(ctx, data, rect, padding, chartWidth, chartHeight, maxPrice, priceRange)

    if (stopLoss) drawPriceLine(ctx, rect, padding, chartHeight, stopLoss, '#ef4444', 'STOP', maxPrice, priceRange, true, 0.8)
    if (entryPoint) drawPriceLine(ctx, rect, padding, chartHeight, entryPoint, '#06b6d4', 'ENTRY', maxPrice, priceRange, false, 0.9)

    targets.forEach((target, i) => {
      const opacity = i === 2 ? 1 : 0.7 + (i * 0.1)
      drawPriceLine(ctx, rect, padding, chartHeight, target, '#10b981', `T${i + 1}`, maxPrice, priceRange, false, opacity)
    })

    const lastPrice = data[data.length - 1].value
    drawCurrentPriceTag(ctx, rect, padding, chartHeight, lastPrice, maxPrice, priceRange)
  }, [data, stopLoss, entryPoint, targets])

  return (
    <div className="trading-card rounded-xl transition-all duration-300 h-full flex flex-col">
      <TradingChartHeader
        symbol={symbol}
        chartType={chartType}
        timeframe={timeframe}
        timeframes={timeframes}
        onChartTypeChange={setChartType}
        onTimeframeChange={setTimeframe}
      />
      <div className="flex-grow relative min-h-0">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
          style={{ display: 'block' }}
        />
      </div>
      <TradingChartFooter currentPrice={currentPrice} symbol={symbol} />
    </div>
  )
}