'use client'

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { Activity, TrendingUp, Maximize2, Settings, ChevronDown } from 'lucide-react'
import { useRouter } from 'next/navigation'

interface ProfessionalChartProps {
  symbol: string
  currentPrice?: number
  stopLoss?: number
  targets?: number[]
  entryPoint?: number
}

interface CandleData {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export const ProfessionalChart: React.FC<ProfessionalChartProps> = ({
  symbol,
  currentPrice = 445.20,
  stopLoss,
  targets = [],
  entryPoint
}) => {
  const router = useRouter()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const volumeCanvasRef = useRef<HTMLCanvasElement>(null)
  const crosshairCanvasRef = useRef<HTMLCanvasElement>(null)
  const [timeframe, setTimeframe] = useState('1H')
  const [chartType, setChartType] = useState<'candles' | 'line'>('candles')
  const [data, setData] = useState<CandleData[]>([])
  const [hoveredCandle, setHoveredCandle] = useState<CandleData | null>(null)
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null)

  // Generate realistic candlestick data
  const generateCandleData = useCallback(() => {
    const candles: CandleData[] = []
    const basePrice = currentPrice
    const numCandles = 50
    const now = Date.now()

    for (let i = 0; i < numCandles; i++) {
      const time = now - (numCandles - i) * 3600000 // 1 hour candles
      const volatility = basePrice * 0.002

      // Generate OHLC values
      const open = i === 0 ? basePrice : candles[i - 1].close
      const change = (Math.random() - 0.5) * volatility
      const close = open + change
      const highExtra = Math.random() * volatility * 0.5
      const lowExtra = Math.random() * volatility * 0.5
      const high = Math.max(open, close) + highExtra
      const low = Math.min(open, close) - lowExtra
      const volume = 50000000 + Math.random() * 50000000 // 50M to 100M

      candles.push({ time, open, high, low, close, volume })
    }

    return candles
  }, [currentPrice])

  useEffect(() => {
    setData(generateCandleData())
  }, [generateCandleData])

  // Draw main chart
  useEffect(() => {
    const canvas = canvasRef.current
    const volumeCanvas = volumeCanvasRef.current
    if (!canvas || !volumeCanvas || data.length === 0) return

    const ctx = canvas.getContext('2d')
    const volCtx = volumeCanvas.getContext('2d')
    if (!ctx || !volCtx) return

    // Setup canvas
    const rect = canvas.getBoundingClientRect()
    const volRect = volumeCanvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1

    // Main chart canvas
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    canvas.style.width = rect.width + 'px'
    canvas.style.height = rect.height + 'px'
    ctx.scale(dpr, dpr)

    // Volume canvas
    volumeCanvas.width = volRect.width * dpr
    volumeCanvas.height = volRect.height * dpr
    volumeCanvas.style.width = volRect.width + 'px'
    volumeCanvas.style.height = volRect.height + 'px'
    volCtx.scale(dpr, dpr)

    // Clear canvases
    ctx.fillStyle = '#0d0e15'
    ctx.fillRect(0, 0, rect.width, rect.height)
    volCtx.fillStyle = '#0d0e15'
    volCtx.fillRect(0, 0, volRect.width, volRect.height)

    // Calculate dimensions
    const padding = { top: 10, right: 80, bottom: 20, left: 10 }
    const chartWidth = rect.width - padding.left - padding.right
    const chartHeight = rect.height - padding.top - padding.bottom
    const volChartHeight = volRect.height - 20

    // Calculate price range
    const allPrices = data.flatMap(d => [d.high, d.low])
    if (stopLoss) allPrices.push(stopLoss)
    if (entryPoint) allPrices.push(entryPoint)
    allPrices.push(...targets)

    const minPrice = Math.min(...allPrices) * 0.998
    const maxPrice = Math.max(...allPrices) * 1.002
    const priceRange = maxPrice - minPrice

    // Calculate volume range
    const maxVolume = Math.max(...data.map(d => d.volume))

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)'
    ctx.lineWidth = 1
    ctx.setLineDash([1, 1])

    // Horizontal grid lines (price levels)
    for (let i = 0; i <= 8; i++) {
      const y = padding.top + (chartHeight / 8) * i
      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(rect.width - padding.right, y)
      ctx.stroke()

      // Price labels
      const price = maxPrice - (priceRange / 8) * i
      ctx.fillStyle = '#6b7280'
      ctx.font = '11px monospace'
      ctx.textAlign = 'right'
      ctx.fillText(price.toFixed(2), rect.width - 5, y + 3)
    }

    // Vertical grid lines (time)
    for (let i = 0; i <= 6; i++) {
      const x = padding.left + (chartWidth / 6) * i
      ctx.beginPath()
      ctx.moveTo(x, padding.top)
      ctx.lineTo(x, rect.height - padding.bottom)
      ctx.stroke()

      volCtx.strokeStyle = 'rgba(255, 255, 255, 0.03)'
      volCtx.beginPath()
      volCtx.moveTo(x, 0)
      volCtx.lineTo(x, volChartHeight)
      volCtx.stroke()
    }

    ctx.setLineDash([])

    // Draw candlesticks
    const candleWidth = chartWidth / data.length
    const candleSpacing = candleWidth * 0.8

    data.forEach((candle, i) => {
      const x = padding.left + i * candleWidth + candleWidth / 2
      const isGreen = candle.close >= candle.open

      // Calculate positions
      const highY = padding.top + ((maxPrice - candle.high) / priceRange) * chartHeight
      const lowY = padding.top + ((maxPrice - candle.low) / priceRange) * chartHeight
      const openY = padding.top + ((maxPrice - candle.open) / priceRange) * chartHeight
      const closeY = padding.top + ((maxPrice - candle.close) / priceRange) * chartHeight

      // Draw high-low line (wick)
      ctx.strokeStyle = isGreen ? '#22c55e' : '#ef4444'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(x, highY)
      ctx.lineTo(x, lowY)
      ctx.stroke()

      // Draw candle body
      const bodyTop = Math.min(openY, closeY)
      const bodyHeight = Math.abs(closeY - openY) || 1

      ctx.fillStyle = isGreen ? '#22c55e' : '#ef4444'
      ctx.fillRect(x - candleSpacing / 2, bodyTop, candleSpacing, bodyHeight)

      // Draw volume bars
      const volHeight = (candle.volume / maxVolume) * volChartHeight * 0.8
      volCtx.fillStyle = isGreen ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)'
      volCtx.fillRect(x - candleSpacing / 2, volChartHeight - volHeight, candleSpacing, volHeight)
    })

    // Draw price levels
    const drawPriceLine = (price: number, color: string, label: string, dashed = false) => {
      const y = padding.top + ((maxPrice - price) / priceRange) * chartHeight

      ctx.strokeStyle = color
      ctx.lineWidth = 1
      if (dashed) {
        ctx.setLineDash([10, 5])
      }

      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(rect.width - padding.right, y)
      ctx.stroke()

      ctx.setLineDash([])

      // Price tag
      ctx.fillStyle = color
      ctx.fillRect(rect.width - padding.right + 2, y - 10, padding.right - 7, 20)
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 10px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(price.toFixed(2), rect.width - padding.right / 2 - 2, y + 3)
    }

    // Draw stop loss
    if (stopLoss) {
      drawPriceLine(stopLoss, '#ef444488', 'SL', true)
    }

    // Draw entry point
    if (entryPoint) {
      drawPriceLine(entryPoint, '#06b6d488', 'ENTRY', false)
    }

    // Draw targets
    targets.forEach((target, i) => {
      drawPriceLine(target, '#22c55e88', `T${i + 1}`, false)
    })

    // Current price line and tag
    const lastCandle = data[data.length - 1]
    const currentY = padding.top + ((maxPrice - lastCandle.close) / priceRange) * chartHeight

    // Price line
    ctx.strokeStyle = '#fbbf24'
    ctx.lineWidth = 1
    ctx.setLineDash([4, 2])
    ctx.beginPath()
    ctx.moveTo(padding.left, currentY)
    ctx.lineTo(rect.width, currentY)
    ctx.stroke()
    ctx.setLineDash([])

    // Price tag
    ctx.fillStyle = '#fbbf24'
    ctx.fillRect(rect.width - padding.right + 2, currentY - 11, padding.right - 7, 22)
    ctx.fillStyle = '#000000'
    ctx.font = 'bold 11px monospace'
    ctx.textAlign = 'center'
    ctx.fillText(lastCandle.close.toFixed(2), rect.width - padding.right / 2 - 2, currentY + 4)

    // Draw volume scale
    volCtx.fillStyle = '#6b7280'
    volCtx.font = '10px monospace'
    volCtx.textAlign = 'right'
    volCtx.fillText('Vol', volRect.width - 5, 12)
    volCtx.fillText((maxVolume / 1000000).toFixed(0) + 'M', volRect.width - 5, 25)

  }, [data, stopLoss, entryPoint, targets])

  // Draw crosshair
  useEffect(() => {
    const canvas = crosshairCanvasRef.current
    if (!canvas || !mousePos) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1

    // Setup canvas
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    canvas.style.width = rect.width + 'px'
    canvas.style.height = rect.height + 'px'
    ctx.scale(dpr, dpr)

    // Clear canvas
    ctx.clearRect(0, 0, rect.width, rect.height)

    if (!mousePos) return

    const padding = { top: 10, right: 80, bottom: 20, left: 10 }
    const chartHeight = rect.height - padding.top - padding.bottom - 80 // Subtract volume chart height

    // Don't draw crosshair outside chart area
    if (mousePos.x < padding.left || mousePos.x > rect.width - padding.right ||
        mousePos.y < padding.top || mousePos.y > padding.top + chartHeight) {
      return
    }

    // Draw crosshair lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)'
    ctx.lineWidth = 0.5
    ctx.setLineDash([3, 3])

    // Vertical line
    ctx.beginPath()
    ctx.moveTo(mousePos.x, padding.top)
    ctx.lineTo(mousePos.x, rect.height - 100) // Stop at volume chart
    ctx.stroke()

    // Horizontal line
    ctx.beginPath()
    ctx.moveTo(padding.left, mousePos.y)
    ctx.lineTo(rect.width - padding.right, mousePos.y)
    ctx.stroke()

    // Calculate price at cursor
    if (data.length > 0) {
      const allPrices = data.flatMap(d => [d.high, d.low])
      const minPrice = Math.min(...allPrices) * 0.998
      const maxPrice = Math.max(...allPrices) * 1.002
      const priceRange = maxPrice - minPrice
      const price = maxPrice - ((mousePos.y - padding.top) / chartHeight) * priceRange

      // Price label on right
      ctx.setLineDash([])
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
      ctx.fillRect(rect.width - padding.right + 2, mousePos.y - 10, padding.right - 7, 20)
      ctx.fillStyle = '#000000'
      ctx.font = 'bold 10px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(price.toFixed(2), rect.width - padding.right / 2 - 2, mousePos.y + 3)

      // Time label on bottom
      const chartWidth = rect.width - padding.left - padding.right
      const candleIndex = Math.floor((mousePos.x - padding.left) / (chartWidth / data.length))

      if (candleIndex >= 0 && candleIndex < data.length) {
        const candle = data[candleIndex]
        const date = new Date(candle.time)
        const timeStr = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })

        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
        ctx.fillRect(mousePos.x - 30, rect.height - 95, 60, 18)
        ctx.fillStyle = '#000000'
        ctx.font = '10px monospace'
        ctx.textAlign = 'center'
        ctx.fillText(timeStr, mousePos.x, rect.height - 82)

        // Update hovered candle info
        setHoveredCandle(candle)
      }
    }
  }, [mousePos, data])

  // Handle mouse events
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    setMousePos({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    })
  }, [])

  const handleMouseLeave = useCallback(() => {
    setMousePos(null)
    setHoveredCandle(null)
  }, [])

  const timeframes = ['6M', '3M', '1M', '5D', '1D', '4H', '1H', '15m']

  return (
    <div className="h-full flex flex-col bg-[#0d0e15]">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-white font-semibold">{symbol}</span>
            {hoveredCandle ? (
              <>
                <span className="text-gray-400 text-sm">O {hoveredCandle.open.toFixed(2)}</span>
                <span className="text-gray-400 text-sm">H {hoveredCandle.high.toFixed(2)}</span>
                <span className="text-gray-400 text-sm">L {hoveredCandle.low.toFixed(2)}</span>
                <span className={`text-sm ${hoveredCandle.close >= hoveredCandle.open ? 'text-green-400' : 'text-red-400'}`}>
                  C {hoveredCandle.close.toFixed(2)}
                </span>
                <span className={`text-sm ${hoveredCandle.close >= hoveredCandle.open ? 'text-green-400' : 'text-red-400'}`}>
                  {hoveredCandle.close >= hoveredCandle.open ? '+' : ''}{((hoveredCandle.close - hoveredCandle.open) / hoveredCandle.open * 100).toFixed(2)}%
                </span>
                <span className="text-gray-400 text-sm">VOL {(hoveredCandle.volume / 1000000).toFixed(1)}M</span>
              </>
            ) : (
              <>
                <span className="text-gray-400 text-sm">O {currentPrice.toFixed(2)}</span>
                <span className="text-gray-400 text-sm">H {(currentPrice * 1.002).toFixed(2)}</span>
                <span className="text-gray-400 text-sm">L {(currentPrice * 0.998).toFixed(2)}</span>
                <span className="text-gray-400 text-sm">C {currentPrice.toFixed(2)}</span>
                <span className="text-green-400 text-sm">+0.02%</span>
                <span className="text-gray-400 text-sm">VOL 50.5M</span>
              </>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {timeframes.map((tf) => (
            <button
              key={tf}
              onClick={() => setTimeframe(tf)}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                timeframe === tf
                  ? 'bg-gray-700 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              {tf}
            </button>
          ))}
          <div className="w-px h-4 bg-gray-700 mx-1" />
          <button className="text-gray-400 hover:text-white p-1">
            <Settings size={14} />
          </button>
          <button className="text-gray-400 hover:text-white p-1">
            <Maximize2 size={14} />
          </button>
        </div>
      </div>

      {/* Main chart area */}
      <div
        className="flex-grow relative cursor-crosshair"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full"
          style={{ height: 'calc(100% - 80px)', pointerEvents: 'none' }}
        />

        {/* Volume chart */}
        <canvas
          ref={volumeCanvasRef}
          className="absolute bottom-0 left-0 right-0"
          style={{ height: '80px', pointerEvents: 'none' }}
        />

        {/* Crosshair overlay */}
        <canvas
          ref={crosshairCanvasRef}
          className="absolute inset-0 w-full h-full"
          style={{ pointerEvents: 'none' }}
        />
      </div>

      {/* Bottom info bar */}
      <div className="flex items-center justify-between px-4 py-1 border-t border-gray-800 text-xs">
        <div className="flex items-center gap-4 text-gray-400">
          <span>22:10:31 (UTC-5)</span>
          <span className="text-gray-600">|</span>
          <button className="hover:text-white">%</button>
          <button className="hover:text-white">LOG</button>
          <button className="text-blue-400">AUTO</button>
        </div>
        <div className="text-gray-400">
          <span>VOLUME SMA 21</span>
        </div>
      </div>
    </div>
  )
}