'use client'

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { Activity, TrendingUp, Maximize2, Settings } from 'lucide-react'
import { useRouter } from 'next/navigation'

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
  const router = useRouter()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [timeframe, setTimeframe] = useState('2h')
  const [chartType, setChartType] = useState<'price' | 'depth'>('price')
  const [data, setData] = useState<{ time: number; value: number }[]>([])

  // Generate mock data
  const generateData = useCallback(() => {
    const newData = []
    const basePrice = currentPrice
    const points = 100
    const now = Date.now()

    for (let i = 0; i < points; i++) {
      const time = now - (points - i) * 3600000 // 1 hour intervals
      const volatility = basePrice * 0.003
      const random = Math.random()
      const change = (random - 0.5) * volatility
      const trend = Math.sin(i * 0.1) * (basePrice * 0.002)
      const value = basePrice + change + trend

      newData.push({ time, value })
    }
    return newData
  }, [currentPrice])

  useEffect(() => {
    setData(generateData())
  }, [generateData])

  // Draw chart
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || data.length === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size - respect container dimensions
    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    canvas.style.width = rect.width + 'px'
    canvas.style.height = rect.height + 'px'
    ctx.scale(dpr, dpr)

    // Clear canvas
    ctx.fillStyle = '#0B0E11'
    ctx.fillRect(0, 0, rect.width, rect.height)

    // Calculate scales
    const padding = { top: 20, right: 60, bottom: 30, left: 10 }
    const chartWidth = rect.width - padding.left - padding.right
    const chartHeight = rect.height - padding.top - padding.bottom

    const minPrice = Math.min(...data.map(d => d.value), stopLoss || Infinity) * 0.998
    const maxPrice = Math.max(...data.map(d => d.value), ...(targets || [])) * 1.002
    const priceRange = maxPrice - minPrice

    // Draw grid lines with dotted pattern
    ctx.strokeStyle = 'rgba(30, 34, 45, 0.5)'
    ctx.lineWidth = 0.5
    ctx.setLineDash([2, 4]) // Dotted pattern

    // Horizontal grid lines
    for (let i = 0; i <= 8; i++) {
      const y = padding.top + (chartHeight / 8) * i
      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(rect.width - padding.right, y)
      ctx.stroke()

      // Price labels for major lines only
      if (i % 2 === 0) {
        const price = maxPrice - (priceRange / 8) * i
        ctx.fillStyle = '#848E9C'
        ctx.font = '11px system-ui, -apple-system, sans-serif'
        ctx.textAlign = 'left'
        ctx.fillText(price.toFixed(2), rect.width - padding.right + 5, y + 3)
      }
    }

    // Vertical grid lines (time)
    for (let i = 0; i <= 6; i++) {
      const x = padding.left + (chartWidth / 6) * i
      ctx.beginPath()
      ctx.moveTo(x, padding.top)
      ctx.lineTo(x, rect.height - padding.bottom)
      ctx.stroke()
    }

    ctx.setLineDash([]) // Reset dash

    // Create gradient for the fill
    const gradient = ctx.createLinearGradient(0, padding.top, 0, rect.height - padding.bottom)
    gradient.addColorStop(0, 'rgba(14, 203, 129, 0.3)')
    gradient.addColorStop(1, 'rgba(14, 203, 129, 0.01)')

    // Draw filled area first
    ctx.beginPath()
    data.forEach((point, i) => {
      const x = padding.left + (chartWidth / (data.length - 1)) * i
      const y = padding.top + ((maxPrice - point.value) / priceRange) * chartHeight

      if (i === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })
    ctx.lineTo(rect.width - padding.right, rect.height - padding.bottom)
    ctx.lineTo(padding.left, rect.height - padding.bottom)
    ctx.closePath()
    ctx.fillStyle = gradient
    ctx.fill()

    // Draw price line on top
    ctx.strokeStyle = '#0ECB81'
    ctx.lineWidth = 2
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.beginPath()

    data.forEach((point, i) => {
      const x = padding.left + (chartWidth / (data.length - 1)) * i
      const y = padding.top + ((maxPrice - point.value) / priceRange) * chartHeight

      if (i === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })
    ctx.stroke()

    // Draw price levels with professional styling
    const drawPriceLine = (price: number, color: string, label: string, dashed = false, opacity = 1) => {
      const y = padding.top + ((maxPrice - price) / priceRange) * chartHeight

      // Draw the line
      ctx.strokeStyle = color + Math.round(opacity * 255).toString(16).padStart(2, '0')
      ctx.lineWidth = 1
      if (dashed) {
        ctx.setLineDash([8, 4])
      } else {
        ctx.setLineDash([])
      }

      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(rect.width - padding.right, y)
      ctx.stroke()

      // Price value on the line (left side)
      ctx.setLineDash([])
      ctx.fillStyle = color
      ctx.fillRect(padding.left, y - 9, 50, 18)
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 10px system-ui, -apple-system, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(price.toFixed(2), padding.left + 25, y + 3)

      // Label on right
      const labelBg = dashed ? color + '99' : color
      ctx.fillStyle = labelBg
      ctx.fillRect(rect.width - padding.right + 2, y - 9, padding.right - 4, 18)
      ctx.fillStyle = '#ffffff'
      ctx.textAlign = 'center'
      ctx.fillText(label, rect.width - padding.right / 2, y + 3)
    }

    // Draw stop loss with dashed line
    if (stopLoss) {
      drawPriceLine(stopLoss, '#ef4444', 'STOP', true, 0.8)
    }

    // Draw entry point
    if (entryPoint) {
      drawPriceLine(entryPoint, '#06b6d4', 'ENTRY', false, 0.9)
    }

    // Draw targets with increasing opacity
    targets.forEach((target, i) => {
      const opacity = i === 2 ? 1 : 0.7 + (i * 0.1)
      drawPriceLine(target, '#10b981', `T${i + 1}`, false, opacity)
    })

    // Reset line dash
    ctx.setLineDash([])

    // Current price indicator with animation effect
    const lastPrice = data[data.length - 1].value
    const lastY = padding.top + ((maxPrice - lastPrice) / priceRange) * chartHeight

    // Draw price line extension
    ctx.strokeStyle = '#0ECB8133'
    ctx.lineWidth = 1
    ctx.setLineDash([4, 4])
    ctx.beginPath()
    ctx.moveTo(rect.width - padding.right, lastY)
    ctx.lineTo(rect.width, lastY)
    ctx.stroke()
    ctx.setLineDash([])

    // Current price tag
    const priceTagWidth = 55
    ctx.fillStyle = '#0ECB81'
    ctx.fillRect(rect.width - priceTagWidth, lastY - 11, priceTagWidth, 22)

    // Arrow pointing left
    ctx.beginPath()
    ctx.moveTo(rect.width - priceTagWidth, lastY)
    ctx.lineTo(rect.width - priceTagWidth - 5, lastY - 5)
    ctx.lineTo(rect.width - priceTagWidth - 5, lastY + 5)
    ctx.closePath()
    ctx.fill()

    ctx.fillStyle = '#ffffff'
    ctx.font = 'bold 11px system-ui, -apple-system, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(lastPrice.toFixed(2), rect.width - priceTagWidth/2, lastY + 4)

  }, [data, stopLoss, entryPoint, targets])

  const timeframes = ['6M', '3M', '1M', '5D', '1D', '4H', '1H', '2h']

  return (
    <div className="trading-card rounded-xl transition-all duration-300 h-full flex flex-col">
      <div className="flex items-center justify-between p-4 border-b border-gray-800/50 flex-shrink-0">
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setChartType('price')}
            className={`px-3 py-1.5 text-sm rounded-lg transition-all duration-200 ${
              chartType === 'price' ? 'bg-gradient-to-r from-gray-700 to-gray-600 text-white shadow-sm' : 'bg-gray-800/30 text-gray-400 hover:text-white hover:bg-gray-800/50'
            }`}
          >
            Price chart
          </button>
          <button
            onClick={() => setChartType('depth')}
            className={`px-3 py-1.5 text-sm rounded-lg transition-all duration-200 ${
              chartType === 'depth' ? 'bg-gradient-to-r from-gray-700 to-gray-600 text-white shadow-sm' : 'bg-gray-800/30 text-gray-400 hover:text-white hover:bg-gray-800/50'
            }`}
          >
            Depth chart
          </button>
        </div>

        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1">
            {timeframes.map((tf) => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-2 py-1 text-xs rounded-lg transition-all duration-200 ${
                  timeframe === tf ? 'bg-gradient-to-r from-gray-700 to-gray-600 text-white shadow-sm' : 'bg-gray-800/30 text-gray-400 hover:text-white hover:bg-gray-800/50'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>

          <div className="flex items-center space-x-2">
            <Activity size={18} className="text-gray-400 cursor-pointer hover:text-white" />
            <TrendingUp size={18} className="text-gray-400 cursor-pointer hover:text-white" />
            <Settings size={18} className="text-gray-400 cursor-pointer hover:text-white" />
            <Maximize2
              size={18}
              className="text-gray-400 cursor-pointer hover:text-white"
              onClick={() => router.push(`/ticker/${symbol}`)}
              title="Open full chart"
            />
          </div>
        </div>
      </div>

      <div className="flex-grow relative min-h-0">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
          style={{ display: 'block' }}
        />
      </div>

      <div className="flex items-center justify-between px-4 py-3 text-xs text-gray-400 border-t border-gray-800/50 bg-gray-900/30 flex-shrink-0">
        <div className="flex items-center space-x-4">
          <span>O: {currentPrice.toFixed(2)}</span>
          <span>H: {(currentPrice * 1.002).toFixed(2)}</span>
          <span>L: {(currentPrice * 0.998).toFixed(2)}</span>
          <span>C: {currentPrice.toFixed(2)}</span>
          <span>VOL: {symbol === 'SPY' ? '85.2M' : symbol === 'QQQ' ? '42.7M' : symbol === 'IWM' ? '31.5M' : 'N/A'}</span>
        </div>
        <div className="flex items-center space-x-2">
          <span>15:08:21 (UTC-5)</span>
          <span>LOG</span>
          <span className="text-blue-400">AUTO</span>
        </div>
      </div>
    </div>
  )
}