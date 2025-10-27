'use client'

import React, { useEffect, useRef } from 'react'
import { CandleData } from './types'

interface CrosshairProps {
  mousePos: { x: number; y: number } | null
  visibleData: CandleData[]
  chartWidth: number
  chartHeight: number
  padding: any
  minPrice: number
  maxPrice: number
  priceRange: number
  baseWidth: number
  isPanning: boolean
}

export const Crosshair: React.FC<CrosshairProps> = ({
  mousePos,
  visibleData,
  chartWidth,
  chartHeight,
  padding,
  minPrice,
  maxPrice,
  priceRange,
  baseWidth,
  isPanning
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * window.devicePixelRatio
    canvas.height = rect.height * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    // Clear canvas
    ctx.clearRect(0, 0, rect.width, rect.height)

    // Don't show crosshair while panning
    if (!mousePos || visibleData.length === 0 || isPanning) return

    // Calculate which candle we're hovering over
    const candleWidth = chartWidth / baseWidth
    const relativeX = mousePos.x - padding.left
    const candleIndex = Math.floor(relativeX / candleWidth)

    // Check if we're within bounds
    if (candleIndex < 0 || candleIndex >= visibleData.length) return

    const candle = visibleData[candleIndex]
    if (!candle) return

    // Snap to center of candle
    const snapX = padding.left + candleIndex * candleWidth + candleWidth / 2

    // Calculate price at mouse Y position
    const relativeY = mousePos.y - padding.top
    const priceAtMouse = maxPrice - (relativeY / chartHeight) * priceRange

    // Draw vertical line (snapped to candle center)
    ctx.strokeStyle = 'rgba(150, 150, 150, 0.6)'
    ctx.lineWidth = 1
    ctx.setLineDash([5, 5])
    ctx.beginPath()
    ctx.moveTo(snapX, 0)
    ctx.lineTo(snapX, rect.height)
    ctx.stroke()

    // Draw horizontal line
    ctx.beginPath()
    ctx.moveTo(0, mousePos.y)
    ctx.lineTo(rect.width, mousePos.y)
    ctx.stroke()
    ctx.setLineDash([])

    // Draw price label on Y axis
    ctx.fillStyle = 'rgba(50, 50, 50, 0.9)'
    ctx.fillRect(rect.width - 80, mousePos.y - 12, 80, 24)
    ctx.fillStyle = '#fff'
    ctx.font = '11px monospace'
    ctx.textAlign = 'center'
    ctx.fillText(priceAtMouse.toFixed(2), rect.width - 40, mousePos.y + 4)

    // Draw OHLC tooltip
    const tooltipX = snapX + 10
    const tooltipY = 10
    const tooltipWidth = 150
    const tooltipHeight = 105

    // Tooltip background
    ctx.fillStyle = 'rgba(30, 30, 30, 0.95)'
    ctx.fillRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight)
    ctx.strokeStyle = 'rgba(100, 100, 100, 0.8)'
    ctx.lineWidth = 1
    ctx.strokeRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight)

    // Tooltip text
    ctx.fillStyle = '#fff'
    ctx.font = '10px monospace'
    ctx.textAlign = 'left'

    const textX = tooltipX + 8
    let textY = tooltipY + 15

    // Format time
    const date = new Date(candle.time)
    const timeStr = date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    })

    ctx.fillText(`Time: ${timeStr}`, textX, textY)
    textY += 15

    // OHLC data with colors
    ctx.fillStyle = '#888'
    ctx.fillText('O:', textX, textY)
    ctx.fillStyle = '#fff'
    ctx.fillText(candle.open.toFixed(2), textX + 15, textY)
    textY += 15

    ctx.fillStyle = '#888'
    ctx.fillText('H:', textX, textY)
    ctx.fillStyle = '#22c55e'
    ctx.fillText(candle.high.toFixed(2), textX + 15, textY)
    textY += 15

    ctx.fillStyle = '#888'
    ctx.fillText('L:', textX, textY)
    ctx.fillStyle = '#ef4444'
    ctx.fillText(candle.low.toFixed(2), textX + 15, textY)
    textY += 15

    ctx.fillStyle = '#888'
    ctx.fillText('C:', textX, textY)
    ctx.fillStyle = candle.close >= candle.open ? '#22c55e' : '#ef4444'
    ctx.fillText(candle.close.toFixed(2), textX + 15, textY)
    textY += 15

    // Volume
    ctx.fillStyle = '#888'
    ctx.fillText('V:', textX, textY)
    ctx.fillStyle = '#9ca3af'
    const volumeStr = candle.volume >= 1000000
      ? `${(candle.volume / 1000000).toFixed(2)}M`
      : candle.volume >= 1000
      ? `${(candle.volume / 1000).toFixed(1)}K`
      : candle.volume.toFixed(0)
    ctx.fillText(volumeStr, textX + 15, textY)

  }, [mousePos, visibleData, chartWidth, chartHeight, padding, minPrice, maxPrice, priceRange, baseWidth, isPanning])

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 10
      }}
    />
  )
}
