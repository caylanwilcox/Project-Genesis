'use client'

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { Activity, TrendingUp, Maximize2, Minimize2, Settings, ChevronDown } from 'lucide-react'
import { useRouter } from 'next/navigation'
import styles from './ProfessionalChart.module.css'
import controlStyles from './ChartControls.module.css'
import interactionStyles from './ChartInteraction.module.css'
import tagStyles from './ChartYAxisTags.module.css'
import toolbarStyles from './ChartToolbar.module.css'

interface ProfessionalChartProps {
  symbol: string
  currentPrice?: number
  stopLoss?: number
  targets?: number[]
  entryPoint?: number
  data?: CandleData[] // Allow external data injection
  onDataUpdate?: (data: CandleData[]) => void
  onTimeframeChange?: (tf: string, displayTf: string) => void
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
  entryPoint,
  data: externalData,
  onDataUpdate,
  onTimeframeChange
}) => {
  const router = useRouter()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const volumeCanvasRef = useRef<HTMLCanvasElement>(null)
  const crosshairCanvasRef = useRef<HTMLCanvasElement>(null)
  const scaleAreaRef = useRef<HTMLDivElement>(null)
  const [timeframe, setTimeframe] = useState('1D')
  const [chartType, setChartType] = useState<'candles' | 'line'>('candles')
  const [data, setData] = useState<CandleData[]>([])
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null)
  const [useExternalData, setUseExternalData] = useState(false)
  const [chartPixelWidth, setChartPixelWidth] = useState<number>(0)
  const [interval, setInterval] = useState('15 min') // Default to 15 min to match 1D timeframe
  const [showIntervalDropdown, setShowIntervalDropdown] = useState(false)
  const [showIndicatorsDropdown, setShowIndicatorsDropdown] = useState(false)
  // Track the data timeframe string used for label formatting (e.g., '15m','1h','1d','1w','1M')
  const [dataTimeframeForLabels, setDataTimeframeForLabels] = useState<string>('15m')
  const [isPanning, setIsPanning] = useState(false)
  const [panOffset, setPanOffset] = useState(0)
  const [panStart, setPanStart] = useState<{ x: number; offset: number } | null>(null)
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 100 })
  const [currentTime, setCurrentTime] = useState(new Date())
  const [priceScale, setPriceScale] = useState(1.0) // 1.0 = normal, > 1 = zoomed in, < 1 = zoomed out
  const [isScaling, setIsScaling] = useState(false)
  const [scaleStart, setScaleStart] = useState<{ y: number; scale: number } | null>(null)
  const [timeScale, setTimeScale] = useState(1.0) // 1.0 = 100 candles, 2.0 = 50 candles (zoomed in)
  const [isTimeScaling, setIsTimeScaling] = useState(false)
  const [timeScaleStart, setTimeScaleStart] = useState<{ x: number; scale: number } | null>(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const chartContainerRef = useRef<HTMLDivElement>(null)
  
  // Y-axis price tags rendered as DOM for CSS styling
  type OverlayKind = 'target' | 'stop' | 'entry' | 'current'
  const [overlayTags, setOverlayTags] = useState<Array<{ y: number; label: string; kind: OverlayKind }>>([])

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

  // Use external data if provided, otherwise generate mock data
  useEffect(() => {
    if (externalData && externalData.length > 0) {
      setData(externalData)
      setUseExternalData(true)
      if (onDataUpdate) {
        onDataUpdate(externalData)
      }
    } else {
      setData(generateCandleData())
      setUseExternalData(false)
    }
    // Reset pan to show most recent data when new data loads
    setPanOffset(0)
  }, [externalData, generateCandleData, onDataUpdate])

  // Calculate visible data range when pan offset or time scale changes
  useEffect(() => {
    if (data.length === 0) return

    // Calculate how many candles fit in the view based on time scale
    // timeScale 1.0 = 100 candles, 2.0 = 50 candles (zoomed in), 0.5 = 200 candles (zoomed out)
    const baseCandlesInView = 100
    const candlesInView = Math.round(baseCandlesInView / timeScale)
    const maxOffset = Math.max(0, data.length - candlesInView)

    // Clamp offset to valid range
    const clampedOffset = Math.max(0, Math.min(maxOffset, Math.round(panOffset)))

    // When panOffset is 0, show the LATEST data (right side)
    // As panOffset increases, we pan backwards in time (left)
    const start = maxOffset - clampedOffset
    const end = Math.min(start + candlesInView, data.length)

    setVisibleRange({
      start: Math.max(0, start),
      end: end
    })
  }, [panOffset, data.length, timeScale])

  // Draw main chart
  useEffect(() => {
    const canvas = canvasRef.current
    const volumeCanvas = volumeCanvasRef.current
    if (!canvas || !volumeCanvas || data.length === 0) return

    const ctx = canvas.getContext('2d')
    const volCtx = volumeCanvas.getContext('2d')
    if (!ctx || !volCtx) return

    // Get visible data slice
    const visibleData = data.slice(visibleRange.start, visibleRange.end)
    if (visibleData.length === 0) return

    // Setup canvas to fit parent container exactly
    const parentWidth = canvas.parentElement ? (canvas.parentElement as HTMLElement).clientWidth : 800
    const dpr = window.devicePixelRatio || 1
    // Reduce padding and label sizes on narrow screens to avoid overlap
    const isNarrow = parentWidth < 420
    // Read CSS variable for y-axis gutter so CSS controls the width
    const cssGutter = getComputedStyle(document.documentElement).getPropertyValue('--chart-y-axis-gutter').trim()
    const gutter = cssGutter.endsWith('px') ? parseInt(cssGutter) : (isNarrow ? 56 : 80)
    const padding = isNarrow ? { top: 6, right: gutter, bottom: 14, left: 6 } : { top: 10, right: gutter, bottom: 20, left: 10 }

    // Use parent width - don't expand beyond container
    canvas.style.width = parentWidth + 'px'
    volumeCanvas.style.width = parentWidth + 'px'
    setChartPixelWidth(parentWidth)

    // Now read rects after style applied
    const rect = canvas.getBoundingClientRect()
    const volRect = volumeCanvas.getBoundingClientRect()

    // Main chart canvas
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    canvas.style.height = rect.height + 'px'
    ctx.scale(dpr, dpr)

    // Volume canvas
    volumeCanvas.width = volRect.width * dpr
    volumeCanvas.height = volRect.height * dpr
    volumeCanvas.style.height = volRect.height + 'px'
    volCtx.scale(dpr, dpr)

    // Clear canvases
    ctx.fillStyle = '#0d0e15'
    ctx.fillRect(0, 0, rect.width, rect.height)
    volCtx.fillStyle = '#0d0e15'
    volCtx.fillRect(0, 0, volRect.width, volRect.height)

    // Calculate dimensions
    const chartWidth = rect.width - padding.left - padding.right
    const chartHeight = rect.height - padding.top - padding.bottom
    const volChartHeight = volRect.height - 20

    // Calculate price range from visible data
    const allPrices = visibleData.flatMap(d => [d.high, d.low])
    if (stopLoss) allPrices.push(stopLoss)
    if (entryPoint) allPrices.push(entryPoint)
    allPrices.push(...targets)

    const dataMinPrice = Math.min(...allPrices)
    const dataMaxPrice = Math.max(...allPrices)
    const dataPriceRange = dataMaxPrice - dataMinPrice
    const priceCenter = (dataMinPrice + dataMaxPrice) / 2

    // Apply price scale - zoom in/out from center
    const scaledRange = dataPriceRange / priceScale
    const minPrice = priceCenter - (scaledRange / 2) * 1.002
    const maxPrice = priceCenter + (scaledRange / 2) * 1.002
    const priceRange = maxPrice - minPrice

    // Calculate volume range from visible data
    const volumes = visibleData.map(d => d.volume).filter(v => v > 0)
    const maxVolume = volumes.length > 0 ? Math.max(...volumes) : 0

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)'
    ctx.lineWidth = 1
    ctx.setLineDash([1, 1])

    // Horizontal grid lines (price levels)
    const horizontalTicks = isNarrow ? 5 : 8
    for (let i = 0; i <= horizontalTicks; i++) {
      const y = padding.top + (chartHeight / 8) * i
      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(rect.width - padding.right, y)
      ctx.stroke()

      // Price labels - centered within the gutter area
      const price = maxPrice - (priceRange / 8) * i
      ctx.fillStyle = '#6b7280'
      ctx.font = isNarrow ? '8px monospace' : '11px monospace'
      ctx.textAlign = 'center'
      // Position label in center of gutter
      const labelX = rect.width - (gutter / 2)
      ctx.fillText(price.toFixed(2), labelX, y + 3)
    }

    // Vertical grid lines (time)
    // Exactly three time labels across the x-axis
    const verticalTicks = 2
    for (let i = 0; i <= verticalTicks; i++) {
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

      // Time labels - drawn on volume canvas below the volume bars
      if (visibleData.length > 0) {
        const index = Math.min(visibleData.length - 1, Math.round((i / 6) * (visibleData.length - 1)))
        const ts = new Date(visibleData[index].time)

        // Format based on timeframe - show appropriate detail (New York timezone)
        let label = ''
        // Format based on data timeframe granularity
        if (dataTimeframeForLabels === '1M') {
          label = ts.toLocaleDateString('en-US', { month: 'short', year: 'numeric', timeZone: 'America/New_York' })
        } else if (dataTimeframeForLabels === '1w' || dataTimeframeForLabels === '1d') {
          label = ts.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'America/New_York' })
        } else {
          // Intraday (minutes/hours)
          label = ts.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true, timeZone: 'America/New_York' })
        }

        // Draw labels on volume canvas below the volume bars
        volCtx.fillStyle = isNarrow ? '#AAB2C5' : '#6b7280'
        volCtx.font = isNarrow ? '9px monospace' : '10px monospace'
        volCtx.textAlign = 'center'
        // Optional subtle backdrop for narrow screens
        if (isNarrow) {
          const textWidth = volCtx.measureText(label).width
          const bx = x - textWidth / 2 - 3
          const by = volChartHeight + 2
          volCtx.fillStyle = 'rgba(13,14,21,0.7)'
          volCtx.fillRect(bx, by, textWidth + 6, 10)
          volCtx.fillStyle = '#AAB2C5'
        }
        volCtx.fillText(label, x, volChartHeight + 12)
      }
    }

    ctx.setLineDash([])

    // Draw candlesticks
    const candleWidth = chartWidth / visibleData.length
    const candleSpacing = candleWidth * 0.8

    visibleData.forEach((candle, i) => {
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
      const volHeight = maxVolume > 0 ? (candle.volume / maxVolume) * volChartHeight * 0.8 : 0
      volCtx.fillStyle = isGreen ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)'
      volCtx.fillRect(x - candleSpacing / 2, volChartHeight - volHeight, candleSpacing, volHeight)
    })

    // Draw price levels (render tag via DOM for CSS control)
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
    }

    const nextTags: Array<{ y: number; label: string; kind: OverlayKind }> = []
    // Draw stop loss
    if (stopLoss) {
      drawPriceLine(stopLoss, '#ef444488', 'SL', true)
      const y = padding.top + ((maxPrice - stopLoss) / priceRange) * chartHeight
      nextTags.push({ y, label: stopLoss.toFixed(2), kind: 'stop' })
    }

    // Draw entry point
    if (entryPoint) {
      drawPriceLine(entryPoint, '#06b6d488', 'ENTRY', false)
      const y = padding.top + ((maxPrice - entryPoint) / priceRange) * chartHeight
      nextTags.push({ y, label: entryPoint.toFixed(2), kind: 'entry' })
    }

    // Draw targets
    targets.forEach((target) => {
      drawPriceLine(target, '#22c55e88', 'T', false)
      const y = padding.top + ((maxPrice - target) / priceRange) * chartHeight
      nextTags.push({ y, label: target.toFixed(2), kind: 'target' })
    })

    // Current price line and tag (use last candle from original data, not just visible)
    const lastCandle = data[data.length - 1]
    // Only show current price line if the last candle is in the visible range
    const isCurrentPriceVisible = visibleRange.end >= data.length
    if (isCurrentPriceVisible) {
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

      // Tag via DOM overlay
      nextTags.push({ y: currentY, label: lastCandle.close.toFixed(2), kind: 'current' })
    }

    // Draw volume scale (accurate, unit-aware)
    volCtx.fillStyle = '#6b7280'
    volCtx.font = isNarrow ? '8px monospace' : '10px monospace'
    volCtx.textAlign = 'center'
    const formatVolume = (v: number): string => {
      const abs = Math.abs(v)
      if (abs >= 1e9) return (abs / 1e9 >= 10 ? (abs / 1e9).toFixed(0) : (abs / 1e9).toFixed(1)) + 'B'
      if (abs >= 1e6) return (abs / 1e6 >= 10 ? (abs / 1e6).toFixed(0) : (abs / 1e6).toFixed(1)) + 'M'
      if (abs >= 1e3) return (abs / 1e3 >= 10 ? (abs / 1e3).toFixed(0) : (abs / 1e3).toFixed(1)) + 'K'
      return abs.toFixed(0)
    }
    // Position volume labels in center of gutter, vertically stacked
    const volLabelX = volRect.width - (gutter / 2)
    volCtx.fillText('Vol', volLabelX, 10)
    volCtx.fillText(formatVolume(maxVolume), volLabelX, 22)

    setOverlayTags(nextTags)

  }, [data, visibleRange, stopLoss, entryPoint, targets, timeframe, priceScale])

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
      const visibleData = data.slice(visibleRange.start, visibleRange.end)
      const candleIndex = Math.floor((mousePos.x - padding.left) / (chartWidth / visibleData.length))

      if (candleIndex >= 0 && candleIndex < visibleData.length) {
        const candle = visibleData[candleIndex]
        const date = new Date(candle.time)

        // Format timestamp based on timeframe (New York timezone)
        let timeStr = ''
        if (dataTimeframeForLabels === '1M') {
          timeStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'America/New_York' })
        } else if (dataTimeframeForLabels === '1w' || dataTimeframeForLabels === '1d') {
          timeStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'America/New_York' })
        } else {
          timeStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'America/New_York' }) + ' ' +
                    date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true, timeZone: 'America/New_York' })
        }

        // Adjust tooltip width based on content
        const tooltipWidth = timeStr.length > 12 ? 100 : 80
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
        ctx.fillRect(mousePos.x - tooltipWidth / 2, rect.height - 95, tooltipWidth, 18)
        ctx.fillStyle = '#000000'
        ctx.font = '10px monospace'
        ctx.textAlign = 'center'
        ctx.fillText(timeStr, mousePos.x, rect.height - 82)

        // Hover data no longer displayed in header
      }
    }
  }, [mousePos, data, visibleRange])

  // Handle mouse events
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    setIsPanning(true)
    setPanStart({ x: e.clientX, offset: panOffset })
  }, [panOffset])

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()

    if (isPanning && panStart) {
      // Calculate pan delta
      const deltaX = e.clientX - panStart.x
      // Convert pixels to candles (approximate)
      const candlesPerPixel = 100 / rect.width // Approximate ratio
      const candleDelta = -deltaX * candlesPerPixel * 2 // Negative because panning right shows older data

      // Update pan offset
      const newOffset = panStart.offset + candleDelta
      const maxOffset = Math.max(0, data.length - 100)
      setPanOffset(Math.max(0, Math.min(maxOffset, newOffset)))
    } else {
      // Update crosshair position
      setMousePos({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      })
    }
  }, [isPanning, panStart, data.length])

  const handleMouseUp = useCallback(() => {
    setIsPanning(false)
    setPanStart(null)
  }, [])

  const handleMouseLeave = useCallback(() => {
    setMousePos(null)
    // Hover data no longer displayed in header
    setIsPanning(false)
    setPanStart(null)
  }, [])

  // Close interval dropdown when clicking outside
  const intervalDropdownRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (showIntervalDropdown && intervalDropdownRef.current) {
        const target = e.target as HTMLElement
        if (!intervalDropdownRef.current.contains(target)) {
          setShowIntervalDropdown(false)
        }
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [showIntervalDropdown])

  // Global mouse up handler for panning and scaling
  useEffect(() => {
    const handleGlobalMouseUp = () => {
      if (isPanning) {
        setIsPanning(false)
        setPanStart(null)
      }
      if (isScaling) {
        setIsScaling(false)
        setScaleStart(null)
      }
      if (isTimeScaling) {
        setIsTimeScaling(false)
        setTimeScaleStart(null)
      }
    }
    document.addEventListener('mouseup', handleGlobalMouseUp)
    return () => document.removeEventListener('mouseup', handleGlobalMouseUp)
  }, [isPanning, isScaling, isTimeScaling])

  // Handle price scale dragging
  const handleScaleMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    e.stopPropagation() // Prevent chart panning
    setIsScaling(true)
    setScaleStart({ y: e.clientY, scale: priceScale })
  }, [priceScale])

  const handleScaleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (isScaling && scaleStart) {
      const deltaY = e.clientY - scaleStart.y
      // Convert pixels to scale factor (negative because dragging down = zoom out)
      const scaleFactor = 1 - (deltaY / 300) // 300px drag = 2x zoom change
      const newScale = Math.max(0.1, Math.min(10, scaleStart.scale * scaleFactor))
      setPriceScale(newScale)
    }
  }, [isScaling, scaleStart])

  const handleScaleDoubleClick = useCallback(() => {
    setPriceScale(1.0) // Reset to default
  }, [])

  // Handle time scale dragging (horizontal zoom)
  const handleTimeScaleMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    e.stopPropagation() // Prevent chart panning
    setIsTimeScaling(true)
    setTimeScaleStart({ x: e.clientX, scale: timeScale })
  }, [timeScale])

  const handleTimeScaleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (isTimeScaling && timeScaleStart) {
      const deltaX = e.clientX - timeScaleStart.x
      // Convert pixels to scale factor (positive = zoom in, negative = zoom out)
      const scaleFactor = 1 + (deltaX / 300) // 300px drag = 2x zoom change
      const newScale = Math.max(0.2, Math.min(5, timeScaleStart.scale * scaleFactor))
      setTimeScale(newScale)
    }
  }, [isTimeScaling, timeScaleStart])

  const handleTimeScaleDoubleClick = useCallback(() => {
    setTimeScale(1.0) // Reset to default
  }, [])

  // Handle fullscreen toggle
  const toggleFullscreen = useCallback(async () => {
    if (!chartContainerRef.current) return

    try {
      if (!document.fullscreenElement) {
        await chartContainerRef.current.requestFullscreen()
        setIsFullscreen(true)
      } else {
        await document.exitFullscreen()
        setIsFullscreen(false)
      }
    } catch (error) {
      console.error('Error toggling fullscreen:', error)
    }
  }, [])

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }

    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  // Update current time every second for the info bar
  useEffect(() => {
    const updateTime = () => setCurrentTime(new Date())
    const timeInterval = window.setInterval(updateTime, 1000)
    return () => window.clearInterval(timeInterval)
  }, [])

  const timeframes = ['1D', '5D', '1M', '3M', '6M', 'YTD', '1Y', '5Y', 'All']
  const intervals = ['1 min', '5 min', '15 min', '30 min', '1 hour', '1 day', '1 week', '1 month', '3 months', '6 months', '1 year']

  const handleTimeframeClick = (tf: string) => {
    console.log('[ProfessionalChart] Timeframe button clicked:', tf);
    setTimeframe(tf)
    // Reset pan to show latest data on the right
    setPanOffset(0)
    setPriceScale(1.0) // Reset zoom when changing timeframe
    setTimeScale(1.0)

    if (onTimeframeChange) {
      // Map display timeframe to data timeframe
      const map: Record<string, string> = {
        '1D': '15m',    // 1 day view with 15min bars
        '5D': '1h',     // 5 day view with 1hr bars
        '1M': '1h',     // 1 month view with 1hr bars for higher fidelity
        '3M': '1d',     // 3 month view with daily bars
        '6M': '1d',     // 6 month view with daily bars
        'YTD': '1d',    // Year to date with daily bars
        '1Y': '1d',     // 1 year view with daily bars
        '5Y': '1w',     // 5 year view with weekly bars
        'All': '1M',    // All time with monthly bars
      }

      // Also update interval dropdown to match the actual data timeframe
      const intervalMap: Record<string, string> = {
        '1D': '15 min',
        '5D': '1 hour',
        '1M': '1 hour',
        '3M': '1 day',
        '6M': '1 day',
        'YTD': '1 day',
        '1Y': '1 day',
        '5Y': '1 week',
        'All': '1 month',
      }

      const mapped = map[tf]
      const intervalMapped = intervalMap[tf]

      console.log('[ProfessionalChart] Mapped to data timeframe:', mapped);
      if (intervalMapped) setInterval(intervalMapped)
      if (mapped) onTimeframeChange(mapped, tf) // Pass both data timeframe and display timeframe
    }
  }

  const handleIntervalChange = (newInterval: string) => {
    console.log('[ProfessionalChart] Interval changed to:', newInterval);
    setInterval(newInterval)
    setShowIntervalDropdown(false)
    // Reset pan to show latest data on the right
    setPanOffset(0)
    if (onTimeframeChange) {
      // Map interval to timeframe
      const map: Record<string, string> = {
        '1 min': '1m',
        '5 min': '5m',
        '15 min': '15m',
        '30 min': '30m',
        '1 hour': '1h',
        '1 day': '1d',
        '1 week': '1w',
        '1 month': '1M',
        '3 months': '3M',
        '6 months': '6M',
        '1 year': '1Y',
      }
      const mapped = map[newInterval]
      console.log('[ProfessionalChart] Mapped interval to timeframe:', mapped);
      if (mapped) onTimeframeChange(mapped, 'Custom') // Pass 'Custom' as display timeframe for interval changes
    }
  }

  return (
    <div ref={chartContainerRef} className={styles.chartContainer}>
      {/* Header */}
      <div className={styles.chartHeader}>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className={styles.chartSymbol}>{symbol}</span>
          </div>
        </div>

        <div className={styles.chartControls}>
            {/* Chart type pill */}
            <button
              onClick={() => setChartType(prev => (prev === 'candles' ? 'line' : 'candles'))}
              className={controlStyles.chartTypeButton}
              title="Toggle chart type"
            >
              {chartType === 'candles' ? 'Candle' : 'Line'}
            </button>

            <div ref={intervalDropdownRef} className={controlStyles.intervalDropdownContainer}>
              <button
                onClick={() => setShowIntervalDropdown(!showIntervalDropdown)}
                className={controlStyles.intervalButton}
              >
                {interval}
                <ChevronDown size={10} className="sm:w-3 sm:h-3" />
              </button>
              {showIntervalDropdown && (
                <div className={controlStyles.intervalDropdown}>
                  {intervals.map((int) => (
                    <button
                      key={int}
                      onClick={() => handleIntervalChange(int)}
                      className={`${controlStyles.intervalOption} ${interval === int ? controlStyles.active : ''}`}
                    >
                      {int}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Indicators pill (hidden on mobile to reduce clutter) */}
            <div className="relative hidden sm:block">
              <button
                onClick={() => setShowIndicatorsDropdown(!showIndicatorsDropdown)}
                className="px-3 py-1.5 text-xs font-medium rounded-full bg-gray-800/60 text-gray-200 hover:bg-gray-800 transition-colors whitespace-nowrap flex items-center gap-1"
              >
                Indicators <ChevronDown size={12} />
              </button>
              {showIndicatorsDropdown && (
                <div className="absolute right-0 top-full mt-1 bg-gray-800 border border-gray-700 rounded shadow-lg z-50 min-w-[160px]">
                  <div className="px-3 py-2 text-xs text-gray-400">Coming soon</div>
                </div>
              )}
            </div>

            <div className={toolbarStyles.toolbarButtons}>
              <button className={toolbarStyles.settingsButton} title="Settings">
                <Settings className="w-3.5 h-3.5 sm:w-3.5 sm:h-3.5" />
              </button>
              <button
                onClick={toggleFullscreen}
                className={toolbarStyles.fullscreenButton}
                title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
              >
                {isFullscreen ? <Minimize2 className="w-3.5 h-3.5 sm:w-3.5 sm:h-3.5" /> : <Maximize2 className="w-3.5 h-3.5 sm:w-3.5 sm:h-3.5" />}
              </button>
            </div>
        </div>
      </div>

      {/* Main chart area */}
      <div
        className={`${styles.chartMainArea} ${isPanning ? styles.panning : styles.idle}`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      >
        <canvas
          ref={canvasRef}
          className={styles.chartCanvas}
        />

        {/* Volume chart */}
        <canvas
          ref={volumeCanvasRef}
          className={styles.volumeCanvas}
        />

        {/* Crosshair overlay */}
        <canvas
          ref={crosshairCanvasRef}
          className={styles.crosshairCanvas}
        />

        {/* CSS-stylable y-axis tags */}
        <div className={tagStyles.yAxisTagsContainer} aria-hidden>
          {overlayTags.map((tag, idx) => (
            <div
              key={idx}
              className={`${tagStyles.yAxisTag} ${
                tag.kind === 'current' ? tagStyles['yAxisTag--current'] :
                tag.kind === 'entry' ? tagStyles['yAxisTag--entry'] :
                tag.kind === 'stop' ? tagStyles['yAxisTag--stop'] : tagStyles['yAxisTag--target']
              }`}
              style={{ top: `${tag.y - 11}px` }}
            >
              <span className={tagStyles.yAxisTagLabel}>{tag.label}</span>
            </div>
          ))}
        </div>

        {/* Price scale drag area - right side of chart */}
        <div
          ref={scaleAreaRef}
          className={interactionStyles.priceScaleDragArea}
          onMouseDown={handleScaleMouseDown}
          onMouseMove={handleScaleMouseMove}
          onDoubleClick={handleScaleDoubleClick}
          title="Drag to zoom price scale (double-click to reset)"
        >
          {/* Visual indicator for drag area */}
          <div className={interactionStyles.dragIndicator}>
            <div className={interactionStyles.dragIndicatorLine}></div>
            <div className={interactionStyles.dragIndicatorLine}></div>
            <div className={interactionStyles.dragIndicatorLine}></div>
          </div>

          {/* Zoom level indicator */}
          {priceScale !== 1.0 && (
            <div className={interactionStyles.zoomIndicator}>
              {priceScale.toFixed(1)}x
            </div>
          )}
        </div>

        {/* Time scale drag area - below volume chart where time labels are */}
        <div
          className={interactionStyles.timeScaleDragArea}
          onMouseDown={handleTimeScaleMouseDown}
          onMouseMove={handleTimeScaleMouseMove}
          onDoubleClick={handleTimeScaleDoubleClick}
          title="Drag horizontally to zoom time scale (double-click to reset)"
        >
          {/* Visual indicator for drag area */}
          <div className={interactionStyles.timeScaleDragIndicator}>
            <div className={interactionStyles.timeScaleDragIndicatorLine}></div>
            <div className={interactionStyles.timeScaleDragIndicatorLine}></div>
            <div className={interactionStyles.timeScaleDragIndicatorLine}></div>
          </div>

          {/* Zoom level indicator */}
          {timeScale !== 1.0 && (
            <div className={interactionStyles.timeZoomIndicator}>
              {timeScale.toFixed(1)}x
            </div>
          )}
        </div>
      </div>

      {/* Bottom info bar */}
      <div className={styles.bottomInfoBar}>
        <div className={styles.timeDisplay}>
          <span>{currentTime.toLocaleTimeString('en-US', { hour12: false, timeZone: 'America/New_York' })} (ET)</span>
          <span className={styles.timeSeparator}>|</span>
          <span>{currentTime.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'America/New_York' })}</span>
        </div>
      </div>
    </div>
  )
}