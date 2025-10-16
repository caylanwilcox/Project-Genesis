'use client'

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { Activity, TrendingUp, Maximize2, Minimize2, Settings, ChevronDown } from 'lucide-react'
import { useRouter } from 'next/navigation'

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
  const [hoveredCandle, setHoveredCandle] = useState<CandleData | null>(null)
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null)
  const [useExternalData, setUseExternalData] = useState(false)
  const [chartPixelWidth, setChartPixelWidth] = useState<number>(0)
  const [interval, setInterval] = useState('15 min') // Default to 15 min to match 1D timeframe
  const [showIntervalDropdown, setShowIntervalDropdown] = useState(false)
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
    const padding = { top: 10, right: 80, bottom: 20, left: 10 }

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
    const maxVolume = Math.max(...visibleData.map(d => d.volume))

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

      // Time labels
      if (visibleData.length > 0) {
        const index = Math.min(visibleData.length - 1, Math.round((i / 6) * (visibleData.length - 1)))
        const ts = new Date(visibleData[index].time)

        // Format based on timeframe - show appropriate detail (New York timezone)
        let label = ''
        if (timeframe === '1Y' || timeframe === '5Y' || timeframe === 'All') {
          // For long timeframes, show year and month
          label = ts.toLocaleDateString('en-US', { month: 'short', year: 'numeric', timeZone: 'America/New_York' })
        } else if (timeframe === '1M' || timeframe === '3M' || timeframe === '6M' || timeframe === 'YTD') {
          // For medium timeframes, show month and day
          label = ts.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'America/New_York' })
        } else if (timeframe === '5D' || timeframe === '1D') {
          // For daily timeframes, show date and time
          label = ts.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'America/New_York' }) + ' ' +
                  ts.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true, timeZone: 'America/New_York' })
        } else {
          // For intraday, show just time
          label = ts.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true, timeZone: 'America/New_York' })
        }

        ctx.fillStyle = '#6b7280'
        ctx.font = '10px monospace'
        ctx.textAlign = 'center'
        ctx.fillText(label, x, rect.height - 5)
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

      // Price tag
      ctx.fillStyle = '#fbbf24'
      ctx.fillRect(rect.width - padding.right + 2, currentY - 11, padding.right - 7, 22)
      ctx.fillStyle = '#000000'
      ctx.font = 'bold 11px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(lastCandle.close.toFixed(2), rect.width - padding.right / 2 - 2, currentY + 4)
    }

    // Draw volume scale
    volCtx.fillStyle = '#6b7280'
    volCtx.font = '10px monospace'
    volCtx.textAlign = 'right'
    volCtx.fillText('Vol', volRect.width - 5, 12)
    volCtx.fillText((maxVolume / 1000000).toFixed(0) + 'M', volRect.width - 5, 25)

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
        if (timeframe === '1Y' || timeframe === '5Y' || timeframe === 'All') {
          // For long timeframes, show full date
          timeStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'America/New_York' })
        } else if (timeframe === '1M' || timeframe === '3M' || timeframe === '6M' || timeframe === 'YTD') {
          // For medium timeframes, show date
          timeStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'America/New_York' })
        } else {
          // For intraday, show date and time
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

        // Update hovered candle info
        setHoveredCandle(candle)
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
    setHoveredCandle(null)
    setIsPanning(false)
    setPanStart(null)
  }, [])

  // Close interval dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (showIntervalDropdown) {
        const target = e.target as HTMLElement
        if (!target.closest('.interval-dropdown-container')) {
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
  const intervals = ['1 min', '5 min', '15 min', '30 min', '1 hour', '1 day', '1 week', '1 month']

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
      }
      const mapped = map[newInterval]
      console.log('[ProfessionalChart] Mapped interval to timeframe:', mapped);
      if (mapped) onTimeframeChange(mapped, 'Custom') // Pass 'Custom' as display timeframe for interval changes
    }
  }

  return (
    <div ref={chartContainerRef} className="h-full flex flex-col bg-[#0d0e15]">
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
          {/* Range/Timeframe buttons */}
          {timeframes.map((tf) => (
            <button
              key={tf}
              onClick={() => handleTimeframeClick(tf)}
              className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                timeframe === tf
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              {tf}
            </button>
          ))}

          <div className="w-px h-4 bg-gray-700 mx-1" />

          {/* Chart type toggle (candles icon) */}
          <button className="text-gray-400 hover:text-white p-1" title="Chart Type">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <rect x="2" y="4" width="2" height="8" />
              <rect x="7" y="2" width="2" height="12" />
              <rect x="12" y="6" width="2" height="6" />
            </svg>
          </button>

          {/* Interval dropdown */}
          <div className="relative interval-dropdown-container">
            <button
              onClick={() => setShowIntervalDropdown(!showIntervalDropdown)}
              className="flex items-center gap-1 px-2 py-1 text-xs font-medium text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
            >
              Interval: {interval}
              <ChevronDown size={12} />
            </button>
            {showIntervalDropdown && (
              <div className="absolute right-0 top-full mt-1 bg-gray-800 border border-gray-700 rounded shadow-lg z-50 min-w-[120px]">
                {intervals.map((int) => (
                  <button
                    key={int}
                    onClick={() => handleIntervalChange(int)}
                    className={`w-full text-left px-3 py-2 text-xs hover:bg-gray-700 transition-colors first:rounded-t last:rounded-b ${
                      interval === int ? 'text-blue-400 bg-gray-700/50' : 'text-gray-300'
                    }`}
                  >
                    {int}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="w-px h-4 bg-gray-700 mx-1" />

          <button className="text-gray-400 hover:text-white p-1" title="Settings">
            <Settings size={14} />
          </button>
          <button
            onClick={toggleFullscreen}
            className="text-gray-400 hover:text-white p-1"
            title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
          >
            {isFullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
          </button>
        </div>
      </div>

      {/* Main chart area */}
      <div
        className="flex-grow relative overflow-hidden"
        style={{ cursor: isPanning ? 'grabbing' : 'grab' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
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

        {/* Price scale drag area - right side of chart */}
        <div
          ref={scaleAreaRef}
          className="absolute right-0 top-0 w-20 hover:bg-blue-500/5 transition-colors"
          style={{
            height: 'calc(100% - 80px)',
            cursor: isScaling ? 'ns-resize' : 'ns-resize',
            pointerEvents: 'auto'
          }}
          onMouseDown={handleScaleMouseDown}
          onMouseMove={handleScaleMouseMove}
          onDoubleClick={handleScaleDoubleClick}
          title="Drag to zoom price scale (double-click to reset)"
        >
          {/* Visual indicator for drag area */}
          <div className="absolute right-1 top-1/2 -translate-y-1/2 flex flex-col gap-1 opacity-30 hover:opacity-70 transition-opacity">
            <div className="w-3 h-0.5 bg-gray-400"></div>
            <div className="w-3 h-0.5 bg-gray-400"></div>
            <div className="w-3 h-0.5 bg-gray-400"></div>
          </div>

          {/* Zoom level indicator */}
          {priceScale !== 1.0 && (
            <div className="absolute right-2 bottom-4 bg-gray-800/90 border border-gray-700 rounded px-2 py-1 text-xs text-gray-300">
              {priceScale.toFixed(1)}x
            </div>
          )}
        </div>

        {/* Time scale drag area - bottom of chart (x-axis) */}
        <div
          className="absolute bottom-20 left-0 right-20 h-8 hover:bg-blue-500/5 transition-colors"
          style={{
            cursor: isTimeScaling ? 'ew-resize' : 'ew-resize',
            pointerEvents: 'auto'
          }}
          onMouseDown={handleTimeScaleMouseDown}
          onMouseMove={handleTimeScaleMouseMove}
          onDoubleClick={handleTimeScaleDoubleClick}
          title="Drag to zoom time scale (double-click to reset)"
        >
          {/* Visual indicator for drag area */}
          <div className="absolute left-1/2 -translate-x-1/2 bottom-1 flex gap-1 opacity-30 hover:opacity-70 transition-opacity">
            <div className="w-0.5 h-3 bg-gray-400"></div>
            <div className="w-0.5 h-3 bg-gray-400"></div>
            <div className="w-0.5 h-3 bg-gray-400"></div>
          </div>

          {/* Zoom level indicator */}
          {timeScale !== 1.0 && (
            <div className="absolute left-4 bottom-1 bg-gray-800/90 border border-gray-700 rounded px-2 py-1 text-xs text-gray-300">
              {timeScale.toFixed(1)}x
            </div>
          )}
        </div>
      </div>

      {/* Bottom info bar */}
      <div className="flex items-center justify-between px-4 py-1 border-t border-gray-800 text-xs">
        <div className="flex items-center gap-4 text-gray-400">
          <span>{currentTime.toLocaleTimeString('en-US', { hour12: false, timeZone: 'America/New_York' })} (ET)</span>
          <span className="text-gray-600">|</span>
          <span>{currentTime.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'America/New_York' })}</span>
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