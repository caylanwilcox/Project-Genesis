import { CandleData, ChartPadding } from './types'

export function formatTimeLabel(timestamp: number, timeframe: string): string {
  const ts = new Date(timestamp)

  if (timeframe === '1Y' || timeframe === '5Y' || timeframe === 'All') {
    return ts.toLocaleDateString('en-US', { month: 'short', year: 'numeric', timeZone: 'America/New_York' })
  } else if (timeframe === '1M' || timeframe === '3M' || timeframe === '6M' || timeframe === 'YTD') {
    return ts.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'America/New_York' })
  } else if (timeframe === '5D' || timeframe === '1D') {
    return ts.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'America/New_York' }) + ' ' +
           ts.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true, timeZone: 'America/New_York' })
  } else {
    return ts.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true, timeZone: 'America/New_York' })
  }
}

export function formatTooltipTime(timestamp: number, timeframe: string): string {
  const date = new Date(timestamp)

  if (timeframe === '1Y' || timeframe === '5Y' || timeframe === 'All') {
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'America/New_York' })
  } else if (timeframe === '1M' || timeframe === '3M' || timeframe === '6M' || timeframe === 'YTD') {
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'America/New_York' })
  } else {
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'America/New_York' }) + ' ' +
           date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true, timeZone: 'America/New_York' })
  }
}

export function calculatePriceRange(
  visibleData: CandleData[],
  priceScale: number,
  stopLoss?: number,
  entryPoint?: number,
  targets: number[] = []
): { minPrice: number; maxPrice: number; priceRange: number } {
  const allPrices = visibleData.flatMap(d => [d.high, d.low])
  if (stopLoss) allPrices.push(stopLoss)
  if (entryPoint) allPrices.push(entryPoint)
  allPrices.push(...targets)

  const dataMinPrice = Math.min(...allPrices)
  const dataMaxPrice = Math.max(...allPrices)
  const dataPriceRange = dataMaxPrice - dataMinPrice
  const priceCenter = (dataMinPrice + dataMaxPrice) / 2

  const scaledRange = dataPriceRange / priceScale
  const minPrice = priceCenter - (scaledRange / 2) * 1.002
  const maxPrice = priceCenter + (scaledRange / 2) * 1.002
  const priceRange = maxPrice - minPrice

  return { minPrice, maxPrice, priceRange }
}

export function drawGrid(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  padding: ChartPadding
) {
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)'
  ctx.lineWidth = 1

  for (let i = 0; i <= 6; i++) {
    const y = padding.top + (i / 6) * (height - padding.top - padding.bottom)
    ctx.beginPath()
    ctx.moveTo(padding.left, y)
    ctx.lineTo(width, y)
    ctx.stroke()
  }

  for (let i = 0; i <= 6; i++) {
    const x = padding.left + (i / 6) * (width - padding.left - padding.right)
    ctx.beginPath()
    ctx.moveTo(x, padding.top)
    ctx.lineTo(x, height - padding.bottom)
    ctx.stroke()
  }
}

export function drawPriceLine(
  ctx: CanvasRenderingContext2D,
  price: number,
  color: string,
  label: string,
  minPrice: number,
  maxPrice: number,
  priceRange: number,
  width: number,
  height: number,
  padding: ChartPadding,
  isDashed: boolean = true
) {
  const y = padding.top + ((maxPrice - price) / priceRange) * (height - padding.top - padding.bottom)

  ctx.strokeStyle = color
  ctx.lineWidth = 1
  if (isDashed) ctx.setLineDash([4, 2])

  ctx.beginPath()
  ctx.moveTo(padding.left, y)
  ctx.lineTo(width - padding.right, y)
  ctx.stroke()

  if (isDashed) ctx.setLineDash([])

  ctx.fillStyle = color
  ctx.fillRect(width - padding.right + 2, y - 11, padding.right - 7, 22)
  ctx.fillStyle = '#000000'
  ctx.font = 'bold 10px monospace'
  ctx.textAlign = 'center'
  ctx.fillText(label, width - padding.right / 2 - 2, y + 3)
}

export function drawCandle(
  ctx: CanvasRenderingContext2D,
  candle: CandleData,
  x: number,
  candleWidth: number,
  candleSpacing: number,
  minPrice: number,
  maxPrice: number,
  priceRange: number,
  padding: ChartPadding,
  chartHeight: number
) {
  const isGreen = candle.close >= candle.open

  const highY = padding.top + ((maxPrice - candle.high) / priceRange) * chartHeight
  const lowY = padding.top + ((maxPrice - candle.low) / priceRange) * chartHeight
  const openY = padding.top + ((maxPrice - candle.open) / priceRange) * chartHeight
  const closeY = padding.top + ((maxPrice - candle.close) / priceRange) * chartHeight

  ctx.strokeStyle = isGreen ? '#22c55e' : '#ef4444'
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(x, highY)
  ctx.lineTo(x, lowY)
  ctx.stroke()

  const bodyTop = Math.min(openY, closeY)
  const bodyHeight = Math.abs(closeY - openY)

  if (bodyHeight < 1) {
    ctx.beginPath()
    ctx.moveTo(x - candleSpacing / 2, openY)
    ctx.lineTo(x + candleSpacing / 2, openY)
    ctx.stroke()
  } else {
    ctx.fillStyle = isGreen ? '#22c55e' : '#ef4444'
    ctx.fillRect(x - candleSpacing / 2, bodyTop, candleSpacing, bodyHeight)
  }
}

export function drawVolumeBar(
  ctx: CanvasRenderingContext2D,
  candle: CandleData,
  x: number,
  candleWidth: number,
  maxVolume: number,
  volChartHeight: number
) {
  const isGreen = candle.close >= candle.open
  const barHeight = (candle.volume / maxVolume) * (volChartHeight - 10)

  ctx.fillStyle = isGreen ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)'
  ctx.fillRect(x - candleWidth / 3, volChartHeight - barHeight, candleWidth * 0.66, barHeight)
}
