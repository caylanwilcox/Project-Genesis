import { CandleData, ChartPadding } from './types'

export function formatTimeLabel(timestamp: number, timeframe: string): string {
  const ts = new Date(timestamp)
  const opts = { timeZone: 'America/New_York' } as const

  if (timeframe === '1Y' || timeframe === '5Y' || timeframe === 'All') {
    return ts.toLocaleDateString('en-US', { ...opts, month: 'short', year: 'numeric' })
  } else if (timeframe === '1M' || timeframe === '3M' || timeframe === '6M' || timeframe === 'YTD') {
    return ts.toLocaleDateString('en-US', { ...opts, month: 'short', day: 'numeric' })
  } else if (timeframe === '5D' || timeframe === '1D') {
    const date = ts.toLocaleDateString('en-US', { ...opts, month: 'short', day: 'numeric' })
    const time = ts.toLocaleTimeString('en-US', { ...opts, hour: 'numeric', minute: '2-digit', hour12: true })
    return `${date} ${time}`
  } else {
    return ts.toLocaleTimeString('en-US', { ...opts, hour: 'numeric', minute: '2-digit', hour12: true })
  }
}

export function formatTooltipTime(timestamp: number, timeframe: string): string {
  const date = new Date(timestamp)
  const opts = { timeZone: 'America/New_York' } as const

  if (timeframe === '1Y' || timeframe === '5Y' || timeframe === 'All') {
    return date.toLocaleDateString('en-US', { ...opts, month: 'short', day: 'numeric', year: 'numeric' })
  } else if (timeframe === '1M' || timeframe === '3M' || timeframe === '6M' || timeframe === 'YTD') {
    return date.toLocaleDateString('en-US', { ...opts, month: 'short', day: 'numeric' })
  } else {
    const d = date.toLocaleDateString('en-US', { ...opts, month: 'short', day: 'numeric' })
    const t = date.toLocaleTimeString('en-US', { ...opts, hour: 'numeric', minute: '2-digit', hour12: true })
    return `${d} ${t}`
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

export function drawGrid(ctx: CanvasRenderingContext2D, width: number, height: number, padding: ChartPadding) {
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

