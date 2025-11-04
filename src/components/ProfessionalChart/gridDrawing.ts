import { CandleData } from './types'

interface GridConfig {
  ctx: CanvasRenderingContext2D
  rect: DOMRect
  padding: any
  chartWidth: number
  chartHeight: number
  minPrice: number
  maxPrice: number
  priceRange: number
  isNarrow: boolean
  gutter: number
}

export function drawPriceGrid(config: GridConfig) {
  const { ctx, rect, padding, chartHeight, maxPrice, priceRange, isNarrow, gutter } = config

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)'; ctx.lineWidth = 1; ctx.setLineDash([1, 1])

  const numTicks = 8
  for (let i = 0; i <= numTicks; i++) {
    const y = padding.top + (chartHeight / numTicks) * i
    ctx.beginPath(); ctx.moveTo(padding.left, y); ctx.lineTo(rect.width - padding.right, y); ctx.stroke()

    const price = maxPrice - (priceRange / numTicks) * i
    const labelX = rect.width - (gutter / 2)
    const fontSize = isNarrow ? 9 : 12
    ctx.font = `bold ${fontSize}px monospace`
    const text = price.toFixed(2)
    const textWidth = ctx.measureText(text).width

    ctx.fillStyle = 'rgba(13, 14, 21, 0.85)'
    ctx.fillRect(labelX - textWidth / 2 - 4, y - fontSize / 2 - 2, textWidth + 8, fontSize + 4)
    ctx.fillStyle = '#ffffff'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
    ctx.fillText(text, labelX, y)
  }
  ctx.setLineDash([])
}

export function drawTimeGrid(
  ctx: CanvasRenderingContext2D,
  volCtx: CanvasRenderingContext2D,
  rect: DOMRect,
  volRect: DOMRect,
  padding: any,
  chartWidth: number,
  chartHeight: number,
  volChartHeight: number,
  visibleData: CandleData[],
  dataTimeframe: string,
  displayTimeframe: string | undefined,
  isNarrow: boolean
) {
  if (visibleData.length === 0) return

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)'; ctx.lineWidth = 1; ctx.setLineDash([1, 1])
  const numLabels = Math.min(Math.max(3, Math.floor(visibleData.length / 15)), 8)

  for (let i = 0; i <= numLabels; i++) {
    const x = padding.left + (chartWidth / numLabels) * i

    ctx.beginPath(); ctx.moveTo(x, padding.top); ctx.lineTo(x, rect.height - padding.bottom); ctx.stroke()
    volCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)'; volCtx.lineWidth = 1; volCtx.setLineDash([1, 1])
    volCtx.beginPath(); volCtx.moveTo(x, 0); volCtx.lineTo(x, volChartHeight); volCtx.stroke()

    const index = Math.min(visibleData.length - 1, Math.round((i / numLabels) * (visibleData.length - 1)))
    const ts = new Date(visibleData[index].time)
    const label = formatTimeLabel(ts, dataTimeframe, displayTimeframe)
    const fontSize = isNarrow ? 10 : 11
    volCtx.font = `bold ${fontSize}px monospace`
    volCtx.textAlign = 'center'; volCtx.textBaseline = 'top'
    const textWidth = volCtx.measureText(label).width
    const bgHeight = fontSize + 6; const bgY = volChartHeight - 20

    volCtx.fillStyle = 'rgba(13, 14, 21, 0.9)'
    volCtx.fillRect(x - textWidth / 2 - 4, bgY, textWidth + 8, bgHeight)
    volCtx.fillStyle = '#ffffff'; volCtx.fillText(label, x, bgY + 3)
  }

  ctx.setLineDash([]); volCtx.setLineDash([])
}

function formatTimeLabel(ts: Date, dataTimeframe: string, displayTimeframe?: string): string {
  if (dataTimeframe === '1M') {
    return ts.toLocaleDateString('en-US', { month: 'short', year: 'numeric', timeZone: 'America/New_York' })
  } else if (dataTimeframe === '1w' || dataTimeframe === '1d') {
    return ts.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'America/New_York' })
  } else if (displayTimeframe === '5D' && dataTimeframe === '1h') {
    // For 5D timeframe with 1h bars, show date number
    return ts.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'America/New_York' })
  } else {
    return ts.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true, timeZone: 'America/New_York' })
  }
}
