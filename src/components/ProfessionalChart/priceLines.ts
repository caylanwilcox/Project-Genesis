type OverlayKind = 'target' | 'stop' | 'entry' | 'current'

export interface PriceTag {
  y: number
  label: string
  kind: OverlayKind
}

export function drawPriceLines(
  ctx: CanvasRenderingContext2D,
  rect: DOMRect,
  padding: any,
  chartHeight: number,
  minPrice: number,
  maxPrice: number,
  priceRange: number,
  stopLoss?: number,
  entryPoint?: number,
  targets: number[] = []
): PriceTag[] {
  const tags: PriceTag[] = []

  const drawLine = (price: number, color: string, label: string, dashed = false) => {
    const y = padding.top + ((maxPrice - price) / priceRange) * chartHeight

    ctx.strokeStyle = color
    ctx.lineWidth = 1
    if (dashed) ctx.setLineDash([10, 5])

    ctx.beginPath()
    ctx.moveTo(padding.left, y)
    ctx.lineTo(rect.width - padding.right, y)
    ctx.stroke()
    ctx.setLineDash([])

    return y
  }

  if (stopLoss) {
    const y = drawLine(stopLoss, '#ef444488', 'SL', true)
    tags.push({ y, label: stopLoss.toFixed(2), kind: 'stop' })
  }

  if (entryPoint) {
    const y = drawLine(entryPoint, '#06b6d488', 'ENTRY', false)
    tags.push({ y, label: entryPoint.toFixed(2), kind: 'entry' })
  }

  targets.forEach((target) => {
    const y = drawLine(target, '#22c55e88', 'T', false)
    tags.push({ y, label: target.toFixed(2), kind: 'target' })
  })

  return tags
}

export function drawCurrentPriceLine(
  ctx: CanvasRenderingContext2D,
  rect: DOMRect,
  padding: any,
  chartHeight: number,
  minPrice: number,
  maxPrice: number,
  priceRange: number,
  lastPrice: number,
  isVisible: boolean
): PriceTag | null {
  if (!isVisible) return null

  const currentY = padding.top + ((maxPrice - lastPrice) / priceRange) * chartHeight

  ctx.strokeStyle = '#fbbf24'
  ctx.lineWidth = 1
  ctx.setLineDash([4, 2])
  ctx.beginPath()
  ctx.moveTo(padding.left, currentY)
  ctx.lineTo(rect.width, currentY)
  ctx.stroke()
  ctx.setLineDash([])

  return { y: currentY, label: lastPrice.toFixed(2), kind: 'current' }
}
