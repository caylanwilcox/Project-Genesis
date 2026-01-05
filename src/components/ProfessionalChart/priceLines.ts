type OverlayKind = 'target' | 'stop' | 'entry' | 'current' | 'rangeHigh' | 'rangeLow' | 'rangeMid' | 'retestHigh' | 'retestLow'

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
  targets: number[] = [],
  rangeHigh?: number,
  rangeLow?: number,
  rangeMid?: number,
  retestHigh?: number,
  retestLow?: number
): PriceTag[] {
  const tags: PriceTag[] = []
  const labelOffset = 10 // Offset from left edge for labels

  // Helper to calculate Y position for a price
  const priceToY = (price: number) => padding.top + ((maxPrice - price) / priceRange) * chartHeight

  // Helper to draw a line with label on the line
  const drawLineWithLabel = (price: number, color: string, label: string, dashed = false) => {
    const y = priceToY(price)

    ctx.strokeStyle = color
    ctx.lineWidth = 1
    if (dashed) ctx.setLineDash([10, 5])

    ctx.beginPath()
    ctx.moveTo(padding.left, y)
    ctx.lineTo(rect.width - padding.right, y)
    ctx.stroke()
    ctx.setLineDash([])

    // Draw label on the line
    ctx.font = 'bold 10px monospace'
    ctx.fillStyle = color.replace('88', 'ff').replace('66', 'cc') // Make label more visible
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'

    // Draw background for label readability
    const textWidth = ctx.measureText(label).width
    ctx.fillStyle = 'rgba(17, 24, 39, 0.85)' // Dark background
    ctx.fillRect(padding.left + labelOffset - 2, y - 7, textWidth + 4, 14)

    // Draw label text
    ctx.fillStyle = color.replace('88', 'ff').replace('66', 'cc')
    ctx.fillText(label, padding.left + labelOffset, y)

    return y
  }

  // Helper to draw shaded zone between two prices
  const drawShadedZone = (topPrice: number, bottomPrice: number, color: string) => {
    const topY = priceToY(topPrice)
    const bottomY = priceToY(bottomPrice)
    ctx.fillStyle = color
    ctx.fillRect(padding.left, topY, rect.width - padding.right - padding.left, bottomY - topY)
  }

  // Draw range levels first (underneath other levels)
  if (rangeHigh && rangeLow) {
    // Draw shaded range zone
    drawShadedZone(rangeHigh, rangeLow, 'rgba(251, 191, 36, 0.08)')

    // Draw range high line with label
    const yHigh = drawLineWithLabel(rangeHigh, '#f59e0b88', 'RANGE HIGH', true)
    tags.push({ y: yHigh, label: rangeHigh.toFixed(2), kind: 'rangeHigh' })

    // Draw range low line with label
    const yLow = drawLineWithLabel(rangeLow, '#f59e0b88', 'RANGE LOW', true)
    tags.push({ y: yLow, label: rangeLow.toFixed(2), kind: 'rangeLow' })
  }

  if (rangeMid) {
    const y = drawLineWithLabel(rangeMid, '#fbbf2488', 'RANGE MID', true)
    tags.push({ y, label: rangeMid.toFixed(2), kind: 'rangeMid' })
  }

  // Draw retest levels (where price retests on range failure)
  if (retestHigh && retestHigh !== rangeHigh) {
    ctx.setLineDash([3, 3])
    const y = drawLineWithLabel(retestHigh, '#06b6d466', 'RETEST HIGH', true)
    tags.push({ y, label: retestHigh.toFixed(2), kind: 'retestHigh' })
    ctx.setLineDash([])
  }

  if (retestLow && retestLow !== rangeLow) {
    ctx.setLineDash([3, 3])
    const y = drawLineWithLabel(retestLow, '#06b6d466', 'RETEST LOW', true)
    tags.push({ y, label: retestLow.toFixed(2), kind: 'retestLow' })
    ctx.setLineDash([])
  }

  // Draw shaded zones for trading levels
  // Risk zone: Stop Loss to Entry (red shading)
  if (stopLoss && entryPoint) {
    if (stopLoss < entryPoint) {
      // Long position: SL below entry
      drawShadedZone(entryPoint, stopLoss, 'rgba(239, 68, 68, 0.06)')
    } else {
      // Short position: SL above entry
      drawShadedZone(stopLoss, entryPoint, 'rgba(239, 68, 68, 0.06)')
    }
  }

  // Profit zones: Entry to Targets (green shading with increasing opacity)
  if (entryPoint && targets.length > 0) {
    const sortedTargets = [...targets].sort((a, b) => a - b)
    const isLong = sortedTargets[0] > entryPoint

    if (isLong) {
      // Long position: targets above entry
      let prevLevel = entryPoint
      sortedTargets.forEach((target, index) => {
        if (target > entryPoint) {
          const opacity = 0.04 + (index * 0.02) // Increasing opacity for further targets
          drawShadedZone(target, prevLevel, `rgba(34, 197, 94, ${opacity})`)
          prevLevel = target
        }
      })
    } else {
      // Short position: targets below entry
      let prevLevel = entryPoint
      const reversedTargets = [...sortedTargets].reverse()
      reversedTargets.forEach((target, index) => {
        if (target < entryPoint) {
          const opacity = 0.04 + (index * 0.02)
          drawShadedZone(prevLevel, target, `rgba(34, 197, 94, ${opacity})`)
          prevLevel = target
        }
      })
    }
  }

  // Draw stop loss line with label
  if (stopLoss) {
    const y = drawLineWithLabel(stopLoss, '#ef444488', 'STOP LOSS', true)
    tags.push({ y, label: stopLoss.toFixed(2), kind: 'stop' })
  }

  // Draw entry line with label
  if (entryPoint) {
    const y = drawLineWithLabel(entryPoint, '#06b6d488', 'ENTRY', false)
    tags.push({ y, label: entryPoint.toFixed(2), kind: 'entry' })
  }

  // Draw target lines with labels
  targets.forEach((target, index) => {
    const label = targets.length > 1 ? `TARGET ${index + 1}` : 'TARGET'
    const y = drawLineWithLabel(target, '#22c55e88', label, false)
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

export function drawLowPriceMarker(
  ctx: CanvasRenderingContext2D,
  rect: DOMRect,
  padding: any,
  chartHeight: number,
  minPrice: number,
  maxPrice: number,
  priceRange: number,
  lowPrice: number,
  chartWidth: number,
  baseWidth: number,
  candleIndex: number
): void {
  const lowY = padding.top + ((maxPrice - lowPrice) / priceRange) * chartHeight

  // Calculate the X position of the specific candle
  const candleWidth = chartWidth / baseWidth
  const candleX = padding.left + (candleIndex + 0.5) * candleWidth

  // Draw triangle marker pointing to the low price (similar to Webull)
  const triangleSize = 6
  const markerOffset = 8 // Space between candle and marker

  ctx.fillStyle = '#ef4444'
  ctx.beginPath()
  ctx.moveTo(candleX, lowY + markerOffset)
  ctx.lineTo(candleX - triangleSize, lowY + markerOffset + triangleSize * 2)
  ctx.lineTo(candleX + triangleSize, lowY + markerOffset + triangleSize * 2)
  ctx.closePath()
  ctx.fill()

  // Draw price label below the triangle
  ctx.font = '10px monospace'
  ctx.fillStyle = '#ef4444'
  ctx.textAlign = 'center'
  ctx.fillText(lowPrice.toFixed(2), candleX, lowY + markerOffset + triangleSize * 2 + 12)
}

export function drawHighPriceMarker(
  ctx: CanvasRenderingContext2D,
  rect: DOMRect,
  padding: any,
  chartHeight: number,
  minPrice: number,
  maxPrice: number,
  priceRange: number,
  highPrice: number,
  chartWidth: number,
  baseWidth: number,
  candleIndex: number
): void {
  const highY = padding.top + ((maxPrice - highPrice) / priceRange) * chartHeight

  // Calculate the X position of the specific candle
  const candleWidth = chartWidth / baseWidth
  const candleX = padding.left + (candleIndex + 0.5) * candleWidth

  // Draw triangle marker pointing to the high price (similar to Webull)
  const triangleSize = 6
  const markerOffset = 8 // Space between candle and marker

  ctx.fillStyle = '#22c55e'
  ctx.beginPath()
  ctx.moveTo(candleX, highY - markerOffset)
  ctx.lineTo(candleX - triangleSize, highY - markerOffset - triangleSize * 2)
  ctx.lineTo(candleX + triangleSize, highY - markerOffset - triangleSize * 2)
  ctx.closePath()
  ctx.fill()

  // Draw price label above the triangle
  ctx.font = '10px monospace'
  ctx.fillStyle = '#22c55e'
  ctx.textAlign = 'center'
  ctx.fillText(highPrice.toFixed(2), candleX, highY - markerOffset - triangleSize * 2 - 4)
}
