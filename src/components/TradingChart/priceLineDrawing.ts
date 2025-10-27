export function drawPriceLine(ctx: CanvasRenderingContext2D, rect: DOMRect, padding: any, chartHeight: number, price: number, color: string, label: string, maxPrice: number, priceRange: number, dashed = false, opacity = 1) {
  const y = padding.top + ((maxPrice - price) / priceRange) * chartHeight

  ctx.strokeStyle = color + Math.round(opacity * 255).toString(16).padStart(2, '0')
  ctx.lineWidth = 1
  if (dashed) ctx.setLineDash([8, 4])
  else ctx.setLineDash([])

  ctx.beginPath()
  ctx.moveTo(padding.left, y)
  ctx.lineTo(rect.width - padding.right, y)
  ctx.stroke()

  ctx.setLineDash([])
  ctx.fillStyle = color
  ctx.fillRect(padding.left, y - 9, 50, 18)
  ctx.fillStyle = '#ffffff'
  ctx.font = 'bold 10px system-ui, -apple-system, sans-serif'
  ctx.textAlign = 'center'
  ctx.fillText(price.toFixed(2), padding.left + 25, y + 3)

  const labelBg = dashed ? color + '99' : color
  ctx.fillStyle = labelBg
  ctx.fillRect(rect.width - padding.right + 2, y - 9, padding.right - 4, 18)
  ctx.fillStyle = '#ffffff'
  ctx.textAlign = 'center'
  ctx.fillText(label, rect.width - padding.right / 2, y + 3)
}

export function drawCurrentPriceTag(ctx: CanvasRenderingContext2D, rect: DOMRect, padding: any, chartHeight: number, lastPrice: number, maxPrice: number, priceRange: number) {
  const lastY = padding.top + ((maxPrice - lastPrice) / priceRange) * chartHeight

  ctx.strokeStyle = '#0ECB8133'
  ctx.lineWidth = 1
  ctx.setLineDash([4, 4])
  ctx.beginPath()
  ctx.moveTo(rect.width - padding.right, lastY)
  ctx.lineTo(rect.width, lastY)
  ctx.stroke()
  ctx.setLineDash([])

  const priceTagWidth = 55
  ctx.fillStyle = '#0ECB81'
  ctx.fillRect(rect.width - priceTagWidth, lastY - 11, priceTagWidth, 22)

  ctx.beginPath()
  ctx.moveTo(rect.width - priceTagWidth, lastY)
  ctx.lineTo(rect.width - priceTagWidth - 5, lastY - 5)
  ctx.lineTo(rect.width - priceTagWidth - 5, lastY + 5)
  ctx.closePath()
  ctx.fill()

  ctx.fillStyle = '#ffffff'
  ctx.font = 'bold 11px system-ui, -apple-system, sans-serif'
  ctx.textAlign = 'center'
  ctx.fillText(lastPrice.toFixed(2), rect.width - priceTagWidth / 2, lastY + 4)
}
