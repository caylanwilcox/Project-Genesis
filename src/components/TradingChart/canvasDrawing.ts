export function drawGrid(ctx: CanvasRenderingContext2D, rect: DOMRect, padding: any, chartWidth: number, chartHeight: number) {
  ctx.strokeStyle = 'rgba(30, 34, 45, 0.5)'
  ctx.lineWidth = 0.5
  ctx.setLineDash([2, 4])

  for (let i = 0; i <= 8; i++) {
    const y = padding.top + (chartHeight / 8) * i
    ctx.beginPath()
    ctx.moveTo(padding.left, y)
    ctx.lineTo(rect.width - padding.right, y)
    ctx.stroke()
  }

  for (let i = 0; i <= 6; i++) {
    const x = padding.left + (chartWidth / 6) * i
    ctx.beginPath()
    ctx.moveTo(x, padding.top)
    ctx.lineTo(x, rect.height - padding.bottom)
    ctx.stroke()
  }
  ctx.setLineDash([])
}

export function drawPriceLabels(ctx: CanvasRenderingContext2D, rect: DOMRect, padding: any, chartHeight: number, minPrice: number, maxPrice: number, priceRange: number) {
  for (let i = 0; i <= 8; i += 2) {
    const y = padding.top + (chartHeight / 8) * i
    const price = maxPrice - (priceRange / 8) * i
    ctx.fillStyle = '#848E9C'
    ctx.font = '11px system-ui, -apple-system, sans-serif'
    ctx.textAlign = 'left'
    ctx.fillText(price.toFixed(2), rect.width - padding.right + 5, y + 3)
  }
}

export function drawAreaChart(ctx: CanvasRenderingContext2D, data: any[], rect: DOMRect, padding: any, chartWidth: number, chartHeight: number, maxPrice: number, priceRange: number) {
  const gradient = ctx.createLinearGradient(0, padding.top, 0, rect.height - padding.bottom)
  gradient.addColorStop(0, 'rgba(14, 203, 129, 0.3)')
  gradient.addColorStop(1, 'rgba(14, 203, 129, 0.01)')

  ctx.beginPath()
  data.forEach((point, i) => {
    const x = padding.left + (chartWidth / (data.length - 1)) * i
    const y = padding.top + ((maxPrice - point.value) / priceRange) * chartHeight
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  })
  ctx.lineTo(rect.width - padding.right, rect.height - padding.bottom)
  ctx.lineTo(padding.left, rect.height - padding.bottom)
  ctx.closePath()
  ctx.fillStyle = gradient
  ctx.fill()

  ctx.strokeStyle = '#0ECB81'
  ctx.lineWidth = 2
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.beginPath()

  data.forEach((point, i) => {
    const x = padding.left + (chartWidth / (data.length - 1)) * i
    const y = padding.top + ((maxPrice - point.value) / priceRange) * chartHeight
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  })
  ctx.stroke()
}
