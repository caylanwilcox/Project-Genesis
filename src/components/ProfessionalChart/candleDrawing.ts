import { CandleData } from './types'

export function drawCandles(
  ctx: CanvasRenderingContext2D,
  visibleData: CandleData[],
  padding: any,
  chartWidth: number,
  chartHeight: number,
  minPrice: number,
  maxPrice: number,
  priceRange: number,
  baseWidth: number = 100 // Use fixed base width for consistent spacing
) {
  if (visibleData.length === 0) return

  // Use the LARGER of baseWidth or actual data length to ensure candles fit
  // This prevents candles from running off the chart when auto-fitting
  const effectiveWidth = Math.max(baseWidth, visibleData.length)
  const candleWidth = chartWidth / effectiveWidth
  const candleSpacing = candleWidth * 0.8

  visibleData.forEach((candle, i) => {
    const x = padding.left + i * candleWidth + candleWidth / 2

    // Ensure candle stays within chart boundaries
    if (x < padding.left || x > padding.left + chartWidth) return

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
    const bodyHeight = Math.abs(closeY - openY) || 1

    ctx.fillStyle = isGreen ? '#22c55e' : '#ef4444'
    ctx.fillRect(x - candleSpacing / 2, bodyTop, candleSpacing, bodyHeight)
  })
}

export function drawVolumeBars(
  volCtx: CanvasRenderingContext2D,
  visibleData: CandleData[],
  candleWidth: number,
  volChartHeight: number,
  maxVolume: number,
  padding: any,
  chartWidth?: number
) {
  if (visibleData.length === 0) return

  const volBarMaxHeight = volChartHeight - 25

  visibleData.forEach((candle, i) => {
    const x = padding.left + i * candleWidth + candleWidth / 2

    // Ensure volume bar stays within chart boundaries
    if (chartWidth && (x < padding.left || x > padding.left + chartWidth)) return

    const isGreen = candle.close >= candle.open

    // Ensure volume is a valid number
    const volume = candle.volume || 0
    const volHeight = maxVolume > 0 ? (volume / maxVolume) * volBarMaxHeight : 0

    // Histogram style: thin bars with small spacing
    const barWidth = candleWidth * 0.85 // Slightly wider bars for histogram look
    const barX = x - barWidth / 2

    // Draw volume histogram bar
    if (volHeight > 0.1) {
      // Use more opaque colors for better visibility
      volCtx.fillStyle = isGreen ? 'rgba(34, 197, 94, 0.75)' : 'rgba(239, 68, 68, 0.75)'
      volCtx.fillRect(barX, volChartHeight - 25 - volHeight, barWidth, volHeight)

      // Add subtle border for definition
      volCtx.strokeStyle = isGreen ? 'rgba(34, 197, 94, 0.9)' : 'rgba(239, 68, 68, 0.9)'
      volCtx.lineWidth = 0.5
      volCtx.strokeRect(barX, volChartHeight - 25 - volHeight, barWidth, volHeight)
    }
  })
}
