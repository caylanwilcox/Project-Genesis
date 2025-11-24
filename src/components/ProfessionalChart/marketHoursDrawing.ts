/**
 * Market Hours Background Drawing
 * Draws colored background overlays for market open (green) and closed (red) hours
 */

import { CandleData, ChartPadding } from './types'
import { getMarketHoursSegments, getMarketHoursSegmentsWithDuration, MarketHoursConfig, DEFAULT_MARKET_HOURS } from '@/utils/marketHours'

/**
 * Draw market hours background overlays
 * @param ctx Canvas rendering context
 * @param visibleData Visible candle data
 * @param padding Chart padding
 * @param chartWidth Chart width in pixels
 * @param chartHeight Chart height in pixels
 * @param baseWidth Base width for candle calculations
 * @param config Market hours configuration
 */
export function drawMarketHoursBackground(
  ctx: CanvasRenderingContext2D,
  visibleData: CandleData[],
  padding: ChartPadding,
  chartWidth: number,
  chartHeight: number,
  baseWidth: number,
  config: MarketHoursConfig = DEFAULT_MARKET_HOURS
) {
  if (visibleData.length === 0) return

  // Estimate bar duration (median diff of consecutive bars). Fallback to 1m.
  let barDurationMs = 60_000
  if (visibleData.length >= 3) {
    const diffs: number[] = []
    const count = Math.min(visibleData.length - 1, 10)
    for (let i = 0; i < count; i++) {
      const dt = visibleData[i + 1].time - visibleData[i].time
      if (dt > 0) diffs.push(dt)
    }
    if (diffs.length > 0) {
      diffs.sort((a, b) => a - b)
      barDurationMs = diffs[Math.floor(diffs.length / 2)]
    }
  }

  // Use duration-aware segmentation so 1h bars overlapping 9:30–4:00 render as open.
  const segments = getMarketHoursSegmentsWithDuration(visibleData, barDurationMs, config)

  // Calculate candle width to match candle drawing (no offset)
  const candleWidth = chartWidth / visibleData.length
  const leftOffset = 0  // No offset needed since baseWidth = visibleData.length
  const chartTop = padding.top
  const chartBottom = padding.top + chartHeight

  segments.forEach(segment => {
    const { startIndex, endIndex, isOpen } = segment

    // Calculate x positions for this segment
    const startX = padding.left + leftOffset + (startIndex * candleWidth)
    const endX = padding.left + leftOffset + ((endIndex + 1) * candleWidth)
    const width = endX - startX

    // Set color based on market status
    // Use very subtle opacity - just a hint of color, not a dark overlay
    ctx.fillStyle = isOpen
      ? 'rgba(34, 197, 94, 0.03)'  // Green with 3% opacity for open hours (very subtle)
      : 'rgba(239, 68, 68, 0.03)'   // Red with 3% opacity for closed hours (very subtle)

    // Draw rectangle covering the time segment
    ctx.fillRect(startX, chartTop, width, chartHeight)
  })
}

/**
 * Draw market hours legend/indicator
 * @param ctx Canvas rendering context
 * @param rect Canvas bounding rect
 * @param padding Chart padding
 * @param config Market hours configuration
 */
export function drawMarketHoursLegend(
  ctx: CanvasRenderingContext2D,
  rect: DOMRect,
  padding: ChartPadding,
  config: MarketHoursConfig = DEFAULT_MARKET_HOURS
) {
  const legendX = padding.left + 10
  const legendY = padding.top + 10

  // Draw legend background
  ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
  ctx.fillRect(legendX, legendY, 140, 50)

  // Draw border
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)'
  ctx.lineWidth = 1
  ctx.strokeRect(legendX, legendY, 140, 50)

  // Draw open hours indicator
  ctx.fillStyle = 'rgba(34, 197, 94, 0.5)'
  ctx.fillRect(legendX + 5, legendY + 8, 12, 12)
  ctx.fillStyle = '#ffffff'
  ctx.font = '10px monospace'
  ctx.fillText('Open (9:30am-4pm ET)', legendX + 22, legendY + 18)

  // Draw closed hours indicator
  ctx.fillStyle = 'rgba(239, 68, 68, 0.5)'
  ctx.fillRect(legendX + 5, legendY + 30, 12, 12)
  ctx.fillStyle = '#ffffff'
  ctx.fillText('Closed', legendX + 22, legendY + 40)
}
