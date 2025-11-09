/**
 * FVG (Fair Value Gap) Drawing Module
 * Detects and visualizes FVG patterns directly on the chart
 */

import { CandleData } from './types'

export interface FvgPattern {
  type: 'bullish' | 'bearish'
  startIndex: number
  gapHigh: number
  gapLow: number
  entryPrice: number
  tp1: number
  tp2: number
  tp3: number
  stopLoss: number
  validationScore: number
  volumeProfile: string
  marketStructure: string
}

/**
 * Detect FVG patterns from candle data
 */
export function detectFvgPatterns(data: CandleData[], options?: {
  recentOnly?: boolean,
  minGapPct?: number,
  maxGapPct?: number
}): FvgPattern[] {
  const patterns: FvgPattern[] = []

  if (data.length < 3) return patterns

  // Default: scan all data, but allow "recent only" mode (last 50 candles)
  const startIndex = options?.recentOnly && data.length > 50 ? data.length - 50 : 2
  const minGapPct = options?.minGapPct ?? 0.1
  const maxGapPct = options?.maxGapPct ?? 5.0

  for (let i = startIndex; i < data.length; i++) {
    const candle1 = data[i - 2]
    const candle2 = data[i - 1]
    const candle3 = data[i]

    // Debug: Log recent candles if we're near the end
    const isRecent = i >= data.length - 10

    // Bullish FVG: candle3.low > candle1.high (gap between them)
    if (candle3.low > candle1.high) {
      const gapSize = candle3.low - candle1.high
      const gapSizePct = (gapSize / candle2.close) * 100

      // Debug logging for recent candles
      if (isRecent) {
        console.log(`[FVG Debug] Bullish gap at index ${i}: ${gapSizePct.toFixed(3)}% (${gapSize.toFixed(2)})`,
          { candle1High: candle1.high, candle3Low: candle3.low, passed: gapSizePct >= minGapPct && gapSizePct <= maxGapPct })
      }

      // Filter: gap size between configurable thresholds
      if (gapSizePct >= minGapPct && gapSizePct <= maxGapPct) {
        const gapLow = candle1.high
        const gapHigh = candle3.low

        // Fibonacci retracement levels (price expected to retrace DOWN into the gap)
        // Entry would be ABOVE the gap, targeting a move DOWN to fill it
        const entryPrice = gapHigh // Enter at top of gap when price returns
        const tp1 = gapHigh - (gapSize * 0.382) // 38.2% fill from top
        const tp2 = gapHigh - (gapSize * 0.618) // 61.8% fill from top
        const tp3 = gapLow // Full fill (100% - reaches bottom of gap)
        const stopLoss = gapHigh + (gapSize * 0.1) // Stop above the gap

        // Volume profile analysis
        const avgVolume = (candle1.volume + candle2.volume + candle3.volume) / 3
        const volumeProfile = candle2.volume > avgVolume * 1.2 ? 'bell_curve' : 'flat'

        // Market structure (simplified)
        const candle2Range = candle2.high - candle2.low
        const avgRange = ((candle1.high - candle1.low) + (candle3.high - candle3.low)) / 2
        const marketStructure = candle2Range > avgRange * 2 ? 'balance_to_imbalance' : 'trending'

        // Validation score
        let score = 0.5
        if (candle2.volume > avgVolume * 1.2) score += 0.2 // Volume confirmation
        if (candle2Range > avgRange * 2) score += 0.15 // Market structure
        if (candle2.close > candle2.open && candle3.close > candle3.open) score += 0.15 // Directional

        patterns.push({
          type: 'bullish',
          startIndex: i - 2,
          gapHigh,
          gapLow,
          entryPrice,
          tp1,
          tp2,
          tp3,
          stopLoss,
          validationScore: Math.min(1.0, score),
          volumeProfile,
          marketStructure
        })
      }
    }

    // Bearish FVG: candle3.high < candle1.low (gap between them)
    if (candle3.high < candle1.low) {
      const gapSize = candle1.low - candle3.high
      const gapSizePct = (gapSize / candle2.close) * 100

      // Debug logging for recent candles
      if (isRecent) {
        console.log(`[FVG Debug] Bearish gap at index ${i}: ${gapSizePct.toFixed(3)}% (${gapSize.toFixed(2)})`,
          { candle1Low: candle1.low, candle3High: candle3.high, passed: gapSizePct >= minGapPct && gapSizePct <= maxGapPct })
      }

      // Filter: gap size between configurable thresholds
      if (gapSizePct >= minGapPct && gapSizePct <= maxGapPct) {
        const gapHigh = candle1.low
        const gapLow = candle3.high

        // Fibonacci retracement levels (price expected to retrace UP into the gap)
        // Entry would be BELOW the gap, targeting a move UP to fill it
        const entryPrice = gapLow // Enter at bottom of gap when price returns
        const tp1 = gapLow + (gapSize * 0.382) // 38.2% fill from bottom
        const tp2 = gapLow + (gapSize * 0.618) // 61.8% fill from bottom
        const tp3 = gapHigh // Full fill (100% - reaches top of gap)
        const stopLoss = gapLow - (gapSize * 0.1) // Stop below the gap

        // Volume profile analysis
        const avgVolume = (candle1.volume + candle2.volume + candle3.volume) / 3
        const volumeProfile = candle2.volume > avgVolume * 1.2 ? 'bell_curve' : 'flat'

        // Market structure (simplified)
        const candle2Range = candle2.high - candle2.low
        const avgRange = ((candle1.high - candle1.low) + (candle3.high - candle3.low)) / 2
        const marketStructure = candle2Range > avgRange * 2 ? 'balance_to_imbalance' : 'trending'

        // Validation score
        let score = 0.5
        if (candle2.volume > avgVolume * 1.2) score += 0.2 // Volume confirmation
        if (candle2Range > avgRange * 2) score += 0.15 // Market structure
        if (candle2.close < candle2.open && candle3.close < candle3.open) score += 0.15 // Directional

        patterns.push({
          type: 'bearish',
          startIndex: i - 2,
          gapHigh,
          gapLow,
          entryPrice,
          tp1,
          tp2,
          tp3,
          stopLoss,
          validationScore: Math.min(1.0, score),
          volumeProfile,
          marketStructure
        })
      }
    }
  }

  // Summary log
  if (patterns.length > 0) {
    const recentPatterns = patterns.filter(p => p.startIndex >= data.length - 50)
    console.log(`[FVG Summary] Found ${patterns.length} total patterns, ${recentPatterns.length} in last 50 candles`,
      { oldestIndex: patterns[0]?.startIndex, newestIndex: patterns[patterns.length - 1]?.startIndex, dataLength: data.length })
  }

  return patterns
}

/**
 * Draw FVG patterns on canvas
 */
export function drawFvgPatterns(
  ctx: CanvasRenderingContext2D,
  patterns: FvgPattern[],
  visibleData: CandleData[],
  padding: { left: number; top: number; right: number; bottom: number },
  chartWidth: number,
  chartHeight: number,
  minPrice: number,
  maxPrice: number,
  priceRange: number,
  baseWidth: number,
  visibleStart: number
) {
  // Match candle spacing logic with drawCandles(): use the larger of baseWidth or visibleData length
  const effectiveWidth = Math.max(baseWidth, visibleData.length)
  const candleWidth = chartWidth / effectiveWidth
  const leftOffset = (effectiveWidth - visibleData.length) * candleWidth

  const priceToY = (price: number) => {
    return padding.top + ((maxPrice - price) / priceRange) * chartHeight
  }

  const gapExtendCandles = 30 // Extend 30 candles forward from detection point

  patterns.forEach(pattern => {
    // Pattern indices are relative to the provided visibleData slice
    // So we use startIndex directly without subtracting visibleStart
    const localIndex = pattern.startIndex

    // Pattern is visible if it starts before the end of visible range AND extends into visible range
    const patternEndIndex = localIndex + gapExtendCandles
    if (patternEndIndex < 0 || localIndex >= visibleData.length + gapExtendCandles) return

    // Start from the first candle of the pattern
    const xStart = padding.left + leftOffset + localIndex * candleWidth
    const gapTop = priceToY(pattern.gapHigh)
    const gapBottom = priceToY(pattern.gapLow)
    const gapHeight = Math.abs(gapBottom - gapTop)

    // Calculate end index for gap box
    const endLocalIndex = Math.min(localIndex + gapExtendCandles, visibleData.length - 1)
    const xEnd = padding.left + leftOffset + (endLocalIndex + 1) * candleWidth // include last candle's full width
    const gapWidth = Math.max(xEnd - xStart, candleWidth * 15) // Minimum 15 candles wide

    // Ensure gap is visible
    if (gapHeight < 2) return

    // Draw semi-transparent gap rectangle with stronger color
    const color = pattern.type === 'bullish'
      ? 'rgba(34, 197, 94, 0.12)'
      : 'rgba(239, 68, 68, 0.12)'
    const borderColor = pattern.type === 'bullish'
      ? 'rgba(34, 197, 94, 0.7)'
      : 'rgba(239, 68, 68, 0.7)'

    ctx.fillStyle = color
    ctx.fillRect(xStart, gapTop, gapWidth, gapHeight)

    // Draw stronger border
    ctx.strokeStyle = borderColor
    ctx.lineWidth = 2
    ctx.setLineDash([8, 4])
    ctx.strokeRect(xStart, gapTop, gapWidth, gapHeight)
    ctx.setLineDash([])

    // Draw entry level (top or bottom edge of gap)
    const entryY = priceToY(pattern.entryPrice)
    ctx.strokeStyle = pattern.type === 'bullish' ? 'rgba(34, 197, 94, 0.9)' : 'rgba(239, 68, 68, 0.9)'
    ctx.lineWidth = 2
    ctx.setLineDash([])
    ctx.beginPath()
    ctx.moveTo(xStart, entryY)
    ctx.lineTo(xStart + gapWidth, entryY)
    ctx.stroke()

    // Draw TP levels with better visibility (INSIDE the gap only)
    const tp1Y = priceToY(pattern.tp1)
    const tp2Y = priceToY(pattern.tp2)
    const tp3Y = priceToY(pattern.tp3)

    // Verify TPs are inside gap zone
    const minY = Math.min(gapTop, gapBottom)
    const maxY = Math.max(gapTop, gapBottom)

    // TP1 - Most important (thicker line)
    if (tp1Y >= minY && tp1Y <= maxY) {
      ctx.strokeStyle = pattern.type === 'bullish'
        ? 'rgba(34, 197, 94, 0.8)'
        : 'rgba(239, 68, 68, 0.8)'
      ctx.lineWidth = 1.5
      ctx.setLineDash([6, 3])
      ctx.beginPath()
      ctx.moveTo(xStart, tp1Y)
      ctx.lineTo(xStart + gapWidth, tp1Y)
      ctx.stroke()
    }

    // TP2
    if (tp2Y >= minY && tp2Y <= maxY) {
      ctx.lineWidth = 1.5
      ctx.setLineDash([4, 4])
      ctx.beginPath()
      ctx.moveTo(xStart, tp2Y)
      ctx.lineTo(xStart + gapWidth, tp2Y)
      ctx.stroke()
    }

    // TP3 - Full gap fill (opposite edge)
    if (tp3Y >= minY && tp3Y <= maxY) {
      ctx.lineWidth = 1.5
      ctx.setLineDash([3, 5])
      ctx.beginPath()
      ctx.moveTo(xStart, tp3Y)
      ctx.lineTo(xStart + gapWidth, tp3Y)
      ctx.stroke()
    }

    ctx.setLineDash([])

    // Compact label - only show on left side of gap
    const labelX = xStart + 6
    const labelY = gapTop + 12

    // Single compact label background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.85)'
    ctx.fillRect(labelX - 3, labelY - 10, 110, 30)

    // FVG type label
    ctx.font = 'bold 10px -apple-system, sans-serif'
    ctx.fillStyle = pattern.type === 'bullish' ? '#22c55e' : '#ef4444'
    ctx.textAlign = 'left'
    const fvgLabel = pattern.type === 'bullish' ? '↑ FVG' : '↓ FVG'
    ctx.fillText(fvgLabel, labelX, labelY)

    // Gap size - compact
    ctx.font = '9px monospace'
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)'
    const gapSize = Math.abs(pattern.gapHigh - pattern.gapLow)
    const gapPct = ((gapSize / pattern.entryPrice) * 100).toFixed(1)
    ctx.fillText(`${gapPct}% gap`, labelX, labelY + 12)

    // Right side - show ENTRY and TP prices vertically aligned
    const rightLabelX = xStart + gapWidth - 65

    // Entry price (at entry line)
    ctx.fillStyle = 'rgba(0, 0, 0, 0.85)'
    ctx.fillRect(rightLabelX - 3, entryY - 10, 62, 12)
    ctx.font = 'bold 9px monospace'
    ctx.fillStyle = pattern.type === 'bullish' ? '#22c55e' : '#ef4444'
    ctx.fillText(`E: ${pattern.entryPrice.toFixed(2)}`, rightLabelX, entryY - 1)

    // Only show TPs if there's enough space (gap height > 30px)
    if (gapHeight > 30) {
      // TP1 label
      if (tp1Y >= minY && tp1Y <= maxY && Math.abs(tp1Y - entryY) > 15) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.85)'
        ctx.fillRect(rightLabelX - 3, tp1Y - 10, 62, 12)
        ctx.font = '9px monospace'
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
        ctx.fillText(`1: ${pattern.tp1.toFixed(2)}`, rightLabelX, tp1Y - 1)
      }

      // TP2 label
      if (tp2Y >= minY && tp2Y <= maxY && Math.abs(tp2Y - entryY) > 15 && Math.abs(tp2Y - tp1Y) > 15) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.85)'
        ctx.fillRect(rightLabelX - 3, tp2Y - 10, 62, 12)
        ctx.font = '9px monospace'
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
        ctx.fillText(`2: ${pattern.tp2.toFixed(2)}`, rightLabelX, tp2Y - 1)
      }

      // TP3 label (only if far from TP2)
      if (tp3Y >= minY && tp3Y <= maxY && Math.abs(tp3Y - tp2Y) > 15 && Math.abs(tp3Y - entryY) > 15) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.85)'
        ctx.fillRect(rightLabelX - 3, tp3Y - 10, 62, 12)
        ctx.font = '9px monospace'
        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)'
        ctx.fillText(`3: ${pattern.tp3.toFixed(2)}`, rightLabelX, tp3Y - 1)
      }
    }

    // Draw vertical line at detection point
    ctx.strokeStyle = borderColor
    ctx.lineWidth = 2
    ctx.setLineDash([])
    ctx.beginPath()
    ctx.moveTo(xStart, gapTop)
    ctx.lineTo(xStart, gapBottom)
    ctx.stroke()
  })
}
