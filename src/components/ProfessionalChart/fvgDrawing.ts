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
  gapSizePct: number
  entryPrice: number
  tp1: number
  tp2: number
  tp3: number
  stopLoss: number
  validationScore: number
  volumeProfile: string
  marketStructure: string
  fib382?: number  // Fibonacci 38.2% level within gap
  fib50?: number   // Fibonacci 50% level within gap
  fib618?: number  // Fibonacci 61.8% level within gap
  expanded?: boolean  // Whether the FVG is expanded to show full details
  id?: string  // Unique identifier for click tracking
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
  const minGapPct = options?.minGapPct ?? 0.2
  const maxGapPct = options?.maxGapPct ?? 5.0

  for (let i = startIndex; i < data.length; i++) {
    const candle1 = data[i - 2]
    const candle2 = data[i - 1]
    const candle3 = data[i]

    // Debug: Log recent candles if we're near the end
    const isRecent = i >= data.length - 10

    // Bullish FVG: candle3.low > candle1.high (upward imbalance - bullish continuation)
    if (candle3.low > candle1.high) {
      const gapSize = candle3.low - candle1.high
      const gapSizePct = (gapSize / candle2.close) * 100

      // Debug logging for recent candles (disabled to reduce console noise)
      // if (isRecent) {
      //   console.log(`[FVG Debug] Bullish FVG at index ${i}: ${gapSizePct.toFixed(3)}% (${gapSize.toFixed(2)})`,
      //     { candle1High: candle1.high, candle3Low: candle3.low, passed: gapSizePct >= minGapPct && gapSizePct <= maxGapPct })
      // }

      // Filter: gap size between configurable thresholds
      if (gapSizePct >= minGapPct && gapSizePct <= maxGapPct) {
        const gapLow = candle1.high  // Bottom of imbalance zone
        const gapHigh = candle3.low  // Top of imbalance zone

        // GAP-ZONE FIBONACCI STRATEGY: Trade WITHIN the gap for 5% returns
        // Bullish FVG: Enter at Fibonacci retracement levels within the gap
        const fib382 = gapLow + (gapSize * 0.382)  // 38.2% Fibonacci level
        const fib50 = gapLow + (gapSize * 0.50)    // 50% Fibonacci level (optimal entry)
        const fib618 = gapLow + (gapSize * 0.618)  // 61.8% Fibonacci level

        // Entry at 50% Fibonacci retracement (middle of gap)
        const entryPrice = fib50

        // Risk = distance from entry to stop below gap
        const riskAmount = gapSize * 0.75  // Risk: 0.75x gap size below entry

        // FIBONACCI-BASED TARGETS for 2:1 to 5:1 returns:
        // TP1: 100% gap fill (gap high) - First resistance
        // TP2: 150% Fibonacci extension (1.5x gap size above gap high)
        // TP3: 200-250% Fibonacci extension (targeting 5% total return)
        const tp1 = gapHigh  // 100% gap fill
        const tp2 = gapHigh + (gapSize * 0.5)  // 150% extension
        const tp3 = gapHigh + (gapSize * 1.0)  // 200% extension (5% move target)

        // Stop loss BELOW the gap (invalidation)
        const stopLoss = gapLow - (gapSize * 0.5)  // SL below gap low

        // Volume profile analysis (Fabio emphasizes order flow)
        const avgVolume = (candle1.volume + candle2.volume + candle3.volume) / 3
        const volumeProfile = candle2.volume > avgVolume * 1.5 ? 'bell_curve' : 'flat'

        // Market structure (balance → imbalance shift)
        const candle2Range = candle2.high - candle2.low
        const avgRange = ((candle1.high - candle1.low) + (candle3.high - candle3.low)) / 2
        const marketStructure = candle2Range > avgRange * 2.5 ? 'balance_to_imbalance' : 'trending'

        // Enhanced validation score (Fabio's criteria)
        let score = 0.3 // Lower base - require proof

        // Volume aggression
        if (candle2.volume > avgVolume * 1.5) score += 0.25
        else if (candle2.volume > avgVolume * 1.2) score += 0.15

        // Market structure
        if (candle2Range > avgRange * 2.5) score += 0.25
        else if (candle2Range > avgRange * 2.0) score += 0.15

        // Directional momentum
        if (candle2.close > candle2.open && candle3.close > candle3.open) score += 0.20

        patterns.push({
          type: 'bullish',
          startIndex: i - 2,
          gapHigh,
          gapLow,
          gapSizePct,
          entryPrice,
          tp1,
          tp2,
          tp3,
          stopLoss,
          validationScore: Math.min(1.0, score),
          volumeProfile,
          marketStructure,
          fib382,
          fib50,
          fib618,
          expanded: false,
          id: `fvg-bullish-${i - 2}`
        })
      }
    }

    // Bearish FVG: candle3.high < candle1.low (gap DOWN leaving imbalance zone ABOVE)
    // FABIO'S CONTINUATION STRATEGY: This is DOWNWARD imbalance → Trade SHORT on pullback UP to gap
    if (candle3.high < candle1.low) {
      const gapSize = candle1.low - candle3.high
      const gapSizePct = (gapSize / candle2.close) * 100

      // Debug logging for recent candles (disabled to reduce console noise)
      // if (isRecent) {
      //   console.log(`[FVG Debug] Bearish gap at index ${i}: ${gapSizePct.toFixed(3)}% (${gapSize.toFixed(2)})`,
      //     { candle1Low: candle1.low, candle3High: candle3.high, passed: gapSizePct >= minGapPct && gapSizePct <= maxGapPct })
      // }

      // Filter: gap size between configurable thresholds
      if (gapSizePct >= minGapPct && gapSizePct <= maxGapPct) {
        const gapHigh = candle1.low  // Top of imbalance zone
        const gapLow = candle3.high   // Bottom of imbalance zone

        // GAP-ZONE FIBONACCI STRATEGY: Trade WITHIN the gap for 5% returns
        // Bearish FVG: Enter at Fibonacci retracement levels within the gap
        const fib382 = gapHigh - (gapSize * 0.382)  // 38.2% Fibonacci level
        const fib50 = gapHigh - (gapSize * 0.50)    // 50% Fibonacci level (optimal entry)
        const fib618 = gapHigh - (gapSize * 0.618)  // 61.8% Fibonacci level

        // Entry at 50% Fibonacci retracement (middle of gap)
        const entryPrice = fib50

        // Risk = distance from entry to stop above gap
        const riskAmount = gapSize * 0.75  // Risk: 0.75x gap size above entry

        // FIBONACCI-BASED TARGETS for 2:1 to 5:1 returns:
        // TP1: 100% gap fill (gap low) - First support
        // TP2: 150% Fibonacci extension (1.5x gap size below gap low)
        // TP3: 200-250% Fibonacci extension (targeting 5% total return)
        const tp1 = gapLow  // 100% gap fill
        const tp2 = gapLow - (gapSize * 0.5)  // 150% extension
        const tp3 = gapLow - (gapSize * 1.0)  // 200% extension (5% move target)

        const stopLoss = gapHigh + (gapSize * 0.5) // SL above gap high (invalidation)

        // Volume & Structure Analysis (Fabio's criteria)
        const avgVolume = (candle1.volume + candle2.volume + candle3.volume) / 3
        const candle2Range = candle2.high - candle2.low
        const avgRange = ((candle1.high - candle1.low) + (candle3.high - candle3.low)) / 2

        // Enhanced confidence scoring aligned with Fabio
        let score = 0.3 // Lower base - require proof of quality

        // Volume Aggression (order flow strength)
        if (candle2.volume > avgVolume * 1.5) score += 0.25      // Strong aggressive flow
        else if (candle2.volume > avgVolume * 1.2) score += 0.15 // Moderate flow

        // Market Structure (balance → imbalance shift)
        if (candle2Range > avgRange * 2.5) score += 0.25      // Strong imbalance creation
        else if (candle2Range > avgRange * 2.0) score += 0.15 // Moderate imbalance

        // Directional Momentum (continuation alignment)
        const bearishMomentum = candle2.close < candle2.open && candle3.close < candle3.open
        if (bearishMomentum) score += 0.20 // Directional confirmation DOWN

        // Volume profile & market structure metadata
        const volumeProfile = candle2.volume > avgVolume * 1.5 ? 'bell_curve' : 'flat'
        const marketStructure = candle2Range > avgRange * 2.5 ? 'balance_to_imbalance' : 'trending'

        patterns.push({
          type: 'bearish',
          startIndex: i - 2,
          gapHigh,
          gapLow,
          gapSizePct,
          entryPrice,
          tp1,
          tp2,
          tp3,
          stopLoss,
          validationScore: Math.min(1.0, score),
          volumeProfile,
          marketStructure,
          fib382,
          fib50,
          fib618,
          expanded: false,
          id: `fvg-bearish-${i - 2}`
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
 * Helper to check if a point is inside a circle
 */
export function isPointInFvgDot(
  x: number,
  y: number,
  dotX: number,
  dotY: number,
  radius: number
): boolean {
  const dx = x - dotX
  const dy = y - dotY
  return Math.sqrt(dx * dx + dy * dy) <= radius
}

/**
 * Find which FVG pattern was clicked (if any)
 */
export function findClickedFvg(
  patterns: FvgPattern[],
  clickX: number,
  clickY: number
): FvgPattern | null {
  for (const pattern of patterns) {
    // If collapsed, check dot click
    if (!pattern.expanded) {
      const dotX = (pattern as any).dotX
      const dotY = (pattern as any).dotY
      const dotRadius = (pattern as any).dotRadius

      if (dotX !== undefined && dotY !== undefined && dotRadius !== undefined) {
        if (isPointInFvgDot(clickX, clickY, dotX, dotY, dotRadius)) {
          return pattern
        }
      }
    } else {
      // If expanded, check if click is within the box
      const boxLeft = (pattern as any).boxLeft
      const boxTop = (pattern as any).boxTop
      const boxRight = (pattern as any).boxRight
      const boxBottom = (pattern as any).boxBottom

      if (boxLeft !== undefined && boxTop !== undefined && boxRight !== undefined && boxBottom !== undefined) {
        if (clickX >= boxLeft && clickX <= boxRight && clickY >= boxTop && clickY <= boxBottom) {
          return pattern
        }
      }
    }
  }
  return null
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
  // Calculate candle width to match candle drawing (no offset)
  const candleWidth = chartWidth / visibleData.length
  const leftOffset = 0  // No offset needed since baseWidth = visibleData.length

  const priceToY = (price: number) => {
    return padding.top + ((maxPrice - price) / priceRange) * chartHeight
  }

  const gapExtendCandles = 30 // Extend 30 candles forward from detection point

  patterns.forEach(pattern => {
    // Pattern indices are relative to the FULL dataset.
    // Convert to local index within the current visible slice by subtracting visibleStart.
    const localIndex = pattern.startIndex - visibleStart

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

    // Store dot/box position for click detection
    const dotRadius = pattern.expanded ? 0 : 8  // 8px radius for easier clicking
    const dotX = xStart + candleWidth / 2
    const dotY = (gapTop + gapBottom) / 2

    ;(pattern as any).dotX = dotX
    ;(pattern as any).dotY = dotY
    ;(pattern as any).dotRadius = dotRadius
    ;(pattern as any).boxLeft = xStart
    ;(pattern as any).boxTop = gapTop
    ;(pattern as any).boxRight = xStart + gapWidth
    ;(pattern as any).boxBottom = gapBottom

    // If not expanded, just draw a dot indicator
    if (!pattern.expanded) {
      const dotColor = pattern.type === 'bullish' ? '#22c55e' : '#ef4444'

      // Draw dot with border
      ctx.beginPath()
      ctx.arc(dotX, dotY, dotRadius, 0, 2 * Math.PI)
      ctx.fillStyle = dotColor
      ctx.fill()
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)'
      ctx.lineWidth = 2
      ctx.stroke()

      console.log(`[FVG Draw] Drew dot for ${pattern.id} at (${dotX.toFixed(1)}, ${dotY.toFixed(1)}) radius=${dotRadius}`)

      return // Skip drawing full box if not expanded
    }

    console.log(`[FVG Draw] Drew expanded box for ${pattern.id}`)

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

    // Draw TP levels - CONTINUATION STRATEGY (targets are OUTSIDE gap zone)
    const tp1Y = priceToY(pattern.tp1)
    const tp2Y = priceToY(pattern.tp2)
    const tp3Y = priceToY(pattern.tp3)

    // Draw TPs as continuation targets (no need to check if inside gap - they're beyond it)
    // Only draw if within visible chart area
    const chartTop = padding.top
    const chartBottom = padding.top + chartHeight

    // TP1 - 0.5:1 R:R (closest target)
    if (tp1Y >= chartTop && tp1Y <= chartBottom) {
      ctx.strokeStyle = pattern.type === 'bullish'
        ? 'rgba(34, 197, 94, 0.7)'
        : 'rgba(239, 68, 68, 0.7)'
      ctx.lineWidth = 1.5
      ctx.setLineDash([6, 3])
      ctx.beginPath()
      ctx.moveTo(xStart, tp1Y)
      ctx.lineTo(xStart + gapWidth, tp1Y)
      ctx.stroke()
    }

    // TP2 - 1:1 R:R
    if (tp2Y >= chartTop && tp2Y <= chartBottom) {
      ctx.strokeStyle = pattern.type === 'bullish'
        ? 'rgba(34, 197, 94, 0.6)'
        : 'rgba(239, 68, 68, 0.6)'
      ctx.lineWidth = 1.5
      ctx.setLineDash([4, 4])
      ctx.beginPath()
      ctx.moveTo(xStart, tp2Y)
      ctx.lineTo(xStart + gapWidth, tp2Y)
      ctx.stroke()
    }

    // TP3 - 2:1 R:R (furthest target)
    if (tp3Y >= chartTop && tp3Y <= chartBottom) {
      ctx.strokeStyle = pattern.type === 'bullish'
        ? 'rgba(34, 197, 94, 0.5)'
        : 'rgba(239, 68, 68, 0.5)'
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

    // Expanded label background for 3 lines
    ctx.fillStyle = 'rgba(0, 0, 0, 0.85)'
    ctx.fillRect(labelX - 3, labelY - 10, 110, 42)

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

    // Confidence score
    const confidencePct = (pattern.validationScore * 100).toFixed(0)
    const confidenceColor = pattern.validationScore >= 0.85 ? '#22c55e' :
                           pattern.validationScore >= 0.65 ? '#eab308' : '#ef4444'
    ctx.fillStyle = confidenceColor
    ctx.fillText(`${confidencePct}% conf`, labelX, labelY + 24)

    // Right side - show ENTRY and TP prices vertically aligned
    const rightLabelX = xStart + gapWidth - 65

    // Entry price (at entry line)
    ctx.fillStyle = 'rgba(0, 0, 0, 0.85)'
    ctx.fillRect(rightLabelX - 3, entryY - 10, 62, 12)
    ctx.font = 'bold 9px monospace'
    ctx.fillStyle = pattern.type === 'bullish' ? '#22c55e' : '#ef4444'
    ctx.fillText(`E: ${pattern.entryPrice.toFixed(2)}`, rightLabelX, entryY - 1)

    // Show TP labels if they're visible in chart area and have enough spacing
    // TP1 label
    if (tp1Y >= chartTop && tp1Y <= chartBottom && Math.abs(tp1Y - entryY) > 15) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.85)'
      ctx.fillRect(rightLabelX - 3, tp1Y - 10, 62, 12)
      ctx.font = '9px monospace'
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
      ctx.fillText(`1: ${pattern.tp1.toFixed(2)}`, rightLabelX, tp1Y - 1)
    }

    // TP2 label
    if (tp2Y >= chartTop && tp2Y <= chartBottom && Math.abs(tp2Y - entryY) > 15 && Math.abs(tp2Y - tp1Y) > 15) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.85)'
      ctx.fillRect(rightLabelX - 3, tp2Y - 10, 62, 12)
      ctx.font = '9px monospace'
      ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
      ctx.fillText(`2: ${pattern.tp2.toFixed(2)}`, rightLabelX, tp2Y - 1)
    }

    // TP3 label (only if far from TP2)
    if (tp3Y >= chartTop && tp3Y <= chartBottom && Math.abs(tp3Y - tp2Y) > 15 && Math.abs(tp3Y - entryY) > 15) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.85)'
      ctx.fillRect(rightLabelX - 3, tp3Y - 10, 62, 12)
      ctx.font = '9px monospace'
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)'
      ctx.fillText(`3: ${pattern.tp3.toFixed(2)}`, rightLabelX, tp3Y - 1)
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
