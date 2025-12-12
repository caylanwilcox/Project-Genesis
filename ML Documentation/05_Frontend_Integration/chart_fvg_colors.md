# Chart FVG Color Integration

## Overview

FVG (Fair Value Gap) boxes on the chart are colored based on ML prediction confidence. This helps traders quickly identify high-probability setups.

## Color Scheme

### Fill Probability Based Colors

| Confidence | Color | Use Case |
|------------|-------|----------|
| >= 70% | `rgba(0, 255, 136, 0.3)` | High confidence - prioritize these trades |
| 50-70% | `rgba(255, 193, 7, 0.3)` | Medium confidence - trade with caution |
| < 50% | `rgba(255, 82, 82, 0.3)` | Low confidence - consider avoiding |

### Direction-Based Colors

| FVG Type | Alignment | Color |
|----------|-----------|-------|
| Bullish FVG | Bullish day | Green (enhanced) |
| Bullish FVG | Bearish day | Green (dimmed) |
| Bearish FVG | Bearish day | Red (enhanced) |
| Bearish FVG | Bullish day | Red (dimmed) |

## Implementation

### FVG Drawing Function

```typescript
function drawFVG(ctx: CanvasRenderingContext2D, fvg: FVG, prediction: FVGPrediction) {
  const { fill_probability, recommendation } = prediction

  // Determine base color
  let baseColor: string
  if (fvg.type === 'bullish') {
    baseColor = fill_probability >= 0.7
      ? 'rgba(0, 255, 136, 0.4)'  // Bright green
      : fill_probability >= 0.5
        ? 'rgba(0, 255, 136, 0.25)'  // Medium green
        : 'rgba(0, 255, 136, 0.1)'   // Dim green
  } else {
    baseColor = fill_probability >= 0.7
      ? 'rgba(255, 82, 82, 0.4)'   // Bright red
      : fill_probability >= 0.5
        ? 'rgba(255, 82, 82, 0.25)' // Medium red
        : 'rgba(255, 82, 82, 0.1)'  // Dim red
  }

  ctx.fillStyle = baseColor
  ctx.fillRect(x, y, width, height)

  // Draw probability label
  if (fill_probability >= 0.7) {
    ctx.fillStyle = '#00ff88'
    ctx.fillText(`${(fill_probability * 100).toFixed(0)}%`, x + 5, y + 15)
  }
}
```

## Daily Bias Integration

When the ML model predicts a strong directional bias:

```typescript
function getFVGOpacity(fvg: FVG, dailyBias: string): number {
  // Enhance FVGs aligned with daily bias
  if (dailyBias === 'BULLISH' && fvg.type === 'bullish') {
    return 1.0  // Full opacity
  }
  if (dailyBias === 'BEARISH' && fvg.type === 'bearish') {
    return 1.0  // Full opacity
  }

  // Dim FVGs against daily bias
  if (dailyBias === 'BULLISH' && fvg.type === 'bearish') {
    return 0.4  // Dimmed
  }
  if (dailyBias === 'BEARISH' && fvg.type === 'bullish') {
    return 0.4  // Dimmed
  }

  return 0.7  // Neutral bias - normal opacity
}
```

## Visual Hierarchy

```
High Priority (Most Visible)
┌────────────────────────────────────────┐
│  1. High conf FVG aligned with bias    │  Bright color, full opacity
├────────────────────────────────────────┤
│  2. High conf FVG against bias         │  Bright color, dimmed
├────────────────────────────────────────┤
│  3. Medium conf FVG aligned with bias  │  Medium color, full opacity
├────────────────────────────────────────┤
│  4. Medium conf FVG against bias       │  Medium color, dimmed
├────────────────────────────────────────┤
│  5. Low conf FVG                       │  Dim color, very transparent
└────────────────────────────────────────┘
Low Priority (Least Visible)
```

## File References

### FVG Drawing
```
src/components/ProfessionalChart/fvgDrawing.ts:1-200
```

### Chart Component
```
src/components/ProfessionalChart/MainChart.tsx
```

### FVG Detection
```
src/utils/fvgDetection.ts
```

---

Last Verified: December 8, 2025
