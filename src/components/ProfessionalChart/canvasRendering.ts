import { CandleData } from './types'

export function setupCanvas(canvas: HTMLCanvasElement): CanvasRenderingContext2D | null {
  const ctx = canvas.getContext('2d')
  if (!ctx) return null

  const rect = canvas.getBoundingClientRect()
  const dpr = window.devicePixelRatio || 1

  canvas.width = rect.width * dpr
  canvas.height = rect.height * dpr
  canvas.style.width = rect.width + 'px'
  canvas.style.height = rect.height + 'px'
  ctx.scale(dpr, dpr)

  return ctx
}

export function clearCanvas(ctx: CanvasRenderingContext2D, width: number, height: number, color: string = '#0d0e15') {
  ctx.fillStyle = color
  ctx.fillRect(0, 0, width, height)
}

export function calculatePadding(parentWidth: number) {
  const isNarrow = parentWidth < 420
  const cssGutter = getComputedStyle(document.documentElement).getPropertyValue('--chart-y-axis-gutter').trim()
  const gutter = cssGutter.endsWith('px') ? parseInt(cssGutter) : (isNarrow ? 56 : 80)
  const padding = isNarrow
    ? { top: 6, right: gutter, bottom: 14, left: 6 }
    : { top: 10, right: gutter, bottom: 20, left: 10 }

  return { padding, gutter, isNarrow }
}

export function calculatePriceRange(visibleData: CandleData[], priceScale: number, priceOffset: number = 0) {
  const highs = visibleData.map(d => d.high)
  const lows = visibleData.map(d => d.low)
  const visibleMin = Math.min(...lows)
  const visibleMax = Math.max(...highs)
  const rawRange = Math.max(1e-6, visibleMax - visibleMin)
  const margin = rawRange * 0.05
  const rangedMin = visibleMin - margin
  const rangedMax = visibleMax + margin
  const centered = (rangedMin + rangedMax) / 2
  const scaledRange = (rangedMax - rangedMin) / priceScale

  // Apply vertical offset
  const minPrice = centered - scaledRange / 2 + priceOffset
  const maxPrice = centered + scaledRange / 2 + priceOffset
  const priceRange = Math.max(1e-6, maxPrice - minPrice)

  return { minPrice, maxPrice, priceRange }
}

export function formatVolume(v: number): string {
  const abs = Math.abs(v)
  if (abs >= 1e9) return (abs / 1e9 >= 10 ? (abs / 1e9).toFixed(0) : (abs / 1e9).toFixed(1)) + 'B'
  if (abs >= 1e6) return (abs / 1e6 >= 10 ? (abs / 1e6).toFixed(0) : (abs / 1e6).toFixed(1)) + 'M'
  if (abs >= 1e3) return (abs / 1e3 >= 10 ? (abs / 1e3).toFixed(0) : (abs / 1e3).toFixed(1)) + 'K'
  return abs.toFixed(0)
}
