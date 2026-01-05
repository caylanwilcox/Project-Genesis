/**
 * V6 ML Prediction data structure
 * Matches the output from /api/v2/trading_directions
 */
export interface V6Prediction {
  direction: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
  probability_a: number  // Close vs Open (Target A)
  probability_b: number  // Close vs 11AM (Target B)
  confidence: number     // 0-100
  session: 'early' | 'late'
  action: 'BUY_CALL' | 'BUY_PUT' | 'NO_TRADE'
}

export interface ProfessionalChartProps {
  symbol: string
  currentPrice?: number
  stopLoss?: number
  targets?: number[]
  entryPoint?: number
  data?: CandleData[]
  onDataUpdate?: (data: CandleData[]) => void
  onTimeframeChange?: (tf: string, displayTf: string, intervalLabel?: string) => void
  showFvg?: boolean
  onFvgCountChange?: (count: number) => void
  onVisibleBarCountChange?: (count: number, visibleData: CandleData[]) => void
  onLoadMoreData?: () => void
  isLoadingMore?: boolean
  fvgPercentage?: number
  // Optional: check if a timeframe is cached for seamless zoom transitions
  isTimeframeCached?: (displayTf: string) => boolean
  // ML Overlay props
  v6Prediction?: V6Prediction
  showMLOverlay?: boolean
  showSessionLines?: boolean
  price11am?: number
  // Range levels for balanced/choppy market structure
  rangeHigh?: number   // recent_high - top of acceptance range
  rangeLow?: number    // recent_low - bottom of acceptance range
  rangeMid?: number    // mid_point - center of acceptance range
  // Retest levels - where price will retest on range failure
  retestHigh?: number  // Next resistance if range high breaks
  retestLow?: number   // Next support if range low breaks
}

export interface CandleData {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface ChartState {
  timeframe: string
  chartType: 'candles' | 'line'
  data: CandleData[]
  hoveredCandle: CandleData | null
  mousePos: { x: number; y: number } | null
  useExternalData: boolean
  chartPixelWidth: number
  interval: string
  showIntervalDropdown: boolean
  isPanning: boolean
  panOffset: number
  panStart: { x: number; offset: number } | null
  visibleRange: { start: number; end: number }
  currentTime: Date
  priceScale: number
  isScaling: boolean
  scaleStart: { y: number; scale: number } | null
  timeScale: number
  isTimeScaling: boolean
  timeScaleStart: { x: number; scale: number } | null
}

export interface ChartPadding {
  top: number
  right: number
  bottom: number
  left: number
}

export interface TimeframeMap {
  [key: string]: string
}

export const TIMEFRAME_CONFIGS = {
  timeframes: ['1D', '5D', '1M', '3M', '6M', 'YTD', '1Y', '5Y', 'All'],
  intervals: [
    '1 min',
    '5 min',
    '15 min',
    '30 min',
    '1 hour',
    '2 hour',
    '4 hour',
    '1 day',
    '1 week',
    '1 month',
  ],
  dataTimeframeMap: {
    '1D': '5m',   // intraday granularity for single day
    '5D': '30m',  // five-day swing with 30m bars
    '1M': '1h',   // one month uses hourly bars (~140)
    '3M': '1h',   // quarter view with 1h aggregation
    '6M': '1d',   // half-year uses daily bars
    'YTD': '1d',  // YTD daily bars
    '1Y': '1d',   // yearly daily bars
    '5Y': '1w',   // multi-year weekly bars
    'All': '1M',  // full history monthly bars
  } as TimeframeMap,
  intervalDisplayMap: {
    '1D': '5 min',
    '5D': '30 min',
    '1M': '1 hour',
    '3M': '1 hour',
    '6M': '1 day',
    'YTD': '1 day',
    '1Y': '1 day',
    '5Y': '1 week',
    'All': '1 month',
  } as TimeframeMap,
  intervalToTimeframeMap: {
    '1 min': '1m',
    '5 min': '5m',
    '15 min': '15m',
    '30 min': '30m',
    '1 hour': '1h',
    '2 hour': '2h',
    '4 hour': '4h',
    '1 day': '1d',
    '1 week': '1w',
    '1 month': '1M',
  } as TimeframeMap,
}

export const CHART_PADDING: ChartPadding = {
  top: 10,
  right: 80,
  bottom: 20,
  left: 10,
}
