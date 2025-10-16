export interface ProfessionalChartProps {
  symbol: string
  currentPrice?: number
  stopLoss?: number
  targets?: number[]
  entryPoint?: number
  data?: CandleData[]
  onDataUpdate?: (data: CandleData[]) => void
  onTimeframeChange?: (tf: string, displayTf: string) => void
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
  intervals: ['1 min', '5 min', '15 min', '30 min', '1 hour', '1 day', '1 week', '1 month'],
  dataTimeframeMap: {
    '1D': '15m',
    '5D': '1h',
    '1M': '4h',
    '3M': '1d',
    '6M': '1d',
    'YTD': '1d',
    '1Y': '1d',
    '5Y': '1w',
    'All': '1M',
  } as TimeframeMap,
  intervalDisplayMap: {
    '1D': '15 min',
    '5D': '1 hour',
    '1M': '1 hour',
    '3M': '1 day',
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
