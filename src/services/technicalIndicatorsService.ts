/**
 * Technical Indicators Service
 *
 * Calculates technical indicators for ML feature engineering:
 * - RSI (Relative Strength Index)
 * - MACD (Moving Average Convergence Divergence)
 * - ATR (Average True Range)
 * - SMA (Simple Moving Average)
 * - EMA (Exponential Moving Average)
 * - Bollinger Bands
 *
 * These indicators will be used as features for the ML model
 * to predict FVG win rates.
 */

export interface OHLCV {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface RSIResult {
  time: number
  rsi: number
}

export interface MACDResult {
  time: number
  macd: number
  signal: number
  histogram: number
}

export interface ATRResult {
  time: number
  atr: number
}

export interface BollingerBandsResult {
  time: number
  upper: number
  middle: number
  lower: number
  bandwidth: number
}

export interface AllIndicators {
  time: number
  rsi_14?: number
  macd?: number
  macd_signal?: number
  macd_histogram?: number
  atr_14?: number
  sma_20?: number
  sma_50?: number
  ema_12?: number
  ema_26?: number
  bb_upper?: number
  bb_middle?: number
  bb_lower?: number
  bb_bandwidth?: number
  volume_sma_20?: number
  volume_ratio?: number
}

export class TechnicalIndicatorsService {
  /**
   * Calculate RSI (Relative Strength Index)
   * RSI = 100 - (100 / (1 + RS))
   * RS = Average Gain / Average Loss
   */
  calculateRSI(data: OHLCV[], period: number = 14): RSIResult[] {
    if (data.length < period + 1) {
      return []
    }

    const results: RSIResult[] = []
    const gains: number[] = []
    const losses: number[] = []

    // Calculate price changes
    for (let i = 1; i < data.length; i++) {
      const change = data[i].close - data[i - 1].close
      gains.push(change > 0 ? change : 0)
      losses.push(change < 0 ? Math.abs(change) : 0)
    }

    // First RSI uses simple average
    let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period
    let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period

    for (let i = period; i < data.length; i++) {
      if (i > period) {
        // Subsequent RSI uses smoothed average (Wilder's smoothing)
        avgGain = (avgGain * (period - 1) + gains[i - 1]) / period
        avgLoss = (avgLoss * (period - 1) + losses[i - 1]) / period
      }

      const rs = avgLoss === 0 ? 100 : avgGain / avgLoss
      const rsi = 100 - (100 / (1 + rs))

      results.push({
        time: data[i].time,
        rsi: Math.round(rsi * 100) / 100,
      })
    }

    return results
  }

  /**
   * Calculate EMA (Exponential Moving Average)
   * EMA = Price * k + EMA(previous) * (1 - k)
   * k = 2 / (period + 1)
   */
  calculateEMA(data: OHLCV[], period: number): { time: number; ema: number }[] {
    if (data.length < period) {
      return []
    }

    const results: { time: number; ema: number }[] = []
    const k = 2 / (period + 1)

    // First EMA is SMA
    let ema = data.slice(0, period).reduce((sum, d) => sum + d.close, 0) / period

    results.push({
      time: data[period - 1].time,
      ema: Math.round(ema * 100) / 100,
    })

    for (let i = period; i < data.length; i++) {
      ema = data[i].close * k + ema * (1 - k)
      results.push({
        time: data[i].time,
        ema: Math.round(ema * 100) / 100,
      })
    }

    return results
  }

  /**
   * Calculate SMA (Simple Moving Average)
   */
  calculateSMA(data: OHLCV[], period: number): { time: number; sma: number }[] {
    if (data.length < period) {
      return []
    }

    const results: { time: number; sma: number }[] = []

    for (let i = period - 1; i < data.length; i++) {
      const sum = data.slice(i - period + 1, i + 1).reduce((s, d) => s + d.close, 0)
      results.push({
        time: data[i].time,
        sma: Math.round((sum / period) * 100) / 100,
      })
    }

    return results
  }

  /**
   * Calculate MACD (Moving Average Convergence Divergence)
   * MACD Line = 12-period EMA - 26-period EMA
   * Signal Line = 9-period EMA of MACD Line
   * Histogram = MACD Line - Signal Line
   */
  calculateMACD(
    data: OHLCV[],
    fastPeriod: number = 12,
    slowPeriod: number = 26,
    signalPeriod: number = 9
  ): MACDResult[] {
    if (data.length < slowPeriod + signalPeriod) {
      return []
    }

    const ema12 = this.calculateEMA(data, fastPeriod)
    const ema26 = this.calculateEMA(data, slowPeriod)

    // Align EMAs by time
    const ema26Times = new Set(ema26.map(e => e.time))
    const alignedEma12 = ema12.filter(e => ema26Times.has(e.time))

    // Calculate MACD line
    const macdLine: { time: number; macd: number }[] = []
    for (let i = 0; i < alignedEma12.length; i++) {
      const ema12Val = alignedEma12[i]
      const ema26Val = ema26.find(e => e.time === ema12Val.time)
      if (ema26Val) {
        macdLine.push({
          time: ema12Val.time,
          macd: Math.round((ema12Val.ema - ema26Val.ema) * 100) / 100,
        })
      }
    }

    if (macdLine.length < signalPeriod) {
      return []
    }

    // Calculate signal line (EMA of MACD)
    const results: MACDResult[] = []
    const k = 2 / (signalPeriod + 1)

    // First signal is SMA of MACD
    let signal = macdLine.slice(0, signalPeriod).reduce((sum, d) => sum + d.macd, 0) / signalPeriod

    for (let i = signalPeriod - 1; i < macdLine.length; i++) {
      if (i > signalPeriod - 1) {
        signal = macdLine[i].macd * k + signal * (1 - k)
      }

      const histogram = macdLine[i].macd - signal

      results.push({
        time: macdLine[i].time,
        macd: macdLine[i].macd,
        signal: Math.round(signal * 100) / 100,
        histogram: Math.round(histogram * 100) / 100,
      })
    }

    return results
  }

  /**
   * Calculate ATR (Average True Range)
   * TR = max(high - low, abs(high - prevClose), abs(low - prevClose))
   * ATR = EMA of TR
   */
  calculateATR(data: OHLCV[], period: number = 14): ATRResult[] {
    if (data.length < period + 1) {
      return []
    }

    const trueRanges: number[] = []

    // Calculate True Range for each bar
    for (let i = 1; i < data.length; i++) {
      const high = data[i].high
      const low = data[i].low
      const prevClose = data[i - 1].close

      const tr = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      )
      trueRanges.push(tr)
    }

    const results: ATRResult[] = []

    // First ATR is simple average
    let atr = trueRanges.slice(0, period).reduce((a, b) => a + b, 0) / period

    results.push({
      time: data[period].time,
      atr: Math.round(atr * 100) / 100,
    })

    // Subsequent ATR uses Wilder's smoothing
    for (let i = period; i < trueRanges.length; i++) {
      atr = (atr * (period - 1) + trueRanges[i]) / period
      results.push({
        time: data[i + 1].time,
        atr: Math.round(atr * 100) / 100,
      })
    }

    return results
  }

  /**
   * Calculate Bollinger Bands
   * Middle = 20-period SMA
   * Upper = Middle + (2 * Standard Deviation)
   * Lower = Middle - (2 * Standard Deviation)
   */
  calculateBollingerBands(
    data: OHLCV[],
    period: number = 20,
    stdDev: number = 2
  ): BollingerBandsResult[] {
    if (data.length < period) {
      return []
    }

    const results: BollingerBandsResult[] = []

    for (let i = period - 1; i < data.length; i++) {
      const slice = data.slice(i - period + 1, i + 1)
      const closes = slice.map(d => d.close)

      const sma = closes.reduce((a, b) => a + b, 0) / period
      const variance = closes.reduce((sum, c) => sum + Math.pow(c - sma, 2), 0) / period
      const sd = Math.sqrt(variance)

      const upper = sma + stdDev * sd
      const lower = sma - stdDev * sd
      const bandwidth = ((upper - lower) / sma) * 100

      results.push({
        time: data[i].time,
        upper: Math.round(upper * 100) / 100,
        middle: Math.round(sma * 100) / 100,
        lower: Math.round(lower * 100) / 100,
        bandwidth: Math.round(bandwidth * 100) / 100,
      })
    }

    return results
  }

  /**
   * Calculate Volume SMA
   */
  calculateVolumeSMA(data: OHLCV[], period: number = 20): { time: number; volumeSma: number }[] {
    if (data.length < period) {
      return []
    }

    const results: { time: number; volumeSma: number }[] = []

    for (let i = period - 1; i < data.length; i++) {
      const sum = data.slice(i - period + 1, i + 1).reduce((s, d) => s + d.volume, 0)
      results.push({
        time: data[i].time,
        volumeSma: Math.round(sum / period),
      })
    }

    return results
  }

  /**
   * Calculate all indicators at once for ML features
   */
  calculateAllIndicators(data: OHLCV[]): AllIndicators[] {
    if (data.length < 50) {
      return []
    }

    const rsi14 = this.calculateRSI(data, 14)
    const macd = this.calculateMACD(data, 12, 26, 9)
    const atr14 = this.calculateATR(data, 14)
    const sma20 = this.calculateSMA(data, 20)
    const sma50 = this.calculateSMA(data, 50)
    const ema12 = this.calculateEMA(data, 12)
    const ema26 = this.calculateEMA(data, 26)
    const bb = this.calculateBollingerBands(data, 20, 2)
    const volumeSma = this.calculateVolumeSMA(data, 20)

    // Create lookup maps for efficient joining
    const rsiMap = new Map(rsi14.map(r => [r.time, r.rsi]))
    const macdMap = new Map(macd.map(m => [m.time, m]))
    const atrMap = new Map(atr14.map(a => [a.time, a.atr]))
    const sma20Map = new Map(sma20.map(s => [s.time, s.sma]))
    const sma50Map = new Map(sma50.map(s => [s.time, s.sma]))
    const ema12Map = new Map(ema12.map(e => [e.time, e.ema]))
    const ema26Map = new Map(ema26.map(e => [e.time, e.ema]))
    const bbMap = new Map(bb.map(b => [b.time, b]))
    const volumeSmaMap = new Map(volumeSma.map(v => [v.time, v.volumeSma]))

    // Combine all indicators for each data point
    const results: AllIndicators[] = []

    for (const bar of data) {
      const time = bar.time
      const macdData = macdMap.get(time)
      const bbData = bbMap.get(time)
      const volSma = volumeSmaMap.get(time)

      results.push({
        time,
        rsi_14: rsiMap.get(time),
        macd: macdData?.macd,
        macd_signal: macdData?.signal,
        macd_histogram: macdData?.histogram,
        atr_14: atrMap.get(time),
        sma_20: sma20Map.get(time),
        sma_50: sma50Map.get(time),
        ema_12: ema12Map.get(time),
        ema_26: ema26Map.get(time),
        bb_upper: bbData?.upper,
        bb_middle: bbData?.middle,
        bb_lower: bbData?.lower,
        bb_bandwidth: bbData?.bandwidth,
        volume_sma_20: volSma,
        volume_ratio: volSma ? Math.round((bar.volume / volSma) * 100) / 100 : undefined,
      })
    }

    return results
  }
}

export const technicalIndicatorsService = new TechnicalIndicatorsService()
