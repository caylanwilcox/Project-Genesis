import type { CandleData } from '../../types'

/**
 * Generates mock candlestick data for testing/fallback
 * @param price - Starting price
 * @param candles - Number of candles to generate
 * @returns Array of mock candle data
 */
export function generateMockData(price = 100, candles = 120): CandleData[] {
  const now = Date.now()
  const mock: CandleData[] = []
  let lastClose = price || 100

  for (let i = candles - 1; i >= 0; i--) {
    const time = now - i * 60_000
    const volatility = lastClose * 0.01
    const change = (Math.random() - 0.5) * volatility
    const open = lastClose
    const close = Math.max(1, open + change)
    const high = Math.max(open, close) + Math.random() * volatility * 0.5
    const low = Math.min(open, close) - Math.random() * volatility * 0.5
    const volume = 500_000 + Math.random() * 750_000

    mock.push({ time, open, high, low, close, volume })
    lastClose = close
  }

  return mock
}
