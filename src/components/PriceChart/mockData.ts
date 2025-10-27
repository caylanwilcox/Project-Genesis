export function generateMockCandlestickData(currentPrice: number, symbol: string) {
  const data = [] as { time: number; value: number }[]
  const basePrice = currentPrice || getDefaultPrice(symbol)
  const startTime = Math.floor(Date.now() / 1000) - 86400 * 30

  for (let i = 0; i < 500; i++) {
    const time = startTime + i * 3600
    const volatility = basePrice * 0.005
    const random = Math.random()
    const change = (random - 0.5) * volatility
    const value = basePrice + change + Math.sin(i * 0.1) * (basePrice * 0.002)

    data.push({ time, value })
  }

  return data
}

function getDefaultPrice(symbol: string): number {
  switch (symbol) {
    case 'SPY': return 445.20
    case 'QQQ': return 385.50
    case 'IWM': return 218.75
    default: return 14.25
  }
}
