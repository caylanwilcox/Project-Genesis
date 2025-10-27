import { useCallback, useEffect, useState } from 'react'

export function useChartData(currentPrice: number) {
  const [data, setData] = useState<{ time: number; value: number }[]>([])

  const generateData = useCallback(() => {
    const newData = []
    const basePrice = currentPrice
    const points = 100
    const now = Date.now()

    for (let i = 0; i < points; i++) {
      const time = now - (points - i) * 3600000
      const volatility = basePrice * 0.003
      const random = Math.random()
      const change = (random - 0.5) * volatility
      const trend = Math.sin(i * 0.1) * (basePrice * 0.002)
      const value = basePrice + change + trend

      newData.push({ time, value })
    }
    return newData
  }, [currentPrice])

  useEffect(() => {
    setData(generateData())
  }, [generateData])

  return data
}
