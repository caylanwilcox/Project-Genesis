import { useMemo } from 'react'
import { format } from 'date-fns'

export function useChartData(marketData: any, selectedSymbol: string | null, signals: any[]) {
  return useMemo(() => {
    if (!selectedSymbol) return []

    const data = marketData.get(selectedSymbol) || []
    return data.map((d: any) => ({
      ...d,
      date: format(new Date(d.date), 'MMM dd'),
      signals: signals.filter(
        s => s.symbol === selectedSymbol &&
        format(new Date(s.ts_emit), 'yyyy-MM-dd') === d.date
      ).length
    }))
  }, [marketData, selectedSymbol, signals])
}
