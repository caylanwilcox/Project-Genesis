'use client'

import React, { useEffect, useRef, useState } from 'react'
import { setupLightweightChart, addPriceLines } from './PriceChart/chartSetup'
import { generateMockCandlestickData } from './PriceChart/mockData'
import { ChartHeader } from './PriceChart/ChartHeader'
import { ChartFooter } from './PriceChart/ChartFooter'

interface PriceChartProps {
  symbol: string
  currentPrice?: number
  stopLoss?: number
  targets?: number[]
  entryPoint?: number
}

export const PriceChart: React.FC<PriceChartProps> = ({
  symbol,
  currentPrice = 445.20,
  stopLoss,
  targets = [],
  entryPoint
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<any>(null)
  const candlestickSeriesRef = useRef<any>(null)
  const [timeframe, setTimeframe] = useState('2h')
  const [chartType, setChartType] = useState<'price' | 'depth'>('price')
  const timeframes = ['6M', '3M', '1M', '5D', '1D', '4H', '1H', '2h']

  useEffect(() => {
    if (!chartContainerRef.current) return

    const containerHeight = chartContainerRef.current.parentElement?.clientHeight || 350
    const { chart, series } = setupLightweightChart(chartContainerRef.current, containerHeight)
    chartRef.current = chart
    candlestickSeriesRef.current = series

    const mockData = generateMockCandlestickData(currentPrice, symbol)
    series.setData(mockData)
    addPriceLines(series, stopLoss, entryPoint, targets)
    chart.timeScale().fitContent()

    const handleResize = () => {
      if (chart && chartContainerRef.current) {
        const height = chartContainerRef.current.parentElement?.clientHeight || 350
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: height - 120,
        })
      }
    }

    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
      chart?.remove()
    }
  }, [stopLoss, entryPoint, targets, currentPrice, symbol])

  return (
    <div className="trading-card rounded-xl transition-all duration-300 h-full flex flex-col">
      <ChartHeader
        symbol={symbol}
        chartType={chartType}
        timeframe={timeframe}
        timeframes={timeframes}
        onChartTypeChange={setChartType}
        onTimeframeChange={setTimeframe}
      />
      <div ref={chartContainerRef} className="w-full flex-grow" />
      <ChartFooter currentPrice={currentPrice} symbol={symbol} />
    </div>
  )
}