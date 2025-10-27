import { createChart, ColorType, LineSeries } from 'lightweight-charts'

export function setupLightweightChart(container: HTMLElement, containerHeight: number) {
  const chart = createChart(container, {
    width: container.clientWidth,
    height: containerHeight - 120,
    layout: {
      background: { type: ColorType.Solid, color: '#0B0E11' },
      textColor: '#848E9C',
    },
    grid: {
      vertLines: { color: '#1E222D' },
      horzLines: { color: '#1E222D' },
    },
    crosshair: { mode: 1 },
    rightPriceScale: {
      borderColor: '#1E222D',
      scaleMargins: { top: 0.1, bottom: 0.2 },
    },
    timeScale: {
      borderColor: '#1E222D',
      timeVisible: true,
    },
  })

  const series = chart.addSeries(LineSeries, {
    color: '#0ECB81',
    lineWidth: 2,
    crosshairMarkerVisible: true,
    crosshairMarkerRadius: 5,
    lastValueVisible: true,
    priceLineVisible: true,
  })

  return { chart, series }
}

export function addPriceLines(
  series: any,
  stopLoss?: number,
  entryPoint?: number,
  targets?: number[]
) {
  if (stopLoss) {
    series.createPriceLine({
      price: stopLoss,
      color: '#ef4444',
      lineWidth: 2,
      lineStyle: 2,
      axisLabelVisible: true,
      title: 'Stop Loss',
    })
  }

  if (entryPoint) {
    series.createPriceLine({
      price: entryPoint,
      color: '#06b6d4',
      lineWidth: 2,
      lineStyle: 0,
      axisLabelVisible: true,
      title: 'Entry',
    })
  }

  targets?.forEach((target, index) => {
    series.createPriceLine({
      price: target,
      color: '#10b981',
      lineWidth: index === 2 ? 2 : 1,
      lineStyle: 0,
      axisLabelVisible: true,
      title: `T${index + 1}`,
    })
  })
}
