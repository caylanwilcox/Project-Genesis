'use client'

import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, LineSeries } from 'lightweight-charts';
import { Activity, TrendingUp, Maximize2, Settings } from 'lucide-react';
import { useRouter } from 'next/navigation';

interface PriceChartProps {
  symbol: string;
  currentPrice?: number;
  stopLoss?: number;
  targets?: number[];
  entryPoint?: number;
}

export const PriceChart: React.FC<PriceChartProps> = ({
  symbol,
  currentPrice = 445.20,
  stopLoss,
  targets = [],
  entryPoint
}) => {
  const router = useRouter();
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const candlestickSeriesRef = useRef<any>(null);
  const [timeframe, setTimeframe] = useState('2h');
  const [chartType, setChartType] = useState<'price' | 'depth'>('price');

  useEffect(() => {
    if (chartContainerRef.current) {
      const containerHeight = chartContainerRef.current.parentElement?.clientHeight || 350;
      chartRef.current = createChart(chartContainerRef.current, {
        width: chartContainerRef.current.clientWidth,
        height: containerHeight - 120, // Subtract header and footer heights
        layout: {
          background: { type: ColorType.Solid, color: '#0B0E11' },
          textColor: '#848E9C',
        },
        grid: {
          vertLines: { color: '#1E222D' },
          horzLines: { color: '#1E222D' },
        },
        crosshair: {
          mode: 1,
        },
        rightPriceScale: {
          borderColor: '#1E222D',
          scaleMargins: {
            top: 0.1,
            bottom: 0.2,
          },
        },
        timeScale: {
          borderColor: '#1E222D',
          timeVisible: true,
        },
      });

      // Create a line series (v5 syntax)
      const series = chartRef.current.addSeries(LineSeries, {
        color: '#0ECB81',
        lineWidth: 2,
        crosshairMarkerVisible: true,
        crosshairMarkerRadius: 5,
        lastValueVisible: true,
        priceLineVisible: true,
      });
      candlestickSeriesRef.current = series;

      // Mock data - adjusted to be around the current price
      const mockData = generateMockCandlestickData();
      candlestickSeriesRef.current.setData(mockData);

      // Add stop loss line
      if (stopLoss) {
        candlestickSeriesRef.current.createPriceLine({
          price: stopLoss,
          color: '#ef4444',
          lineWidth: 2,
          lineStyle: 2, // Dashed
          axisLabelVisible: true,
          title: 'Stop Loss',
        });
      }

      // Add entry point line
      if (entryPoint) {
        candlestickSeriesRef.current.createPriceLine({
          price: entryPoint,
          color: '#06b6d4',
          lineWidth: 2,
          lineStyle: 0, // Solid
          axisLabelVisible: true,
          title: 'Entry',
        });
      }

      // Add target lines
      targets.forEach((target, index) => {
        candlestickSeriesRef.current.createPriceLine({
          price: target,
          color: '#10b981',
          lineWidth: index === 2 ? 2 : 1, // Target 3 is thicker
          lineStyle: 0, // Solid
          axisLabelVisible: true,
          title: `T${index + 1}`,
        });
      });

      chartRef.current.timeScale().fitContent();

      const handleResize = () => {
        if (chartRef.current && chartContainerRef.current) {
          const containerHeight = chartContainerRef.current.parentElement?.clientHeight || 350;
          chartRef.current.applyOptions({
            width: chartContainerRef.current.clientWidth,
            height: containerHeight - 120, // Maintain consistent height with header/footer
          });
        }
      };

      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
        chartRef.current?.remove();
      };
    }
  }, [stopLoss, entryPoint, targets]);

  const generateMockCandlestickData = () => {
    const data = [] as { time: number; value: number }[];
    // Use the current price or a default based on symbol
    const basePrice = currentPrice || (symbol === 'SPY' ? 445.20 : symbol === 'QQQ' ? 385.50 : symbol === 'IWM' ? 218.75 : 14.25);
    const startTime = Math.floor(Date.now() / 1000) - 86400 * 30;

    for (let i = 0; i < 500; i++) {
      const time = startTime + i * 3600;
      const volatility = basePrice * 0.005; // 0.5% volatility
      const random = Math.random();
      const change = (random - 0.5) * volatility;
      const value = basePrice + change + Math.sin(i * 0.1) * (basePrice * 0.002);

      data.push({
        time,
        value,
      });
    }

    return data;
  };

  const timeframes = ['6M', '3M', '1M', '5D', '1D', '4H', '1H', '2h'];

  return (
    <div className="trading-card rounded-xl transition-all duration-300 h-full flex flex-col">
      <div className="flex items-center justify-between p-4 border-b border-gray-800/50 flex-shrink-0">
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setChartType('price')}
            className={`px-3 py-1.5 text-sm rounded-lg transition-all duration-200 ${
              chartType === 'price' ? 'bg-gradient-to-r from-gray-700 to-gray-600 text-white shadow-sm' : 'bg-gray-800/30 text-gray-400 hover:text-white hover:bg-gray-800/50'
            }`}
          >
            Price chart
          </button>
          <button
            onClick={() => setChartType('depth')}
            className={`px-3 py-1.5 text-sm rounded-lg transition-all duration-200 ${
              chartType === 'depth' ? 'bg-gradient-to-r from-gray-700 to-gray-600 text-white shadow-sm' : 'bg-gray-800/30 text-gray-400 hover:text-white hover:bg-gray-800/50'
            }`}
          >
            Depth chart
          </button>
        </div>

        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1">
            {timeframes.map((tf) => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-2 py-1 text-xs rounded-lg transition-all duration-200 ${
                  timeframe === tf ? 'bg-gradient-to-r from-gray-700 to-gray-600 text-white shadow-sm' : 'bg-gray-800/30 text-gray-400 hover:text-white hover:bg-gray-800/50'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>

          <div className="flex items-center space-x-2">
            <Activity size={18} className="text-gray-400 cursor-pointer hover:text-white" />
            <TrendingUp size={18} className="text-gray-400 cursor-pointer hover:text-white" />
            <Settings size={18} className="text-gray-400 cursor-pointer hover:text-white" />
            <div title="Open full chart">
              <Maximize2
                size={18}
                className="text-gray-400 cursor-pointer hover:text-white"
                onClick={() => router.push(`/ticker/${symbol}`)}
              />
            </div>
          </div>
        </div>
      </div>

      <div ref={chartContainerRef} className="w-full flex-grow" />

      <div className="flex items-center justify-between px-4 py-3 text-xs text-gray-400 border-t border-gray-800/50 bg-gray-900/30 flex-shrink-0">
        <div className="flex items-center space-x-4">
          <span>O: {currentPrice.toFixed(2)}</span>
          <span>H: {(currentPrice * 1.002).toFixed(2)}</span>
          <span>L: {(currentPrice * 0.998).toFixed(2)}</span>
          <span>C: {currentPrice.toFixed(2)}</span>
          <span>VOL: {symbol === 'SPY' ? '85.2M' : symbol === 'QQQ' ? '42.7M' : symbol === 'IWM' ? '31.5M' : 'N/A'}</span>
        </div>
        <div className="flex items-center space-x-2">
          <span>15:08:21 (UTC-5)</span>
          <span>LOG</span>
          <span className="text-blue-400">AUTO</span>
        </div>
      </div>
    </div>
  );
};