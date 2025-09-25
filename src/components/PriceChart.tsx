'use client'

import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, LineSeries } from 'lightweight-charts';
import { Activity, TrendingUp, Maximize2, Settings } from 'lucide-react';

interface PriceChartProps {
  symbol: string;
}

export const PriceChart: React.FC<PriceChartProps> = ({ symbol }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const candlestickSeriesRef = useRef<any>(null);
  const [timeframe, setTimeframe] = useState('2h');
  const [chartType, setChartType] = useState<'price' | 'depth'>('price');

  useEffect(() => {
    if (chartContainerRef.current) {
      chartRef.current = createChart(chartContainerRef.current, {
        width: chartContainerRef.current.clientWidth,
        height: 500,
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

      // Mock data
      const mockData = generateMockCandlestickData();
      candlestickSeriesRef.current.setData(mockData);

      chartRef.current.timeScale().fitContent();

      const handleResize = () => {
        if (chartRef.current && chartContainerRef.current) {
          chartRef.current.applyOptions({
            width: chartContainerRef.current.clientWidth,
          });
        }
      };

      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
        chartRef.current?.remove();
      };
    }
  }, []);

  const generateMockCandlestickData = () => {
    const data = [] as { time: number; value: number }[];
    const basePrice = 0.0274;
    const startTime = Math.floor(Date.now() / 1000) - 86400 * 30;

    for (let i = 0; i < 500; i++) {
      const time = startTime + i * 3600;
      const volatility = 0.002;
      const random = Math.random();
      const change = (random - 0.5) * volatility;
      const value = basePrice + change + Math.sin(i * 0.1) * 0.001;

      data.push({
        time,
        value,
      });
    }

    return data;
  };

  const timeframes = ['6M', '3M', '1M', '5D', '1D', '4H', '1H', '2h'];

  return (
    <div className="trading-card rounded-xl transition-all duration-300">
      <div className="flex items-center justify-between p-4 border-b border-gray-800/50">
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
            <Maximize2 size={18} className="text-gray-400 cursor-pointer hover:text-white" />
          </div>
        </div>
      </div>

      <div ref={chartContainerRef} className="w-full" />

      <div className="flex items-center justify-between px-4 py-3 text-xs text-gray-400 border-t border-gray-800/50 bg-gray-900/30">
        <div className="flex items-center space-x-4">
          <span>O: 0.0275</span>
          <span>H: 0.0275</span>
          <span>L: 0.0274</span>
          <span>C: 0.0274</span>
          <span>VOL: 140.015K</span>
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