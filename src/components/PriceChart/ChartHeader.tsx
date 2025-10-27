import React from 'react'
import { Activity, TrendingUp, Maximize2, Settings } from 'lucide-react'
import { useRouter } from 'next/navigation'

interface ChartHeaderProps {
  symbol: string
  chartType: 'price' | 'depth'
  timeframe: string
  timeframes: string[]
  onChartTypeChange: (type: 'price' | 'depth') => void
  onTimeframeChange: (tf: string) => void
}

export const ChartHeader: React.FC<ChartHeaderProps> = ({
  symbol,
  chartType,
  timeframe,
  timeframes,
  onChartTypeChange,
  onTimeframeChange,
}) => {
  const router = useRouter()

  return (
    <div className="flex items-center justify-between p-4 border-b border-gray-800/50 flex-shrink-0">
      <div className="flex items-center space-x-2">
        <button
          onClick={() => onChartTypeChange('price')}
          className={`px-3 py-1.5 text-sm rounded-lg transition-all duration-200 ${
            chartType === 'price' ? 'bg-gradient-to-r from-gray-700 to-gray-600 text-white shadow-sm' : 'bg-gray-800/30 text-gray-400 hover:text-white hover:bg-gray-800/50'
          }`}
        >
          Price chart
        </button>
        <button
          onClick={() => onChartTypeChange('depth')}
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
              onClick={() => onTimeframeChange(tf)}
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
  )
}
