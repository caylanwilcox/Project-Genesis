import React from 'react'

interface TradingChartFooterProps {
  currentPrice: number
  symbol: string
}

export const TradingChartFooter: React.FC<TradingChartFooterProps> = ({ currentPrice, symbol }) => {
  const getVolume = (sym: string) => {
    switch (sym) {
      case 'SPY': return '85.2M'
      case 'QQQ': return '42.7M'
      case 'IWM': return '31.5M'
      default: return 'N/A'
    }
  }

  return (
    <div className="flex items-center justify-between px-4 py-3 text-xs text-gray-400 border-t border-gray-800/50 bg-gray-900/30 flex-shrink-0">
      <div className="flex items-center space-x-4">
        <span>O: {currentPrice.toFixed(2)}</span>
        <span>H: {(currentPrice * 1.002).toFixed(2)}</span>
        <span>L: {(currentPrice * 0.998).toFixed(2)}</span>
        <span>C: {currentPrice.toFixed(2)}</span>
        <span>VOL: {getVolume(symbol)}</span>
      </div>
      <div className="flex items-center space-x-2">
        <span>15:08:21 (UTC-5)</span>
        <span>LOG</span>
        <span className="text-blue-400">AUTO</span>
      </div>
    </div>
  )
}
