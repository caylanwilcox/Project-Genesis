'use client'

import React from 'react';
import { TradingPair } from '../types/trading';
import { TrendingUp, TrendingDown, ChevronDown, Bell, Settings, User } from 'lucide-react';

interface TradingHeaderProps {
  pair: TradingPair;
}

export const TradingHeader: React.FC<TradingHeaderProps> = ({ pair }) => {
  const isPositive = pair.priceChangePercent >= 0;

  return (
    <header className="bg-gradient-to-r from-gray-900/95 to-gray-900/90 border-b border-gray-800/50 backdrop-blur-sm shadow-xl">
      <div className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center space-x-6">
          <div className="flex items-center gap-2 cursor-pointer hover:bg-gray-800/50 px-3 py-2 rounded-lg transition-all duration-200 group">
            <span className="text-xl font-bold">{pair.baseAsset}/{pair.quoteAsset}</span>
            <ChevronDown size={20} className="text-gray-400" />
          </div>

          <div className="flex items-center space-x-4">
            <div>
              <div className="text-xs text-gray-400">Last Price (24H)</div>
              <div className={`text-2xl font-semibold ${isPositive ? 'text-green-500' : 'text-red-500'}`}>
                ${pair.lastPrice.toLocaleString()}
              </div>
            </div>

            <div className={`flex items-center space-x-1 ${isPositive ? 'text-green-500' : 'text-red-500'}`}>
              {isPositive ? <TrendingUp size={20} /> : <TrendingDown size={20} />}
              <span className="text-lg font-medium">
                {isPositive ? '+' : ''}{pair.priceChangePercent.toFixed(2)}%
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-4 text-sm">
            <div>
              <div className="text-xs text-gray-400">24H Volume</div>
              <div className="font-medium">${(pair.volume24h / 1000000).toFixed(2)}M</div>
            </div>
            <div>
              <div className="text-xs text-gray-400">24H High</div>
              <div className="font-medium">${pair.high24h.toLocaleString()}</div>
            </div>
            <div>
              <div className="text-xs text-gray-400">24H Low</div>
              <div className="font-medium">${pair.low24h.toLocaleString()}</div>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <button className="px-6 py-3 bg-gradient-to-br from-yellow-400 via-yellow-500 to-amber-500 text-black rounded-xl font-bold hover:from-yellow-300 hover:via-yellow-400 hover:to-amber-400 transition-all duration-300 transform hover:scale-105 active:scale-95 shadow-xl hover:shadow-yellow-500/40 border-2 border-yellow-300/50">
            Deposit
          </button>
          <button className="px-6 py-3 bg-gradient-to-br from-gray-700 via-gray-600 to-gray-500 rounded-xl font-bold hover:from-gray-600 hover:via-gray-500 hover:to-gray-400 transition-all duration-300 transform hover:scale-105 active:scale-95 shadow-lg hover:shadow-gray-500/30 text-white border border-gray-400/30">
            Manage Funds
          </button>
          <Bell size={20} className="text-gray-400 cursor-pointer hover:text-white" />
          <Settings size={20} className="text-gray-400 cursor-pointer hover:text-white" />
          <div className="w-12 h-12 bg-gradient-to-br from-blue-500 via-blue-600 to-purple-600 rounded-full flex items-center justify-center cursor-pointer hover:scale-110 transition-all duration-300 shadow-lg hover:shadow-blue-500/40 border-2 border-blue-300/40">
            <User size={18} />
          </div>
        </div>
      </div>

      <div className="flex items-center px-4 py-2 space-x-4 text-sm text-gray-400">
        <span className="hover:text-white cursor-pointer">We have lowered our Advanced fees</span>
        <span className="text-xs">•</span>
        <span>Trade as low as 0.00% maker and 0.05% taker fee with Coinbase Advanced.</span>
        <a href="#" className="text-blue-400 hover:text-blue-300">Learn more.</a>
      </div>
    </header>
  );
};