'use client'

import React from 'react';
import { Trade } from '../types/trading';
import { format } from 'date-fns';

interface TradeHistoryProps {
  trades: Trade[];
}

export const TradeHistory: React.FC<TradeHistoryProps> = ({ trades }) => {
  return (
    <div className="trading-card rounded-xl h-full flex flex-col transition-all duration-300">
      <div className="p-4 border-b border-gray-800/50">
        <h3 className="text-sm font-semibold text-white">Trade History</h3>
      </div>

      <div className="grid grid-cols-3 gap-2 px-4 py-2 text-xs font-medium text-gray-500 border-b border-gray-800/50 bg-gray-900/30">
        <span>Price (USD)</span>
        <span className="text-right">Amount (DNT)</span>
        <span className="text-right">Time</span>
      </div>

      <div className="flex-1 overflow-y-auto">
        {trades.map((trade) => (
          <div
            key={trade.id}
            className="grid grid-cols-3 gap-2 px-4 py-1.5 text-xs cursor-pointer transition-all duration-200 hover:bg-gradient-to-r hover:from-transparent hover:via-gray-800/20 hover:to-transparent"
          >
            <span className={trade.side === 'buy' ? 'text-green-500' : 'text-red-500'}>
              {trade.price.toFixed(4)}
            </span>
            <span className="text-right text-gray-300">{trade.amount.toFixed(2)}</span>
            <span className="text-right text-gray-400">
              {format(trade.time, 'HH:mm:ss')}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};