'use client'

import React, { useState } from 'react';
import { OrderBook as OrderBookType, OrderBookEntry } from '../types/trading';

interface OrderBookProps {
  orderBook: OrderBookType;
}

export const OrderBook: React.FC<OrderBookProps> = ({ orderBook }) => {
  const [orderType, setOrderType] = useState<'all' | 'buy' | 'sell'>('all');

  const maxTotal = Math.max(
    ...orderBook.bids.map(b => b.total),
    ...orderBook.asks.map(a => a.total)
  );

  const renderOrderBookSide = (orders: OrderBookEntry[], side: 'buy' | 'sell') => {
    return orders.slice(0, 15).map((order, index) => {
      const depthPercent = (order.total / maxTotal) * 100;

      return (
        <div
          key={index}
          className="relative grid grid-cols-3 gap-2 px-3 py-1.5 text-xs cursor-pointer transition-all duration-200 hover:bg-gradient-to-r hover:from-transparent hover:via-gray-800/20 hover:to-transparent"
        >
          <div
            className={`absolute inset-0 ${
              side === 'buy' ? 'bg-gradient-to-r from-transparent to-green-500' : 'bg-gradient-to-r from-transparent to-red-500'
            } opacity-20`}
            style={{ width: `${depthPercent}%` }}
          />
          <span className={side === 'buy' ? 'text-green-500' : 'text-red-500'}>
            {order.price.toFixed(4)}
          </span>
          <span className="text-right text-gray-300">{order.amount.toFixed(2)}</span>
          <span className="text-right text-gray-300">{order.total.toFixed(2)}</span>
        </div>
      );
    });
  };

  return (
    <div className="trading-card rounded-xl h-full flex flex-col transition-all duration-300">
      <div className="p-4 border-b border-gray-800/50">
        <h3 className="text-sm font-semibold mb-3 text-white">Order Book</h3>
        <div className="flex gap-1">
          <button
            onClick={() => setOrderType('all')}
            className={`flex-1 py-1.5 text-xs rounded-lg font-medium transition-all duration-200 ${
              orderType === 'all'
                ? 'bg-gradient-to-r from-gray-700 to-gray-600 text-white shadow-sm'
                : 'bg-gray-800/30 text-gray-400 hover:text-white hover:bg-gray-800/50'
            }`}
          >
            All
          </button>
          <button
            onClick={() => setOrderType('buy')}
            className={`flex-1 py-1.5 text-xs rounded-lg font-medium transition-all duration-200 ${
              orderType === 'buy'
                ? 'bg-gradient-to-r from-green-600/30 to-green-500/30 text-green-400 shadow-sm'
                : 'bg-gray-800/30 hover:bg-gray-800/50'
            }`}
          >
            <span className={orderType === 'buy' ? '' : 'text-green-500'}>Buy</span>
          </button>
          <button
            onClick={() => setOrderType('sell')}
            className={`flex-1 py-1.5 text-xs rounded-lg font-medium transition-all duration-200 ${
              orderType === 'sell'
                ? 'bg-gradient-to-r from-red-600/30 to-red-500/30 text-red-400 shadow-sm'
                : 'bg-gray-800/30 hover:bg-gray-800/50'
            }`}
          >
            <span className={orderType === 'sell' ? '' : 'text-red-500'}>Sell</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 px-4 py-2 text-xs font-medium text-gray-500 border-b border-gray-800/50 bg-gray-900/30">
        <span>Price (USD)</span>
        <span className="text-right">Amount (DNT)</span>
        <span className="text-right">Total</span>
      </div>

      <div className="flex-1 overflow-y-auto">
        {(orderType === 'all' || orderType === 'sell') && (
          <div className="border-b border-gray-800/30">
            {renderOrderBookSide(orderBook.asks, 'sell')}
          </div>
        )}

        {orderType === 'all' && (
          <div className="px-4 py-3 bg-gradient-to-r from-gray-800/30 to-gray-800/10 backdrop-blur-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-gray-400">USD Spread</span>
              <span className="text-sm font-bold bg-gradient-to-r from-green-400 to-blue-500 bg-clip-text text-transparent">{orderBook.spread.toFixed(4)}</span>
            </div>
          </div>
        )}

        {(orderType === 'all' || orderType === 'buy') && (
          <div>
            {renderOrderBookSide(orderBook.bids, 'buy')}
          </div>
        )}
      </div>
    </div>
  );
};