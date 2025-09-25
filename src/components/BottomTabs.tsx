'use client'

import React, { useState } from 'react';
import { Order, Asset } from '../types/trading';
import { X, ChevronDown } from 'lucide-react';

interface BottomTabsProps {
  orders: Order[];
  assets: Asset[];
}

export const BottomTabs: React.FC<BottomTabsProps> = ({ orders, assets }) => {
  const [activeTab, setActiveTab] = useState<'orders' | 'assets'>('orders');
  const [orderFilter, setOrderFilter] = useState<'all' | 'open' | 'filled'>('all');

  const filteredOrders = orders.filter((order) => {
    if (orderFilter === 'all') return true;
    return order.status === orderFilter;
  });

  return (
    <div className="trading-card rounded-xl transition-all duration-300">
      <div className="flex border-b border-gray-800/50">
        <button
          onClick={() => setActiveTab('orders')}
          className={`px-4 py-3 text-sm font-medium ${
            activeTab === 'orders'
              ? 'text-white border-b-2 border-blue-500'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          Orders
        </button>
        <button
          onClick={() => setActiveTab('assets')}
          className={`px-4 py-3 text-sm font-medium ${
            activeTab === 'assets'
              ? 'text-white border-b-2 border-blue-500'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          Assets
        </button>
      </div>

      {activeTab === 'orders' ? (
        <div>
          <div className="flex items-center justify-between p-3 border-b border-gray-800/50">
            <div className="flex space-x-2">
              <button
                onClick={() => setOrderFilter('all')}
                className={`px-3 py-1 text-xs rounded ${
                  orderFilter === 'all' ? 'bg-gray-800/50 text-white' : 'text-gray-400'
                }`}
              >
                All
              </button>
              <button
                onClick={() => setOrderFilter('open')}
                className={`px-3 py-1 text-xs rounded ${
                  orderFilter === 'open' ? 'bg-gray-800/50 text-white' : 'text-gray-400'
                }`}
              >
                Open
              </button>
              <button
                onClick={() => setOrderFilter('filled')}
                className={`px-3 py-1 text-xs rounded ${
                  orderFilter === 'filled' ? 'bg-gray-800/50 text-white' : 'text-gray-400'
                }`}
              >
                Filled
              </button>
            </div>
            <div className="flex items-center space-x-2">
              <button className="text-xs text-gray-400 hover:text-white">Cancel all</button>
              <button className="text-xs text-gray-400 hover:text-white">View all</button>
              <button className="flex items-center space-x-1 text-xs bg-gray-800/50 hover:bg-gray-800/70 px-3 py-1.5 rounded-lg transition-all duration-200">
                <span>DNT-USD</span>
                <ChevronDown size={14} />
              </button>
              <button className="flex items-center space-x-1 text-xs bg-gray-800/50 hover:bg-gray-800/70 px-3 py-1.5 rounded-lg transition-all duration-200">
                <span>ALL STATUSES</span>
                <ChevronDown size={14} />
              </button>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="text-gray-400 border-b border-gray-800/50">
                <tr>
                  <th className="text-left p-3 font-normal">Time Placed</th>
                  <th className="text-left p-3 font-normal">Name</th>
                  <th className="text-left p-3 font-normal">Type</th>
                  <th className="text-left p-3 font-normal">Side</th>
                  <th className="text-left p-3 font-normal">Price</th>
                  <th className="text-left p-3 font-normal">Amount</th>
                  <th className="text-left p-3 font-normal">% Filled</th>
                  <th className="text-left p-3 font-normal">Total</th>
                  <th className="text-left p-3 font-normal">TP/SL</th>
                  <th className="text-left p-3 font-normal">Status</th>
                  <th className="text-left p-3 font-normal">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredOrders.length === 0 ? (
                  <tr>
                    <td colSpan={11} className="text-center py-8 text-gray-400">
                      <div className="flex flex-col items-center">
                        <div className="w-16 h-16 bg-gray-800/50 rounded-full flex items-center justify-center mb-2">
                          <span className="text-2xl">📊</span>
                        </div>
                        <span>No orders yet</span>
                      </div>
                    </td>
                  </tr>
                ) : (
                  filteredOrders.map((order) => (
                    <tr key={order.id} className="border-b border-gray-800/50 hover:bg-gray-800/50">
                      <td className="p-3">{order.createdAt.toLocaleDateString()}</td>
                      <td className="p-3">{order.symbol}</td>
                      <td className="p-3 capitalize">{order.type}</td>
                      <td className="p-3">
                        <span className={order.side === 'buy' ? 'text-green-500' : 'text-red-500'}>
                          {order.side.toUpperCase()}
                        </span>
                      </td>
                      <td className="p-3">${order.price.toFixed(4)}</td>
                      <td className="p-3">{order.amount.toFixed(2)}</td>
                      <td className="p-3">{((order.filled / order.amount) * 100).toFixed(0)}%</td>
                      <td className="p-3">${order.total.toFixed(2)}</td>
                      <td className="p-3">--</td>
                      <td className="p-3 capitalize">{order.status}</td>
                      <td className="p-3">
                        <button className="text-gray-400 hover:text-white">
                          <X size={14} />
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-gray-400 border-b border-gray-800/50">
              <tr>
                <th className="text-left p-3 font-normal">Asset</th>
                <th className="text-left p-3 font-normal">Total Balance</th>
                <th className="text-left p-3 font-normal">Available</th>
                <th className="text-left p-3 font-normal">Locked</th>
                <th className="text-left p-3 font-normal">USD Value</th>
              </tr>
            </thead>
            <tbody>
              {assets.map((asset) => (
                <tr key={asset.symbol} className="border-b border-gray-800/50 hover:bg-gray-800/50">
                  <td className="p-3">
                    <div className="flex items-center space-x-2">
                      <div className="w-6 h-6 bg-blue-500 rounded-full" />
                      <div>
                        <div className="font-medium">{asset.symbol}</div>
                        <div className="text-gray-400">{asset.name}</div>
                      </div>
                    </div>
                  </td>
                  <td className="p-3">{asset.balance.toFixed(4)}</td>
                  <td className="p-3">{asset.available.toFixed(4)}</td>
                  <td className="p-3">{asset.locked.toFixed(4)}</td>
                  <td className="p-3">${asset.usdValue.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};