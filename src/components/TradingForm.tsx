'use client'

import React, { useState } from 'react';
import { Info } from 'lucide-react';

interface TradingFormProps {
  availableBalance: number;
  currentPrice: number;
}

export const TradingForm: React.FC<TradingFormProps> = ({ availableBalance, currentPrice }) => {
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market');
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [amount, setAmount] = useState('0.0');
  const [price, setPrice] = useState(currentPrice.toString());
  const [percentage, setPercentage] = useState(0);
  const [slippage, setSlippage] = useState('0.5');
  const [takeProfitLoss, setTakeProfitLoss] = useState(false);

  const percentages = [25, 50, 75, 100];

  const handlePercentageClick = (percent: number) => {
    setPercentage(percent);
    const calculatedAmount = (availableBalance * (percent / 100)) / parseFloat(price);
    setAmount(calculatedAmount.toFixed(2));
  };

  return (
    <div className="trading-card rounded-xl p-6 transition-all duration-300">
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setSide('buy')}
          className={`flex-1 py-3 px-6 font-bold rounded-lg transition-all duration-300 ${
            side === 'buy'
              ? 'btn-buy-premium text-white'
              : 'bg-gradient-to-br from-gray-800/80 to-gray-700/60 text-gray-300 hover:text-white hover:from-gray-700/90 hover:to-gray-600/70 border border-gray-600/30 hover:border-gray-500/50'
          }`}
        >
          Buy
        </button>
        <button
          onClick={() => setSide('sell')}
          className={`flex-1 py-3 px-6 font-bold rounded-lg transition-all duration-300 ${
            side === 'sell'
              ? 'btn-sell-premium text-white'
              : 'bg-gradient-to-br from-gray-800/80 to-gray-700/60 text-gray-300 hover:text-white hover:from-gray-700/90 hover:to-gray-600/70 border border-gray-600/30 hover:border-gray-500/50'
          }`}
        >
          Sell
        </button>
      </div>

      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setOrderType('market')}
          className={`px-5 py-2.5 text-sm rounded-lg font-semibold transition-all duration-300 transform hover:scale-[1.02] ${
            orderType === 'market'
              ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white shadow-lg hover:shadow-blue-500/30 border border-blue-400/30'
              : 'bg-gradient-to-br from-gray-800/50 to-gray-700/30 text-gray-300 hover:text-white hover:from-gray-700/60 hover:to-gray-600/40 border border-gray-600/20 hover:border-gray-500/40'
          }`}
        >
          Market
        </button>
        <button
          onClick={() => setOrderType('limit')}
          className={`px-5 py-2.5 text-sm rounded-lg font-semibold transition-all duration-300 transform hover:scale-[1.02] ${
            orderType === 'limit'
              ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white shadow-lg hover:shadow-blue-500/30 border border-blue-400/30'
              : 'bg-gradient-to-br from-gray-800/50 to-gray-700/30 text-gray-300 hover:text-white hover:from-gray-700/60 hover:to-gray-600/40 border border-gray-600/20 hover:border-gray-500/40'
          }`}
        >
          Limit
        </button>
      </div>

      <div className="space-y-4">
        <div>
          <label className="text-xs font-medium text-gray-400 block mb-2">Amount</label>
          <div className="relative">
            <input
              type="text"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              className="w-full bg-gray-900/50 border border-gray-800 rounded-lg px-4 py-2.5 text-white placeholder-gray-500 focus:border-green-500/50 focus:bg-gray-900/70 focus:outline-none focus:ring-2 focus:ring-green-500/20 transition-all duration-200 text-sm"
              placeholder="0.0"
            />
            <span className="absolute right-3 top-1/2 transform -translate-y-1/2 text-xs text-gray-400">
              DNT
            </span>
          </div>
        </div>

        {orderType === 'limit' && (
          <div>
            <label className="text-xs font-medium text-gray-400 block mb-2">Price (USD)</label>
            <div className="relative">
              <input
                type="text"
                value={price}
                onChange={(e) => setPrice(e.target.value)}
                className="w-full bg-gray-900/50 border border-gray-800 rounded-lg px-4 py-2.5 text-white placeholder-gray-500 focus:border-green-500/50 focus:bg-gray-900/70 focus:outline-none focus:ring-2 focus:ring-green-500/20 transition-all duration-200 text-sm"
              />
              <span className="absolute right-3 top-1/2 transform -translate-y-1/2 text-xs text-gray-400">
                USD
              </span>
            </div>
          </div>
        )}

        <div className="flex gap-2">
          {percentages.map((percent) => (
            <button
              key={percent}
              onClick={() => handlePercentageClick(percent)}
              className={`flex-1 py-2.5 px-3 text-xs rounded-lg font-semibold transition-all duration-300 transform hover:scale-[1.05] ${
                percentage === percent
                  ? 'bg-gradient-to-br from-purple-500/80 to-purple-600/80 text-white shadow-lg shadow-purple-500/25 border border-purple-400/40'
                  : 'bg-gradient-to-br from-gray-800/40 to-gray-700/20 text-gray-400 hover:text-white hover:from-gray-700/50 hover:to-gray-600/30 border border-gray-600/10 hover:border-gray-500/30'
              }`}
            >
              {percent}%
            </button>
          ))}
        </div>

        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            id="takeProfitLoss"
            checked={takeProfitLoss}
            onChange={(e) => setTakeProfitLoss(e.target.checked)}
            className="rounded"
          />
          <label htmlFor="takeProfitLoss" className="text-xs text-gray-400">
            Take profit / Stop loss
          </label>
        </div>

        {orderType === 'market' && (
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs text-gray-400">Slippage</label>
              <Info size={14} className="text-gray-400" />
            </div>
            <input
              type="text"
              value={slippage}
              onChange={(e) => setSlippage(e.target.value)}
              className="w-full bg-gray-900/50 border border-gray-800 rounded-lg px-4 py-2.5 text-white placeholder-gray-500 focus:border-green-500/50 focus:bg-gray-900/70 focus:outline-none focus:ring-2 focus:ring-green-500/20 transition-all duration-200 text-sm"
              placeholder="0.5%"
            />
          </div>
        )}

        <div className="space-y-2 py-4 border-t border-gray-800/50">
          <div className="flex justify-between text-xs">
            <span className="text-gray-400">Fee</span>
            <span>--</span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-gray-400">Total</span>
            <span>--</span>
          </div>
        </div>

        <button
          className={`w-full py-4 px-8 font-bold text-lg rounded-xl transition-all duration-500 btn-press ${
            side === 'buy'
              ? 'btn-buy-premium glow-green float'
              : 'btn-sell-premium glow-red float'
          }`}
        >
          {side === 'buy' ? 'Buy DNT' : 'Sell DNT'}
        </button>

        <div className="flex items-center justify-between text-xs text-gray-400 pt-4 border-t border-gray-800/30">
          <span>Available to trade</span>
          <span className="text-white">${availableBalance.toFixed(2)} USD</span>
        </div>
      </div>
    </div>
  );
};