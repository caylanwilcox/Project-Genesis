'use client'

import React, { useState } from 'react';
import { TradingHeader } from './TradingHeader';
import { PriceChart } from './PriceChart';
import { OrderBook } from './OrderBook';
import { TradeHistory } from './TradeHistory';
import { TradingForm } from './TradingForm';
import { BottomTabs } from './BottomTabs';
import { TradingPair, OrderBook as OrderBookType, Trade, Order, Asset } from '../types/trading';

export const TradingView: React.FC = () => {
  const [tradingPair] = useState<TradingPair>({
    symbol: 'DNT-USD',
    baseAsset: 'DNT',
    quoteAsset: 'USD',
    lastPrice: 0.0274,
    priceChange: -0.0006,
    priceChangePercent: -1.08,
    volume24h: 39347654.9,
    high24h: 0.0277,
    low24h: 0.0268,
  });

  const [orderBook] = useState<OrderBookType>({
    bids: generateOrderBookData('buy'),
    asks: generateOrderBookData('sell'),
    spread: 0.0001,
    spreadPercent: 0.36,
  });

  const [trades] = useState<Trade[]>(generateTradeHistory());
  const [orders] = useState<Order[]>([]);
  const [assets] = useState<Asset[]>([
    {
      symbol: 'USD',
      name: 'US Dollar',
      balance: 10000,
      available: 10000,
      locked: 0,
      usdValue: 10000,
    },
    {
      symbol: 'DNT',
      name: 'District0x',
      balance: 2131,
      available: 2131,
      locked: 0,
      usdValue: 58.39,
    },
  ]);

  function generateOrderBookData(side: 'buy' | 'sell') {
    const data = [];
    const basePrice = side === 'buy' ? 0.0274 : 0.0275;

    for (let i = 0; i < 20; i++) {
      const priceOffset = side === 'buy'
        ? -0.0001 * i
        : 0.0001 * i;

      const price = basePrice + priceOffset;
      const amount = Math.random() * 100000;
      const total = price * amount;

      data.push({
        price,
        amount,
        total,
      });
    }

    return data;
  }

  function generateTradeHistory(): Trade[] {
    const trades = [];
    const now = new Date();

    for (let i = 0; i < 50; i++) {
      const time = new Date(now.getTime() - i * 30000);
      trades.push({
        id: `trade-${i}`,
        price: 0.0274 + (Math.random() - 0.5) * 0.0010,
        amount: Math.random() * 10000,
        time,
        side: Math.random() > 0.5 ? 'buy' : 'sell',
      });
    }

    return trades;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950 flex flex-col">
      <TradingHeader pair={tradingPair} />

      <div className="flex-1 flex gap-1 p-1">
        {/* Main Chart and Bottom Tabs */}
        <div className="flex-1 flex flex-col gap-1">
          <div className="flex-1">
            <PriceChart symbol={tradingPair.symbol} />
          </div>

          <div className="h-60">
            <BottomTabs orders={orders} assets={assets} />
          </div>
        </div>

        {/* Right Side Panel */}
        <div className="w-[400px] flex gap-1">
          {/* Order Book and Trade History */}
          <div className="flex-1 flex flex-col gap-1">
            <div className="flex-1">
              <OrderBook orderBook={orderBook} />
            </div>
            <div className="h-60">
              <TradeHistory trades={trades} />
            </div>
          </div>

          {/* Trading Form */}
          <div className="w-[300px]">
            <TradingForm
              availableBalance={10000}
              currentPrice={tradingPair.lastPrice}
            />
          </div>
        </div>
      </div>
    </div>
  );
};