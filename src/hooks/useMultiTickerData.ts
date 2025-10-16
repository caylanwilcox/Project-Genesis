'use client'

import { useState, useEffect, useCallback } from 'react';
import { polygonService } from '../services/polygonService';
import { NormalizedChartData } from '../types/polygon';

interface TickerSnapshot {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: string;
  open: number;
  high: number;
  low: number;
  prevClose: number;
}

interface UseMultiTickerDataResult {
  tickers: Map<string, TickerSnapshot>;
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

export function useMultiTickerData(
  symbols: string[],
  autoRefresh: boolean = true,
  refreshInterval: number = 10000 // 10 seconds
): UseMultiTickerDataResult {
  const [tickers, setTickers] = useState<Map<string, TickerSnapshot>>(new Map());
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [isFetching, setIsFetching] = useState(false);

  const fetchTickerData = useCallback(async (symbol: string): Promise<TickerSnapshot | null> => {
    try {
      // Prefer snapshot endpoint for near real-time data (works on paid plans).
      // If it fails (e.g., free plan), gracefully fall back to aggregates.
      try {
        const snap = await polygonService.getSnapshot(symbol);
        if (snap) {
          const price = snap.min?.c ?? snap.day?.c ?? 0;
          const prevClose = snap.prevDay?.c ?? snap.day?.o ?? price;

          const change = price - prevClose;
          const changePercent = (change / prevClose) * 100;
          const volumeNum = snap.day?.v ?? 0;
          const volumeStr = volumeNum >= 1000000
            ? `${(volumeNum / 1000000).toFixed(1)}M`
            : volumeNum >= 1000
            ? `${(volumeNum / 1000).toFixed(1)}K`
            : volumeNum.toString();

          return {
            symbol,
            price,
            change,
            changePercent,
            volume: volumeStr,
            open: snap.day?.o ?? price,
            high: snap.day?.h ?? price,
            low: snap.day?.l ?? price,
            prevClose,
          };
        }
      } catch (e) {
        console.warn('Snapshot unavailable, using aggregates fallback:', e);
      }

      // Free tier or fallback: Use previous close + recent aggregates
      // Get previous close for baseline
      const prevCloseData = await polygonService.getPreviousClose(symbol);

      if (!prevCloseData) {
        console.error(`No previous close data for ${symbol}`);
        return null;
      }

      // Get most recent 1-minute bars for current price
      const recentAggregates = await polygonService.getAggregates(symbol, '1m', 2);

      let currentData = prevCloseData;

      // If we have recent data, use it; otherwise use previous close
      if (recentAggregates.length > 0) {
        const latest = recentAggregates[recentAggregates.length - 1];
        currentData = {
          time: latest.time,
          open: latest.open,
          high: latest.high,
          low: latest.low,
          close: latest.close,
          volume: latest.volume,
        };
      }

      // Calculate change from previous close
      const change = currentData.close - prevCloseData.close;
      const changePercent = (change / prevCloseData.close) * 100;

      // Format volume
      const volumeNum = currentData.volume;
      const volumeStr = volumeNum >= 1000000
        ? `${(volumeNum / 1000000).toFixed(1)}M`
        : volumeNum >= 1000
        ? `${(volumeNum / 1000).toFixed(1)}K`
        : volumeNum.toString();

      return {
        symbol,
        price: currentData.close,
        change,
        changePercent,
        volume: volumeStr,
        open: currentData.open,
        high: currentData.high,
        low: currentData.low,
        prevClose: prevCloseData.close,
      };
    } catch (err) {
      console.error(`Error fetching data for ${symbol}:`, err);
      return null;
    }
  }, []);

  const fetchAllTickers = useCallback(async () => {
    // Prevent multiple simultaneous fetches
    if (isFetching) {
      console.log('Already fetching, skipping...');
      return;
    }

    try {
      setIsFetching(true);
      setError(null);

      if (!polygonService.isConfigured()) {
        throw new Error('Polygon.io API key not configured');
      }

      // Fetch tickers - the polygonService handles rate limiting and queueing
      const tickerMap = new Map<string, TickerSnapshot>(tickers); // Start with existing data

      // Fetch all tickers in parallel - the service will queue and rate-limit them
      const promises = symbols.map(async (symbol) => {
        try {
          const result = await fetchTickerData(symbol);
          if (result) {
            tickerMap.set(symbol, result);
            // Update state after each successful fetch so UI shows progress
            setTickers(new Map(tickerMap));
          }
        } catch (err) {
          console.error(`Error fetching ${symbol}:`, err);
          // Continue with other tickers even if one fails
        }
      });

      // Wait for all fetches to complete (service handles rate limiting)
      await Promise.all(promises);

      setIsLoading(false);
      setIsFetching(false);
    } catch (err) {
      console.error('Error fetching ticker data:', err);
      setError(err instanceof Error ? err : new Error('Failed to fetch ticker data'));
      setIsLoading(false);
      setIsFetching(false);
    }
  }, [symbols, fetchTickerData, isFetching, tickers]);

  // Initial fetch - only once on mount
  useEffect(() => {
    fetchAllTickers();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-refresh - separate effect with stable interval
  useEffect(() => {
    if (!autoRefresh || isLoading) return;

    const interval = setInterval(() => {
      console.log('Auto-refresh triggered');
      fetchAllTickers();
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, isLoading]); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    tickers,
    isLoading,
    error,
    refetch: fetchAllTickers,
  };
}
