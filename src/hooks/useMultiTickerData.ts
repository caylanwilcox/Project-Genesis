'use client'

import { useState, useEffect, useCallback, useRef } from 'react';
import { polygonService } from '../services/polygonService';
import { NormalizedChartData } from '../types/polygon';

interface TickerSnapshot {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: string;
  volumeNum: number;
  volumeHigh1M: number;
  volumeHigh1MStr: string;
  volumeHigh1MArrow: 'up' | 'down' | 'flat';
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

  const formatCompact = useCallback((value: number) => {
    if (!Number.isFinite(value)) return '0';
    if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B`;
    if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
    if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
    return Math.round(value).toString();
  }, []);

  // Cache 1M volume highs to avoid spamming daily aggregates (esp. paid plan refresh).
  const volumeHighCacheRef = useRef(new Map<string, { value: number; ts: number }>());
  const VOLUME_HIGH_CACHE_MS = 10 * 60 * 1000; // 10 minutes

  const getOrFetch1MVolumeHigh = useCallback(async (symbol: string): Promise<number> => {
    const key = symbol.toUpperCase();
    const now = Date.now();
    const existing = volumeHighCacheRef.current.get(key);
    if (existing && now - existing.ts < VOLUME_HIGH_CACHE_MS) {
      return existing.value;
    }

    const daily = await polygonService.getAggregates(key, '1d', 40, '1M');
    const high = daily.reduce((max, b) => Math.max(max, b.volume || 0), 0);
    volumeHighCacheRef.current.set(key, { value: high, ts: now });
    return high;
  }, []);

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
          const volumeStr = formatCompact(volumeNum);
          const volumeHigh1M = await getOrFetch1MVolumeHigh(symbol);
          const volumeHigh1MStr = formatCompact(volumeHigh1M);
          const volumeHigh1MArrow: TickerSnapshot['volumeHigh1MArrow'] =
            volumeNum > volumeHigh1M ? 'up' : volumeNum < volumeHigh1M ? 'down' : 'flat';

          return {
            symbol,
            price,
            change,
            changePercent,
            volume: volumeStr,
            volumeNum,
            volumeHigh1M,
            volumeHigh1MStr,
            volumeHigh1MArrow,
            open: snap.day?.o ?? price,
            high: snap.day?.h ?? price,
            low: snap.day?.l ?? price,
            prevClose,
          };
        }
      } catch (e) {
        console.warn('Snapshot unavailable, using aggregates fallback:', e);
      }

      // Fallback (free tier): use daily aggregates for latest daily bar + previous close.
      // This is rate-limit friendly and gives a coherent daily volume series for 1M high comparisons.
      const daily = await polygonService.getAggregates(symbol, '1d', 40, '1M');
      if (daily.length < 2) {
        console.error(`Not enough daily aggregate data for ${symbol}`);
        return null;
      }

      const latest = daily[daily.length - 1];
      const prev = daily[daily.length - 2];
      const price = latest.close;
      const prevClose = prev.close;
      const change = price - prevClose;
      const changePercent = (change / prevClose) * 100;

      const volumeNum = latest.volume || 0;
      const volumeStr = formatCompact(volumeNum);
      const volumeHigh1M = daily.reduce((max, b) => Math.max(max, b.volume || 0), 0);
      volumeHighCacheRef.current.set(symbol.toUpperCase(), { value: volumeHigh1M, ts: Date.now() });
      const volumeHigh1MStr = formatCompact(volumeHigh1M);
      const volumeHigh1MArrow: TickerSnapshot['volumeHigh1MArrow'] =
        volumeNum > volumeHigh1M ? 'up' : volumeNum < volumeHigh1M ? 'down' : 'flat';

      return {
        symbol,
        price,
        change,
        changePercent,
        volume: volumeStr,
        volumeNum,
        volumeHigh1M,
        volumeHigh1MStr,
        volumeHigh1MArrow,
        open: latest.open,
        high: latest.high,
        low: latest.low,
        prevClose,
      };
    } catch (err) {
      console.error(`Error fetching data for ${symbol}:`, err);
      return null;
    }
  }, [formatCompact, getOrFetch1MVolumeHigh]);

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
