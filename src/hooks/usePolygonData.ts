'use client'

import { useState, useEffect, useCallback } from 'react';
import { polygonService } from '../services/polygonService';
import { NormalizedChartData, Timeframe } from '../types/polygon';
import {
  aggregateBarsToDuration,
  FALLBACK_INTRADAY_INTERVAL,
  mergeCandlesReplacing,
  shouldPatchIntradaySession,
  TIMEFRAME_IN_MS,
  INTRADAY_TIMEFRAMES,
} from './polygonRealtimeUtils';

const FALLBACK_TIMEFRAME_CHAIN: Partial<Record<Timeframe, Timeframe[]>> = {
  '5m': ['1m'],
  '15m': ['5m', '1m'],
  '30m': ['15m', '5m', '1m'],
  '1h': ['30m', '15m', '5m', '1m'],
  '2h': ['30m', '15m', '5m', '1m'],
  '4h': ['30m', '15m', '5m', '1m'],
  '1d': ['4h', '1h', '30m'],
  '1w': ['1d', '4h', '1h'],
  '1M': ['1w', '1d'],
};

const RECENT_INTRADAY_PATCH_DAYS = 10;
const RECENT_INTRADAY_PATCH_DISPLAY = 'recent_intraday_patch';

const STALE_DISPLAY_OVERRIDES: Partial<Record<Timeframe, string>> = {
  '1m': '1D',
  '5m': '5D',
  '15m': '5D',
  '30m': '1M',
};

interface UsePolygonDataOptions {
  ticker: string;
  timeframe?: Timeframe;
  limit?: number;
  autoRefresh?: boolean;
  refreshInterval?: number; // in milliseconds
  displayTimeframe?: string; // Display timeframe like 'YTD', '1Y', '5Y' for special date handling
}

interface UsePolygonDataResult {
  data: NormalizedChartData[];
  currentPrice: number | null;
  priceChange: number | null;
  priceChangePercent: number | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

export function usePolygonData({
  ticker,
  timeframe = '1h',
  limit = 100,
  autoRefresh = false,
  refreshInterval = 60000, // 1 minute default
  displayTimeframe,
}: UsePolygonDataOptions): UsePolygonDataResult {
  const [data, setData] = useState<NormalizedChartData[]>([]);
  const [currentPrice, setCurrentPrice] = useState<number | null>(null);
  const [priceChange, setPriceChange] = useState<number | null>(null);
  const [priceChangePercent, setPriceChangePercent] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    if (!ticker) return;

    try {
      setIsLoading(true);
      setError(null);

      console.log(`[usePolygonData] Fetching data for ${ticker} - Timeframe: ${timeframe}, Limit: ${limit}, Display: ${displayTimeframe}`);

      // Check if API is configured
      if (!polygonService.isConfigured()) {
        throw new Error('Polygon.io API key not configured. Please set NEXT_PUBLIC_POLYGON_API_KEY in your environment variables.');
      }

      // Fetch aggregate data for chart
      let aggregates = await polygonService.getAggregates(ticker, timeframe, limit, displayTimeframe);

      // IMPORTANT: refresh stale data BEFORE patching intraday bars so we don't mask multi-day gaps
      aggregates = await refreshStaleDataIfNeeded({
        ticker,
        timeframe,
        limit,
        displayTimeframe,
        aggregates,
      });

      if (shouldPatchIntradaySession(timeframe, aggregates)) {
        try {
          const intradayPatch = await fetchTodayIntradayBars(ticker, timeframe);
          if (intradayPatch.length > 0) {
            aggregates = mergeCandlesReplacing(aggregates, intradayPatch);
            console.log(`[usePolygonData] Patched ${intradayPatch.length} intraday bars to include today's session for ${ticker} (${timeframe}).`);
          } else {
            console.warn(`[usePolygonData] Intraday patch triggered but no bars returned for ${ticker} (${timeframe}).`);
          }
        } catch (patchError) {
          console.warn(`[usePolygonData] Failed to patch intraday data for ${ticker} (${timeframe}):`, patchError);
        }
      }

      console.log(`[usePolygonData] Fetched ${aggregates.length} bars for ${ticker}`);

      // Set the data (service ensures it's not empty)
      setData(aggregates);

      // Get current price from latest bar
      const latest = aggregates[aggregates.length - 1];

      if (latest && aggregates.length > 0) {
        setCurrentPrice(latest.close);

        // Fetch previous close to calculate accurate daily change
        try {
          const prevCloseData = await polygonService.getPreviousClose(ticker);

          if (prevCloseData) {
            // Calculate change from previous day's close (not from previous bar)
            const change = latest.close - prevCloseData.close;
            const changePercent = (change / prevCloseData.close) * 100;
            setPriceChange(change);
            setPriceChangePercent(changePercent);
          } else {
            // Fallback: use bar-to-bar change if prev close not available
            const previous = aggregates[aggregates.length - 2];
            if (previous) {
              const change = latest.close - previous.close;
              const changePercent = (change / previous.close) * 100;
              setPriceChange(change);
              setPriceChangePercent(changePercent);
            }
          }
        } catch (prevCloseError) {
          console.warn('Could not fetch previous close, using bar-to-bar change:', prevCloseError);
          // Fallback: use bar-to-bar change
          const previous = aggregates[aggregates.length - 2];
          if (previous) {
            const change = latest.close - previous.close;
            const changePercent = (change / previous.close) * 100;
            setPriceChange(change);
            setPriceChangePercent(changePercent);
          }
        }
      }
    } catch (err) {
      console.error('Error fetching polygon data:', err);
      setError(err instanceof Error ? err : new Error('Failed to fetch data'));
    } finally {
      setIsLoading(false);
    }
  }, [ticker, timeframe, limit, displayTimeframe]);

  // Initial fetch
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      fetchData();
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, fetchData]);

  return {
    data,
    currentPrice,
    priceChange,
    priceChangePercent,
    isLoading,
    error,
    refetch: fetchData,
  };
}

/**
 * Hook for fetching real-time snapshot data
 */
export function usePolygonSnapshot(ticker: string, refreshInterval: number = 5000) {
  const [snapshot, setSnapshot] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchSnapshot = useCallback(async () => {
    if (!ticker) return;

    try {
      setError(null);
      const data = await polygonService.getSnapshot(ticker);
      setSnapshot(data);
      setIsLoading(false);
    } catch (err) {
      console.error('Error fetching polygon snapshot:', err);
      setError(err instanceof Error ? err : new Error('Failed to fetch snapshot'));
      setIsLoading(false);
    }
  }, [ticker]);

  useEffect(() => {
    fetchSnapshot();
    const interval = setInterval(fetchSnapshot, refreshInterval);
    return () => clearInterval(interval);
  }, [fetchSnapshot, refreshInterval]);

  return { snapshot, isLoading, error, refetch: fetchSnapshot };
}

async function fetchTodayIntradayBars(ticker: string, timeframe: Timeframe): Promise<NormalizedChartData[]> {
  const interval = FALLBACK_INTRADAY_INTERVAL[timeframe];
  if (!interval) return [];

  const rawBars = await polygonService.getIntradayData(ticker, interval);
  if (rawBars.length === 0) return [];

  if (timeframe === '1m' || timeframe === '5m' || timeframe === '15m' || timeframe === '30m') {
    return rawBars;
  }

  const bucketMs = TIMEFRAME_IN_MS[timeframe];
  if (!bucketMs) {
    return rawBars;
  }

  return aggregateBarsToDuration(rawBars, bucketMs);
}

async function refreshStaleDataIfNeeded({
  ticker,
  timeframe,
  limit,
  displayTimeframe,
  aggregates,
}: {
  ticker: string;
  timeframe: Timeframe;
  limit: number;
  displayTimeframe?: string;
  aggregates: NormalizedChartData[];
}): Promise<NormalizedChartData[]> {
  if (aggregates.length === 0) {
    return aggregates;
  }

  const intervalMs = TIMEFRAME_IN_MS[timeframe];
  if (!intervalMs) {
    return aggregates;
  }

  const maxStaleness = Math.max(intervalMs * 1.5, 30 * 60 * 1000);
  const isFresh = (data: NormalizedChartData[]) => {
    if (!data.length) return false;
    const latest = data[data.length - 1]?.time ?? 0;
    return Date.now() - latest <= maxStaleness;
  };

  if (isFresh(aggregates)) {
    return aggregates;
  }

  console.log(
    `[usePolygonData] Detected stale ${timeframe} data (${Math.round(
      (Date.now() - aggregates[aggregates.length - 1].time) / 1000,
    )}s old). Attempting fallback refresh.`,
  );

  let workingData = aggregates;

  const fallbackChain = FALLBACK_TIMEFRAME_CHAIN[timeframe] ?? [];
  for (const fallbackTimeframe of fallbackChain) {
    const fallbackData = await fetchFallbackAggregates({
      ticker,
      fallbackTimeframe,
      targetTimeframe: timeframe,
      targetLimit: limit,
      displayTimeframe,
    });

    if (!fallbackData.length) {
      continue;
    }

    workingData = fallbackData;
    if (isFresh(workingData)) {
      return workingData;
    }
  }

  if (INTRADAY_TIMEFRAMES.includes(timeframe)) {
    const recentPatch = await fetchRecentIntradayPatch(ticker, timeframe);
    if (recentPatch.length) {
      workingData = mergeCandlesReplacing(workingData, recentPatch);
      if (isFresh(workingData)) {
        return workingData;
      }
    }
  }

  return workingData;
}

async function fetchFallbackAggregates({
  ticker,
  fallbackTimeframe,
  targetTimeframe,
  targetLimit,
  displayTimeframe,
}: {
  ticker: string;
  fallbackTimeframe: Timeframe;
  targetTimeframe: Timeframe;
  targetLimit: number;
  displayTimeframe?: string;
}): Promise<NormalizedChartData[]> {
  const fallbackMs = TIMEFRAME_IN_MS[fallbackTimeframe];
  const targetMs = TIMEFRAME_IN_MS[targetTimeframe];
  if (!fallbackMs || !targetMs) {
    return [];
  }

  const ratio = Math.max(1, Math.ceil((targetMs / fallbackMs) * 1.1));
  const fallbackLimit = Math.min(targetLimit * ratio, 50000);
  const fallbackDisplay =
    fallbackTimeframe === targetTimeframe
      ? STALE_DISPLAY_OVERRIDES[targetTimeframe] ?? displayTimeframe
      : displayTimeframe;

  try {
    const rawFallback = await polygonService.getAggregates(
      ticker,
      fallbackTimeframe,
      fallbackLimit,
      fallbackDisplay,
    );

    if (!rawFallback.length) {
      return [];
    }

    if (fallbackTimeframe === targetTimeframe) {
      return rawFallback;
    }

    return aggregateBarsToDuration(rawFallback, targetMs);
  } catch (error) {
    console.warn(
      `[usePolygonData] Fallback refresh failed via ${fallbackTimeframe}:`,
      error,
    );
    return [];
  }
}

async function fetchRecentIntradayPatch(
  ticker: string,
  targetTimeframe: Timeframe,
): Promise<NormalizedChartData[]> {
  const bucketMs = TIMEFRAME_IN_MS[targetTimeframe];
  if (!bucketMs) {
    return [];
  }

  const perDayBars = 24 * 60; // 1-minute bars per day (over-fetch to cover extended hours)
  const limit = Math.min(perDayBars * RECENT_INTRADAY_PATCH_DAYS, 50000);
  const cutoff = Date.now() - RECENT_INTRADAY_PATCH_DAYS * 24 * 60 * 60 * 1000;

  try {
    const recentRaw = await polygonService.getAggregates(
      ticker,
      '1m',
      limit,
      RECENT_INTRADAY_PATCH_DISPLAY,
    );

    if (!recentRaw.length) {
      return [];
    }

    const filtered = recentRaw.filter(bar => bar.time >= cutoff);
    const slice = filtered.length ? filtered : recentRaw.slice(-limit);
    return aggregateBarsToDuration(slice, bucketMs);
  } catch (error) {
    console.warn(
      `[usePolygonData] Recent intraday patch failed for ${ticker} (${targetTimeframe}):`,
      error,
    );
    return [];
  }
}
