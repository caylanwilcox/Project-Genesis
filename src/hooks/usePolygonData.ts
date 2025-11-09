'use client'

import { useState, useEffect, useCallback } from 'react';
import { polygonService } from '../services/polygonService';
import { NormalizedChartData, Timeframe } from '../types/polygon';

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
      const aggregates = await polygonService.getAggregates(ticker, timeframe, limit, displayTimeframe);
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
