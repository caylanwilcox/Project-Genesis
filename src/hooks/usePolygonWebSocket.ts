import { useEffect, useState, useCallback, useRef } from 'react';
import { polygonWebSocketService, AggregateBar } from '@/services/polygonWebSocket';
import { NormalizedChartData } from '@/types/polygon';

interface UsePolygonWebSocketOptions {
  ticker: string;
  enabled?: boolean; // Whether to connect and subscribe
  onBar?: (bar: AggregateBar) => void; // Callback for each new bar
}

interface UsePolygonWebSocketReturn {
  isConnected: boolean;
  latestBar: AggregateBar | null;
  error: Error | null;
  reconnect: () => void;
}

/**
 * Hook to connect to Polygon.io WebSocket and receive real-time per-minute aggregates
 */
export function usePolygonWebSocket({
  ticker,
  enabled = true,
  onBar,
}: UsePolygonWebSocketOptions): UsePolygonWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [latestBar, setLatestBar] = useState<AggregateBar | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const unsubscribeRef = useRef<(() => void) | null>(null);
  const connectionAttempted = useRef(false);

  // Connect to WebSocket
  useEffect(() => {
    if (!enabled || !ticker || connectionAttempted.current) return;

    const connectToWebSocket = async () => {
      try {
        console.log(`[usePolygonWebSocket] Connecting for ${ticker}...`);
        await polygonWebSocketService.connect();
        setIsConnected(true);
        setError(null);
        connectionAttempted.current = true;
      } catch (err) {
        console.error('[usePolygonWebSocket] Connection failed:', err);
        setError(err as Error);
        setIsConnected(false);
      }
    };

    connectToWebSocket();

    // Cleanup on unmount
    return () => {
      // Don't disconnect - keep connection alive for other components
      // Just unsubscribe from this symbol
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
        unsubscribeRef.current = null;
      }
    };
  }, [enabled, ticker]);

  // Subscribe to symbol updates
  useEffect(() => {
    if (!enabled || !ticker || !isConnected) return;

    console.log(`[usePolygonWebSocket] Subscribing to ${ticker}`);

    // Subscribe and store unsubscribe function
    const unsubscribe = polygonWebSocketService.onUpdate(ticker, (bar) => {
      console.log(`[usePolygonWebSocket] Received bar for ${ticker}:`, bar);
      setLatestBar(bar);

      // Call custom callback if provided
      if (onBar) {
        onBar(bar);
      }
    });

    unsubscribeRef.current = unsubscribe;

    // Cleanup: unsubscribe when ticker changes or component unmounts
    return () => {
      if (unsubscribeRef.current) {
        console.log(`[usePolygonWebSocket] Unsubscribing from ${ticker}`);
        unsubscribeRef.current();
        unsubscribeRef.current = null;
      }
    };
  }, [ticker, isConnected, enabled, onBar]);

  // Reconnect function
  const reconnect = useCallback(async () => {
    connectionAttempted.current = false;
    setError(null);
    try {
      await polygonWebSocketService.connect();
      setIsConnected(true);
    } catch (err) {
      setError(err as Error);
      setIsConnected(false);
    }
  }, []);

  return {
    isConnected,
    latestBar,
    error,
    reconnect,
  };
}

/**
 * Convert AggregateBar to NormalizedChartData format
 */
export function aggregateBarToChartData(bar: AggregateBar): NormalizedChartData {
  return {
    time: bar.timestamp,
    open: bar.open,
    high: bar.high,
    low: bar.low,
    close: bar.close,
    volume: bar.volume,
  };
}
