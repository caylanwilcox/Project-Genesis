// Polygon.io API Type Definitions

export interface PolygonAggregateBar {
  v: number;  // Volume
  vw: number; // Volume weighted average price
  o: number;  // Open
  c: number;  // Close
  h: number;  // High
  l: number;  // Low
  t: number;  // Timestamp (milliseconds)
  n: number;  // Number of transactions
}

export interface PolygonAggregatesResponse {
  ticker: string;
  queryCount: number;
  resultsCount: number;
  adjusted: boolean;
  results: PolygonAggregateBar[];
  status: string;
  request_id: string;
  count: number;
}

export interface PolygonPreviousCloseResponse {
  ticker: string;
  queryCount: number;
  resultsCount: number;
  adjusted: boolean;
  results: Array<{
    T: string;
    v: number;
    vw: number;
    o: number;
    c: number;
    h: number;
    l: number;
    t: number;
    n: number;
  }>;
  status: string;
  request_id: string;
}

export interface PolygonTickerSnapshot {
  ticker: string;
  todaysChangePerc: number;
  todaysChange: number;
  updated: number;
  day: {
    o: number;
    h: number;
    l: number;
    c: number;
    v: number;
    vw: number;
  };
  min: {
    av: number;
    t: number;
    n: number;
    o: number;
    h: number;
    l: number;
    c: number;
    v: number;
    vw: number;
  };
  prevDay: {
    o: number;
    h: number;
    l: number;
    c: number;
    v: number;
    vw: number;
  };
}

export interface PolygonQuote {
  ask_exchange: number;
  ask_price: number;
  ask_size: number;
  bid_exchange: number;
  bid_price: number;
  bid_size: number;
  participant_timestamp: number;
  sip_timestamp: number;
  conditions: number[];
  indicators: number[];
  sequence_number: number;
  tape: number;
}

export interface PolygonQuotesResponse {
  status: string;
  results: PolygonQuote[];
  next_url?: string;
}

// Normalized chart data format
export interface NormalizedChartData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface TimeframeConfig {
  multiplier: number;
  timespan: 'minute' | 'hour' | 'day' | 'week' | 'month' | 'quarter' | 'year';
  fromDate: Date;
  toDate: Date;
}

export type Timeframe = '1m' | '5m' | '15m' | '30m' | '1h' | '2h' | '4h' | '1d' | '1w' | '1M';

export const TIMEFRAME_CONFIGS: Record<Timeframe, Partial<TimeframeConfig>> = {
  '1m': { multiplier: 1, timespan: 'minute' },
  '5m': { multiplier: 5, timespan: 'minute' },
  '15m': { multiplier: 15, timespan: 'minute' },
  '30m': { multiplier: 30, timespan: 'minute' },
  '1h': { multiplier: 1, timespan: 'hour' },
  '2h': { multiplier: 2, timespan: 'hour' },
  '4h': { multiplier: 4, timespan: 'hour' },
  '1d': { multiplier: 1, timespan: 'day' },
  '1w': { multiplier: 1, timespan: 'week' },
  '1M': { multiplier: 1, timespan: 'month' },
};
