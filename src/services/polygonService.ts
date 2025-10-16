import { restClient } from '@polygon.io/client-js';
import {
  PolygonAggregatesResponse,
  PolygonPreviousCloseResponse,
  PolygonTickerSnapshot,
  NormalizedChartData,
  TimeframeConfig,
  Timeframe,
  TIMEFRAME_CONFIGS,
} from '../types/polygon';

class PolygonService {
  private client: ReturnType<typeof restClient>;
  private cache: Map<string, { data: any; timestamp: number }> = new Map();
  private cacheDuration: number = 30000; // default 30 seconds cache (free tier)
  private requestQueue: Array<() => Promise<any>> = [];
  private isProcessingQueue: boolean = false;
  private minRequestInterval: number = 13000; // default 13 seconds between requests (free tier)
  private lastRequestTime: number = 0;
  private plan: 'free' | 'starter' | 'developer' = 'free';

  constructor() {
    const apiKey = process.env.NEXT_PUBLIC_POLYGON_API_KEY || '';

    this.client = restClient(apiKey);

    // Detect plan from env and tune limits accordingly
    // NEXT_PUBLIC_POLYGON_PLAN can be 'free' | 'starter' | 'developer'
    const planEnv = (process.env.NEXT_PUBLIC_POLYGON_PLAN || '').toLowerCase();
    if (planEnv === 'starter' || planEnv === 'developer') {
      this.plan = planEnv as any;
    }

    this.applyPlanSettings();

    console.log(`[PolygonService] Plan: ${this.plan}. minInterval=${this.minRequestInterval}ms cache=${this.cacheDuration}ms`);
  }

  private applyPlanSettings(): void {
    if (this.plan !== 'free') {
      // Paid plans: relax spacing and increase cache throughput
      this.minRequestInterval = 0; // fire requests back-to-back on paid plans
      this.cacheDuration = 3000; // fresher cache for near real-time
    } else {
      // Free defaults
      this.minRequestInterval = 13000;
      this.cacheDuration = 30000;
    }
  }

  /**
   * Get data from cache if available and not expired
   */
  private getCached<T>(key: string): T | null {
    const cached = this.cache.get(key);
    if (!cached) return null;

    const now = Date.now();
    if (now - cached.timestamp > this.cacheDuration) {
      this.cache.delete(key);
      return null;
    }

    console.log(`[Cache HIT] ${key}`);
    return cached.data as T;
  }

  /**
   * Store data in cache
   */
  private setCache(key: string, data: any): void {
    console.log(`[Cache SET] ${key}`);
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
    });
  }

  /**
   * Add request to queue and process with rate limiting
   */
  private async queueRequest<T>(key: string, fn: () => Promise<T>): Promise<T> {
    // Check cache first
    const cached = this.getCached<T>(key);
    if (cached !== null) {
      return cached;
    }

    return new Promise((resolve, reject) => {
      this.requestQueue.push(async () => {
        try {
          const result = await fn();
          // Only cache successful results (not errors or empty arrays)
          if (result !== null && result !== undefined) {
            this.setCache(key, result);
          }
          resolve(result);
        } catch (error) {
          // Don't cache errors - let them propagate
          reject(error);
        }
      });

      this.processQueue();
    });
  }

  /**
   * Process queued requests with rate limiting
   */
  private async processQueue(): Promise<void> {
    if (this.isProcessingQueue || this.requestQueue.length === 0) {
      return;
    }

    this.isProcessingQueue = true;

    while (this.requestQueue.length > 0) {
      const now = Date.now();
      const timeSinceLastRequest = now - this.lastRequestTime;

      // Wait if we need to respect rate limit
      if (timeSinceLastRequest < this.minRequestInterval) {
        const waitTime = this.minRequestInterval - timeSinceLastRequest;
        console.log(`[Rate Limit] Waiting ${(waitTime/1000).toFixed(1)}s before next request...`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
      }

      const request = this.requestQueue.shift();
      if (request) {
        this.lastRequestTime = Date.now();
        await request();
      }
    }

    this.isProcessingQueue = false;
  }

  /**
   * Make API request with retry logic for 429 errors
   */
  private async makeRequestWithRetry<T>(
    requestFn: () => Promise<T>,
    maxRetries: number = 2
  ): Promise<T> {
    let lastError: any;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await requestFn();
      } catch (error: any) {
        lastError = error;

        // Check if it's a 429 error (rate limit)
        const is429 = error.message?.includes('429') || error.status === 429;
        if (is429) {
          const waitTime = Math.pow(2, attempt) * 20000; // Exponential backoff: 20s, 40s, 80s
          console.warn(`[429 Rate Limit] Waiting ${waitTime/1000}s before retry ${attempt + 1}/${maxRetries}...`);
          await new Promise(resolve => setTimeout(resolve, waitTime));
          continue;
        }

        // For other errors, don't retry
        throw error;
      }
    }

    throw lastError;
  }

  /**
   * Get aggregates (bars/candles) for a stock ticker
   * @param ticker Stock symbol (e.g., 'AAPL', 'SPY')
   * @param timeframe Timeframe string (e.g., '1h', '1d')
   * @param limit Number of bars to fetch (default: 100)
   * @returns Normalized chart data
   */
  async getAggregates(
    ticker: string,
    timeframe: Timeframe = '1h',
    limit: number = 100
  ): Promise<NormalizedChartData[]> {
    const cacheKey = `aggs_${ticker}_${timeframe}_${limit}`;

    return this.queueRequest(cacheKey, async () => {
      try {
        const config = TIMEFRAME_CONFIGS[timeframe];
        if (!config) {
          throw new Error(`Invalid timeframe: ${timeframe}`);
        }

        // Calculate date range - go back far enough to get data even when market is closed
        // For intraday timeframes, we need to account for weekends and market hours
        // For daily+ timeframes, we can use simpler calculation
        const now = Date.now();
        const toDate = new Date(now);
        const fromDate = this.calculateHistoricalFromDate(toDate, timeframe, limit);

        console.log(`[PolygonService] Current timestamp: ${now}, toDate: ${toDate.toISOString()}`);
        console.log(`[PolygonService] Requesting ${limit} bars of ${timeframe} data for ${ticker} from ${this.formatDate(fromDate)} to ${this.formatDate(toDate)}`);

        const response = await this.makeRequestWithRetry(async () => {
          // Using the Polygon.io REST client
          const result = await this.client.getStocksAggregates(
            ticker,
            config.multiplier,
            config.timespan as any, // Polygon client uses enum, we use string
            this.formatDate(fromDate),
            this.formatDate(toDate),
            true, // adjusted
            'asc' as any, // sort
            50000 // limit
          );
          return result;
        });

        console.log(`[PolygonService] API Response:`, {
          status: response.status,
          resultsCount: response.resultsCount,
          queryCount: response.queryCount,
          hasResults: !!response.results,
          resultsLength: response.results?.length || 0,
        });

        // Trust presence of results over status (Polygon may return status 'DELAYED')
        if (response && Array.isArray(response.results) && response.results.length > 0) {
          const normalized = this.normalizeAggregates(response.results);
          console.log(`[PolygonService] Received ${normalized.length} bars, returning last ${Math.min(limit, normalized.length)}`);
          // Return up to the requested limit (most recent bars)
          return normalized.slice(-limit);
        }

        // If no results, throw error with helpful message
        console.error(`[PolygonService] No results from API. Full response:`, response);
        throw new Error(`No market data available for ${ticker}. The ticker symbol may be invalid or data may not be available for this timeframe.`);
      } catch (error: any) {
        console.error('Error fetching aggregates from Polygon.io:', error);
        throw error;
      }
    });
  }

  /**
   * Get previous day's close data
   */
  async getPreviousClose(ticker: string): Promise<NormalizedChartData | null> {
    const cacheKey = `prev_${ticker}`;

    return this.queueRequest(cacheKey, async () => {
      try {
        const response = await this.makeRequestWithRetry(async () => {
          return await this.client.getPreviousStocksAggregates(ticker, true);
        });

        if (response.status === 'OK' && response.results?.length > 0) {
          const result = response.results[0];
          return {
            time: result.t,
            open: result.o,
            high: result.h,
            low: result.l,
            close: result.c,
            volume: result.v,
          };
        }

        return null;
      } catch (error) {
        console.error('Error fetching previous close from Polygon.io:', error);
        throw error;
      }
    });
  }

  /**
   * Get real-time snapshot of a ticker
   */
  async getSnapshot(ticker: string): Promise<PolygonTickerSnapshot | null> {
    const cacheKey = `snapshot_${ticker}`;
    return this.queueRequest(cacheKey, async () => {
      try {
        const response = await this.makeRequestWithRetry(async () => {
          return await this.client.getStocksSnapshotTicker(ticker);
        });

        if (response.status === 'OK' && response.ticker) {
          // If snapshot works while we're in free mode, auto-upgrade plan heuristically
          if (this.plan === 'free') {
            this.plan = 'starter';
            this.applyPlanSettings();
            console.log(`[PolygonService] Detected paid capability via snapshot. Switched plan to ${this.plan}. minInterval=${this.minRequestInterval}ms cache=${this.cacheDuration}ms`);
          }
          return response.ticker as any; // Type compatibility - Polygon client types are slightly different
        }

        return null;
      } catch (error) {
        console.error('Error fetching snapshot from Polygon.io:', error);
        throw error;
      }
    });
  }

  /**
   * Get intraday data for today
   */
  async getIntradayData(
    ticker: string,
    interval: '1' | '5' | '15' | '30' = '5'
  ): Promise<NormalizedChartData[]> {
    try {
      const today = new Date();
      const startOfDay = new Date(today);
      startOfDay.setHours(0, 0, 0, 0);

      const response = await this.client.getStocksAggregates(
        ticker,
        parseInt(interval),
        'minute' as any,
        this.formatDate(startOfDay),
        this.formatDate(today),
        true, // adjusted
        'asc' as any, // sort
        50000 // limit
      );

      if (response.status === 'OK' && response.results) {
        return this.normalizeAggregates(response.results);
      }

      return [];
    } catch (error) {
      console.error('Error fetching intraday data from Polygon.io:', error);
      throw error;
    }
  }

  /**
   * Normalize Polygon.io aggregate data to our standard format
   */
  private normalizeAggregates(aggregates: any[]): NormalizedChartData[] {
    return aggregates.map((bar) => ({
      time: bar.t,
      open: bar.o,
      high: bar.h,
      low: bar.l,
      close: bar.c,
      volume: bar.v,
    }));
  }

  /**
   * Calculate from date for historical data
   * Goes back far enough in calendar time to ensure we get enough trading bars
   * Accounts for weekends, holidays, and market hours
   */
  private calculateHistoricalFromDate(toDate: Date, timeframe: Timeframe, limit: number): Date {
    const date = new Date(toDate.getTime()); // Create new date object to avoid mutation

    // Simple approach: go back enough days to cover the requested bars
    // We'll request more data than needed and let Polygon return what's available
    let daysToGoBack = 0;

    switch (timeframe) {
      case '1m':
      case '5m':
        // Intraday minute bars: go back enough days
        // For 200 bars of 5min data, we need about 1-2 trading days
        daysToGoBack = Math.max(7, Math.ceil(limit * 5 / (60 * 6.5))); // Convert to days
        break;
      case '15m':
      case '30m':
        // 15-30 min bars: need several trading days
        daysToGoBack = Math.max(14, Math.ceil(limit * (timeframe === '15m' ? 15 : 30) / (60 * 6.5)));
        break;
      case '1h':
      case '2h':
      case '4h':
        // Hourly bars: need weeks of data
        const hoursPerBar = parseInt(timeframe.replace('h', ''));
        daysToGoBack = Math.max(30, Math.ceil((limit * hoursPerBar) / 6.5));
        break;
      case '1d':
        // Daily bars: simple - about 1.4x the limit to account for weekends
        daysToGoBack = Math.ceil(limit * 1.5);
        break;
      case '1w':
        // Weekly bars
        daysToGoBack = limit * 7;
        break;
      case '1M':
        // Monthly bars
        date.setMonth(date.getMonth() - limit - 1); // Extra month for safety
        return date;
    }

    // Go back the calculated number of days
    date.setDate(date.getDate() - daysToGoBack);

    console.log(`[PolygonService] Calculated from date: going back ${daysToGoBack} days for ${limit} bars of ${timeframe}`);
    return date;
  }

  /**
   * Format date to YYYY-MM-DD for Polygon.io API
   */
  private formatDate(date: Date): string {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  }

  /**
   * Check if API key is configured
   */
  isConfigured(): boolean {
    const apiKey = process.env.NEXT_PUBLIC_POLYGON_API_KEY || '';
    return apiKey !== '' && apiKey !== undefined;
  }

  /** Plan helpers */
  getPlan(): 'free' | 'starter' | 'developer' {
    return this.plan;
  }
  isPaid(): boolean {
    return this.plan !== 'free';
  }
}

export const polygonService = new PolygonService();
