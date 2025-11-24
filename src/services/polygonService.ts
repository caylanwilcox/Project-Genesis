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
  private plan: 'free' | 'starter' | 'developer' | 'advanced' = 'free';

  constructor() {
    const apiKey = process.env.NEXT_PUBLIC_POLYGON_API_KEY || '';

    this.client = restClient(apiKey);

    // Detect plan from env and tune limits accordingly
    // NEXT_PUBLIC_POLYGON_PLAN can be 'free' | 'starter' | 'developer' | 'advanced'
    const planEnv = (process.env.NEXT_PUBLIC_POLYGON_PLAN || '').toLowerCase();
    if (planEnv === 'starter' || planEnv === 'developer' || planEnv === 'advanced') {
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
   * @param displayTimeframe Display timeframe for special date handling (e.g., 'YTD', '1Y', '5Y')
   * @returns Normalized chart data
   */
  async getAggregates(
    ticker: string,
    timeframe: Timeframe = '1h',
    limit: number = 100,
    displayTimeframe?: string
  ): Promise<NormalizedChartData[]> {
    // Include displayTimeframe in cache key to ensure different requests for 3M, 6M, YTD, etc.
    const cacheKey = `aggs_${ticker}_${timeframe}_${limit}_${displayTimeframe || 'default'}`;

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
        const toDate = this.getLastTradingDate(new Date(now));

        // Use special date calculation for specific display timeframes
        const fromDate = this.calculateDateRange(toDate, timeframe, limit, displayTimeframe);

        // For intraday timeframes (1m, 5m, 15m, 30m, 1h, 2h, 4h), use current date/time to get all data up to now
        // For daily+ timeframes, use today's date
        // Previously we added +1 day but this caused issues getting latest candles after market close
        const adjustedToDate = toDate;
        const toDateStr = this.formatDate(adjustedToDate);
        const fromDateStr = this.formatDate(fromDate);

        // Simplified logging
        console.log(`[🔍 FETCH] ${ticker} ${timeframe} (${displayTimeframe}): ${fromDateStr} to ${toDateStr}, limit=${limit}`);

        const response = await this.makeRequestWithRetry(async () => {
          // Using the Polygon.io REST client
          const result = await this.client.getStocksAggregates(
            ticker,
            config.multiplier,
            config.timespan as any, // Polygon client uses enum, we use string
            fromDateStr,
            toDateStr,
            true, // adjusted
            'asc' as any, // sort
            50000 // limit
          );
          return result;
        });

        // Trust presence of results over status (Polygon may return status 'DELAYED')
        if (response && Array.isArray(response.results) && response.results.length > 0) {
          const normalized = this.normalizeAggregates(response.results);
          const firstBar = normalized[0];
          const lastBar = normalized[normalized.length - 1];
          const firstDate = new Date(firstBar.time).toLocaleDateString();
          const lastDate = new Date(lastBar.time).toLocaleDateString();
          console.log(`[✅ DATA] Got ${normalized.length} bars: ${firstDate} → ${lastDate}`);
          return normalized;
        }

        // If no results, throw error with helpful message
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
   * Calculate date range based on display timeframe AND interval
   * The key insight: we need enough calendar time to get enough bars at the given interval
   * @param toDate End date (usually now)
   * @param timeframe Data interval (1m, 5m, 1h, 1d, etc.) - the granularity of bars
   * @param limit Number of bars to fetch
   * @param displayTimeframe Display period (1D, 5D, 1M, etc.) - what the user wants to view
   * @returns From date
   */
  private calculateDateRange(toDate: Date, timeframe: Timeframe, limit: number, displayTimeframe?: string): Date {
    const date = new Date(toDate.getTime());

    // Calculate trading hours per day (6.5 hours: 9:30am - 4:00pm ET)
    const tradingHoursPerDay = 6.5;

    // Helper: calculate calendar days needed for a given number of bars at this interval
    const getDaysForBars = (bars: number, interval: Timeframe): number => {
      switch (interval) {
        case '1m':
          return Math.ceil(bars / (tradingHoursPerDay * 60) * 1.5); // 1.5x for weekends
        case '5m':
          return Math.ceil(bars / (tradingHoursPerDay * 12) * 1.5);
        case '15m':
          return Math.ceil(bars / (tradingHoursPerDay * 4) * 1.5);
        case '30m':
          return Math.ceil(bars / (tradingHoursPerDay * 2) * 1.5);
        case '1h':
          return Math.ceil(bars / tradingHoursPerDay * 1.5);
        case '2h':
          return Math.ceil(bars / (tradingHoursPerDay / 2) * 1.5);
        case '4h':
          return Math.ceil(bars / (tradingHoursPerDay / 4) * 1.5);
        case '1d':
          return Math.ceil(bars * 1.5); // 1.5x for weekends
        case '1w':
          return Math.ceil(bars * 7);
        case '1M':
          return Math.ceil(bars * 30);
        default:
          return Math.ceil(bars * 1.5);
      }
    };

    // For display timeframes, calculate appropriate lookback with buffer for panning
    if (displayTimeframe) {
      let daysToGoBack = 0;

      switch (displayTimeframe) {
        case '1D':
          // 1 day view: load enough days to get all the bars we need
          daysToGoBack = getDaysForBars(limit, timeframe);
          break;

        case '5D':
          // 5 day view: explicitly request the last calendar week to cover 5 trading days
          daysToGoBack = Math.max(7, getDaysForBars(limit, timeframe));
          break;

        case '1M':
          // 1 month view: load enough days to get all the bars we need
          daysToGoBack = Math.max(90, getDaysForBars(limit, timeframe));
          break;

        case '3M':
          // 3 month view: only go back ~4 months plus buffer so we stay near current date
          daysToGoBack = Math.max(120, getDaysForBars(limit, timeframe));
          break;

        case '6M':
          // 6 month view: load enough days to get all the bars we need
          daysToGoBack = Math.max(540, getDaysForBars(limit, timeframe));
          break;

        case 'YTD':
          // Year to date - go to January 1st of current year
          date.setMonth(0);
          date.setDate(1);
          date.setHours(0, 0, 0, 0);
          console.log(`[PolygonService] YTD (${timeframe}): From ${date.toISOString()} to ${toDate.toISOString()}`);
          return date;

        case '1Y':
          // 1 year view: go back exactly 1 year
          date.setFullYear(date.getFullYear() - 1);
          console.log(`[PolygonService] 1Y (${timeframe}): From ${date.toISOString()} to ${toDate.toISOString()}`);
          return date;

        case '5Y':
          // 5 year view: go back exactly 5 years
          date.setFullYear(date.getFullYear() - 5);
          console.log(`[PolygonService] 5Y (${timeframe}): From ${date.toISOString()} to ${toDate.toISOString()}`);
          return date;

        case 'All':
          // All time: go back 20 years (or as far as data exists)
          date.setFullYear(date.getFullYear() - 20);
          console.log(`[PolygonService] All (${timeframe}): From ${date.toISOString()} to ${toDate.toISOString()}`);
          return date;

        default:
          daysToGoBack = getDaysForBars(limit, timeframe);
      }

      if (daysToGoBack > 0) {
        date.setDate(date.getDate() - daysToGoBack);
        console.log(`[PolygonService] ${displayTimeframe} (${timeframe}, ${limit} bars): Going back ${daysToGoBack} days from ${toDate.toISOString()} to ${date.toISOString()}`);
        return date;
      }
    }

    // Default: use bar-count-based calculation
    return this.calculateHistoricalFromDate(toDate, timeframe, limit);
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

  /**
   * Roll any provided date back to the most recent weekday (Mon-Fri)
   * so that aggregate queries always anchor on an actual trading session.
   */
  private getLastTradingDate(date: Date): Date {
    const adjusted = new Date(date.getTime());
    let safety = 7; // prevent infinite loops in extreme scenarios
    while (safety > 0) {
      const day = adjusted.getDay();
      if (day !== 0 && day !== 6) {
        return adjusted;
      }
      adjusted.setDate(adjusted.getDate() - 1);
      safety -= 1;
    }
    return adjusted;
  }
}

export const polygonService = new PolygonService();
