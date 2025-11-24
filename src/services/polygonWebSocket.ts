import { NormalizedChartData } from '@/types/polygon';

export type AggregateBar = {
  symbol: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp: number;
  vwap?: number;
};

export type Trade = {
  symbol: string;
  price: number;
  size: number;
  timestamp: number;
  conditions?: number[];
  exchange?: number;
};

type WebSocketCallback = (bar: AggregateBar) => void;
type TradeCallback = (trade: Trade) => void;

class PolygonWebSocketService {
  private ws: WebSocket | null = null;
  private apiKey: string = '';
  private isConnected: boolean = false;
  private subscribedSymbols: Set<string> = new Set();
  private callbacks: Map<string, WebSocketCallback[]> = new Map();
  private tradeCallbacks: Map<string, TradeCallback[]> = new Map();
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectDelay: number = 2000; // 2 seconds
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private connectPromise: Promise<void> | null = null;

  constructor() {
    if (typeof window !== 'undefined') {
      this.apiKey = process.env.NEXT_PUBLIC_POLYGON_API_KEY || '';
    }
  }

  /**
   * Initialize WebSocket connection
   */
  async connect(): Promise<void> {
    if (typeof window === 'undefined') {
      console.log('[PolygonWebSocket] Not in browser environment');
      return;
    }

    if (this.isConnected || !this.apiKey) {
      console.log('[PolygonWebSocket] Already connected or no API key');
      return;
    }

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.log('[PolygonWebSocket] WebSocket already open');
      this.isConnected = true;
      return;
    }

    if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
      console.log('[PolygonWebSocket] WebSocket connection already in progress');
      return;
    }

    if (this.connectPromise) {
      return;
    }

    try {
      console.log('[PolygonWebSocket] Connecting to Polygon.io WebSocket...');

      // Connect to Polygon WebSocket with API key
      const wsUrl = `wss://socket.polygon.io/stocks`;
      const socket = new WebSocket(wsUrl);
      this.ws = socket;

      // Handle connection open
      socket.onopen = () => {
        console.log('[PolygonWebSocket] ✓ WebSocket connected');

        // Authenticate with API key
        socket.send(JSON.stringify({ action: 'auth', params: this.apiKey }));
      };

      // Handle incoming messages
      socket.onmessage = (event) => {
        try {
          const messages = JSON.parse(event.data);
          if (!Array.isArray(messages)) return;

          messages.forEach((msg: any) => {
            this.handleMessage(msg);
          });
        } catch (error) {
          console.error('[PolygonWebSocket] Failed to parse message:', error);
        }
      };

      // Handle connection close
      socket.onclose = (event) => {
        console.log('[PolygonWebSocket] Connection closed', event.code, event.reason);
        this.isConnected = false;
        this.connectPromise = null;
        this.handleReconnect();
      };

      // Handle errors
      socket.onerror = (error) => {
        console.error('[PolygonWebSocket] Error:', error);
        this.isConnected = false;
      };

      // Wait for connection to be established
      this.connectPromise = new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Connection timeout'));
        }, 10000);

        const checkConnection = () => {
          if (this.isConnected) {
            clearTimeout(timeout);
            resolve();
          }
        };

        // Check periodically
        const interval = setInterval(() => {
          if (this.isConnected) {
            clearInterval(interval);
            clearTimeout(timeout);
            resolve();
          }
        }, 100);
      });
      await this.connectPromise;
      this.connectPromise = null;
    } catch (error) {
      console.error('[PolygonWebSocket] Failed to connect:', error);
      this.connectPromise = null;
      this.handleReconnect();
      throw error;
    }
  }

  /**
   * Handle automatic reconnection
   */
  private handleReconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }

    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[PolygonWebSocket] Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    console.log(`[PolygonWebSocket] Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    this.reconnectTimeout = setTimeout(() => {
      this.connect().catch(console.error);
    }, delay);
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(msg: any): void {
    // Handle different event types
    switch (msg.ev) {
      case 'status':
        console.log('[PolygonWebSocket] Status:', msg);
        if (msg.status === 'auth_success') {
          console.log('[PolygonWebSocket] ✓ Authenticated successfully');
          this.isConnected = true;
          this.reconnectAttempts = 0;

          // Resubscribe to any previously subscribed symbols
          this.subscribedSymbols.forEach(symbol => {
            this.subscribeToSymbol(symbol);
          });
        } else if (msg.status === 'auth_failed') {
          console.error('[PolygonWebSocket] ✗ Authentication failed');
        } else if (msg.status === 'success' && msg.message) {
          console.log('[PolygonWebSocket] Success:', msg.message);
        }
        break;

      case 'AM': // Per-minute aggregate
      case 'A':  // Per-second aggregate
        this.handleAggregateMessage(msg);
        break;

      case 'T': // Trade
        this.handleTradeMessage(msg);
        break;

      case 'Q': // Quote
        // Handle quotes if needed in the future
        break;

      default:
        // Log unknown message types for debugging
        if (msg.ev && msg.ev !== 'status') {
          console.log('[PolygonWebSocket] Unknown message type:', msg.ev, msg);
        }
        break;
    }
  }

  /**
   * Process aggregate bar messages
   */
  private handleAggregateMessage(msg: any): void {
    const symbol = msg.sym || msg.s;
    if (!symbol) {
      console.warn('[PolygonWebSocket] Aggregate message missing symbol:', msg);
      return;
    }

    const bar: AggregateBar = {
      symbol,
      open: msg.o || msg.op || 0,
      high: msg.h || 0,
      low: msg.l || 0,
      close: msg.c || msg.a || 0,
      volume: msg.v || 0,
      timestamp: msg.e || msg.t || msg.s || Date.now(),
      vwap: msg.vw,
    };

    console.log(`[PolygonWebSocket] Received ${msg.ev} for ${symbol}:`, bar);

    // Notify all callbacks for this symbol
    const callbacks = this.callbacks.get(symbol);
    if (callbacks && callbacks.length > 0) {
      callbacks.forEach(callback => callback(bar));
    }
  }

  /**
   * Process trade messages for live price updates
   */
  private handleTradeMessage(msg: any): void {
    const symbol = msg.sym || msg.s;
    if (!symbol) {
      console.warn('[PolygonWebSocket] Trade message missing symbol:', msg);
      return;
    }

    const trade: Trade = {
      symbol,
      price: msg.p || 0,
      size: msg.s || msg.sz || 0,
      timestamp: msg.t || Date.now(),
      conditions: msg.c,
      exchange: msg.x,
    };

    // Only log occasionally to avoid spam
    if (Math.random() < 0.1) {
      console.log(`[PolygonWebSocket] Trade for ${symbol}: $${trade.price.toFixed(2)}`);
    }

    // Notify all trade callbacks for this symbol
    const callbacks = this.tradeCallbacks.get(symbol);
    if (callbacks && callbacks.length > 0) {
      callbacks.forEach(callback => callback(trade));
    }
  }

  /**
   * Subscribe to per-second and per-minute aggregates and trades for a symbol
   */
  subscribeToSymbol(symbol: string): void {
    if (!this.ws || !this.isConnected) {
      console.warn('[PolygonWebSocket] Not connected, storing symbol for later subscription:', symbol);
      this.subscribedSymbols.add(symbol);
      return;
    }

    console.log(`[PolygonWebSocket] Subscribing to ${symbol} per-second aggregates, per-minute aggregates, and trades`);

    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('[PolygonWebSocket] Socket not ready, deferring subscription for', symbol);
      this.subscribedSymbols.add(symbol);
      return;
    }

    // Subscribe to per-second aggregates (A.*), per-minute aggregates (AM.*), and trades (T.*)
    // Per-second aggregates provide the most rapid chart updates
    this.ws.send(JSON.stringify({
      action: 'subscribe',
      params: `A.${symbol},AM.${symbol},T.${symbol}`
    }));

    this.subscribedSymbols.add(symbol);
  }

  /**
   * Unsubscribe from a symbol
   */
  unsubscribeFromSymbol(symbol: string): void {
    if (!this.ws || !this.isConnected) {
      this.subscribedSymbols.delete(symbol);
      return;
    }

    console.log(`[PolygonWebSocket] Unsubscribing from ${symbol}`);

    this.ws.send(JSON.stringify({
      action: 'unsubscribe',
      params: `A.${symbol},AM.${symbol},T.${symbol}`
    }));

    this.subscribedSymbols.delete(symbol);
    this.callbacks.delete(symbol);
    this.tradeCallbacks.delete(symbol);
  }

  /**
   * Register a callback for symbol updates
   */
  onUpdate(symbol: string, callback: WebSocketCallback): () => void {
    if (!this.callbacks.has(symbol)) {
      this.callbacks.set(symbol, []);
    }

    this.callbacks.get(symbol)!.push(callback);

    // Auto-subscribe if not already subscribed
    if (!this.subscribedSymbols.has(symbol)) {
      if (this.isConnected) {
        this.subscribeToSymbol(symbol);
      } else {
        // Store for subscription after connection
        this.subscribedSymbols.add(symbol);
        this.connect().catch(console.error);
      }
    }

    // Return unsubscribe function
    return () => {
      const callbacks = this.callbacks.get(symbol);
      if (callbacks) {
        const index = callbacks.indexOf(callback);
        if (index > -1) {
          callbacks.splice(index, 1);
        }

        // If no more callbacks, unsubscribe from symbol
        if (callbacks.length === 0) {
          this.unsubscribeFromSymbol(symbol);
        }
      }
    };
  }

  /**
   * Disconnect WebSocket
   */
  disconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      console.log('[PolygonWebSocket] Disconnecting...');
      this.subscribedSymbols.clear();
      this.callbacks.clear();
      this.ws.close();
      this.ws = null;
      this.isConnected = false;
    }
  }

  /**
   * Check if connected
   */
  getIsConnected(): boolean {
    return this.isConnected && this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Get subscribed symbols
   */
  getSubscribedSymbols(): string[] {
    return Array.from(this.subscribedSymbols);
  }

  /**
   * Register a callback for live trade updates (price updates)
   */
  onTrade(symbol: string, callback: TradeCallback): () => void {
    if (!this.tradeCallbacks.has(symbol)) {
      this.tradeCallbacks.set(symbol, []);
    }

    this.tradeCallbacks.get(symbol)!.push(callback);

    // Auto-subscribe if not already subscribed
    if (!this.subscribedSymbols.has(symbol)) {
      if (this.isConnected) {
        this.subscribeToSymbol(symbol);
      } else {
        // Store for subscription after connection
        this.subscribedSymbols.add(symbol);
        this.connect().catch(console.error);
      }
    }

    // Return unsubscribe function
    return () => {
      const callbacks = this.tradeCallbacks.get(symbol);
      if (callbacks) {
        const index = callbacks.indexOf(callback);
        if (index > -1) {
          callbacks.splice(index, 1);
        }
      }
    };
  }
}

// Singleton instance
export const polygonWebSocketService = new PolygonWebSocketService();
