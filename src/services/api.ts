import axios from 'axios';
import { Signal, Engine, MarketData, Report } from '../types/Signal';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

class ApiService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  async getSignals(symbol?: string, engine?: string): Promise<Signal[]> {
    try {
      const params = new URLSearchParams();
      if (symbol) params.append('symbol', symbol);
      if (engine) params.append('engine', engine);

      const response = await axios.get(`${this.baseUrl}/signals?${params}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching signals:', error);
      return [];
    }
  }

  async createSignal(signal: Omit<Signal, 'id' | 'hash'>): Promise<Signal | null> {
    try {
      const response = await axios.post(`${this.baseUrl}/signals`, signal);
      return response.data;
    } catch (error) {
      console.error('Error creating signal:', error);
      return null;
    }
  }

  async getEngines(): Promise<Engine[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/engines`);
      return response.data;
    } catch (error) {
      console.error('Error fetching engines:', error);
      return [];
    }
  }

  async updateEngine(engineId: string, updates: Partial<Engine>): Promise<Engine | null> {
    try {
      const response = await axios.patch(`${this.baseUrl}/engines/${engineId}`, updates);
      return response.data;
    } catch (error) {
      console.error('Error updating engine:', error);
      return null;
    }
  }

  async getMarketData(symbol: string, period: string = '1d'): Promise<MarketData[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/market/${symbol}?period=${period}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching market data:', error);
      return [];
    }
  }

  async getReports(type?: 'premarket' | 'midday' | 'eod'): Promise<Report[]> {
    try {
      const params = type ? `?type=${type}` : '';
      const response = await axios.get(`${this.baseUrl}/reports${params}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching reports:', error);
      return [];
    }
  }

  async generateReport(type: 'premarket' | 'midday' | 'eod', date?: string): Promise<Report | null> {
    try {
      const response = await axios.post(`${this.baseUrl}/reports/generate`, {
        type,
        date: date || new Date().toISOString().split('T')[0]
      });
      return response.data;
    } catch (error) {
      console.error('Error generating report:', error);
      return null;
    }
  }

  async runBacktest(symbol: string, engineId: string, startDate: string, endDate: string): Promise<any> {
    try {
      const response = await axios.post(`${this.baseUrl}/backtest`, {
        symbol,
        engine: engineId,
        start_date: startDate,
        end_date: endDate
      });
      return response.data;
    } catch (error) {
      console.error('Error running backtest:', error);
      return null;
    }
  }

  async getWeights(regime: 'bull' | 'bear' | 'neutral'): Promise<Record<string, number>> {
    try {
      const response = await axios.get(`${this.baseUrl}/weights/${regime}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching weights:', error);
      return {};
    }
  }
}

export const apiService = new ApiService();