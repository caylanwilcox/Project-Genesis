export interface Target {
  tp: number;
  prob: number;
}

export interface Stop {
  sl: number;
  prob: number;
}

export interface Feature {
  name: string;
  weight: number;
}

export interface Signal {
  id: string;
  ts_emit: string;
  symbol: string;
  engine: string;
  direction: 'long' | 'short' | 'neutral';
  confidence: number;
  horizon: string;
  targets: Target[];
  stops: Stop[];
  explain?: string;
  features?: Feature[];
  hash: string;
}

export interface Engine {
  id: string;
  name: string;
  type: 'core' | 'background';
  active: boolean;
  weight: number;
}

export interface MarketData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Report {
  id: string;
  date: string;
  type: 'premarket' | 'midday' | 'eod';
  content: string;
  signals: Signal[];
  timestamp: string;
}