import { Signal, Engine, MarketData, Report } from '../types/Signal';

export const mockEngines: Engine[] = [
  { id: 'breakout_core', name: 'Breakout Core', type: 'core', active: true, weight: 0.25 },
  { id: 'divergence_core', name: 'Divergence Core', type: 'core', active: true, weight: 0.20 },
  { id: 'momentum_core', name: 'Momentum Core', type: 'core', active: false, weight: 0.15 },
  { id: 'mean_reversion_core', name: 'Mean Reversion', type: 'core', active: true, weight: 0.18 },
  { id: 'rsi_background', name: 'RSI Strategy', type: 'background', active: true, weight: 0.10 },
  { id: 'macd_background', name: 'MACD Strategy', type: 'background', active: false, weight: 0.08 },
  { id: 'bollinger_background', name: 'Bollinger Bands', type: 'background', active: true, weight: 0.12 },
];

export const mockSignals: Signal[] = [
  {
    id: 'sig-001',
    ts_emit: new Date().toISOString(),
    symbol: 'AAPL',
    engine: 'breakout_core',
    direction: 'long',
    confidence: 0.75,
    horizon: '1d',
    targets: [
      { tp: 195.50, prob: 0.7 },
      { tp: 198.00, prob: 0.3 }
    ],
    stops: [{ sl: 188.00, prob: 0.8 }],
    explain: 'Strong breakout above resistance with volume confirmation',
    features: [
      { name: 'volume_spike', weight: 0.4 },
      { name: 'resistance_break', weight: 0.6 }
    ],
    hash: 'abc123'
  },
  {
    id: 'sig-002',
    ts_emit: new Date(Date.now() - 3600000).toISOString(),
    symbol: 'TSLA',
    engine: 'divergence_core',
    direction: 'short',
    confidence: 0.62,
    horizon: '4h',
    targets: [
      { tp: 240.00, prob: 0.6 },
      { tp: 235.00, prob: 0.4 }
    ],
    stops: [{ sl: 252.00, prob: 0.7 }],
    explain: 'Bearish divergence on RSI with declining volume',
    hash: 'def456'
  },
  {
    id: 'sig-003',
    ts_emit: new Date(Date.now() - 7200000).toISOString(),
    symbol: 'SPY',
    engine: 'mean_reversion_core',
    direction: 'neutral',
    confidence: 0.55,
    horizon: '1h',
    targets: [],
    stops: [],
    explain: 'Market in consolidation, waiting for clear direction',
    hash: 'ghi789'
  }
];

export const mockMarketData: MarketData[] = Array.from({ length: 30 }, (_, i) => {
  const date = new Date();
  date.setDate(date.getDate() - (29 - i));
  const basePrice = 190 + Math.random() * 10;

  return {
    date: date.toISOString().split('T')[0],
    open: basePrice + Math.random() * 2 - 1,
    high: basePrice + Math.random() * 3,
    low: basePrice - Math.random() * 3,
    close: basePrice + Math.random() * 2 - 1,
    volume: Math.floor(50000000 + Math.random() * 30000000)
  };
});

export const mockReports: Report[] = [
  {
    id: 'report-001',
    date: new Date().toISOString().split('T')[0],
    type: 'premarket',
    content: `# Premarket Report

## Market Overview
- Futures pointing higher with SPX +0.5%
- VIX at 15.2, suggesting low volatility
- Dollar Index stable at 103.5

## Key Signals
- AAPL: Long signal with high confidence (75%)
- TSLA: Short signal on divergence
- SPY: Neutral, awaiting direction

## Regime Assessment
Current regime: NEUTRAL with slight bullish bias

## Risk Factors
- FOMC minutes release at 2:00 PM ET
- Earnings: NVDA, CRM after close`,
    signals: mockSignals.slice(0, 2),
    timestamp: new Date().toISOString()
  },
  {
    id: 'report-002',
    date: new Date().toISOString().split('T')[0],
    type: 'midday',
    content: `# Midday Market Update

## Performance
- SPY: +0.3%
- QQQ: +0.5%
- IWM: -0.1%

## Active Signals Update
- AAPL long performing well, approaching first target
- TSLA short signal triggered, monitoring position

## Volume Analysis
Above average volume in tech sector`,
    signals: mockSignals,
    timestamp: new Date(Date.now() - 14400000).toISOString()
  }
];