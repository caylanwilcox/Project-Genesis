# ML Morning Briefing Widget - Frontend Integration

## Component Overview

| Property | Value |
|----------|-------|
| **Component** | `MLMorningBriefing` |
| **Location** | `src/components/MLMorningBriefing.tsx` |
| **Type** | Client Component ('use client') |
| **API Endpoint** | `/morning_briefing` |

## Features

1. **Market Direction Display** - Shows bullish/bearish/neutral for each ticker
2. **Probability Bars** - Visual representation of bullish probability
3. **Price Range Prediction** - Wide and shrinking range with capture rates
4. **Info Modal** - Explains how ML signals work
5. **Best Opportunity** - Highlights strongest signal
6. **Auto-refresh** - Fetches data on component mount

## Component Structure

```
MLMorningBriefing
├── InfoIcon (button)
├── InfoModal (overlay)
├── Header (Market Outlook Today)
├── Best Opportunity Card
├── Ticker Cards (SPY, QQQ, IWM)
│   ├── Direction Badge
│   ├── Probability Bar
│   ├── Current Price
│   ├── Wide Range
│   ├── Shrinking Range
│   └── Time Remaining
└── Overall Bias Footer
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   MLMorningBriefing.tsx                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. useEffect → fetch('/morning_briefing')                  │
│                                                             │
│  2. Parse response:                                         │
│     - tickers: { SPY, QQQ, IWM }                           │
│     - overall_bias, best_opportunity                        │
│                                                             │
│  3. For each ticker, display:                               │
│     - direction, emoji, bullish_probability                 │
│     - predicted_range.wide (full day)                       │
│     - predicted_range.shrinking (time-adjusted)             │
│     - capture rates for both ranges                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Interface Definitions

### TickerPrediction
```typescript
interface TickerPrediction {
  direction: string           // 'BULLISH' | 'BEARISH' | 'NEUTRAL'
  emoji: string               // '🟢' | '🔴' | '🟡'
  bullish_probability: number // 0.0 - 1.0
  confidence: number          // 0.0 - 1.0
  fvg_recommendation: string  // 'BULLISH' | 'BEARISH' | 'EITHER'
  current_price: number
  today_high?: number
  today_low?: number
  today_open?: number
  predicted_range: {
    wide?: { low: number; high: number }
    shrinking?: { low: number; high: number }
    time_remaining_pct?: number
    wide_capture_rate?: number
    shrinking_capture_rate?: number
    ml_predicted?: boolean
  }
  model_accuracy: number
  error?: string
}
```

### MorningBriefing
```typescript
interface MorningBriefing {
  generated_at: string
  market_day: string
  tickers: Record<string, TickerPrediction>
  overall_bias: string
  overall_emoji: string
  best_opportunity: {
    ticker: string
    confidence: number
    direction: string
  } | null
}
```

## Visual Components

### Direction Badge
```tsx
<span className={`px-2 py-1 rounded text-xs font-bold ${
  direction === 'BULLISH' ? 'bg-green-900 text-green-300' :
  direction === 'BEARISH' ? 'bg-red-900 text-red-300' :
  'bg-yellow-900 text-yellow-300'
}`}>
  {emoji} {direction}
</span>
```

### Probability Bar
```tsx
<div className="h-2 bg-gray-700 rounded">
  <div
    className={`h-full rounded ${
      bullish_probability > 0.6 ? 'bg-green-500' :
      bullish_probability < 0.4 ? 'bg-red-500' :
      'bg-yellow-500'
    }`}
    style={{ width: `${bullish_probability * 100}%` }}
  />
</div>
```

### Range Display
```tsx
<div className="text-xs">
  <div className="flex justify-between">
    <span className="text-gray-400">Wide Range:</span>
    <span className="text-white">
      ${wide.low.toFixed(2)} - ${wide.high.toFixed(2)}
      <span className="text-green-400 ml-1">({wide_capture_rate}%)</span>
    </span>
  </div>
  <div className="flex justify-between">
    <span className="text-gray-400">Target Range:</span>
    <span className="text-cyan-300">
      ${shrinking.low.toFixed(2)} - ${shrinking.high.toFixed(2)}
      <span className="text-green-400 ml-1">({shrinking_capture_rate}%)</span>
    </span>
  </div>
</div>
```

## Environment Variables

```env
NEXT_PUBLIC_ML_SERVER_URL=https://genesis-production-c1e9.up.railway.app
```

Fallback if not set: `https://genesis-production-c1e9.up.railway.app`

## Error Handling

```tsx
if (loading) {
  return <LoadingSpinner />
}

if (error) {
  return <ErrorMessage message={error} />
}

if (!briefing) {
  return <NoDataMessage />
}
```

## Code Reference

```
src/components/MLMorningBriefing.tsx:1-300
```

---

Last Verified: December 8, 2025
