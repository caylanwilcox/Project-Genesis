# System Analysis - Project Genesis Trading Platform

**Document Version:** 1.0
**Last Updated:** December 23, 2025

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Data Flow](#data-flow)
4. [Component Deep Dive](#component-deep-dive)
5. [ML Pipeline](#ml-pipeline)
6. [API Reference](#api-reference)
7. [Deployment](#deployment)

---

## System Overview

Project Genesis is an intraday trading signal platform that combines:

- **Machine Learning** - V6 ensemble models for price direction prediction
- **Real-time Data** - Polygon.io API for live market data
- **Web Dashboard** - Next.js frontend for signal visualization
- **Python Backend** - Flask server for ML inference

### Core Value Proposition

The system predicts whether ETF prices (SPY, QQQ, IWM) will close higher or lower than the 11 AM price, with historical accuracy of **80%+ at high confidence levels**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│                    (Next.js Dashboard)                           │
│                   app/dashboard/page.tsx                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      NEXT.JS API LAYER                           │
│              app/api/v2/trading-directions/route.ts              │
│                                                                  │
│  • Proxies requests to ML server                                 │
│  • Handles errors and fallbacks                                  │
│  • Caches responses                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML PREDICTION SERVER                          │
│                   ml/predict_server.py                           │
│                                                                  │
│  • Flask REST API                                                │
│  • Loads V6 models on startup                                    │
│  • Fetches live data from Polygon                                │
│  • Computes features in real-time                                │
│  • Returns trading signals                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   V6 MODEL SPY   │ │   V6 MODEL QQQ   │ │   V6 MODEL IWM   │
│  spy_intraday    │ │  qqq_intraday    │ │  iwm_intraday    │
│     _v6.pkl      │ │     _v6.pkl      │ │     _v6.pkl      │
└──────────────────┘ └──────────────────┘ └──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POLYGON.IO DATA API                           │
│                                                                  │
│  • Hourly OHLCV bars                                            │
│  • Daily OHLCV bars                                             │
│  • Real-time quotes                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. Signal Request Flow

```
User opens dashboard
        │
        ▼
Dashboard calls /api/v2/trading-directions
        │
        ▼
Next.js API proxies to ML server
        │
        ▼
ML server fetches live data from Polygon
        │
        ▼
ML server computes 29 features for each ticker
        │
        ▼
Features scaled and passed to ensemble models
        │
        ▼
Ensemble outputs probability (0-1)
        │
        ▼
Server determines action (LONG/SHORT/NO_TRADE)
        │
        ▼
JSON response returned to dashboard
        │
        ▼
Dashboard displays signal with historical accuracy
```

### 2. Feature Computation Flow

```
Polygon API Response (Hourly Bars)
        │
        ├─► Extract today's OHLCV
        ├─► Extract previous day's OHLCV
        ├─► Extract 11 AM price
        │
        ▼
Compute Raw Features:
        │
        ├─► Gap features (gap, gap_size, gap_direction)
        ├─► Previous day features (prev_return, prev_range, etc.)
        ├─► Current session features (current_vs_open, position_in_range)
        ├─► Multi-day features (return_3d, return_5d, volatility_5d)
        ├─► Target B features (current_vs_11am, above_11am)
        │
        ▼
Scale Features (RobustScaler)
        │
        ▼
Pass to 4 Ensemble Models:
        │
        ├─► XGBoost (40% weight)
        ├─► Random Forest (25% weight)
        ├─► Gradient Boosting (20% weight)
        └─► Extra Trees (15% weight)
        │
        ▼
Weighted Average = Final Probability
```

---

## Component Deep Dive

### Frontend: Dashboard (app/dashboard/page.tsx)

**Purpose:** Display trading signals in real-time

**Key Functions:**

```typescript
// Fetches signals every 60 seconds
useEffect(() => {
  const fetchDirections = async () => {
    const response = await fetch('/api/v2/trading-directions')
    const data = await response.json()
    setTradingData(data)
  }
  fetchDirections()
  const interval = setInterval(fetchDirections, 60000)
}, [])

// Computes historical accuracy from confidence
const getHistoricalAccuracy = (probB: number) => {
  const confidence = Math.max(probB, 1 - probB) * 100
  if (confidence >= 90) return { accuracy: 100, label: 'Very Strong' }
  if (confidence >= 85) return { accuracy: 79, label: 'Strong Signal' }
  // ... etc
}
```

**Data Displayed:**
- Current price (from Polygon)
- Target A probability (Close > Open)
- Target B probability (Close > 11 AM)
- Action (LONG/SHORT/WAIT)
- Historical win rate
- Position size recommendation
- Stop loss / Take profit levels

---

### API Proxy: (app/api/v2/trading-directions/route.ts)

**Purpose:** Bridge between frontend and ML server

```typescript
export async function GET() {
  const ML_SERVER_URL = process.env.ML_SERVER_URL ||
    'https://genesis-production-c1e9.up.railway.app'

  const response = await fetch(`${ML_SERVER_URL}/trading_directions`, {
    cache: 'no-store',  // Always fresh data
  })

  const data = await response.json()
  return NextResponse.json(data)
}
```

**Why This Layer Exists:**
1. Hides ML server URL from client
2. Can add caching/rate limiting
3. Handles CORS automatically
4. Adds error handling

---

### ML Server: (ml/predict_server.py)

**Purpose:** Core prediction engine

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/trading_directions` | GET | Get signals for all tickers |
| `/predict/<ticker>` | GET | Get signal for specific ticker |

**Startup Sequence:**
```python
# Load models into memory on startup
MODELS = {}
for ticker in ['SPY', 'QQQ', 'IWM']:
    path = f'models/{ticker.lower()}_intraday_v6.pkl'
    with open(path, 'rb') as f:
        MODELS[ticker] = pickle.load(f)
```

**Prediction Logic:**
```python
def get_prediction(ticker):
    # 1. Fetch live data from Polygon
    hourly_bars = fetch_hourly_data(ticker)
    daily_bars = fetch_daily_data(ticker)

    # 2. Compute features
    features = compute_features(hourly_bars, daily_bars)

    # 3. Determine session
    current_hour = datetime.now(ET).hour
    session = 'late' if current_hour >= 12 else 'early'

    # 4. Get model and scaler
    model_data = MODELS[ticker]
    if session == 'late':
        scaler = model_data['scaler_late']
        models = model_data['models_late_b']  # Target B
        weights = model_data['weights_late_b']
    else:
        scaler = model_data['scaler_early']
        models = model_data['models_early']
        weights = model_data['weights_early']

    # 5. Scale features
    X_scaled = scaler.transform(features)

    # 6. Ensemble prediction
    prob = 0
    for name, model in models.items():
        prob += model.predict_proba(X_scaled)[:, 1][0] * weights[name]

    # 7. Determine action
    action = 'LONG' if prob > 0.5 else 'SHORT'
    if 0.45 <= prob <= 0.55:
        action = 'NO_TRADE'

    return prob, action
```

---

### V6 Model Files

**Location:** `ml/models/*.pkl`

**Contents of Each Pickle:**
```python
{
    'ticker': 'SPY',
    'version': 'v6_time_split',
    'trained_at': '2025-12-22',
    'feature_cols': [...],  # 29 feature names

    # Early session (9:30 AM - 12 PM)
    'scaler_early': RobustScaler(),
    'models_early': {
        'xgb': XGBClassifier(),
        'rf': RandomForestClassifier(),
        'gb': GradientBoostingClassifier(),
        'et': ExtraTreesClassifier(),
    },
    'weights_early': {'xgb': 0.4, 'rf': 0.25, 'gb': 0.2, 'et': 0.15},

    # Late session (12 PM - 4 PM)
    'scaler_late': RobustScaler(),
    'models_late_a': {...},  # Target A models
    'weights_late_a': {...},
    'models_late_b': {...},  # Target B models (PRIMARY)
    'weights_late_b': {...},
}
```

---

## ML Pipeline

### Training Flow (ml/train_time_split.py)

```
1. FETCH DATA
   └─► Polygon API: 3 years of hourly + daily data

2. COMPUTE FEATURES
   └─► 29 features for each trading day

3. CREATE TARGETS
   ├─► Target A: Close > Open (binary)
   └─► Target B: Close > 11 AM price (binary)

4. SPLIT BY TIME
   ├─► Early session samples (9:30 AM - 12 PM)
   └─► Late session samples (12 PM - 4 PM)

5. TRAIN ENSEMBLES
   For each (session, target):
   ├─► Train XGBoost
   ├─► Train Random Forest
   ├─► Train Gradient Boosting
   └─► Train Extra Trees

6. OPTIMIZE WEIGHTS
   └─► Grid search for best ensemble weights

7. SAVE MODEL
   └─► Pickle entire model + scalers + weights
```

### Feature Engineering Details

**Gap Features:**
```python
gap = (today_open - prev_close) / prev_close
gap_size = abs(gap)
gap_direction = 1 if gap > 0 else (-1 if gap < 0 else 0)
gap_filled = 1 if price crossed prev_close else 0
```

**Position Features:**
```python
position_in_range = (current_price - low_so_far) / (high_so_far - low_so_far)
above_open = 1 if current_price > today_open else 0
near_high = 1 if closer_to_high else 0
```

**Multi-day Features:**
```python
return_3d = (close_today - close_3d_ago) / close_3d_ago
volatility_5d = std(daily_returns[-5:])
mean_reversion_signal = -prev_return  # Contrarian signal
```

---

## API Reference

### GET /trading_directions

**Response:**
```json
{
  "current_time_et": "12:30 PM ET",
  "session": "late",
  "market_open": true,
  "best_ticker": "IWM",
  "tickers": {
    "SPY": {
      "action": "LONG",
      "probability_a": 0.85,
      "probability_b": 0.72,
      "confidence": 72,
      "position_pct": 15,
      "current_price": 595.23,
      "today_open": 594.10,
      "today_change_pct": 0.19,
      "stop_loss": 593.75,
      "take_profit": 598.21,
      "session": "late",
      "reason": "Good Signal - Target B (vs 11AM)",
      "model_accuracy": {
        "early": 0.65,
        "late_a": 0.70,
        "late_b": 0.76
      }
    },
    "QQQ": {...},
    "IWM": {...}
  },
  "summary": {
    "recommendation": "IWM showing strongest signal at 83% confidence"
  }
}
```

---

## Deployment

### Production Stack

| Component | Platform | URL |
|-----------|----------|-----|
| ML Server | Railway | genesis-production-c1e9.up.railway.app |
| Frontend | Vercel | (your-domain.vercel.app) |
| Data | Polygon.io | api.polygon.io |

### Environment Variables

**ML Server (Railway):**
```
POLYGON_API_KEY=your_key
PORT=8080
```

**Frontend (Vercel):**
```
ML_SERVER_URL=https://genesis-production-c1e9.up.railway.app
NEXT_PUBLIC_ML_SERVER_URL=https://genesis-production-c1e9.up.railway.app
```

### Deployment Commands

```bash
# Deploy ML server to Railway
cd ml && railway up

# Deploy frontend (auto-deploys on git push to main)
git push origin main
```

---

## System Limitations

1. **Market Hours Only** - Signals only valid during 9:30 AM - 4:00 PM ET
2. **ETFs Only** - Trained specifically for SPY, QQQ, IWM
3. **Intraday Only** - Not designed for overnight holds
4. **Data Dependency** - Requires Polygon.io API access
5. **Latency** - ~1-2 second delay from Polygon data

---

## Future Improvements

1. **Real-time Websockets** - Replace polling with live updates
2. **More Tickers** - Add individual stocks
3. **Options Integration** - Suggest options strategies
4. **Backtesting UI** - In-app historical analysis
5. **Alert System** - Push notifications for high-confidence signals
