# Predict Server / API Orchestration Layer - Current State

## What This Component Is

The Predict Server is the **central nervous system** of the trading platform - it orchestrates all components, routes requests, and produces the final trading signal. It's the Flask server that exposes HTTP endpoints and coordinates RPE, V6, and Policy Engine.

---

## What Predict Server Owns

| Responsibility | Implementation |
|----------------|----------------|
| **HTTP API** | Flask routes for all endpoints |
| **Component Orchestration** | Calls RPE → V6 → Policy in order |
| **Response Packaging** | Formats output for consumers |
| **Signal Caching** | 1-hour lock to prevent flip-flopping |
| **Model Loading** | Loads V6 models at startup |
| **Replay Mode** | Historical signal reconstruction |

---

## What Predict Server Does NOT Own

| Responsibility | Owned By |
|----------------|----------|
| Market structure analysis | RPE (Phases 1-4) |
| ML prediction | V6 Model |
| Sizing and targets | Policy Engine |
| Data fetching | Polygon API integration |
| Dashboard rendering | Next.js frontend |

---

## API Endpoints

### Primary Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/trading-directions` | GET | Main signal endpoint |
| `/daily-signals` | GET | Multi-ticker batch signals |
| `/signal-breakdown` | GET | Detailed signal analysis |
| `/rpe` | GET | RPE-only output |
| `/replay` | GET | Historical replay |
| `/health` | GET | Server health check |

### Trading Directions (Primary)

```
GET /trading-directions?ticker=SPY

Response:
{
    "ticker": "SPY",
    "action": "BUY_CALL",
    "direction": "LONG",
    "probability_a": 0.78,
    "probability_b": 0.82,
    "session": "late",
    "bucket": "strong",
    "position_pct": 22.5,
    "entry": {"price": 595.50},
    "exit": {"take_profit": 597.00, "stop_loss": 593.50},
    "northstar": {...},  # RPE output
    "spec_version": "2026-01-03",
    "engine_version": "V6.1"
}
```

### Daily Signals (Batch)

```
GET /daily-signals

Response:
{
    "signals": [
        {"ticker": "SPY", "action": "BUY_CALL", ...},
        {"ticker": "QQQ", "action": "NO_TRADE", ...},
        {"ticker": "IWM", "action": "BUY_PUT", ...}
    ],
    "timestamp": "2026-01-03T14:30:00",
    "market_status": "OPEN"
}
```

### Replay Mode

```
GET /replay?date=2025-12-20&time=14:30&ticker=SPY

Response:
{
    "ticker": "SPY",
    "replay_timestamp": "2025-12-20T14:30:00",
    "signal": {...},  # Signal as it would have appeared
    "northstar": {...}  # RPE as it would have been
}
```

---

## Orchestration Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    HTTP REQUEST                              │
│                 GET /trading-directions                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                1. CHECK CACHE                                │
│           If cached signal exists and valid, return it       │
└─────────────────────────┬───────────────────────────────────┘
                          │ (cache miss)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                2. FETCH DATA                                 │
│           Polygon API: hourly_bars, daily_bars               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                3. RUN RPE (Phases 1-4)                       │
│           Market structure, health, density, posture         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                4. RUN V6 PREDICTION                          │
│           Build features, run ensemble, get probs            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                5. RUN POLICY ENGINE                          │
│           Determine action, sizing, targets                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                6. CACHE SIGNAL                               │
│           Lock for 1 hour to prevent flip-flopping           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                7. RETURN RESPONSE                            │
│           JSON with signal, northstar, metadata              │
└─────────────────────────────────────────────────────────────┘
```

---

## Signal Caching

### Purpose
Prevent signal flip-flopping within the same hour.

### Implementation

```python
# Cache key: "{ticker}_{date}_{hour}"
signal_cache = {}
SIGNAL_LOCK_MINUTES = 60

def get_cached_signal(ticker, hour):
    cache_key = f"{ticker}_{now.strftime('%Y-%m-%d')}_{hour}"
    if cache_key in signal_cache:
        cached = signal_cache[cache_key]
        if (now - cached['locked_at']).seconds < SIGNAL_LOCK_MINUTES * 60:
            return cached['data']
    return None

def cache_signal(ticker, hour, data):
    cache_key = f"{ticker}_{now.strftime('%Y-%m-%d')}_{hour}"
    signal_cache[cache_key] = {'locked_at': now, 'data': data}
```

### No-Repainting Invariant (SPEC NR-1, NR-2)

| Spec ID | Rule | Implementation |
|---------|------|----------------|
| NR-1 | Signal locked after generation | 1-hour cache lock |
| NR-2 | Different hours = different signals | Hour in cache key |

---

## Model Loading

### Startup Sequence

```
1. Load V6 models for SPY, QQQ, IWM
2. Verify feature columns match V6_FEATURE_COLS
3. Log model versions and metrics
4. Start Flask server
```

### Error Handling

```python
if ticker not in intraday_v6_models:
    return None, None, None, None  # No prediction

# Caller handles None:
if prob_a is None:
    return {'action': 'NO_TRADE', 'reason': 'Model not available'}
```

---

## Spec Version Lock (SPEC SV-1 through SV-3)

| Spec ID | Rule | Implementation |
|---------|------|----------------|
| SV-1 | Response includes spec_version + engine_version | In all responses |
| SV-2 | spec_version matches locked YYYY-MM-DD | "2026-01-03" |
| SV-3 | engine_version follows V{major}.{minor} | "V6.1" |

### Implementation

```python
response = {
    ...
    'spec_version': '2026-01-03',
    'engine_version': 'V6.1'
}
```

---

## Output Contract (SPEC OC-1 through OC-4)

| Spec ID | Rule | Implementation |
|---------|------|----------------|
| OC-1 | Required response fields | action, ticker, session, probs |
| OC-2 | action enum valid | BUY_CALL / BUY_PUT / NO_TRADE |
| OC-3 | prob in [0,1] | Clamped to range |
| OC-4 | session valid | 'early' / 'late' |

---

## Deployment

### Railway Configuration

| Setting | Value |
|---------|-------|
| Runtime | Python 3.12 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python predict_server.py` |
| Port | 5000 (auto-detected) |
| Environment | `POLYGON_API_KEY`, `MODEL_VERSION` |

### Health Check

```
GET /health

Response:
{
    "status": "healthy",
    "models_loaded": ["SPY", "QQQ", "IWM"],
    "uptime_seconds": 3600,
    "version": "V6.1"
}
```

---

## Current Production Status

| Component | Status | Notes |
|-----------|--------|-------|
| Flask server | ✅ Production | Railway deployment |
| /trading-directions | ✅ Production | Primary endpoint |
| /daily-signals | ✅ Production | Batch endpoint |
| /replay | ✅ Production | Historical replay |
| /rpe | ✅ Production | RPE-only output |
| Signal caching | ✅ Production | 1-hour lock |
| Spec version | ✅ Production | In all responses |

---

## Test Coverage

| Spec ID | Rule | Status |
|---------|------|--------|
| OC-1 | Required response fields | ✅ Tested |
| OC-2 | action enum valid | ✅ Tested |
| OC-3 | prob in [0,1] | ✅ Tested |
| OC-4 | session valid | ✅ Tested |
| NR-1 | Signal locked after generation | ✅ Tested |
| NR-2 | Different hours = different signals | ✅ Tested |
| SV-1 | spec_version declared | ✅ Tested |
| SV-2 | spec_version matches locked | ✅ Tested |
| SV-3 | engine_version format | ✅ Tested |

---

*The Predict Server is the central orchestrator that coordinates all trading system components.*
