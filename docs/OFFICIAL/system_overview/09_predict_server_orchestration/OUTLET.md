# Predict Server / API Orchestration Layer - System Connections

## The Body Metaphor

The Predict Server is the **central nervous system** - the spinal cord and brainstem that coordinates all body functions. It receives sensory input (requests), processes it through specialized regions (RPE, V6, Policy), and produces motor output (trading signals).

Just as the central nervous system is the highway connecting brain to body, the Predict Server is the highway connecting ML models to trading actions.

---

## Upstream Connections

### What Predict Server Receives

| Source | Data | Usage |
|--------|------|-------|
| **Next.js Frontend** | HTTP requests | Trigger signal generation |
| **Dashboard UI** | Ticker selection | Route to correct model |
| **Replay Mode** | Historical params | Time-travel reconstruction |
| **Health Monitors** | /health requests | Status checks |

### Interface Contracts

**Frontend → Predict Server**
```
Request:
  GET /trading-directions?ticker=SPY

Response:
  {
    "ticker": "SPY",
    "action": "BUY_CALL",
    "direction": "LONG",
    "probability_a": 0.78,
    "probability_b": 0.82,
    "session": "late",
    ...
  }
```

**Replay → Predict Server**
```
Request:
  GET /replay?date=2025-12-20&time=14:30&ticker=SPY

Response:
  {
    "replay_timestamp": "2025-12-20T14:30:00",
    "signal": {...},
    "northstar": {...}
  }
```

---

## Downstream Connections

### What Predict Server Calls

| Component | Call | Purpose |
|-----------|------|---------|
| **Polygon API** | fetch_hourly_bars, fetch_daily_bars | Get market data |
| **RPE Pipeline** | NorthstarPipeline.run() | Market structure analysis |
| **V6 Model** | get_v6_prediction() | ML probabilities |
| **Policy Engine** | determine_action(), calculate_sizing() | Trading decisions |

### Orchestration Sequence

```
1. Predict Server receives request
2. Check signal cache (return if hit)
3. Call Polygon API for data
4. Pass data to RPE pipeline
5. If RPE allows, call V6 model
6. Pass V6 output to Policy Engine
7. Aggregate response
8. Cache signal
9. Return response
```

---

## Data Flow Position

```
┌─────────────────────────────────────────────────────────────┐
│                    EXTERNAL CLIENTS                          │
│              (Dashboard, API consumers)                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    PREDICT SERVER                            ┃
┃                                                              ┃
┃   ┌─────────────────────────────────────────────────────┐   ┃
┃   │                  HTTP LAYER                          │   ┃
┃   │  /trading-directions, /daily-signals, /replay, etc. │   ┃
┃   └──────────────────────┬──────────────────────────────┘   ┃
┃                          │                                   ┃
┃   ┌──────────────────────▼──────────────────────────────┐   ┃
┃   │                 ORCHESTRATOR                         │   ┃
┃   │  Cache check → Data fetch → RPE → V6 → Policy       │   ┃
┃   └──────────────────────┬──────────────────────────────┘   ┃
┃                          │                                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                          │
          ┌───────────────┼───────────────┬───────────────┐
          │               │               │               │
          ▼               ▼               ▼               ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   POLYGON   │   │     RPE     │   │  V6 MODEL   │   │   POLICY    │
│    API      │   │  (Phases    │   │  (Phase 5)  │   │   ENGINE    │
│             │   │   1-4)      │   │             │   │             │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
```

---

## Request Lifecycle

### Complete Request Flow

```
1. CLIENT REQUEST
   │
   └─► Parse parameters (ticker, date, time)
       │
2. CACHE CHECK
   │
   ├─► Cache hit? Return cached signal
   │
   └─► Cache miss? Continue...
       │
3. DATA FETCH
   │
   └─► Polygon: hourly_bars, daily_bars
       │
       ├─► Fetch error? Return NO_TRADE
       │
4. RPE PIPELINE
   │
   └─► Phase 1: Market structure
       └─► Phase 2: Signal health
           └─► Phase 3: Signal density
               └─► Phase 4: Execution posture
                   │
                   ├─► allowed=False? Return NO_TRADE
                   │
5. V6 PREDICTION
   │
   └─► Build features (29)
       └─► Scale features
           └─► Ensemble prediction
               └─► prob_a, prob_b, session
                   │
6. POLICY ENGINE
   │
   └─► Determine action
       └─► Calculate sizing
           └─► Calculate targets
               │
7. RESPONSE
   │
   └─► Cache signal
       └─► Return JSON
```

---

## Error Handling

### Error Categories

| Category | Example | Response |
|----------|---------|----------|
| Data error | Polygon API timeout | NO_TRADE + reason |
| Model error | Model not loaded | NO_TRADE + reason |
| Validation error | Invalid ticker | 400 Bad Request |
| Server error | Unexpected exception | 500 + logged |

### Error Response Format

```json
{
    "action": "NO_TRADE",
    "reason": "Data fetch failed: Polygon API timeout",
    "error_code": "DATA_FETCH_ERROR",
    "ticker": "SPY",
    "timestamp": "2026-01-03T14:30:00"
}
```

---

## Caching Strategy

### Signal Cache

| Key | Value | TTL |
|-----|-------|-----|
| `{ticker}_{date}_{hour}` | Full signal response | 60 minutes |

### Purpose
- Prevent flip-flopping within hour
- Reduce computation load
- Ensure consistent responses

### Cache Invalidation
- Automatic after 60 minutes
- Manual: Restart server
- New hour: New cache key

---

## System Health Indicators

### When Predict Server Is Healthy
- All endpoints responding
- Latency < 500ms (p95)
- Error rate < 1%
- All models loaded
- Polygon API reachable

### When Predict Server Signals Distress
- Endpoints timing out
- High error rate
- Model loading failures
- Polygon API unreachable
- Cache miss rate high

### System Response to Distress
1. Return NO_TRADE for safety
2. Log detailed error
3. Alert on-call if persistent
4. Circuit breaker activates

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAILWAY PLATFORM                          │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              genesis-production                      │   │
│   │                                                      │   │
│   │   Flask App ──► Port 5000 ──► Railway Load Balancer │   │
│   │                                                      │   │
│   │   Environment:                                       │   │
│   │   - POLYGON_API_KEY                                 │   │
│   │   - MODEL_VERSION                                   │   │
│   │   - PORT (auto)                                     │   │
│   │                                                      │   │
│   │   Deployed from: ml/ directory                      │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

*The Predict Server is the central orchestrator that ties all components into a cohesive trading system.*
