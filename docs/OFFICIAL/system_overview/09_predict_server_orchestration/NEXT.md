# Predict Server / API Orchestration Layer - Future Development

## Planned Improvements

### 1. Async Request Handling

**Current Gap**: Flask is synchronous; slow requests block others.

**Improvement**: Migrate to async framework (FastAPI or async Flask).

```python
# FastAPI example
@app.get("/trading-directions")
async def get_trading_directions(ticker: str):
    data = await fetch_data_async(ticker)
    signal = await compute_signal_async(data)
    return signal
```

**Impact**: Higher throughput; better latency under load.

---

### 2. Request Rate Limiting

**Current Gap**: No rate limiting; susceptible to abuse.

**Improvement**:
```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=get_remote_address)

@app.route("/trading-directions")
@limiter.limit("60 per minute")
def trading_directions():
    ...
```

**Impact**: Protection against abuse; fair resource allocation.

---

### 3. Response Streaming

**Current Gap**: Full response computed before return.

**Improvement**: Stream partial results.

```python
def stream_signal():
    yield '{"status": "computing"}\n'

    rpe = compute_rpe()
    yield f'{{"rpe": {json.dumps(rpe)}}}\n'

    v6 = compute_v6()
    yield f'{{"v6": {json.dumps(v6)}}}\n'

    yield '{"status": "complete"}\n'
```

**Impact**: Faster perceived response; progressive display.

---

### 4. WebSocket Support

**Current Gap**: Polling required for updates.

**Improvement**: WebSocket for real-time signals.

```python
@socketio.on('subscribe')
def handle_subscribe(data):
    ticker = data['ticker']
    while True:
        signal = get_signal(ticker)
        emit('signal', signal)
        sleep(30)  # Update every 30 seconds
```

**Impact**: Real-time updates without polling.

---

### 5. Request Tracing

**Current Gap**: Limited visibility into request flow.

**Improvement**: Distributed tracing.

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@app.route("/trading-directions")
def trading_directions():
    with tracer.start_as_current_span("trading-directions") as span:
        span.set_attribute("ticker", ticker)

        with tracer.start_as_current_span("fetch-data"):
            data = fetch_data(ticker)

        with tracer.start_as_current_span("compute-rpe"):
            rpe = compute_rpe(data)

        with tracer.start_as_current_span("compute-v6"):
            v6 = compute_v6(data)

        return response
```

**Impact**: Debugging; performance analysis.

---

### 6. Circuit Breaker Pattern

**Current Gap**: No graceful degradation when dependencies fail.

**Improvement**:
```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30)
def fetch_polygon_data(ticker):
    return requests.get(polygon_url).json()

# If Polygon fails 5 times, circuit opens for 30 seconds
# During open state, returns cached/default data
```

**Impact**: Graceful degradation; faster recovery.

---

### 7. Multi-Region Deployment

**Current Gap**: Single Railway deployment.

**Improvement**:
```
US-East: genesis-us-east.railway.app
US-West: genesis-us-west.railway.app
EU:      genesis-eu.railway.app

Load Balancer: Route to nearest region
```

**Impact**: Lower latency; higher availability.

---

## Priority Matrix

| Improvement | Complexity | Impact | Priority |
|-------------|------------|--------|----------|
| Async handling | High | High | P1 |
| Rate limiting | Low | Medium | P1 |
| Response streaming | Medium | Medium | P2 |
| WebSocket support | High | High | P2 |
| Request tracing | Medium | High | P1 |
| Circuit breaker | Medium | Medium | P2 |
| Multi-region | High | Medium | P3 |

---

## Dependencies

| Improvement | Requires |
|-------------|----------|
| Async handling | Framework migration |
| Rate limiting | Redis or in-memory store |
| Response streaming | Client-side support |
| WebSocket support | Socket.IO or similar |
| Request tracing | OpenTelemetry, Jaeger |
| Circuit breaker | State management |
| Multi-region | Infrastructure changes |

---

## API Versioning Strategy

### Current
All endpoints are unversioned (implicit v1).

### Future
```
/v1/trading-directions  # Current behavior
/v2/trading-directions  # Future changes

# Version in header
X-API-Version: 2
```

### Breaking Change Policy
1. New versions for breaking changes
2. 6-month deprecation window
3. V1 maintained until EOL

---

## Monitoring Improvements

### Metrics to Add

| Metric | Description |
|--------|-------------|
| `request_latency_p99` | 99th percentile latency |
| `cache_hit_rate` | Signal cache effectiveness |
| `model_prediction_latency` | V6 prediction time |
| `rpe_computation_latency` | RPE computation time |
| `polygon_api_latency` | External API latency |
| `error_rate` | Errors per minute |

### Alerting Rules

| Alert | Condition | Action |
|-------|-----------|--------|
| High latency | p99 > 2s | Page on-call |
| High error rate | > 5% errors | Page on-call |
| Cache miss spike | < 50% hit rate | Investigate |
| Model load failure | Any model missing | Critical alert |

---

*Predict Server improvements focus on performance, reliability, and observability.*
