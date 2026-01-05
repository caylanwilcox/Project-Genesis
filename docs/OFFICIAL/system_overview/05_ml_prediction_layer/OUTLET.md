# ML Prediction Layer (Phase 5) - System Connections

## The Body Metaphor

Phase 5 is the **prefrontal cortex** - the region of the brain responsible for prediction, planning, and probabilistic reasoning. While the sensory systems (data sources) perceive, and the motor cortex (Policy Engine) acts, the prefrontal cortex decides what is likely to happen.

Unlike the reflexive amygdala (RPE phases 1-4, which react to structure), the prefrontal cortex weighs evidence, considers context, and estimates probabilities. It is the highest cognitive function in the system.

**Key insight**: The prefrontal cortex doesn't work alone. It receives input from all other brain regions and produces guidance - not commands. The motor cortex (Policy Engine) makes the final movement decision.

---

## Position in the System

```
┌─────────────────────────────────────────────────────────────┐
│                    EXTERNAL DATA                             │
│              (Polygon: hourly_bars, daily_bars)              │
│                   [The Senses]                               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASES 1-4: RPE PIPELINE                        │
│              (Market structure analysis)                     │
│                   [The Amygdala - Reactive]                  │
│                                                              │
│   Phase 1: Reality truth (levels, VWAP, auction)            │
│   Phase 2: Signal health scoring                             │
│   Phase 3: Density control (spam filter)                     │
│   Phase 4: Execution permission gate                         │
│                                                              │
│   OUTPUT: allowed: true/false, session, direction           │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            │                           │
            ▼ allowed=false             ▼ allowed=true
    ┌───────────────┐           ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    │   NO_TRADE    │           ┃     PHASE 5: V6 ML        ┃
    │   (Skip ML)   │           ┃   [The Prefrontal Cortex] ┃
    └───────────────┘           ┃                           ┃
                                ┃  Input:                   ┃
                                ┃   - hourly_bars           ┃
                                ┃   - daily_bars            ┃
                                ┃   - current_hour          ┃
                                ┃                           ┃
                                ┃  Processing:              ┃
                                ┃   - build_v6_features()   ┃
                                ┃   - scaler.transform()    ┃
                                ┃   - ensemble.predict()    ┃
                                ┃                           ┃
                                ┃  Output:                  ┃
                                ┃   - prob_a (Close>Open)   ┃
                                ┃   - prob_b (Close>11AM)   ┃
                                ┃   - session (early/late)  ┃
                                ┃   - price_11am            ┃
                                ┗━━━━━━━━━━━━━┯━━━━━━━━━━━━━┛
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    POLICY ENGINE                             │
│              (Action + Sizing + Targets)                     │
│                   [The Motor Cortex - Action]                │
│                                                              │
│   Input: prob_a, prob_b, session, price_11am                │
│   Output: action, sizing, entry, exit                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    PREDICT SERVER                            │
│              (API Response Packaging)                        │
│                   [The Brainstem - Coordination]             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    DASHBOARD / API                           │
│              (User-facing signal display)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Upstream Connections

### What Phase 5 Receives

| Source | Data | Contract | Evidence |
|--------|------|----------|----------|
| **Polygon API** | hourly_bars | List of {t, o, h, l, c, v} dicts | [predictions.py:33](ml/server/v6/predictions.py#L33) |
| **Polygon API** | daily_bars | List of {t, o, h, l, c, v} dicts | [predictions.py:33](ml/server/v6/predictions.py#L33) |
| **Predict Server** | current_hour | int (0-23, ET timezone) | [predictions.py:13](ml/server/v6/predictions.py#L13) |
| **RPE Phase 4** | allowed | bool - if False, V6 not called | RPE pipeline |
| **RPE Phase 1** | session | str - "early" or "late" | Session detection |

### Data Quality Requirements

| Requirement | Enforcement | Failure Mode |
|-------------|-------------|--------------|
| hourly_bars ≥ 1 | [predictions.py:33](ml/server/v6/predictions.py#L33) | Return None (NO_TRADE) |
| daily_bars ≥ 3 | [predictions.py:33](ml/server/v6/predictions.py#L33) | Return None (NO_TRADE) |
| All prices > 0 | Feature building | NaN replaced with 0 |
| Timestamps in ET | Feature timezone handling | Correct session detection |

---

## Downstream Connections

### What Phase 5 Produces

| Consumer | Data | Contract | Usage |
|----------|------|----------|-------|
| **Policy Engine** | prob_a, prob_b | float [0, 1] | Action determination |
| **Policy Engine** | session | "early" or "late" | Target selection |
| **Policy Engine** | price_11am | float or None | Target B reference |
| **Predict Server** | Full tuple | (prob_a, prob_b, session, price_11am) | API response |
| **Dashboard** | Probabilities | Displayed as confidence % | User signal |

### Interface Contract: Phase 5 → Policy Engine

```python
# V6 Output
(prob_a: float, prob_b: float, session: str, price_11am: Optional[float])

# Policy Engine Processing
def determine_action(prob: float) -> str:
    """
    NZ-1: prob > 0.55 → 'LONG'
    NZ-2: prob < 0.45 → 'SHORT'
    NZ-3-5: 0.45 ≤ prob ≤ 0.55 → 'NO_TRADE'
    """
    if prob > NEUTRAL_ZONE_HIGH:  # 0.55
        return 'LONG'
    elif prob < NEUTRAL_ZONE_LOW:  # 0.45
        return 'SHORT'
    else:
        return 'NO_TRADE'

# Session determines which probability to use
if session == 'early':
    action = determine_action(prob_a)  # Only Target A available
else:
    action = determine_action(prob_b)  # Use Target B in late session
```

### Interface Contract: Phase 5 → Dashboard

```json
{
    "ticker": "SPY",
    "action": "BUY_CALL",
    "probability_a": 0.78,
    "probability_b": 0.82,
    "confidence": 82,
    "session": "late",
    "price_11am": 592.50,
    "spec_version": "2026-01-03",
    "engine_version": "V6.1"
}
```

---

## Model Expansion Interface

### How New Models Plug In

The ML Prediction Layer is designed for expansion. New models must implement this interface:

```python
class PredictionModel(Protocol):
    """Standard interface for all prediction models"""

    def predict(
        self,
        hourly_bars: List[Dict],
        daily_bars: List[Dict],
        current_hour: int
    ) -> PredictionResult:
        """
        Args:
            hourly_bars: List of hourly OHLCV dicts
            daily_bars: List of daily OHLCV dicts
            current_hour: Current hour in ET

        Returns:
            PredictionResult with required fields
        """
        ...

@dataclass
class PredictionResult:
    """Standard output for all prediction models"""
    prob_primary: float      # Main probability [0, 1]
    prob_secondary: float    # Secondary probability [0, 1] or 0.5 if N/A
    session: str             # 'early' or 'late'
    reference_price: float   # Reference price for target (e.g., price_11am)
    model_id: str            # e.g., 'v6', 'gap_specialist', 'regime_v1'
    confidence_calibrated: bool  # Is probability calibrated?
    feature_importance: Optional[Dict[str, float]]  # For explainability
```

### Model Registration

```python
# In ml/server/models/registry.py (future)

model_registry = {
    'v6': {
        'module': 'server.v6.predictions',
        'function': 'get_v6_prediction',
        'weight': 1.0,      # Production weight
        'status': 'production',
        'sessions': ['early', 'late'],  # When active
        'spec_id': 'ML-V6'
    },
    # Future models register here
    'gap_specialist': {
        'module': 'server.gap.predictions',
        'function': 'get_gap_prediction',
        'weight': 0.0,      # Not yet active
        'status': 'testing',
        'sessions': ['early'],  # Morning only
        'spec_id': 'ML-GAP-1'
    }
}

def get_prediction(ticker, hourly_bars, daily_bars, current_hour):
    """Route to appropriate model(s) based on registry"""
    results = []
    for model_id, config in model_registry.items():
        if config['weight'] > 0:
            if session in config['sessions']:
                result = load_model(config).predict(...)
                results.append((result, config['weight']))

    # Weighted combination
    final_prob = sum(r.prob_primary * w for r, w in results) / sum(w for _, w in results)
    return final_prob
```

---

## Failure Modes & System Response

### Phase 5 Failure Scenarios

| Failure | Detection | System Response | Spec ID |
|---------|-----------|-----------------|---------|
| Model not loaded | `ticker not in models` | Return (None, None, None, None) → NO_TRADE | ML-FAIL-1 |
| Feature building fails | `result is None` | Return (None, None, None, None) → NO_TRADE | ML-FAIL-2 |
| NaN in features | `np.isnan(X)` | Replace with 0, log warning | ML-FAIL-3 |
| Probability out of range | `prob < 0 or prob > 1` | Clamp to [0, 1], log error | ML-FAIL-4 |
| Session ambiguous | `current_hour == 11` | Treat as 'late' (SC-3) | SC-3 |

### Failure Cascade Prevention

```
Phase 5 Failure
       │
       ├─► Can recover? (NaN → 0, clamp probs)
       │       │
       │       └─► Yes → Continue with degraded confidence
       │
       └─► Cannot recover? (model missing, features fail)
               │
               └─► Return (None, None, None, None)
                       │
                       └─► Predict Server → NO_TRADE
                               │
                               └─► Dashboard → "Model unavailable"
```

---

## Integration Points for Future Models

### 1. Pre-Prediction Hook (Regime Detection)

```python
# Before V6 runs
def pre_prediction_hook(hourly_bars, daily_bars):
    regime = detect_regime(hourly_bars, daily_bars)
    return {
        'regime': regime,
        'neutral_zone_adjustment': get_nz_adjustment(regime)
    }

# In V6 prediction
context = pre_prediction_hook(hourly_bars, daily_bars)
if context['regime'] == 'volatile':
    NEUTRAL_ZONE_LOW = 0.40  # Wider
    NEUTRAL_ZONE_HIGH = 0.60
```

### 2. Post-Prediction Hook (Confidence Adjustment)

```python
# After V6 runs
def post_prediction_hook(prob_a, prob_b, hourly_bars, daily_bars):
    cross_asset = get_cross_asset_signal(hourly_bars)

    if cross_asset_confirms(prob_a, cross_asset):
        return prob_a * 1.1, prob_b * 1.1  # Boost confidence
    elif cross_asset_conflicts(prob_a, cross_asset):
        return prob_a * 0.8, prob_b * 0.8  # Reduce confidence

    return prob_a, prob_b
```

### 3. Parallel Prediction (Model Ensemble)

```python
# Run multiple models in parallel
def ensemble_prediction(hourly_bars, daily_bars, current_hour):
    results = {}

    # V6 (always runs)
    results['v6'] = get_v6_prediction(...)

    # Gap specialist (early session only)
    if current_hour < 11:
        results['gap'] = get_gap_prediction(...)

    # Meta-model combines
    final_prob = meta_combine(results)
    return final_prob
```

---

## System Health Indicators

### When Phase 5 Is Healthy
- Models loaded for all tickers (SPY, QQQ, IWM)
- Feature building succeeds > 99% of requests
- Predictions fall in valid range [0, 1]
- Latency < 100ms (p95)
- Accuracy tracking shows no degradation

### When Phase 5 Signals Distress
- Model loading fails
- Feature building returns None frequently
- Predictions stuck at 0.5 (high uncertainty)
- High NaN rate in features
- Accuracy drops below 70%

### Health Monitoring (Future)

```python
ml_health = {
    'models_loaded': ['SPY', 'QQQ', 'IWM'],
    'feature_success_rate': 0.998,
    'avg_latency_ms': 45,
    'rolling_accuracy_7d': 0.85,
    'last_prediction_time': '2026-01-03T14:30:00',
    'status': 'healthy'
}
```

---

## Invariant Enforcement

| Invariant | How Enforced | Test |
|-----------|--------------|------|
| No future data leakage | Features use only past data | test_no_future_leakage |
| Same input = same output | Deterministic feature building | test_determinism |
| Session determines target | Explicit if/else | test_session_target_mapping |
| Probability in [0, 1] | Model output clamping | test_prob_bounds |
| Feature order matches training | V6_FEATURE_COLS constant | test_feature_order |
| Only Phase 5 produces ML | RPE is deterministic | test_p5_3, test_p5_4 |

---

## Connection Ladder

```
Individual Feature
    │
    └─► 29 Features (build_v6_features)
            │
            └─► Scaled Features (StandardScaler)
                    │
                    └─► Ensemble Prediction (XGB + RF + LR)
                            │
                            └─► Raw Probabilities (prob_a, prob_b)
                                    │
                                    └─► Phase 5 Output Tuple
                                            │
                                            └─► Policy Engine
                                                    │
                                                    └─► Action + Sizing + Targets
                                                            │
                                                            └─► Predict Server Response
                                                                    │
                                                                    └─► Dashboard / API
                                                                            │
                                                                            └─► User
```

---

## Future Model Integration Checklist

When adding a new model, verify these connections:

- [ ] Implements `PredictionModel` protocol
- [ ] Returns `PredictionResult` dataclass
- [ ] Registered in model registry
- [ ] Has spec ID assigned (e.g., ML-V7-1)
- [ ] Has entry in SPEC_TEST_TRACE.md
- [ ] Failure mode returns None (fails closed)
- [ ] Latency < 200ms
- [ ] Accuracy > 55% (beats random)
- [ ] Backtest across all regimes
- [ ] Policy Engine integration tested
- [ ] Dashboard display verified
- [ ] No interference with existing models

---

*Phase 5 is the thinking layer. Its connections define how intelligence flows through the system - and how new intelligence will join.*
