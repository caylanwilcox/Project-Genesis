# ML Prediction Layer (Phase 5) - Future Development

## The Vision

V6 proves prediction is possible. What follows is **systematic expansion** - not replacing what works, but layering complementary intelligence that compounds into something greater than any single model.

The future is not one perfect model. It's an **orchestra of specialized predictors**, each tuned to a different aspect of market behavior, governed by rigorous testing and spec compliance.

---

## Evolution Principles

### 1. Additive, Not Replacement
New models complement V6. They don't replace it until proven superior across all dimensions.

### 2. Test Before Trust
Every new model must pass the same spec governance as V6:
- Feature integrity verified (no leakage)
- Backtest across multiple market regimes
- Integration tests with Policy Engine
- Spec ID assigned and traced

### 3. Fail Closed
If a new model can't produce a signal, fall back to V6. If V6 can't produce, fall back to NO_TRADE. Never guess.

### 4. Observable Intelligence
Every model must explain its confidence. Black boxes are not permitted in production.

---

## Model Evolution Roadmap

### Tier 1: V6 Enhancement (Foundation Hardening)

#### 1.1 Automated Retraining Pipeline

**Current Gap**: Models trained manually, no continuous learning.

**Improvement**:
```
Weekly Pipeline:
├── Fetch last 90 days of data
├── Split: Train (70%) / Validation (15%) / Test (15%)
├── Train ensemble on each ticker
├── Compare accuracy to production models
├── IF improved > 2%:
│   ├── Stage new model
│   ├── Run parallel predictions for 5 days
│   └── IF still better: Promote to production
└── Log results to TRAINING_LOG.md
```

**Invariants**: DS-1, DS-4, FS-1 must hold in retrained models.

**Tests to Add**:
- `test_retrained_model_feature_order`
- `test_retrained_model_accuracy_threshold`

**Priority**: P0 | **Complexity**: Medium | **Impact**: High

---

#### 1.2 Confidence Calibration

**Current Gap**: Raw probabilities may not reflect true outcome frequencies.

**Improvement**: Apply Platt scaling or isotonic regression.

```python
# Before: raw model output
prob_a = 0.78

# After: calibrated probability
# If model says 78%, outcomes should be 78% bullish
prob_a_calibrated = calibrator.predict_proba(prob_a)
```

**Evidence Required**: Calibration curve plots showing raw vs calibrated.

**Priority**: P1 | **Complexity**: Low | **Impact**: Medium

---

#### 1.3 Real-Time Degradation Detection

**Current Gap**: No monitoring for model drift or accuracy degradation.

**Improvement**:
```python
model_health = {
    'rolling_accuracy_7d': 0.85,
    'rolling_accuracy_30d': 0.88,
    'bias_direction': 0.02,  # Slight bullish bias
    'prediction_entropy': 0.42,  # Healthy variety
    'degradation_alert': False
}

# Alert conditions:
if rolling_accuracy_7d < 0.70:
    trigger_alert('MODEL_DEGRADATION')
if abs(bias_direction) > 0.10:
    trigger_alert('MODEL_BIAS')
```

**Priority**: P1 | **Complexity**: Medium | **Impact**: High

---

### Tier 2: Complementary Models (Specialized Intelligence)

#### 2.1 Regime Detection Model

**Purpose**: Classify current market regime before prediction.

**Targets**:
| Regime | Characteristics | V6 Behavior |
|--------|-----------------|-------------|
| Trending Up | Strong momentum, low pullbacks | Use momentum features heavily |
| Trending Down | Strong downward momentum | Use momentum features heavily |
| Mean Reverting | Oscillating, gap fills common | Use gap features heavily |
| Volatile | High ATR, unpredictable | Widen neutral zone |
| Low Vol | Tight ranges, slow moves | Narrow neutral zone |

**Implementation**:
```python
regime = detect_regime(daily_bars, hourly_bars)
# Returns: 'trending_up', 'trending_down', 'mean_reverting', 'volatile', 'low_vol'

if regime == 'volatile':
    NEUTRAL_ZONE_LOW = 0.40  # Wider
    NEUTRAL_ZONE_HIGH = 0.60
```

**Integration**: Runs BEFORE V6. Modifies V6's behavior or confidence thresholds.

**Priority**: P1 | **Complexity**: High | **Impact**: High

---

#### 2.2 Gap Fade Specialist

**Purpose**: Dedicated model for gap dynamics (morning session focus).

**Why Separate**: Gap dynamics are fundamentally different from trend continuation. V6 is optimized for late session when gaps have settled. A specialist can dominate the morning.

**Targets**:
| Target | Definition | Window |
|--------|------------|--------|
| Gap Fill | P(price returns to prev close) | 9:30-11:00 AM |
| Gap Extension | P(gap continues in same direction) | 9:30-11:00 AM |

**Features** (12 gap-specific):
- `gap_size_percentile` - Where does this gap rank historically?
- `gap_vs_atr` - Is gap larger than typical range?
- `opening_drive_direction` - First 5 minutes direction
- `opening_drive_strength` - First 5 minutes magnitude
- `overnight_sentiment` - Futures direction
- `gap_type` - Breakaway, continuation, exhaustion, common
- `previous_gap_fill_rate` - How often has this ticker filled gaps?
- `vix_level` - Volatility context
- `sector_gap_alignment` - Is sector gapping same way?
- `news_flag` - Is there a news catalyst?
- `earnings_proximity` - Days to/from earnings
- `first_bar_reversal` - Did first bar reverse the gap?

**Integration**: Active only during early session. V6 handles late.

**Priority**: P2 | **Complexity**: High | **Impact**: Medium

---

#### 2.3 Volatility Expansion Predictor

**Purpose**: Predict when large moves will occur (not direction, just magnitude).

**Why Valuable**: Position sizing optimization. Size up when big moves expected, size down when range-bound.

**Target**:
```python
Target = 1 if abs(close - open) / open > ATR_5d else 0
# P(today will have above-average range)
```

**Features**:
- `vix_change` - Is VIX expanding or contracting?
- `overnight_range` - Futures range overnight
- `gap_size` - Large gaps often precede large ranges
- `prev_day_range_vs_avg` - Was yesterday compressed or expanded?
- `options_volume_ratio` - Unusual options activity
- `time_to_event` - FOMC, CPI, earnings proximity
- `historical_volatility` - Rolling 10-day realized vol
- `implied_vs_realized` - Is market pricing more or less vol?

**Integration**: Feeds into Policy Engine for sizing multiplier.

**Priority**: P2 | **Complexity**: Medium | **Impact**: Medium

---

#### 2.4 Cross-Asset Confirmation Model

**Purpose**: Use related assets to confirm or deny signals.

**Insight**: SPY, QQQ, IWM often move together. Divergence is information.

**Features**:
- `spy_qqq_correlation_5d` - Are they moving together?
- `spy_iwm_ratio` - Large cap vs small cap preference
- `sector_rotation_signal` - Which sectors leading?
- `bond_equity_divergence` - TLT vs SPY disconnect
- `vix_spy_divergence` - Fear index vs equity index

**Target**:
```python
# When SPY signal aligns with cross-asset signals
Target = 1 if direction_confirmed else 0
```

**Integration**: Multiplier on V6 confidence.
- Cross-asset confirms: multiply confidence by 1.2
- Cross-asset diverges: multiply confidence by 0.7

**Priority**: P3 | **Complexity**: Medium | **Impact**: Medium

---

### Tier 3: Advanced Architecture (System Upgrades)

#### 3.1 Model Registry & A/B Testing

**Purpose**: Infrastructure to safely test new models in production.

```python
model_registry = {
    'v6_production': {
        'weight': 0.95,
        'status': 'production',
        'created': '2025-01-01'
    },
    'v7_candidate': {
        'weight': 0.05,  # 5% of traffic
        'status': 'testing',
        'created': '2026-01-03'
    }
}

# Production prediction uses weighted combination
final_prob = (
    prob_v6 * registry['v6_production']['weight'] +
    prob_v7 * registry['v7_candidate']['weight']
)
```

**Priority**: P2 | **Complexity**: Medium | **Impact**: High

---

#### 3.2 Meta-Learning: Model of Models

**Purpose**: Learn which model to trust in which conditions.

**How It Works**:
```python
# Meta-model inputs
meta_features = {
    'regime': 'trending_up',
    'hour': 14,
    'vix_level': 18,
    'recent_v6_accuracy': 0.82,
    'recent_gap_model_accuracy': 0.75
}

# Meta-model output
model_weights = meta_model.predict(meta_features)
# {'v6': 0.6, 'gap_specialist': 0.2, 'vol_predictor': 0.2}

# Weighted ensemble of ensembles
final_prob = sum(prob[model] * weight for model, weight in model_weights.items())
```

**Priority**: P3 | **Complexity**: Very High | **Impact**: Very High

---

#### 3.3 Neural Network Exploration

**Purpose**: Explore deep learning for pattern recognition.

**Constraints**:
- Must still produce interpretable probability [0, 1]
- Must integrate with existing Policy Engine
- Must have feature importance / attention weights for explainability
- Must not degrade to black box

**Architecture Ideas**:
- Transformer for sequential hourly bar data
- LSTM for momentum pattern recognition
- Attention mechanism over features for explainability

**Priority**: P3 | **Complexity**: Very High | **Impact**: Unknown

---

## Priority Matrix

| Improvement | Complexity | Impact | Priority | Prerequisite |
|-------------|------------|--------|----------|--------------|
| Retraining pipeline | Medium | High | P0 | None |
| Confidence calibration | Low | Medium | P1 | None |
| Degradation detection | Medium | High | P1 | None |
| Regime detection | High | High | P1 | Degradation detection |
| Model registry | Medium | High | P2 | Retraining pipeline |
| Gap fade specialist | High | Medium | P2 | Regime detection |
| Volatility predictor | Medium | Medium | P2 | None |
| Cross-asset model | Medium | Medium | P3 | Model registry |
| Meta-learning | Very High | Very High | P3 | Model registry, 2+ models |
| Neural network | Very High | Unknown | P3 | All above |

---

## Testing Requirements for New Models

Every new model MUST have:

### 1. Feature Integrity Tests
```python
def test_no_future_data_leakage():
    """Verify no feature uses future data"""
    for feature in model.feature_cols:
        assert feature_timestamp <= prediction_timestamp
```

### 2. Accuracy Threshold Tests
```python
def test_accuracy_exceeds_random():
    """Model must beat 50% baseline"""
    accuracy = backtest_model(model, test_data)
    assert accuracy > 0.55  # At least 5% edge
```

### 3. Determinism Tests
```python
def test_same_input_same_output():
    """Verify deterministic predictions"""
    prob1 = model.predict(features)
    prob2 = model.predict(features)
    assert prob1 == prob2
```

### 4. Policy Integration Tests
```python
def test_policy_engine_handles_model_output():
    """Verify Policy Engine correctly interprets model output"""
    prob = model.predict(features)
    action = policy_engine.determine_action(prob)
    assert action in ['LONG', 'SHORT', 'NO_TRADE']
```

### 5. Spec Trace Entry
Every new model must have:
- Spec ID assigned (e.g., `ML-V7-1`, `ML-GAP-1`)
- Entry in `SPEC_TEST_TRACE.md`
- Minimum 5 tests before production

---

## Governance for Model Expansion

### Adding a New Model
1. Write spec proposal in `docs/OFFICIAL/proposals/YYYY-MM-DD_model_<name>.md`
2. Define targets, features, integration point
3. Implement with full test suite
4. Backtest across all regimes (trending, reverting, volatile)
5. Run parallel with production for minimum 20 trading days
6. Review results
7. If approved: Add spec ID, update trace, deploy

### Deprecating a Model
1. Verify no production traffic
2. Move to `Deprecated/YYYY-MM/ml/models/<name>/`
3. Add to `DEPRECATION_LOG.md`
4. Preserve training code for reference

---

## Success Metrics for Model Evolution

| Metric | V6 Current | V7+ Target |
|--------|------------|------------|
| Late session accuracy | 89% | 92%+ |
| Early session accuracy | 67% | 75%+ |
| Regime-adaptive accuracy | N/A | 85%+ across all regimes |
| Calibration error | Unknown | < 5% |
| Model count in production | 1 | 3-5 specialized |
| A/B test infrastructure | None | Full registry |

---

## The Compounding Vision

```
Year 1 (V6):     Single model, proves edge exists
Year 2 (V7-V9):  Specialized models, regime awareness
Year 3 (V10+):   Meta-learning, model orchestration
Year 4+:         Self-improving system, continuous adaptation
```

Each improvement compounds. Regime detection makes gap specialist better. Gap specialist makes morning predictions better. Better morning predictions provide more training data. More data makes all models better.

The system gets smarter every day.

---

*V6 is the seed. What grows from it is unlimited - but only if we plant it in the soil of rigorous testing and spec governance.*
