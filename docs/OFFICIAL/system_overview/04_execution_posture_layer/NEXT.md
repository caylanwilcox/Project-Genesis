# Execution Posture Layer (Phase 4) - Future Development

## Planned Improvements

### 1. Unit Test Coverage for Phase 4

**Current Gap**: Phase 4 logic is tested implicitly through integration tests only.

**Improvement**:
```python
# tests/unit/test_phase4.py
def test_p4_denied_when_not_structural_edge():
    reality = RealityState(confidence_band=ConfidenceBand.CONTEXT_ONLY)
    health = SignalHealthState(stand_down=False)
    density = SignalDensityState(throttle=Throttle.OPEN)

    result = Phase4Engine().calculate(reality, health, density)

    assert result.allowed == False
    assert "Confidence band" in result.invalidation_context[0]

def test_p4_mode_is_trend_when_healthy_trend():
    reality = RealityState(
        confidence_band=ConfidenceBand.STRUCTURAL_EDGE,
        range=Range(state=RangeState.TREND)
    )
    health = SignalHealthState(tier=HealthTier.HEALTHY, stand_down=False)
    density = SignalDensityState(throttle=Throttle.OPEN)

    result = Phase4Engine().calculate(reality, health, density)

    assert result.execution_mode == ExecutionMode.TREND_CONTINUATION
```

**Impact**: Spec compliance verification.

---

### 2. Execution Mode Refinement

**Current Gap**: Only 4 modes; doesn't distinguish sub-types.

**Improvement**:

| Mode | Sub-Mode | Condition |
|------|----------|-----------|
| TREND_CONTINUATION | BREAKOUT | First expansion beyond level |
| TREND_CONTINUATION | PULLBACK | Retest after expansion |
| MEAN_REVERSION | RANGE_FADE | At range extreme |
| MEAN_REVERSION | VWAP_REVERSION | Extended from VWAP |
| SCALP | MOMENTUM | Quick momentum play |
| SCALP | FADE | Counter-trend scalp |

---

### 3. Dynamic Invalidation Thresholds

**Current Gap**: Invalidation context is static list of concepts.

**Improvement**: Include specific price levels.

```python
{
    'invalidation_context': [
        {
            'condition': 'Break acceptance',
            'trigger_price': 594.50,
            'direction': 'below'
        },
        {
            'condition': 'MTF conflict',
            'trigger_price': 600.00,
            'direction': 'above'
        }
    ]
}
```

**Impact**: Downstream systems can set precise stops/alerts.

---

### 4. Confidence Scoring

**Current Gap**: Binary allowed/denied; no confidence gradient.

**Improvement**:
```python
{
    'allowed': True,
    'execution_confidence': 85,  # 0-100
    'confidence_breakdown': {
        'structural': 90,
        'health': 80,
        'density': 85
    }
}
```

**Impact**: Policy Engine can scale sizing by confidence.

---

### 5. Historical Posture Tracking

**Current Gap**: No memory of previous posture decisions.

**Improvement**:
```python
{
    'current_posture': {...},
    'posture_history': {
        'last_change': '2026-01-03T14:30:00',
        'previous_bias': 'NEUTRAL',
        'transitions_last_hour': 2,
        'stability_score': 75
    }
}
```

**Impact**: Prevents whipsawing, enables posture stability checks.

---

## Priority Matrix

| Improvement | Complexity | Impact | Priority |
|-------------|------------|--------|----------|
| Unit test coverage | Low | High | P0 |
| Execution mode refinement | Medium | Medium | P2 |
| Dynamic invalidation | Medium | High | P1 |
| Confidence scoring | Medium | Medium | P2 |
| Posture tracking | High | Medium | P3 |

---

## Dependencies

| Improvement | Requires |
|-------------|----------|
| Unit test coverage | Test framework setup |
| Mode refinement | Enhanced Phase 1 context |
| Dynamic invalidation | Level prices from Phase 1 |
| Confidence scoring | Aggregation logic |
| Posture tracking | State persistence |

---

## Test Gap Analysis

| Category | Current Tests | Missing Tests |
|----------|---------------|---------------|
| Permission gating | 0 unit tests | Deny conditions |
| Bias determination | 0 unit tests | Direction mapping |
| Execution mode | 0 unit tests | Mode selection logic |
| Risk state | 0 unit tests | State transitions |
| Invalidation | 0 unit tests | Context generation |

---

*Phase 4 improvements focus on test coverage, richer mode selection, and actionable invalidation.*
