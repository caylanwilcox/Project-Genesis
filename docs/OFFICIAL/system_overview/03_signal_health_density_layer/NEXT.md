# Signal Health & Density Layer (Phases 2-3) - Future Development

## Planned Improvements

### 1. Unit Test Coverage for Phase 2-3

**Current Gap**: Health and density scoring are tested implicitly through integration tests. No dedicated unit tests verify scoring logic in isolation.

**Improvement**:
```python
# tests/unit/test_phase2.py
def test_p2_no_acceptance_penalty():
    reality = RealityState(acceptance=Acceptance(accepted=False))
    health = Phase2Engine().calculate(reality)
    assert health.structural_integrity <= 75  # -25 penalty

def test_p2_stand_down_threshold():
    # health_score < 45 → stand_down = True
    ...

# tests/unit/test_phase3.py
def test_p3_same_level_spam():
    # >3 signals in 10m, <=1 level → -40 penalty
    ...
```

**Impact**: Spec compliance verification for health/density rules.

---

### 2. Dynamic Health Thresholds by Regime

**Current Gap**: Fixed threshold (45) for stand_down regardless of market regime.

**Improvement**:

| Regime | Stand-Down Threshold |
|--------|----------------------|
| High volatility | 35 (more lenient) |
| Normal | 45 (current) |
| Low volatility | 55 (stricter) |

**Rationale**: High-vol regimes naturally produce more "degraded" signals; low-vol regimes should be stricter.

---

### 3. Sliding Window Density Tracking

**Current Gap**: Density is computed per-request; no persistent window tracking.

**Improvement**:
```python
class DensityTracker:
    def __init__(self, window_minutes=10):
        self.signals: List[Signal] = []
        self.window = timedelta(minutes=window_minutes)

    def add_signal(self, signal: Signal):
        self.signals.append(signal)
        self._prune_old()

    def get_density_context(self) -> Dict:
        return {
            'signals_in_window': len(self.signals),
            'distinct_levels': len(set(s.level for s in self.signals)),
            'distinct_tfs': len(set(s.timeframe for s in self.signals))
        }
```

**Impact**: Accurate spam detection across requests.

---

### 4. Health Recovery Tracking

**Current Gap**: No concept of "recovery" from degraded state.

**Improvement**:
```python
{
    'health_score': 78,
    'tier': 'HEALTHY',
    'recovery': {
        'was_degraded': True,
        'degraded_at': '2026-01-03T13:45:00',
        'recovered_at': '2026-01-03T14:02:00',
        'recovery_bars': 5
    }
}
```

**Impact**: Avoids whipsawing between HEALTHY/DEGRADED on edge cases.

---

### 5. Participation Volume Profiling

**Current Gap**: Conviction is LOW/MEDIUM/HIGH based on simple volume ratio.

**Improvement**: Volume-at-price participation scoring.

| Volume Zone | Participation Score |
|-------------|---------------------|
| Above POC with volume | +20 |
| Below POC with volume | +20 |
| At POC (consolidation) | 0 |
| Against volume trend | -15 |

---

## Priority Matrix

| Improvement | Complexity | Impact | Priority |
|-------------|------------|--------|----------|
| Unit test coverage | Low | High | P0 |
| Dynamic thresholds | Medium | Medium | P2 |
| Sliding window density | Medium | High | P1 |
| Health recovery | Medium | Medium | P2 |
| Volume profiling | High | Medium | P3 |

---

## Dependencies

| Improvement | Requires |
|-------------|----------|
| Unit test coverage | Test framework setup |
| Dynamic thresholds | ATR regime from Phase 1 |
| Sliding window density | Persistent state (Redis/in-memory) |
| Health recovery | State persistence |
| Volume profiling | Volume-at-price data |

---

## Test Gap Analysis

| Category | Current Tests | Missing Tests |
|----------|---------------|---------------|
| Phase 2 Health | 0 unit tests | Dimension scoring, tier thresholds |
| Phase 2 Stand-Down | 0 unit tests | Threshold boundary tests |
| Phase 3 Density | 0 unit tests | Spam detection, throttle thresholds |
| Phase 2-3 Integration | Via predict_server | Isolated pipeline tests |

---

*Phases 2-3 improvements focus on explicit test coverage and adaptive thresholds.*
