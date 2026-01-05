# Signal Health & Density Layer (Phases 2-3) - Current State

## What This Component Is

Phases 2 and 3 are the **immune system** of the trading platform. Phase 2 (Health) detects degraded or unreliable signals. Phase 3 (Density) prevents signal spam and clustering. Together they act as gatekeepers, ensuring only healthy, material signals reach Phase 4.

---

## What Phases 2-3 Own

| Responsibility | Phase | Implementation |
|----------------|-------|----------------|
| **Health Scoring** | Phase 2 | `northstar_pipeline.py` - Phase2Engine |
| **Stand-Down Flag** | Phase 2 | Blocks signal generation when degraded |
| **Structural Integrity** | Phase 2 | Acceptance + MTF + Failure scoring |
| **Density Scoring** | Phase 3 | `northstar_pipeline.py` - Phase3Engine |
| **Throttle Control** | Phase 3 | OPEN / LIMITED / BLOCKED |
| **Spam Detection** | Phase 3 | Same-level clustering, TF saturation |

---

## What Phases 2-3 Do NOT Own

| Responsibility | Owned By |
|----------------|----------|
| Market structure observation | Phase 1 |
| Trade permission | Phase 4 |
| ML predictions | Phase 5 (V6) |
| Position sizing | Policy Engine |

---

## Phase 2: Signal Health State

### Data Structure

```python
@dataclass
class SignalHealthState:
    health_score: int = 0          # 0-100 aggregate score
    tier: HealthTier               # HEALTHY / DEGRADED / UNSTABLE
    stand_down: bool = True        # If True, no signals generated
    reasons: List[str]             # Human-readable degradation reasons

    # Individual dimensions (0-100 each)
    structural_integrity: int      # Acceptance + MTF + Failures
    time_persistence: int          # Stall detection
    volatility_alignment: int      # Vol dislocation check
    participation_consistency: int # Volume conviction
    failure_risk: int              # Active failure patterns
```

### Health Tiers

| Tier | Score Range | Behavior |
|------|-------------|----------|
| `HEALTHY` | >= 75 | Full signal generation |
| `DEGRADED` | 45-74 | Signals with reduced confidence |
| `UNSTABLE` | < 45 | stand_down = True, no signals |

### Dimension Weights

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| structural_integrity | 30% | Phase 1 acceptance + MTF alignment |
| time_persistence | 15% | Signal staleness, stall flags |
| volatility_alignment | 15% | ATR regime consistency |
| participation_consistency | 20% | Volume conviction + effort/result |
| failure_risk | 20% | Active failure patterns |

### Scoring Penalties

| Condition | Dimension | Penalty |
|-----------|-----------|---------|
| No acceptance | structural_integrity | -25 |
| MTF conflict | structural_integrity | -15 |
| Failure present | structural_integrity | -30 |
| Failed expansion | structural_integrity | -20 |
| Stall detected | time_persistence | -20 |
| Long time since acceptance | time_persistence | -10 |
| Vol dislocation | volatility_alignment | -25 |
| Low conviction | participation_consistency | -15 |
| Effort/result mismatch | participation_consistency | -20 |
| Active failures | failure_risk | -40 |

---

## Phase 3: Signal Density State

### Data Structure

```python
@dataclass
class SignalDensityState:
    density_score: int = 0         # 0-100 aggregate score
    throttle: Throttle             # OPEN / LIMITED / BLOCKED
    allowed_signals: int = 0       # Signals allowed this window
    reasons: List[str]             # Throttle reasons
```

### Throttle Levels

| Throttle | Score Range | allowed_signals |
|----------|-------------|-----------------|
| `OPEN` | >= 70 | Unlimited (999) |
| `LIMITED` | 40-69 | 1 signal |
| `BLOCKED` | < 40 | 0 signals |

### Density Penalties

| Condition | Penalty | Reason |
|-----------|---------|--------|
| Same-level spam (>3 signals in 10m, <=1 level) | -40 | Clustering |
| Too many TFs firing (>=3) | -30 | Timeframe saturation |
| Noise in balance (>4 signals in 10m, balanced market) | -30 | Regime incompatibility |

---

## Phase 2-3 Integration

```
Phase 1 Output (RealityState)
         │
         ▼
┌─────────────────────────────────────────┐
│           PHASE 2: HEALTH               │
│                                         │
│  structural_integrity ────┐             │
│  time_persistence ────────┼──► health_  │
│  volatility_alignment ────┤    score    │
│  participation_consistency┤             │
│  failure_risk ────────────┘             │
│                                         │
│  IF health_score < 45 → stand_down=True │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│           PHASE 3: DENSITY              │
│                                         │
│  clustering_check ─────┐                │
│  tf_saturation ────────┼──► density_    │
│  regime_compatibility ─┘    score       │
│                                         │
│  density_score → throttle (OPEN/LTD/BLK)│
└─────────────────────────────────────────┘
         │
         ▼
Phase 4 (Execution Posture)
```

---

## Invariants

1. **Phase 2 is gating**: If stand_down=True, Phase 3-4 receive degraded input
2. **Phase 3 is throttling**: If throttle=BLOCKED, no signals pass to Phase 4
3. **Reasons are auditable**: Every score penalty has a human-readable reason
4. **Scores are additive**: Penalties subtract from 100, min 0

---

## Current Production Status

| Component | Status | Notes |
|-----------|--------|-------|
| Health scoring | ✅ Production | 5 dimensions, weighted |
| Stand-down logic | ✅ Production | Tier-based gating |
| Density scoring | ✅ Production | 3 penalty categories |
| Throttle control | ✅ Production | OPEN/LIMITED/BLOCKED |
| Reason tracking | ✅ Production | Human-readable audit trail |

---

## Test Coverage

| Spec ID | Rule | Status |
|---------|------|--------|
| P2-1 | Health score >= 75 is HEALTHY | ⚠️ Implicit in integration tests |
| P2-2 | Health score < 45 sets stand_down | ⚠️ Implicit |
| P3-1 | Density >= 70 is OPEN | ⚠️ Implicit |
| P3-2 | Density < 40 is BLOCKED | ⚠️ Implicit |

---

*Phases 2-3 are the gatekeepers. They ensure only healthy, material signals reach execution.*
