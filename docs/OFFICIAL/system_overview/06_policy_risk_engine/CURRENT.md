# Policy & Risk Engine - Current State

## What This Component Is

The Policy Engine is the **spinal cord** of the trading platform - it translates high-level decisions into specific actions. While the brain (RPE + V6) decides "should we trade and in what direction?", the Policy Engine decides "exactly how much, at what price, with what stops?"

---

## What Policy Engine Owns

| Responsibility | Implementation |
|----------------|----------------|
| **Action Translation** | prob → BUY_CALL / BUY_PUT / NO_TRADE |
| **Confidence Buckets** | very_strong / strong / moderate / weak |
| **Position Sizing** | Base size × multipliers |
| **Time Multipliers** | Hour-based sizing adjustment |
| **Agreement Multipliers** | Target A/B alignment adjustment |
| **Entry/Exit Targets** | Take profit / Stop loss percentages |

---

## What Policy Engine Does NOT Own

| Responsibility | Owned By |
|----------------|----------|
| Market structure | Phase 1 |
| Signal health | Phase 2 |
| Spam filtering | Phase 3 |
| Trade permission | Phase 4 |
| Probability prediction | Phase 5 (V6) |
| Order execution | External broker |

---

## Action Translation Rules

### Neutral Zone (Updated 2026-01-03)

**IMPORTANT:** The neutral zone was widened from 45-55% to 25-75% because model accuracy is only reliable at probability extremes. See [config.py:35-38](ml/server/config.py#L35-L38) and [predictions.py:156-168](ml/server/v6/predictions.py#L156-L168).

| Probability | Action | Evidence |
|-------------|--------|----------|
| prob > 0.75 | LONG (BUY_CALL) | [predictions.py:188](ml/server/v6/predictions.py#L188) |
| prob < 0.25 | SHORT (BUY_PUT) | [predictions.py:191](ml/server/v6/predictions.py#L191) |
| 0.25 <= prob <= 0.75 | NO_TRADE | [predictions.py:186-187](ml/server/v6/predictions.py#L186-L187) |

### Boundary Precision

| prob | Action | Reason |
|------|--------|--------|
| 0.7500000001 | BULLISH | Above threshold |
| 0.7500000000 | NO_TRADE | At threshold (inclusive) |
| 0.2499999999 | BEARISH | Below threshold |
| 0.2500000000 | NO_TRADE | At threshold (inclusive) |

---

## Confidence Buckets (Updated 2026-01-03)

Buckets align with the widened neutral zone. See [predictions.py:74-107](ml/server/v6/predictions.py#L74-L107).

| prob Range | Bucket | Behavior |
|------------|--------|----------|
| >= 0.90 or <= 0.10 | very_strong_bull / very_strong_bear | Highest confidence |
| >= 0.85 or <= 0.15 | strong_bull / strong_bear | High confidence |
| >= 0.80 or <= 0.20 | moderate_bull / moderate_bear | Medium confidence |
| >= 0.75 or <= 0.25 | weak_bull / weak_bear | Minimum for trade |
| (0.25, 0.75) | neutral | NO_TRADE zone |

---

## Time Multipliers

Position size multipliers based on time of day. See [predictions.py:110-128](ml/server/v6/predictions.py#L110-L128).

| Hour (ET) | Multiplier | Reason |
|-----------|------------|--------|
| 13, 14 | 1.2 | Peak accuracy hours |
| 12 | 1.0 | Baseline |
| 15 | 0.8 | Late afternoon |
| < 12 | 0.7 | Morning session |
| >= 16 | 0.5 | After hours |

---

## Agreement Multipliers (SPEC AM-1 through AM-4)

| prob_a | prob_b | Agreement | Multiplier |
|--------|--------|-----------|------------|
| > 0.5 | > 0.5 | aligned_bullish | 1.2 |
| < 0.5 | < 0.5 | aligned_bearish | 1.2 |
| > 0.5 | < 0.5 | conflicting | 0.6 |
| < 0.5 | > 0.5 | conflicting | 0.6 |
| = 0.5 | any | neutral | 1.0 |

---

## Position Size Calculation

```
final_size = base_size × confidence_mult × time_mult × agreement_mult × risk_mult

Where:
- base_size: Account-defined base position (e.g., $1000)
- confidence_mult: From confidence bucket (weak to very_strong)
- time_mult: From time of day (0.5 to 1.2)
- agreement_mult: From A/B agreement (0.6 to 1.2)
- risk_mult: From Phase 4 risk_state
  - NORMAL: 1.0
  - REDUCED: 0.5
  - DEFENSIVE: 0.0 (no trade)
```

---

## Entry/Exit Targets (SPEC EX-1 through EX-4)

### Per-Ticker Targets

| Ticker | Take Profit | Stop Loss | Volatility |
|--------|-------------|-----------|------------|
| SPY | 0.25% | 0.33% | Lowest |
| QQQ | 0.34% | 0.45% | Medium |
| IWM | 0.45% | 0.60% | Highest |

### Direction Application

| Action | Take Profit | Stop Loss |
|--------|-------------|-----------|
| BUY_CALL | entry × (1 + TP%) | entry × (1 - SL%) |
| BUY_PUT | entry × (1 - TP%) | entry × (1 + SL%) |

---

## Target Selection (SPEC TS-1 through TS-3)

| Session | Primary Target |
|---------|----------------|
| early (hour < 11) | Target A (Close > Open) |
| late (hour >= 11) | Target B (Close > 11AM) |

---

## Output Structure

```json
{
    "ticker": "SPY",
    "action": "BUY_CALL",
    "direction": "LONG",

    "probabilities": {
        "probability_a": 0.78,
        "probability_b": 0.82
    },

    "session": "late",
    "bucket": "strong",

    "sizing": {
        "confidence_mult": 0.75,
        "time_mult": 1.0,
        "agreement_mult": 1.2,
        "position_pct": 22.5
    },

    "targets": {
        "entry_price": 595.50,
        "take_profit": 597.00,
        "stop_loss": 593.50
    },

    "reason": "Strong Bull - Target B at 82%, A/B aligned"
}
```

---

## Invariants

1. **Neutral zone is exclusive**: 25-75% always returns NO_TRADE (widened 2026-01-03)
2. **Sizing is multiplicative**: All factors multiply, never add
3. **Targets are ticker-specific**: Never use wrong ticker's percentages
4. **Session determines target**: Early=A, Late=B, no exceptions
5. **Defensive = no trade**: risk_state DEFENSIVE means position_pct=0

---

## Current Production Status

| Component | Status | Notes |
|-----------|--------|-------|
| Neutral zone | ✅ Production | 8 boundary tests pass |
| Confidence buckets | ✅ Production | 6 bucket tests pass |
| Time multipliers | ✅ Production | 4 time tests pass |
| Agreement multipliers | ✅ Production | 4 agreement tests pass |
| Entry/exit targets | ✅ Production | 4 target tests pass |
| Target selection | ✅ Production | 3 session tests pass |

---

## Test Coverage

| Category | Tests | Status |
|----------|-------|--------|
| Neutral Zone (NZ) | 8 | ✅ All pass |
| Confidence Buckets (BK) | 6 | ✅ All pass |
| Time Multiplier (TM) | 4 | ✅ All pass |
| Agreement Multiplier (AM) | 4 | ✅ All pass |
| Entry/Exit (EX) | 4 | ✅ All pass |
| Target Selection (TS) | 3 | ✅ All pass |

---

*The Policy Engine translates predictions into precise trading instructions.*
