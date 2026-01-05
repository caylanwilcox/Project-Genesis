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

### Neutral Zone (SPEC NZ-1 through NZ-8)

| Probability | Action |
|-------------|--------|
| prob > 0.55 | LONG (BUY_CALL) |
| prob < 0.45 | SHORT (BUY_PUT) |
| 0.45 <= prob <= 0.55 | NO_TRADE |

### Boundary Precision

| prob | Action | Reason |
|------|--------|--------|
| 0.5500000001 | BULLISH | Above threshold |
| 0.5500000000 | NO_TRADE | At threshold (inclusive) |
| 0.4499999999 | BEARISH | Below threshold |
| 0.4500000000 | NO_TRADE | At threshold (inclusive) |

---

## Confidence Buckets (SPEC BK-1 through BK-6)

| prob Range | Bucket | size_mult |
|------------|--------|-----------|
| >= 0.90 or <= 0.10 | very_strong | 1.00 (100%) |
| >= 0.70 or <= 0.30 | strong | 0.75 (75%) |
| >= 0.60 or <= 0.40 | moderate | 0.50 (50%) |
| >= 0.55 or <= 0.45 | weak | 0.25 (25%) |
| (0.45, 0.55) | neutral | N/A (NO_TRADE) |

---

## Time Multipliers (SPEC TM-1 through TM-4)

| Hour (ET) | Multiplier | Reason |
|-----------|------------|--------|
| 13, 14, 15 | 1.0 | Peak accuracy hours |
| 11, 12 | 0.8 | Good accuracy |
| 10 | 0.6 | Moderate accuracy |
| < 10 | 0.4 | Early session, lower confidence |

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
- confidence_mult: From confidence bucket (0.25 to 1.0)
- time_mult: From time of day (0.4 to 1.0)
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

1. **Neutral zone is exclusive**: 45-55% always returns NO_TRADE
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
