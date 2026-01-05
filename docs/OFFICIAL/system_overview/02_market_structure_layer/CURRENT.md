# Market Structure Layer (Phase 1) - Current State

## What This Component Is

Phase 1 is the **sensory input layer** of the RPE - the raw market observation system. It detects structural truth: where price is, where it was, and what levels matter. Phase 1 produces **immutable facts** that no downstream component can override.

---

## What Phase 1 Owns

| Responsibility | Implementation |
|----------------|----------------|
| **VWAP Calculation** | `rpe/vwap.py` - Volume-weighted average price |
| **Level Detection** | `rpe/levels.py` - Opening range, prior day, HTF levels |
| **Acceptance State** | `rpe/acceptance.py` - Level acceptance/rejection |
| **Auction Classification** | `rpe/auction_state.py` - RESOLVED/BALANCED/FAILED |
| **Context Computation** | `rpe/compute.py` - Intraday + Swing context aggregation |

---

## What Phase 1 Does NOT Own

| Responsibility | Owned By |
|----------------|----------|
| Health scoring | Phase 2 (Signal Health) |
| Spam filtering | Phase 3 (Signal Density) |
| Trade permission | Phase 4 (Execution Posture) |
| Probability prediction | Phase 5 (V6 ML Model) |
| Position sizing | Policy Engine |

---

## Core Data Structures

### Intraday Context

```python
{
    'version': '3.0',
    'ticker': 'SPY',
    'session': 'EARLY' | 'MID' | 'LATE',
    'current_price': 595.50,

    'auction': {
        'state': 'RESOLVED' | 'BALANCED' | 'FAILED_EXPANSION',
        'resolved_direction': 'UP' | 'DOWN' | 'BALANCED',
        'rotation_complete': bool,
        'expansion_quality': 'CLEAN' | 'DIRTY' | 'NONE'
    },

    'levels': {
        'set': [...],  # All active levels
        'nearest_above': {'name': str, 'price': float, 'distance_pct': float},
        'nearest_below': {'name': str, 'price': float, 'distance_pct': float}
    },

    'swing_link': {
        'alignment': 'ALIGNED' | 'CONFLICT' | 'NEUTRAL',
        'swing_bias': 'BULLISH_CONTEXT' | 'BEARISH_CONTEXT' | 'NEUTRAL_CONTEXT'
    }
}
```

### Swing Context

```python
{
    'version': '3.0',
    'ticker': 'SPY',
    'as_of_date': '2026-01-03',

    'auction': {
        'state': 'RESOLVED' | 'BALANCED',
        'resolved_direction': 'UP' | 'DOWN' | 'BALANCED',
        'dominant_tf': 'YEARLY' | 'QUARTERLY' | 'MONTHLY' | 'WEEKLY'
    },

    'bias': {
        'context': 'BULLISH_CONTEXT' | 'BEARISH_CONTEXT' | 'NEUTRAL_CONTEXT',
        'strength': 'STRONG' | 'MODERATE' | 'WEAK',
        'invalidation': {'description': str, 'level': float}
    },

    'levels': {'set': [...]}  # HTF levels
}
```

---

## Acceptance Logic

### Intraday Acceptance

Acceptance requires price to **close** beyond a level and **hold** for minimum bars:

| Closes Held | Strength | Time (5m bars) |
|-------------|----------|----------------|
| 3-4 | WEAK | 15-20 min |
| 5-8 | MODERATE | 25-40 min |
| 9+ | STRONG | 45+ min |

### Swing Acceptance

| Daily Closes | Strength |
|--------------|----------|
| 1 | WEAK |
| 2 | MODERATE |
| 3+ | STRONG |

### Acceptance States

| State | Meaning |
|-------|---------|
| `ACCEPTED` | Price closed beyond level and held |
| `REJECTED` | Price tested level but couldn't hold |
| `TESTING` | Price within threshold of level |
| `UNTESTED` | Price hasn't approached level |
| `FAILED_ACCEPTANCE` | Was accepted, now reversed |

---

## Auction State Classification

### Intraday Classification

| State | Condition |
|-------|-----------|
| `RESOLVED` | Broke opening range and accepted |
| `BALANCED` | Rotation complete or rejected at extremes |
| `FAILED_EXPANSION` | Broke out but failed acceptance |

### Direction Resolution

| Condition | Direction |
|-----------|-----------|
| OR high accepted, price > OR high | UP |
| OR low accepted, price < OR low | DOWN |
| Failed high or rotation complete | BALANCED |

---

## Level Types

### Intraday Levels

| Level | Source | Calculation |
|-------|--------|-------------|
| `vwap` | Cumulative | Σ(TP × Volume) / Σ(Volume) |
| `rth_open` | First 9:30 bar | bars_1m[0].open |
| `open_30m_high` | First 30 min | max(high) for 9:30-9:59 |
| `open_30m_low` | First 30 min | min(low) for 9:30-9:59 |
| `prior_day_high` | Yesterday | daily_bars[-2].high |
| `prior_day_low` | Yesterday | daily_bars[-2].low |
| `prior_day_close` | Yesterday | daily_bars[-2].close |

### Swing Levels

| Level | Source |
|-------|--------|
| `week_open` | First trading day of week |
| `month_open` | First trading day of month |
| `quarter_open` | First trading day of quarter |
| `year_open` | First trading day of year |
| `prior_week_high/low` | Previous week H/L |
| `prior_month_high/low` | Previous month H/L |

---

## Session Labels

| Time (ET) | Label |
|-----------|-------|
| 9:30 - 11:30 | EARLY |
| 11:30 - 14:00 | MID |
| 14:00 - 16:00 | LATE |

---

## Invariants

1. **Immutability**: Phase 1 outputs cannot be overridden by ML or policy
2. **Determinism**: Same inputs always produce same outputs
3. **No Future Leakage**: Only uses data available at computation time
4. **Context ID**: SHA256 hash ensures uniqueness and cacheability

---

## Current Production Status

| Component | Status | Notes |
|-----------|--------|-------|
| VWAP calculation | ✅ Production | RTH-only filtering |
| Level detection | ✅ Production | All intraday + swing |
| Acceptance logic | ✅ Production | Configurable thresholds |
| Auction classification | ✅ Production | Full state machine |
| Swing context | ✅ Production | Multi-timeframe |
| Context caching | ✅ Production | Deterministic IDs |

---

*Phase 1 is the foundation of market perception. Its accuracy determines all downstream decisions.*
