# Northstar UI Display Standard

**Version:** 1.0
**Last Updated:** 2026-01-03
**Applies To:** Live Dashboard (`NorthstarPanel.tsx`) and Replay Mode (`app/replay/page.tsx`)

---

## Overview

This document defines the standard for displaying Northstar 4-phase pipeline data in the UI. The key principle is **SPECIFIC, DYNAMIC data** - never generic descriptions. Every piece of information must include actual prices, percentages, and calculated values.

---

## Core Principles

### 1. Show Actual Numbers, Not Descriptions

**BAD (Generic):**
```
Signal quality score: 79%. Data is clean, volatility is normal, patterns are clear.
```

**GOOD (Specific):**
```
Health Score: 79%
├── Structure:     85% ×30%
├── Time Persist:  72% ×15%
├── Volatility:    68% ×15%
├── Participation: 80% ×20%
└── Failure Risk:  65% ×20%
```

### 2. Include Price Levels

**BAD:**
```
FAILED_HIGH_BREAKOUT
```

**GOOD:**
```
FAILED_HIGH_BREAKOUT @ $595.43 (rejected, now $593.12)
```

### 3. Clickable Expandable Sections

All sections should be clickable to reveal detailed data. Collapsed state shows summary, expanded shows specifics.

---

## Data Structure

### Key Levels (Required in Phase 1)

```typescript
interface KeyLevels {
  recent_high: number      // 30-bar intraday high
  recent_low: number       // 30-bar intraday low
  mid_point: number        // (recent_high + recent_low) / 2
  pivot: number            // Classic pivot: (prev_high + prev_low + prev_close) / 3
  pivot_r1: number         // Resistance 1: 2 * pivot - prev_low
  pivot_s1: number         // Support 1: 2 * pivot - prev_high
  current_price: number    // Latest price
  today_open: number       // Daily bar open
  prev_close: number       // Previous day close
}
```

### Failure Patterns (With Price Context)

```python
# Format for failure types:
failure_types = [
    "FAILED_HIGH_BREAKOUT @ $595.43 (rejected, now $593.12)",
    "FAILED_LOW_BREAKOUT @ $588.20 (rejected, now $590.45)",
    "BEARISH_REVERSAL from $596.80 high",
    "BULLISH_REVERSAL from $585.30 low"
]
```

### Failed Levels (With Price Context)

```python
# Format for acceptance.failed_levels:
failed_levels = [
    "Rejected at $595.43 resistance (now $593.12, -0.39% off high)",
    "Bounced off $585.30 support (now $587.45, +0.37% off low)"
]
```

---

## UI Components

### 1. Direction Section (Phase 1)

**Collapsed View:**
```
▲ Bullish Structure                    Strong Edge
```

**Expanded View (on click):**
```
● Price Acceptance: YES - STRONG
    8/10 bars above $590.50 mid → price accepted ABOVE range
● Range State: TREND (CLEAN expansion) - Rotation complete
● Multi-Timeframe: ALIGNED - INTRADAY dominant
● Volume Conviction: HIGH
⚠ FAILURE DETECTED: FAILED_HIGH_BREAKOUT @ $595.43 (rejected, now $593.12)
```

### 2. Key Levels Section

**Collapsed View:**
```
KEY LEVELS    R1: $597.20  P: $594.50  S1: $591.80
```

**Expanded View (on click):**
```
Price Position in Range:
  Range High: $596.00              ← AT RESISTANCE (if near)
  ┌─────────────────────────────┐
  │ $593.12 (72% of range)      │  [visual bar]
  └─────────────────────────────┘
  Range Low:  $585.30              ← AT SUPPORT (if near)

Pivot Levels (from prev day):     Reference Prices:
  R1:    $597.20                    Open:       $592.10
  Pivot: $594.50                    Prev Close: $591.80
  S1:    $591.80                    Mid Point:  $590.65

Distance to Levels:
  To R1: +0.69%  |  To Pivot: +0.23%  |  To S1: -0.22%
```

### 3. Health Section (Phase 2)

**Collapsed View:**
```
Signal Health: Healthy (79%)
```

**Expanded View (on click):**
```
Health Score Breakdown:
  Structure      [████████░░] 85%  ×30%
  Time Persist   [███████░░░] 72%  ×15%
  Volatility     [██████░░░░] 68%  ×15%
  Participation  [████████░░] 80%  ×20%
  Failure Risk   [██████░░░░] 65%  ×20%

Issues Detected:
  • Volatility slightly elevated
  • Recent failure pattern detected
```

### 4. Trade Window Section (Phase 3)

**Collapsed View:**
```
Trade Window: Open
```

**Expanded View (on click):**
```
Density Score:    75%
Signals Allowed:  Unlimited

Throttle Reasons: (none)
```

### 5. Execution Section (Phase 4)

**Collapsed View:**
```
Play: Trend Following    Risk: Normal Size
```

**Expanded View (on click):**
```
Bias: LONG
Mode: TREND CONTINUATION

Exit if you see:
  ⚠ Break below $590.00
  ⚠ Volume divergence
```

### 6. Blocked Section (When Phase 4 Not Allowed)

**Collapsed View:**
```
⚠ Market conditions unfavorable - wait for better setup
```

**Expanded View (on click):**
```
Blocking Factors:
  ● Health score 42% (needs ≥45%)
  ● Density score 35% (needs ≥40%)
  ● Failure patterns: FAILED_HIGH_BREAKOUT @ $595.43

Wait for: Higher health score, clearer structure, or failure patterns to resolve.
```

---

## Implementation Files

| Component | File | Description |
|-----------|------|-------------|
| Live Dashboard | `src/components/NorthstarPanel.tsx` | Real-time Northstar display |
| Replay Mode | `app/replay/page.tsx` | Historical replay Northstar |
| Backend Pipeline | `ml/rpe/northstar_pipeline.py` | Key levels calculation |
| API Route | `app/api/v2/northstar/route.ts` | Frontend API proxy |

---

## Backend Requirements

### Phase 1 Output Must Include:

```python
{
    "direction": "UP" | "DOWN" | "BALANCED",
    "confidence_band": "STRUCTURAL_EDGE" | "CONTEXT_ONLY" | "NO_TRADE",
    "acceptance": {
        "accepted": bool,
        "acceptance_strength": "STRONG" | "MODERATE" | "WEAK",
        "acceptance_reason": "7/10 bars above $590.50 mid → price accepted ABOVE range",  # WHY it's strong/moderate
        "failed_levels": ["Rejected at $X.XX resistance (now $Y.YY, -Z.ZZ% off high)"]
    },
    "range": {
        "state": "TREND" | "BALANCE" | "FAILED_EXPANSION",
        "rotation_complete": bool,
        "expansion_quality": "CLEAN" | "DIRTY" | "NONE"
    },
    "mtf": {
        "aligned": bool,
        "dominant_tf": "INTRADAY" | "DAILY" | "WEEKLY",
        "conflict_tf": null | "description"
    },
    "participation": {
        "conviction": "HIGH" | "MEDIUM" | "LOW",
        "effort_result_match": bool
    },
    "failure": {
        "present": bool,
        "failure_types": ["FAILED_HIGH_BREAKOUT @ $X.XX (rejected, now $Y.YY)"]
    },
    "key_levels": {
        "recent_high": float,
        "recent_low": float,
        "mid_point": float,
        "pivot": float,
        "pivot_r1": float,
        "pivot_s1": float,
        "current_price": float,
        "today_open": float,
        "prev_close": float
    }
}
```

### Phase 2 Output Must Include:

```python
{
    "health_score": int,  # 0-100
    "tier": "HEALTHY" | "DEGRADED" | "UNSTABLE",
    "stand_down": bool,
    "reasons": ["Specific issue descriptions"],
    "dimensions": {
        "structural_integrity": int,      # 0-100
        "time_persistence": int,          # 0-100
        "volatility_alignment": int,      # 0-100
        "participation_consistency": int, # 0-100
        "failure_risk": int               # 0-100
    }
}
```

### Phase 3 Output Must Include:

```python
{
    "throttle": "OPEN" | "LIMITED" | "BLOCKED",
    "density_score": int,  # 0-100
    "allowed_signals": int,  # 0, 1, 999 for unlimited
    "reasons": ["Throttle reason descriptions"]
}
```

### Phase 4 Output Must Include:

```python
{
    "allowed": bool,
    "bias": "LONG" | "SHORT" | "NEUTRAL",
    "execution_mode": "TREND_CONTINUATION" | "MEAN_REVERSION" | "SCALP" | "NO_TRADE",
    "risk_state": "NORMAL" | "REDUCED" | "DEFENSIVE",
    "invalidation_context": ["Exit if you see X", "Break below $Y.YY"]
}
```

---

## Testing

Both live and replay modes should render identically given the same data. Tests should verify:

1. **Key levels display** - All price levels show correctly
2. **Failure patterns** - Include specific price context
3. **Expandable sections** - All sections expand/collapse
4. **Color coding** - Green/yellow/red based on state
5. **Distance calculations** - Correct percentage math

---

## Change History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-03 | Initial standard with key levels, specific failure patterns |

---

*This is the official standard. All Northstar UI implementations must follow these specifications.*
