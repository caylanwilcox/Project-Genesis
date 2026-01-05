# Execution Posture Layer (Phase 4) - Current State

## What This Component Is

Phase 4 is the **motor cortex** of the trading platform - the final gate before action. It synthesizes all upstream phases into a single execution decision: should we trade, and if so, with what posture? Phase 4 outputs framing (bias, mode, risk state) but NOT specific entries, exits, or sizes - those belong to the Policy Engine.

---

## What Phase 4 Owns

| Responsibility | Implementation |
|----------------|----------------|
| **Execution Permission** | allowed (bool) - final trade gate |
| **Directional Bias** | LONG / SHORT / NEUTRAL |
| **Execution Mode** | TREND_CONTINUATION / MEAN_REVERSION / SCALP / NO_TRADE |
| **Risk State** | NORMAL / REDUCED / DEFENSIVE |
| **Invalidation Context** | List of conditions that would invalidate the trade |

---

## What Phase 4 Does NOT Own

| Responsibility | Owned By |
|----------------|----------|
| Market structure | Phase 1 |
| Health scoring | Phase 2 |
| Spam filtering | Phase 3 |
| ML predictions | Phase 5 (V6) |
| Entry/exit prices | Policy Engine |
| Position sizing | Policy Engine |
| Stop loss / Take profit | Policy Engine |

---

## Core Data Structure

```python
@dataclass
class ExecutionState:
    allowed: bool = False                          # Final permission flag
    bias: Bias = Bias.NEUTRAL                      # LONG / SHORT / NEUTRAL
    execution_mode: ExecutionMode = ExecutionMode.NO_TRADE  # Trade style
    risk_state: RiskState = RiskState.DEFENSIVE    # Risk posture
    invalidation_context: List[str] = []           # What would invalidate
```

---

## Execution Permission Logic

### Deny Conditions (Any = DENY)

| Condition | Denial Reason |
|-----------|---------------|
| `reality.confidence_band != STRUCTURAL_EDGE` | "Confidence band: {band}" |
| `health.stand_down == True` | "Health stand_down active" |
| `density.throttle == BLOCKED` | "Density throttle BLOCKED" |

### Permission Flow

```python
def calculate(self, reality, health, density):
    deny = False
    invalidation = []

    if reality.confidence_band != ConfidenceBand.STRUCTURAL_EDGE:
        deny = True
        invalidation.append(f"Confidence band: {reality.confidence_band}")

    if health.stand_down:
        deny = True
        invalidation.append("Health stand_down active")

    if density.throttle == Throttle.BLOCKED:
        deny = True
        invalidation.append("Density throttle BLOCKED")

    if deny:
        return ExecutionState(
            allowed=False,
            execution_mode=ExecutionMode.NO_TRADE,
            bias=Bias.NEUTRAL,
            risk_state=RiskState.DEFENSIVE,
            invalidation_context=invalidation
        )

    # Proceed to framing...
```

---

## Bias Determination

| Phase 1 Direction | Bias |
|-------------------|------|
| `Direction.UP` | `Bias.LONG` |
| `Direction.DOWN` | `Bias.SHORT` |
| `Direction.BALANCED` | `Bias.NEUTRAL` |

---

## Execution Mode Selection

| Condition | Execution Mode |
|-----------|----------------|
| `range.state == TREND` AND `health.tier == HEALTHY` | TREND_CONTINUATION |
| `range.state == BALANCE` AND `health.tier != UNSTABLE` | MEAN_REVERSION |
| All other cases | SCALP |
| Permission denied | NO_TRADE |

### Mode Descriptions

| Mode | Trading Style |
|------|---------------|
| `TREND_CONTINUATION` | Trade with the trend, wider stops, larger targets |
| `MEAN_REVERSION` | Fade extremes, tighter stops, mean targets |
| `SCALP` | Quick in/out, tight stops, small targets |
| `NO_TRADE` | No position allowed |

---

## Risk State Determination

| Condition | Risk State |
|-----------|------------|
| `health.tier == HEALTHY` AND `density.throttle == OPEN` | NORMAL |
| `health.tier == DEGRADED` OR `density.throttle == LIMITED` | REDUCED |
| All other cases | DEFENSIVE |

### Risk State Behaviors

| Risk State | Downstream Impact |
|------------|-------------------|
| `NORMAL` | Full position sizing, standard stops |
| `REDUCED` | 50% position sizing, tighter stops |
| `DEFENSIVE` | No trading, or 25% max sizing |

---

## Invalidation Context

Phase 4 outputs a list of conditions that would invalidate the current posture:

```python
invalidation_context = [
    "Break acceptance",        # Level breaks
    "MTF conflict",           # Timeframe disagreement
    "Failure pattern activation"  # New failure signals
]
```

This provides downstream systems (and humans) with clear criteria for exiting or adjusting positions.

---

## Invariants

1. **Phase 4 is deterministic**: Same inputs always produce same outputs
2. **Phase 4 is gating**: If allowed=False, no downstream action
3. **Phase 4 does not size**: No position percentages or dollar amounts
4. **Phase 4 does not price**: No entry, exit, stop, or target prices
5. **Invalidation is forward-looking**: Lists conditions, not past events

---

## Current Production Status

| Component | Status | Notes |
|-----------|--------|-------|
| Permission gating | ✅ Production | Three deny conditions |
| Bias determination | ✅ Production | From Phase 1 direction |
| Execution mode | ✅ Production | Four modes |
| Risk state | ✅ Production | Three states |
| Invalidation context | ✅ Production | Human-readable |

---

## Test Coverage

| Spec ID | Rule | Status |
|---------|------|--------|
| P4-1 | allowed=False when confidence_band != STRUCTURAL_EDGE | ⚠️ Implicit |
| P4-2 | allowed=False when stand_down=True | ⚠️ Implicit |
| P4-3 | allowed=False when throttle=BLOCKED | ⚠️ Implicit |
| P4-4 | bias matches direction | ⚠️ Implicit |
| P4-5 | mode determined by range.state + health.tier | ⚠️ Implicit |

---

*Phase 4 is the final gate. Its decision determines whether any trade can proceed.*
