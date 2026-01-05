# Execution Posture Layer (Phase 4) - System Connections

## The Body Metaphor

Phase 4 is the **motor cortex** - the final decision point before muscle activation. Just as the motor cortex receives processed sensory information and decides whether and how to move, Phase 4 receives processed market structure and decides whether and how to trade.

The motor cortex doesn't control individual muscle fibers (that's the spinal cord) - it decides on the action. Similarly, Phase 4 doesn't set position sizes or prices (that's the Policy Engine) - it decides on the posture.

---

## Upstream Connections

### What Phase 4 Enables

| Consumer | What It Receives | How It Uses It |
|----------|------------------|----------------|
| **V6 ML Model** | allowed, bias | Gates prediction usage |
| **Policy Engine** | execution_mode, risk_state | Determines sizing and targets |
| **Predict Server** | Full ExecutionState | Packages in API response |
| **Dashboard UI** | bias, risk_state | Visual posture indicator |

### Interface Contracts

**Phase 4 → V6 Model**
```
Input:  allowed, bias
Output: V6 behavior

IF allowed == False:
    THEN skip V6 prediction
    THEN action = NO_TRADE

IF bias == LONG:
    THEN V6 prediction used for CALL signals
IF bias == SHORT:
    THEN V6 prediction used for PUT signals
IF bias == NEUTRAL:
    THEN V6 prediction still used, but policy reduces size
```

**Phase 4 → Policy Engine**
```
Input:  execution_mode, risk_state
Output: Sizing and target parameters

IF risk_state == NORMAL:
    THEN position_size = base_size * confidence_mult * time_mult
IF risk_state == REDUCED:
    THEN position_size = base_size * 0.5 * confidence_mult * time_mult
IF risk_state == DEFENSIVE:
    THEN position_size = 0

IF execution_mode == TREND_CONTINUATION:
    THEN take_profit_mult = 1.5
    THEN stop_loss_mult = 1.0
IF execution_mode == MEAN_REVERSION:
    THEN take_profit_mult = 1.0
    THEN stop_loss_mult = 0.75
IF execution_mode == SCALP:
    THEN take_profit_mult = 0.5
    THEN stop_loss_mult = 0.5
```

---

## Downstream Protection

### What Phase 4 Protects

| Downstream System | Protection Provided |
|-------------------|---------------------|
| **V6 ML Model** | Prevents predictions when structure is weak |
| **Policy Engine** | Ensures appropriate risk posture |
| **Trade Execution** | Blocks trades when conditions are unfavorable |
| **User Interface** | Clear posture indication |

### Failure Modes and Impact

| Failure | System Impact |
|---------|---------------|
| Phase 4 always returns allowed=False | No trades ever execute |
| Phase 4 always returns allowed=True | Trades in degraded conditions |
| Wrong bias assignment | Opposite direction trades |
| Wrong risk_state | Over/under-sized positions |
| Empty invalidation_context | No exit criteria for positions |

---

## Data Flow Position

```
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: Market Structure                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│              PHASE 2: Signal Health                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│              PHASE 3: Signal Density                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           PHASE 4: EXECUTION POSTURE                         ┃
┃                                                              ┃
┃   Inputs:                                                    ┃
┃   - reality.confidence_band                                  ┃
┃   - reality.direction                                        ┃
┃   - reality.range.state                                      ┃
┃   - health.tier                                              ┃
┃   - health.stand_down                                        ┃
┃   - density.throttle                                         ┃
┃                                                              ┃
┃   Outputs:                                                   ┃
┃   - allowed (bool)                                           ┃
┃   - bias (LONG/SHORT/NEUTRAL)                                ┃
┃   - execution_mode (TREND/REVERSION/SCALP/NO_TRADE)          ┃
┃   - risk_state (NORMAL/REDUCED/DEFENSIVE)                    ┃
┃   - invalidation_context []                                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   V6 MODEL  │   │   POLICY    │   │   PREDICT   │
│   (Phase 5) │   │   ENGINE    │   │   SERVER    │
└─────────────┘   └─────────────┘   └─────────────┘
```

---

## Decision Matrix

### Complete State Machine

```
                    ┌─────────────────────────┐
                    │      INPUTS             │
                    │                         │
                    │  confidence_band        │
                    │  direction              │
                    │  range.state            │
                    │  health.tier            │
                    │  health.stand_down      │
                    │  density.throttle       │
                    └───────────┬─────────────┘
                                │
                                ▼
              ┌─────────────────────────────────┐
              │  confidence_band !=             │
              │  STRUCTURAL_EDGE?               │
              └───────────┬─────────────────────┘
                    YES   │   NO
                    ▼     │
              DENY ───────┤
                          │
                          ▼
              ┌─────────────────────────────────┐
              │  stand_down == True?            │
              └───────────┬─────────────────────┘
                    YES   │   NO
                    ▼     │
              DENY ───────┤
                          │
                          ▼
              ┌─────────────────────────────────┐
              │  throttle == BLOCKED?           │
              └───────────┬─────────────────────┘
                    YES   │   NO
                    ▼     │
              DENY ───────┤
                          │
                          ▼
                    ┌─────────────┐
                    │   ALLOW     │
                    │             │
                    │  Set bias   │
                    │  Set mode   │
                    │  Set risk   │
                    └─────────────┘
```

---

## System Health Indicators

### When Phase 4 Is Healthy
- allowed=True with clear bias
- execution_mode matches market regime
- risk_state reflects upstream health
- invalidation_context is non-empty

### When Phase 4 Signals Distress
- Frequent allowed flips (ALLOW → DENY → ALLOW)
- Bias conflicts with V6 predictions
- risk_state stuck in DEFENSIVE
- Empty invalidation_context

---

*Phase 4 is the final gate. Its posture decision shapes all downstream trading behavior.*
