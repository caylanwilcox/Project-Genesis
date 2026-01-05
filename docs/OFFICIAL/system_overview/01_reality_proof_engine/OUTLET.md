# Reality Proof Engine (RPE) - System Connections

## The Body Metaphor

The Reality Proof Engine is the **sensory nervous system** of the trading platform. Just as the human nervous system continuously observes and classifies stimuli before the brain decides on action, RPE observes and classifies market conditions before the Policy Engine decides on trades.

The phases flow like sensory processing:
- **Phase 1 (Truth)**: The sensory receptors - raw observation of market structure
- **Phase 2 (Health)**: The spinal cord check - is the signal pathway intact?
- **Phase 3 (Density)**: The attention filter - should this signal reach the brain?
- **Phase 4 (Posture)**: The reflex arc - what is the body's readiness state?
- **Phase 5 (ML)**: The cortex - probabilistic prediction and learning

If RPE fails, the entire system goes blind. No market observation means no valid predictions, no valid sizing, no valid trades.

---

## Upstream Connections

### What RPE Enables

| Consumer | What It Receives | How It Uses It |
|----------|------------------|----------------|
| **V6 ML Model** | Session classification, market context | Selects early/late model, adjusts feature weights |
| **Policy Engine** | Execution posture (bias, mode, risk state) | Gates trade actions, adjusts sizing multipliers |
| **Predict Server** | Stand-down flag, allowed tickers | Prevents signal generation when health is degraded |
| **Dashboard UI** | Phase visualizations | Displays market structure, health indicators |
| **Replay Mode** | Historical phase outputs | Enables time-travel debugging |

### Interface Contracts

**Phase 1 → V6 Model**
```
Input:  session (early/late), direction (UP/DOWN/BALANCED)
Output: V6 selects appropriate model weights
```

**Phase 4 → Policy Engine**
```
Input:  allowed (bool), bias (LONG/SHORT/NEUTRAL), risk_state (NORMAL/REDUCED/DEFENSIVE)
Output: Policy gates actions, applies risk multipliers
```

**Phase 2 → Predict Server**
```
Input:  stand_down (bool), health_score (0-100)
Output: Server skips signal generation if stand_down=true
```

---

## Downstream Protection

### What RPE Protects

| Downstream System | Protection Provided |
|-------------------|---------------------|
| **V6 ML Model** | Prevents predictions during degraded market conditions |
| **Policy Engine** | Ensures bias alignment, prevents conflicting positions |
| **Trade Execution** | Blocks trades when risk state is DEFENSIVE |
| **User Interface** | Prevents display of stale or invalid signals |

### Failure Modes and Impact

| RPE Failure | System Impact |
|-------------|---------------|
| Phase 1 stale | V6 uses outdated market context, predictions may be inverted |
| Phase 2 disabled | Degraded signals pass through unchecked |
| Phase 3 disabled | Signal spam floods downstream, no throttling |
| Phase 4 disabled | No execution gating, trades may fire during risk events |
| All phases fail | System should enter safe mode: NO_TRADE for all tickers |

---

## Data Flow Position

```
┌─────────────────────────────────────────────────────────────┐
│                     External Data                           │
│              (Polygon API: bars, quotes)                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 DATA SOURCE LAYER                           │
│        (fetch, validate, normalize, cache)                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃              REALITY PROOF ENGINE (RPE)                      ┃
┃                                                              ┃
┃   Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4               ┃
┃   (Truth)    (Health)    (Density)   (Posture)              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  V6 MODEL   │   │   POLICY    │   │   PREDICT   │
│  (Phase 5)  │   │   ENGINE    │   │   SERVER    │
└─────────────┘   └─────────────┘   └─────────────┘
```

---

## Nervous System Analogy

### Sensory Input (Phase 1)
Like skin receptors detecting pressure, temperature, and pain, Phase 1 detects:
- Price acceptance or rejection at levels
- Range expansion or contraction
- Multi-timeframe alignment or conflict
- Participation strength (volume conviction)

### Signal Integrity Check (Phase 2)
Like the spinal cord verifying signal pathway health, Phase 2 checks:
- Is the data feed intact? (no missing bars)
- Is the signal consistent? (no trend invalidation)
- Is the risk level manageable? (not defensive state)

### Attention Gating (Phase 3)
Like the reticular activating system filtering stimuli, Phase 3 gates:
- Is this signal material? (significant change)
- Is this signal new? (not clustering/spam)
- Does budget allow? (throttle not blocked)

### Reflex Readiness (Phase 4)
Like muscle tone indicating readiness to act, Phase 4 classifies:
- What is the bias? (LONG/SHORT/NEUTRAL)
- What is the execution mode? (trend/mean-reversion/scalp)
- What is the risk state? (normal/reduced/defensive)

### Cortical Decision (Phase 5 - V6 ML)
Like the prefrontal cortex making probabilistic decisions:
- What is the probability of Target A? (close > open)
- What is the probability of Target B? (close > 11am)
- What is the confidence level?

---

## System Health Indicators

### When RPE Is Healthy
- Phase 1 resolves with clear direction
- Phase 2 health_score >= 80
- Phase 3 throttle is OPEN
- Phase 4 allowed is true with clear bias

### When RPE Signals Distress
- Phase 1 unresolved (conflicting signals)
- Phase 2 health_score < 50 or stand_down true
- Phase 3 throttle is BLOCKED
- Phase 4 allowed is false

### System Response to Distress
When any phase signals distress, downstream systems should:
1. Reduce position sizing (Policy Engine)
2. Increase confirmation requirements (Policy Engine)
3. Display warning indicators (Dashboard UI)
4. Log detailed diagnostics (Predict Server)

---

*RPE is the foundation of market perception. Its health determines system health.*
