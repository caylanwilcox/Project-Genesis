# Signal Health & Density Layer (Phases 2-3) - System Connections

## The Body Metaphor

Phases 2-3 are the **immune system** of the trading platform. Phase 2 is the white blood cell count - measuring overall signal health. Phase 3 is the inflammation response - detecting when too many signals are firing and throttling the noise.

Just as the immune system protects the body from infections and overreactions, Phases 2-3 protect downstream systems from degraded signals and spam.

---

## Upstream Connections

### What Phases 2-3 Enable

| Consumer | What It Receives | How It Uses It |
|----------|------------------|----------------|
| **Phase 4 (Posture)** | health.tier, density.throttle | Gates execution permission |
| **Predict Server** | health.stand_down | Skips signal generation if True |
| **Dashboard UI** | health_score, tier, reasons | Health indicator display |
| **Logging** | reasons[], penalties | Audit trail for debugging |

### Interface Contracts

**Phase 2 → Phase 4**
```
Input:  health.tier, health.stand_down
Output: Phase 4 risk_state (NORMAL/REDUCED/DEFENSIVE)

IF health.stand_down == True:
    THEN risk_state = DEFENSIVE
    THEN allowed = False
```

**Phase 3 → Phase 4**
```
Input:  density.throttle
Output: Phase 4 allowed flag

IF density.throttle == BLOCKED:
    THEN allowed = False
    THEN invalidation_context.append("Density throttle BLOCKED")
```

**Phase 2 → Predict Server**
```
Input:  health.stand_down
Output: Server behavior

IF stand_down == True:
    THEN skip V6 prediction
    THEN return NO_TRADE with reason
```

---

## Downstream Protection

### What Phases 2-3 Protect

| Downstream System | Protection Provided |
|-------------------|---------------------|
| **Phase 4** | Ensures only healthy, non-spam signals reach execution |
| **V6 ML Model** | Prevents predictions during degraded conditions |
| **Policy Engine** | Blocks trades when signal quality is poor |
| **User Interface** | Prevents display of low-quality signals |

### Failure Modes and Impact

| Failure | System Impact |
|---------|---------------|
| Phase 2 always returns HEALTHY | Degraded signals pass through |
| Phase 2 always returns UNSTABLE | No signals ever generated (over-protective) |
| Phase 3 always BLOCKED | System paralyzed, no signals |
| Phase 3 always OPEN | Spam floods downstream |
| Reasons not tracked | Debugging impossible |

---

## Data Flow Position

```
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: Market Structure                       │
│              (RealityState output)                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃         PHASE 2: SIGNAL HEALTH (Immune Response)            ┃
┃                                                              ┃
┃   RealityState ──► Dimension Scoring ──► health_score       ┃
┃                                                              ┃
┃   IF health_score < 45 → stand_down = True                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                          │
                          ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃         PHASE 3: SIGNAL DENSITY (Spam Filter)               ┃
┃                                                              ┃
┃   Clustering ──► TF Saturation ──► Regime Check ──► throttle┃
┃                                                              ┃
┃   IF density_score < 40 → throttle = BLOCKED                ┃
┗━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 4: Execution Posture                      │
│              (Uses health.tier + density.throttle)           │
└─────────────────────────────────────────────────────────────┘
```

---

## Gating Logic

### Phase 2-3 as Sequential Gates

```
Signal Flow:
                 ┌─────────────┐
                 │  Phase 1    │
                 │  (Truth)    │
                 └──────┬──────┘
                        │
                        ▼
              ┌─────────────────────┐
              │     Phase 2         │
              │  stand_down = ?     │
              └─────────┬───────────┘
                        │
           ┌────────────┴────────────┐
           │ stand_down = True       │ stand_down = False
           ▼                         ▼
    ┌──────────────┐        ┌─────────────────────┐
    │   BLOCKED    │        │      Phase 3        │
    │  (NO SIGNAL) │        │  throttle = ?       │
    └──────────────┘        └─────────┬───────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   │ throttle = BLOCKED   │ throttle != BLOCKED
                   ▼                      ▼
            ┌──────────────┐        ┌─────────────┐
            │   BLOCKED    │        │   Phase 4   │
            │  (NO SIGNAL) │        │  (Continue) │
            └──────────────┘        └─────────────┘
```

---

## System Health Indicators

### When Phases 2-3 Are Healthy
- Health score >= 75 consistently
- Throttle is OPEN
- Reasons list is empty or contains minor warnings
- No stand_down in last N minutes

### When Phases 2-3 Signal Distress
- Health score fluctuating wildly
- Frequent tier changes (HEALTHY → DEGRADED → HEALTHY)
- Throttle frequently BLOCKED
- Reasons list growing with multiple penalties

### System Response to Distress
When health or density degrades:
1. Phase 4 receives restricted input
2. V6 predictions may be skipped
3. Dashboard shows health warnings
4. Logging captures reasons for audit

---

## Monitoring Recommendations

| Metric | Alert Threshold |
|--------|-----------------|
| stand_down frequency | > 3 per hour |
| BLOCKED throttle duration | > 5 minutes continuous |
| Health score volatility | Std dev > 15 in 1 hour |
| Reason accumulation | > 5 unique reasons in 10 min |

---

*Phases 2-3 are the immune system. Their vigilance protects the system from bad signals and noise.*
