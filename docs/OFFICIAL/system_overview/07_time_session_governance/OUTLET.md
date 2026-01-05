# Time & Session Governance - System Connections

## The Body Metaphor

Time Governance is the **circadian rhythm** and **internal clock** of the trading platform. Just as the body knows when to wake, when to eat, and when to sleep, Time Governance knows when to trade, when to be cautious, and when to stop entirely.

The circadian rhythm affects every organ in the body. Similarly, Time Governance affects every component of the trading system - from data fetching to signal generation to position management.

---

## Upstream Connections

### What Time Governance Enables

| Consumer | What It Receives | How It Uses It |
|----------|------------------|----------------|
| **V6 Model** | session (early/late) | Selects appropriate model |
| **Policy Engine** | hour, time_mult | Sizing adjustments |
| **Phase 1** | session_label | Context classification |
| **Predict Server** | is_market_open | Gates signal generation |
| **Dashboard UI** | market_status | Display state |

### Interface Contracts

**Time → V6 Model**
```
Input:  current_hour (ET)
Output: session determination

IF current_hour < 11:
    THEN session = 'early'
    THEN use models_early
    THEN action_prob = prob_a

IF current_hour >= 11:
    THEN session = 'late'
    THEN use models_late
    THEN action_prob = prob_b
```

**Time → Policy Engine**
```
Input:  current_hour (ET)
Output: time_mult

hour 13-15 → 1.0 (peak)
hour 11-12 → 0.8
hour 10 → 0.6
hour < 10 → 0.4
```

**Time → Predict Server**
```
Input:  current_time (ET)
Output: market_status

IF day_of_week in [5, 6]:
    THEN market_status = 'CLOSED'
    THEN skip_prediction = True

IF hour < 9 OR (hour == 9 AND minute < 30):
    THEN market_status = 'PRE_MARKET'
    THEN skip_prediction = True

IF hour >= 16:
    THEN market_status = 'AFTER_HOURS'
    THEN skip_prediction = True

ELSE:
    THEN market_status = 'OPEN'
    THEN skip_prediction = False
```

---

## Downstream Protection

### What Time Governance Protects

| Downstream System | Protection Provided |
|-------------------|---------------------|
| **V6 Model** | Correct session/model selection |
| **Policy Engine** | Appropriate sizing by time of day |
| **Signal Generation** | No signals outside market hours |
| **User Interface** | Accurate market status display |

### Failure Modes and Impact

| Failure | System Impact |
|---------|---------------|
| Wrong timezone | Signals at wrong times |
| Session boundary off | Wrong target used |
| Market hours wrong | Trades in closed market |
| Time multiplier error | Incorrect sizing |

---

## Data Flow Position

```
┌─────────────────────────────────────────────────────────────┐
│                    SYSTEM CLOCK                              │
│                 (datetime.now(ET))                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃              TIME & SESSION GOVERNANCE                       ┃
┃                                                              ┃
┃   is_market_open() ──────► market_status                    ┃
┃   get_session() ─────────► session (early/late)             ┃
┃   get_session_label() ───► EARLY/MID/LATE                   ┃
┃   get_time_multiplier() ─► time_mult                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                          │
          ┌───────────────┼───────────────┬───────────────┐
          │               │               │               │
          ▼               ▼               ▼               ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   PHASE 1   │   │  V6 MODEL   │   │   POLICY    │   │   PREDICT   │
│ (session_   │   │  (session)  │   │  (time_     │   │   SERVER    │
│  label)     │   │             │   │   mult)     │   │ (is_open)   │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
```

---

## Time Dependencies

### Components That Depend on Time

| Component | Time Dependency |
|-----------|-----------------|
| V6 Model | session determines model selection |
| Phase 1 | session_label for context |
| Policy Engine | time_mult for sizing |
| Predict Server | market hours for signal generation |
| Data Fetcher | RTH filter for bars |
| VWAP Calculator | RTH start for reset |

### Time-Based Behavior Matrix

| Time (ET) | Market Status | Session | V6 Model | time_mult |
|-----------|---------------|---------|----------|-----------|
| 04:00-09:29 | PRE_MARKET | N/A | Disabled | N/A |
| 09:30-10:59 | OPEN | early | models_early | 0.4-0.6 |
| 11:00-12:59 | OPEN | late | models_late | 0.8-1.0 |
| 13:00-15:59 | OPEN | late | models_late | 0.8-1.2 |
| 16:00-20:00 | AFTER_HOURS | N/A | Disabled | N/A |
| Saturday | CLOSED | N/A | Disabled | N/A |
| Sunday | CLOSED | N/A | Disabled | N/A |

---

## Invariant Enforcement

| Invariant | Enforcement |
|-----------|-------------|
| All times in ET | `pytz.timezone('America/New_York')` |
| Session boundary at 11:00 | `EARLY_SESSION_END_HOUR = 11` |
| Market hours 9:30-16:00 | Constants in config.py |
| Weekend is closed | Day of week check |
| Time multipliers are deterministic | Pure function of hour |

---

## System Health Indicators

### When Time Governance Is Healthy
- System clock matches reality
- Session transitions at correct times
- Market hours enforced correctly
- Time multipliers applied consistently

### When Time Governance Signals Distress
- System clock drift detected
- Session transitions at wrong times
- Signals generated outside market hours
- Time multiplier mismatches

---

## Synchronization Points

### Daily Reset Points

| Time (ET) | Event | Action |
|-----------|-------|--------|
| 00:00 | Midnight | Clear signal cache |
| 04:00 | Pre-market open | Start data collection |
| 09:30 | Market open | Enable signal generation |
| 11:00 | Session transition | Switch to late models |
| 16:00 | Market close | Disable signal generation |
| 20:00 | After-hours close | Finalize daily data |

---

*Time Governance is the internal clock that synchronizes all system behavior.*
