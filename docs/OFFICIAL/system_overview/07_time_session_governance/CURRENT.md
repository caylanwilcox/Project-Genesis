# Time & Session Governance - Current State

## What This Component Is

Time & Session Governance is the **circadian rhythm** of the trading platform - the system that knows when to be active, when to rest, and how behavior should change throughout the trading day. It enforces market hours, session boundaries, and time-based rules.

---

## What Time Governance Owns

| Responsibility | Implementation |
|----------------|----------------|
| **Market Hours Detection** | 09:30-16:00 ET = OPEN |
| **Session Classification** | Early (hour < 11) vs Late (hour >= 11) |
| **Time Zone Normalization** | All times in America/New_York |
| **Time Multipliers** | Hour-based sizing adjustments |
| **Pre-Market Handling** | No trade signals before 9:30 AM |
| **After-Hours Handling** | No trade signals after 4:00 PM |

---

## What Time Governance Does NOT Own

| Responsibility | Owned By |
|----------------|----------|
| Market structure | Phase 1 |
| Signal health | Phase 2 |
| Trade permission | Phase 4 |
| Probability prediction | Phase 5 (V6) |
| Position sizing | Policy Engine |

---

## Market Hours (SPEC MH-1 through MH-4)

| Spec ID | Rule | Implementation |
|---------|------|----------------|
| MH-1 | 09:30:00 ET = OPEN | `if hour >= 9 and (hour > 9 or minute >= 30)` |
| MH-2 | 15:59:59 ET = OPEN | `if hour < 16` |
| MH-3 | 16:00:00 ET = CLOSED | `if hour >= 16: return CLOSED` |
| MH-4 | Weekend = CLOSED | `if day_of_week in [5, 6]: return CLOSED` |

### Implementation

```python
# From server/config.py
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
```

---

## Session Classification (SPEC SC-1 through SC-3)

| Spec ID | Rule | Implementation |
|---------|------|----------------|
| SC-1 | hour < 11 → "early" | Early morning session |
| SC-2 | hour >= 11 → "late" | Late session |
| SC-3 | Boundary at 11:00:00 | 11:00 is LATE |

### Implementation

```python
# From server/config.py
EARLY_SESSION_END_HOUR = 11

# In prediction logic
if current_hour < EARLY_SESSION_END_HOUR:
    session = 'early'
    action_prob = prob_a  # Target A
else:
    session = 'late'
    action_prob = prob_b  # Target B
```

---

## Session Labels in Phase 1

| Time (ET) | Label | Description |
|-----------|-------|-------------|
| 9:30 - 11:30 | EARLY | Opening range development |
| 11:30 - 14:00 | MID | Midday consolidation |
| 14:00 - 16:00 | LATE | Afternoon resolution |

### Implementation

```python
# From rpe/compute.py
def get_session_label(current_time: str) -> str:
    hour, minute = map(int, current_time.split(':'))
    t = time(hour, minute)

    if t < time(11, 30):
        return "EARLY"
    elif t < time(14, 0):
        return "MID"
    else:
        return "LATE"
```

---

## Time Multipliers

| Hour (ET) | Multiplier | Reason |
|-----------|------------|--------|
| 13, 14, 15 | 1.0 | Peak accuracy hours |
| 11, 12 | 0.8 | Good accuracy |
| 10 | 0.6 | Moderate accuracy |
| < 10 | 0.4 | Early session, lower confidence |

### Implementation

```python
# From server/v6/predictions.py
def get_time_multiplier(hour: int) -> float:
    if hour < 12:
        return 0.7
    elif hour == 12:
        return 1.0
    elif hour in [13, 14]:
        return 1.2  # Peak accuracy
    elif hour == 15:
        return 0.8
    else:
        return 0.5
```

---

## Time Zone Handling

### Single Source of Truth

All times in the system are normalized to America/New_York (Eastern Time).

```python
import pytz
ET = pytz.timezone('America/New_York')

def get_current_time():
    return datetime.now(ET)
```

### Data Source Time Handling

| Source | Time Format | Normalization |
|--------|-------------|---------------|
| Polygon API | Unix timestamp | Convert to ET |
| Hourly bars | ISO 8601 | Parse and convert to ET |
| User requests | ET assumed | Validate format |

---

## Pre-Market and After-Hours

### Pre-Market (4:00 AM - 9:29 AM ET)

| Behavior | Implementation |
|----------|----------------|
| Predictions | Not generated |
| Signals | NO_TRADE |
| Dashboard | Shows "Pre-Market" status |
| Data fetch | Allowed (for feature building) |

### After-Hours (4:00 PM - 8:00 PM ET)

| Behavior | Implementation |
|----------|----------------|
| Predictions | Not generated |
| Signals | NO_TRADE |
| Dashboard | Shows "After Hours" status |
| Historical replay | Allowed |

---

## Invariants

1. **All times are ET**: No UTC, no local time
2. **Session boundary is fixed**: 11:00 AM sharp, no drift
3. **Market hours are strict**: 9:30-16:00, no exceptions
4. **Weekend is always closed**: Saturday and Sunday
5. **Time multipliers are deterministic**: Same hour = same multiplier

---

## Current Production Status

| Component | Status | Notes |
|-----------|--------|-------|
| Market hours detection | ✅ Production | 4 tests pass |
| Session classification | ✅ Production | 3 tests pass |
| Time zone handling | ✅ Production | ET throughout |
| Time multipliers | ✅ Production | 4 tests pass |
| Pre-market handling | ✅ Production | NO_TRADE |
| After-hours handling | ✅ Production | NO_TRADE |

---

## Test Coverage

| Spec ID | Rule | Status |
|---------|------|--------|
| MH-1 | 09:30:00 ET = OPEN | ✅ Tested |
| MH-2 | 15:59:59 ET = OPEN | ✅ Tested |
| MH-3 | 16:00:00 ET = CLOSED | ✅ Tested |
| MH-4 | Weekend = CLOSED | ✅ Tested |
| SC-1 | hour < 11 → "early" | ✅ Tested |
| SC-2 | hour >= 11 → "late" | ✅ Tested |
| SC-3 | Boundary at 11:00:00 | ✅ Tested |
| TM-1 | hour 13-15 → 1.0 | ✅ Tested |
| TM-2 | hour 11-12 → 0.8 | ✅ Tested |
| TM-3 | hour 10 → 0.6 | ✅ Tested |
| TM-4 | hour < 10 → 0.4 | ✅ Tested |

---

*Time Governance is the clock that synchronizes all trading behavior.*
