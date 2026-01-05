# Time & Session Governance - Future Development

## Planned Improvements

### 1. Holiday Calendar Integration

**Current Gap**: No awareness of market holidays.

**Improvement**:
```python
MARKET_HOLIDAYS_2026 = [
    '2026-01-01',  # New Year's Day
    '2026-01-19',  # MLK Day
    '2026-02-16',  # Presidents Day
    '2026-04-03',  # Good Friday
    '2026-05-25',  # Memorial Day
    '2026-07-03',  # Independence Day (observed)
    '2026-09-07',  # Labor Day
    '2026-11-26',  # Thanksgiving
    '2026-12-25',  # Christmas
]

def is_market_open(date: str, time: str) -> bool:
    if date in MARKET_HOLIDAYS_2026:
        return False
    # ... existing logic
```

**Impact**: Prevents signals on closed days.

---

### 2. Early Close Detection

**Current Gap**: No awareness of early close days (e.g., day after Thanksgiving).

**Improvement**:
```python
EARLY_CLOSE_DAYS = {
    '2026-11-27': '13:00',  # Day after Thanksgiving
    '2026-12-24': '13:00',  # Christmas Eve
}

def get_market_close_time(date: str) -> str:
    if date in EARLY_CLOSE_DAYS:
        return EARLY_CLOSE_DAYS[date]
    return '16:00'
```

**Impact**: Correct time-based behavior on early close days.

---

### 3. Dynamic Session Boundaries

**Current Gap**: Fixed 11 AM session boundary.

**Improvement**: Adapt session boundary based on opening range completion.

```python
def get_session_boundary(bars_1m: List[Dict]) -> time:
    # If opening range resolved early, shift session boundary
    or_resolved_at = detect_or_resolution(bars_1m)
    if or_resolved_at and or_resolved_at < time(10, 30):
        return time(10, 30)  # Earlier boundary
    return time(11, 0)  # Default
```

**Impact**: Session aligns with actual market behavior, not just clock.

---

### 4. Session Quality Scoring

**Current Gap**: Sessions are binary (early/late); no quality assessment.

**Improvement**:
```python
{
    'session': 'late',
    'session_quality': {
        'score': 85,
        'factors': {
            'or_resolved': True,
            'volume_participation': 'HIGH',
            'range_development': 'NORMAL',
            'time_to_close_minutes': 120
        }
    }
}
```

**Impact**: Downstream can adjust behavior based on session quality.

---

### 5. Time-Based Alert System

**Current Gap**: No proactive alerts for time-based events.

**Improvement**:
```python
alerts = [
    {
        'time': '15:30',
        'message': 'Position review: 30 min to close',
        'action': 'REVIEW_POSITIONS'
    },
    {
        'time': '15:45',
        'message': 'Final exit window',
        'action': 'PREPARE_EXIT'
    },
    {
        'time': '15:55',
        'message': 'Force close imminent',
        'action': 'FORCE_CLOSE'
    }
]
```

**Impact**: Automated position management near close.

---

### 6. Timezone-Aware Replay

**Current Gap**: Replay mode doesn't account for historical timezone changes (DST).

**Improvement**:
```python
def get_historical_time(date: str, time_str: str) -> datetime:
    # Account for DST transitions
    dt = datetime.strptime(f"{date} {time_str}", "%Y-%m-%d %H:%M")
    et = pytz.timezone('America/New_York')
    return et.localize(dt)
```

**Impact**: Accurate historical replay across DST boundaries.

---

## Priority Matrix

| Improvement | Complexity | Impact | Priority |
|-------------|------------|--------|----------|
| Holiday calendar | Low | High | P0 |
| Early close detection | Low | Medium | P1 |
| Dynamic session boundaries | High | Medium | P3 |
| Session quality scoring | Medium | Medium | P2 |
| Time-based alerts | Medium | High | P1 |
| Timezone-aware replay | Medium | Medium | P2 |

---

## Dependencies

| Improvement | Requires |
|-------------|----------|
| Holiday calendar | Static data or API |
| Early close detection | Static data |
| Dynamic session boundaries | Phase 1 OR detection |
| Session quality scoring | Multiple Phase 1 inputs |
| Time-based alerts | Event system |
| Timezone-aware replay | Historical timezone data |

---

## DST Handling

### Current Issues
- DST transitions can cause 1-hour shifts in data alignment
- Polygon timestamps are UTC; conversion is required

### Improvement
```python
def normalize_timestamp(ts: int, to_tz: str = 'America/New_York') -> datetime:
    utc_dt = datetime.utcfromtimestamp(ts / 1000)
    utc_tz = pytz.utc.localize(utc_dt)
    local_dt = utc_tz.astimezone(pytz.timezone(to_tz))
    return local_dt
```

---

*Time Governance improvements focus on calendar awareness and dynamic session detection.*
