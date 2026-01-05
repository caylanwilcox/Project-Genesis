# Data Source & Feature Integrity Layer - Current State

## What This Component Is

The Data Source & Feature Integrity Layer is the **digestive system** of the trading platform - it takes raw external data, breaks it down, validates it, and transforms it into usable features. Just as the digestive system protects the body from harmful substances, this layer protects the system from bad data.

---

## What This Layer Owns

| Responsibility | Implementation |
|----------------|----------------|
| **Data Fetching** | Polygon API integration |
| **Bar Validation** | Required fields, value ranges |
| **Feature Building** | 29 V6 features |
| **Feature Ordering** | Matches training schema exactly |
| **NaN Handling** | Replace with 0, no inf values |
| **Today's Open** | Uses daily_bars[-1]['o'] (9:30 AM) |

---

## What This Layer Does NOT Own

| Responsibility | Owned By |
|----------------|----------|
| Market structure analysis | Phase 1 |
| Signal health | Phase 2 |
| ML prediction | Phase 5 (V6) |
| Trading decisions | Policy Engine |

---

## Data Source Rules (SPEC DS-1 through DS-4)

| Spec ID | Rule | Implementation |
|---------|------|----------------|
| DS-1 | today_open = daily_bars[-1]['o'] | 9:30 AM open, NOT pre-market |
| DS-2 | hourly_bars filtered by hour | Only RTH bars used |
| DS-3 | daily_bars excludes today for prev-day | prev_day = daily_bars[-2] |
| DS-4 | V6 expects 29 features | Fixed feature count |

### Critical: Today's Open

```python
# CORRECT: Use daily bar open (9:30 AM regular market)
today_open = daily_bars[-1]['o']

# WRONG: This is 4 AM pre-market open
# today_open = hourly_bars[0]['o']  # DON'T USE
```

This is the most common source of training/serving skew.

---

## Polygon API Integration

### Endpoints Used

| Data Type | Endpoint | Parameters |
|-----------|----------|------------|
| Daily Bars | `/v2/aggs/ticker/{ticker}/range/1/day` | `limit=50, adjusted=true` |
| Hourly Bars | `/v2/aggs/ticker/{ticker}/range/1/hour` | `limit=50000, adjusted=true` |
| Current Quote | `/v2/aggs/ticker/{ticker}/prev` | Last traded quote |

### API Response Structure

```python
{
    'results': [
        {
            't': 1704067200000,  # Unix timestamp (ms)
            'o': 595.00,         # Open
            'h': 597.50,         # High
            'l': 594.00,         # Low
            'c': 596.25,         # Close
            'v': 1234567         # Volume
        }
    ]
}
```

---

## Feature Schema (SPEC FS-1 through FS-4)

### 29 V6 Features (Fixed Order)

| # | Feature | Category |
|---|---------|----------|
| 1 | `hour` | Temporal |
| 2 | `day_of_week` | Temporal |
| 3 | `week_of_year` | Temporal |
| 4 | `month` | Temporal |
| 5 | `open_to_current` | Price |
| 6 | `open_to_high` | Price |
| 7 | `open_to_low` | Price |
| 8 | `current_range` | Price |
| 9 | `prev_day_close` | Price |
| 10 | `prev_day_range` | Price |
| 11 | `gap_pct` | Price |
| 12 | `gap_direction` | Volume |
| 13 | `volume_ratio` | Volume |
| 14 | `hourly_momentum` | Momentum |
| 15 | `hourly_volatility` | Momentum |
| 16 | `high_low_ratio` | Structure |
| 17 | `body_to_range` | Structure |
| 18 | `upper_wick_pct` | Structure |
| 19 | `lower_wick_pct` | Structure |
| 20 | `prev_hour_close` | Momentum |
| 21 | `prev_hour_range` | Momentum |
| 22 | `two_hour_momentum` | Momentum |
| 23 | `day_range_pct` | Structure |
| 24 | `dist_from_high` | Structure |
| 25 | `dist_from_low` | Structure |
| 26 | `morning_momentum` | Momentum |
| 27 | `morning_volatility` | Momentum |
| 28 | `prev_day_body` | Previous Day |
| 29 | `prev_day_direction` | Previous Day |

### Feature Order Enforcement

```python
# From server/config.py
V6_FEATURE_COLS = [
    'hour', 'day_of_week', 'week_of_year', 'month',
    'open_to_current', 'open_to_high', 'open_to_low',
    'current_range', 'prev_day_close', 'prev_day_range',
    'gap_pct', 'gap_direction', 'volume_ratio',
    'hourly_momentum', 'hourly_volatility',
    'high_low_ratio', 'body_to_range', 'upper_wick_pct', 'lower_wick_pct',
    'prev_hour_close', 'prev_hour_range', 'two_hour_momentum',
    'day_range_pct', 'dist_from_high', 'dist_from_low',
    'morning_momentum', 'morning_volatility',
    'prev_day_body', 'prev_day_direction'
]

# Usage
X = np.array([[features.get(col, 0) for col in V6_FEATURE_COLS]])
```

---

## Feature Building

### Key Calculations

```python
def build_v6_features(hourly_bars, daily_bars, today_open, current_hour):
    # Previous day data
    prev_day = daily_bars[-2]  # NOT daily_bars[-1]
    prev_close = prev_day['c']
    prev_high = prev_day['h']
    prev_low = prev_day['l']

    # Current day data
    current_price = hourly_bars[-1]['c']
    day_high = max(bar['h'] for bar in hourly_bars if is_today(bar))
    day_low = min(bar['l'] for bar in hourly_bars if is_today(bar))

    features = {
        'open_to_current': (current_price - today_open) / today_open,
        'open_to_high': (day_high - today_open) / today_open,
        'open_to_low': (today_open - day_low) / today_open,
        'gap_pct': (today_open - prev_close) / prev_close,
        'gap_direction': 1 if today_open > prev_close else -1,
        # ... remaining features
    }

    return features
```

---

## Data Validation

### Required Bar Fields

| Field | Type | Validation |
|-------|------|------------|
| `t` | int | Unix timestamp in milliseconds |
| `o` | float | > 0 |
| `h` | float | >= o, >= l, >= c |
| `l` | float | <= o, <= h, <= c, > 0 |
| `c` | float | > 0 |
| `v` | int | >= 0 |

### NaN Handling

```python
# Replace all NaN and inf with 0
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
```

---

## Daily Open Hard Gate (SPEC DO-1 through DO-3)

| Spec ID | Rule | Implementation |
|---------|------|----------------|
| DO-1 | Missing daily bar → NO_TRADE + reason | Hard gate |
| DO-2 | Valid daily_bars[-1]['o'] allows trade | Required |
| DO-3 | NO_TRADE reason is deterministic | "Missing daily bar for open price" |

### Implementation

```python
if not daily_bars or len(daily_bars) < 1:
    return {
        'action': 'NO_TRADE',
        'reason': 'Missing daily bar for open price',
        'data_quality': 'FAILED'
    }

today_open = daily_bars[-1]['o']
if today_open is None or today_open <= 0:
    return {
        'action': 'NO_TRADE',
        'reason': 'Invalid daily bar open price',
        'data_quality': 'FAILED'
    }
```

---

## Invariants

1. **Feature count is fixed**: Always 29 features, no more, no less
2. **Feature order matches training**: V6_FEATURE_COLS constant is authoritative
3. **Today's open from daily bar**: daily_bars[-1]['o'], not hourly
4. **Previous day excludes today**: daily_bars[-2] for prev_day features
5. **No NaN in model input**: All replaced with 0

---

## Current Production Status

| Component | Status | Notes |
|-----------|--------|-------|
| Polygon integration | ✅ Production | API key in env |
| Daily bar fetching | ✅ Production | limit=50 |
| Hourly bar fetching | ✅ Production | limit=50000 |
| Feature building | ✅ Production | 29 features |
| NaN handling | ✅ Production | nan_to_num |
| Daily open gate | ✅ Production | 3 tests pass |

---

## Test Coverage

| Spec ID | Rule | Status |
|---------|------|--------|
| DS-1 | today_open = daily_bars[-1]['o'] | ✅ Tested |
| DS-2 | hourly_bars filtered by hour | ✅ Tested |
| DS-3 | daily_bars excludes today | ✅ Tested |
| DS-4 | V6 expects 29 features | ✅ Tested |
| FS-1 | Feature names match training | ✅ Tested |
| FS-2 | Returns dict with features | ✅ Tested |
| FS-3 | Features are numeric | ✅ Tested |
| FS-4 | No NaN features | ✅ Tested |
| DO-1 | Missing daily bar → NO_TRADE | ✅ Tested |
| DO-2 | Valid daily bar allows trade | ✅ Tested |
| DO-3 | Reason is deterministic | ✅ Tested |

---

*Data Integrity is the foundation of prediction accuracy. Bad data = bad predictions.*
