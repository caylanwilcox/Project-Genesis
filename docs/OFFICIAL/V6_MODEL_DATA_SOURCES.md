# V6 Model Data Sources - Official Specification

## Critical: Training/Serving Alignment

The V6 time-split model was trained with specific data sources. **Production must use the same sources to avoid training/serving skew.**

## Open Price Definition

### Correct (Matches Training)
```python
today_open = daily_bars[-1]['o']  # Regular market open from Polygon daily bar
```

### WRONG (Causes Skew)
```python
today_open = hourly_bars[0]['o']  # Pre-market open at 4 AM - DO NOT USE
```

## Why This Matters

| Data Source | Time | Example (Jan 2, 2026) |
|------------|------|----------------------|
| Daily bar `o` | 9:30 AM ET (Regular Market) | $685.71 |
| First hourly bar `o` | 4:00 AM ET (Pre-Market) | $683.90 |

The V6 model was trained using **daily bar open** (9:30 AM regular market). Using pre-market open causes:
- Incorrect `gap` calculation
- Incorrect `current_vs_open` feature
- Incorrect `above_open` feature
- **Reduced prediction accuracy**

## Training Data Source (train_time_split.py)

```python
# Line 238 - Uses daily bar open
today_open = today['Open']  # From daily_df

# Line 245-247 - Filters hourly bars to regular market hours only
day_bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index <= day_end)]
# where day_start = 9:00 AM, day_end = 4:30 PM
```

## Production Implementation (predict_server.py)

### get_v6_prediction() - Line 3126
```python
# CRITICAL: Use daily bar open (9:30 AM regular market open) to match training
today_open = daily_bars[-1]['o']
```

### /trading_directions endpoint - Line 3381
```python
# CRITICAL: Use daily bar open (9:30 AM regular market) to match V6 training
today_open = daily_bars[-1]['o'] if daily_bars else hourly_bars[0]['o']
```

### /replay endpoint - Line 3793
```python
# CRITICAL: Use daily bar open (9:30 AM regular market) to match V6 training
today_open = daily_bars[-1]['o'] if daily_bars else None
```

## Features Affected by Open Price

These V6 model features depend on `today_open`:

| Feature | Formula |
|---------|---------|
| `gap` | `(today_open - prev_close) / prev_close` |
| `current_vs_open` | `(current_close - today_open) / today_open` |
| `current_vs_open_direction` | `1 if current_close > today_open else -1` |
| `above_open` | `1 if current_close > today_open else 0` |
| `first_hour_return` | `(hourly_bars[0]['c'] - today_open) / today_open` |
| `morning_reversal` | Uses `today_open` for gap fill detection |

## Target Definitions

| Target | Definition | Data Source |
|--------|------------|-------------|
| Target A | Close > Open | Daily bar `c` vs Daily bar `o` (9:30 AM) |
| Target B | Close > 11 AM | Daily bar `c` vs 11 AM hourly bar `c` |

## Polygon API Data Sources

| Endpoint | Returns |
|----------|---------|
| `/v2/aggs/ticker/{ticker}/range/1/day/{date}/{date}` | Daily bar with regular market open at `o` |
| `/v2/aggs/ticker/{ticker}/range/1/hour/{date}/{date}` | Hourly bars starting from 4 AM pre-market |
| `/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}` | Real-time snapshot with `day.o` = regular market open |

## Verification

To verify the data sources match:

```bash
# Daily bar (correct)
curl "https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2026-01-02/2026-01-02?apiKey=KEY"
# Returns: "o": 685.71

# Hourly bar (pre-market - WRONG for V6)
curl "https://api.polygon.io/v2/aggs/ticker/SPY/range/1/hour/2026-01-02/2026-01-02?apiKey=KEY"
# First bar returns: "o": 683.90 (4 AM pre-market)
```

## Fixes Applied: January 2, 2026

### Fix 1: Open Price Skew
Fixed training/serving skew by updating all endpoints to use `daily_bars[-1]['o']` instead of `hourly_bars[0]['o']`.

### Fix 2: Feature Discrepancies

#### Missing Features (10 features were missing from production)

| Feature | Formula | Status |
|---------|---------|--------|
| `gap_direction` | `1 if gap > 0 else (-1 if gap < 0 else 0)` | Added |
| `gap_size` | `abs(gap)` | Added |
| `gap_filled` | `1 if gap filled (price crossed prev_close)` | Added |
| `prev_body` | `(prev_close - prev_open) / prev_open` | Added |
| `prev_bullish` | `1 if prev_close > prev_open else 0` | Added |
| `range_so_far_pct` | `(current_high - current_low) / today_open` | Added |
| `morning_reversal` | `1 if gap direction reversed` | Added |
| `is_monday` | `1 if Monday else 0` | Added |
| `is_friday` | `1 if Friday else 0` | Added |
| `mean_reversion_signal` | `-1 if gap > 1%, +1 if gap < -1%` | Added |

#### Wrong Feature Name

| Production (WRONG) | Training (CORRECT) | Status |
|-------------------|-------------------|--------|
| `prev_day_return` | `prev_return` | Fixed |

#### Calculation Differences

| Feature | Production (WRONG) | Training (CORRECT) | Status |
|---------|-------------------|-------------------|--------|
| `time_pct` | `len(hourly_bars) / 8` | `(hours_since_9AM) / 6.5` | Fixed |
| `near_high` | `current >= high * 0.995` | `(high - current) < (current - low)` | Fixed |
| `current_vs_open_direction` | `1 if > open else 0` | `1 if > open, -1 if < open, 0 if equal` | Fixed |

## Complete V6 Feature List (29 Features)

The V6 model expects exactly these features in this order:

```python
feature_cols = [
    'gap', 'gap_size', 'gap_direction',
    'prev_return', 'prev_range', 'prev_body', 'prev_bullish',
    'current_vs_open', 'current_vs_open_direction', 'above_open',
    'position_in_range', 'range_so_far_pct', 'near_high',
    'gap_filled', 'morning_reversal',
    'time_pct', 'first_hour_return', 'last_hour_return', 'bullish_bar_ratio',
    'is_monday', 'is_friday', 'mean_reversion_signal',
    'current_vs_11am', 'above_11am',
    'return_3d', 'return_5d', 'volatility_5d',
    'consecutive_up', 'consecutive_down'
]
```

## Reference Implementation

All production code in `predict_server.py` now matches `train_time_split.py` exactly:

```python
# predict_server.py - get_v6_prediction() (lines 3143-3191)
gap = (today_open - prev_day['c']) / prev_day['c']
range_so_far = max(current_high - current_low, 0.0001)

# time_pct - matches training exactly
last_bar_time = pd.Timestamp(hourly_bars[-1]['t'], unit='ms', tz='America/New_York')
hours_since_open = (last_bar_time.hour - 9) + (last_bar_time.minute / 60)
time_pct = min(max(hours_since_open / 6.5, 0), 1)

features = {
    'gap': gap,
    'gap_size': abs(gap),
    'gap_direction': 1 if gap > 0 else (-1 if gap < 0 else 0),
    'prev_return': (prev_day['c'] - prev_prev_day['c']) / prev_prev_day['c'],
    'prev_range': (prev_day['h'] - prev_day['l']) / prev_day['c'],
    'prev_body': (prev_day['c'] - prev_day['o']) / prev_day['o'],
    'prev_bullish': 1 if prev_day['c'] > prev_day['o'] else 0,
    'current_vs_open': (current_close - today_open) / today_open,
    'current_vs_open_direction': 1 if current_close > today_open else (-1 if current_close < today_open else 0),
    'above_open': 1 if current_close > today_open else 0,
    'position_in_range': (current_close - current_low) / range_so_far,
    'range_so_far_pct': range_so_far / today_open,
    'near_high': 1 if (current_high - current_close) < (current_close - current_low) else 0,
    'gap_filled': 1 if (gap > 0 and current_low <= prev_day['c']) or (gap <= 0 and current_high >= prev_day['c']) else 0,
    'morning_reversal': 1 if (gap > 0 and current_close < today_open) or (gap < 0 and current_close > today_open) else 0,
    'time_pct': time_pct,
    'first_hour_return': (hourly_bars[0]['c'] - today_open) / today_open,
    'last_hour_return': (hourly_bars[-1]['c'] - hourly_bars[-2]['c']) / hourly_bars[-2]['c'],
    'bullish_bar_ratio': sum(1 for b in hourly_bars if b['c'] > b['o']) / len(hourly_bars),
    'is_monday': 1 if last_bar_time.dayofweek == 0 else 0,
    'is_friday': 1 if last_bar_time.dayofweek == 4 else 0,
    'mean_reversion_signal': -1 if gap > 0.01 else (1 if gap < -0.01 else 0),
}
```

---

*Last Updated: January 2, 2026*
