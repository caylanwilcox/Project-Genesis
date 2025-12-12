# Shrinking Range (Time-Decay) - Feature Documentation

## Overview
The shrinking range model uses 10 features to predict the REMAINING upside and downside potential from the current price at different times during the trading day.

## Feature List

### Time Features (2)

| Feature | Description | Calculation | File Reference |
|---------|-------------|-------------|----------------|
| `time_remaining` | Fraction of day remaining | `1 - (hours_elapsed / 6.5)` | `ml/train_shrinking_range_model.py:149` |
| `hours_elapsed` | Hours since market open | `0.5 to 6.0` in 30-min increments | `ml/train_shrinking_range_model.py:148` |

### Current State Features (4)

| Feature | Description | Calculation | File Reference |
|---------|-------------|-------------|----------------|
| `current_vs_open_pct` | Current price vs open | `(current - open) / open * 100` | `ml/train_shrinking_range_model.py:178` |
| `high_so_far_pct` | Today's high achieved | `(high_so_far - open) / open * 100` | `ml/train_shrinking_range_model.py:179` |
| `low_so_far_pct` | Today's low achieved | `(open - low_so_far) / open * 100` | `ml/train_shrinking_range_model.py:180` |
| `range_so_far_pct` | Total range so far | `high_so_far_pct + low_so_far_pct` | `ml/train_shrinking_range_model.py:181` |

### Daily Context Features (4)

| Feature | Description | Calculation | File Reference |
|---------|-------------|-------------|----------------|
| `gap_pct` | Overnight gap | `(Open - prev_close) / prev_close * 100` | `ml/train_shrinking_range_model.py:119` |
| `prev_range_pct` | Previous day's range | `(prev_high - prev_low) / prev_close * 100` | `ml/train_shrinking_range_model.py:116` |
| `prev_return` | Previous day return | `Close.pct_change().shift(1) * 100` | `ml/train_shrinking_range_model.py:117` |
| `volatility_5d` | 5-day volatility | `std(returns, 5).shift(1) * 100` | `ml/train_shrinking_range_model.py:118` |

## Feature Importance

Most predictive features for shrinking range:

1. **time_remaining** - Primary driver of range shrinkage
2. **high_so_far_pct / low_so_far_pct** - Boundary constraints
3. **volatility_5d** - Baseline volatility expectation
4. **current_vs_open_pct** - Directional bias
5. **prev_range_pct** - Historical range context

## Target Variables

The model predicts two values:

```python
# Remaining upside from current price to day high
remaining_upside = ((day_high - current_price) / current_price) * 100

# Remaining downside from current price to day low
remaining_downside = ((current_price - day_low) / current_price) * 100
```

## Time Slice Simulation

Since intraday minute data is limited, training samples are simulated:

```python
# Time slices (hours after market open)
time_slices = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

# For each time slice, simulate:
# - High/low achieved so far (using sqrt(time) approximation)
# - Current price position (random 30-70% of range)
# - Actual remaining upside/downside to EOD
```

## Real-Time Feature Calculation

At inference time, features are calculated from live market data:

```python
def calculate_shrinking_features(
    current_price: float,
    today_open: float,
    today_high: float,
    today_low: float,
    prev_range_pct: float,
    prev_return: float,
    gap_pct: float,
    volatility_5d: float,
    hours_elapsed: float
):
    return {
        'time_remaining': 1 - (hours_elapsed / 6.5),
        'hours_elapsed': hours_elapsed,
        'current_vs_open_pct': (current_price - today_open) / today_open * 100,
        'high_so_far_pct': (today_high - today_open) / today_open * 100,
        'low_so_far_pct': (today_open - today_low) / today_open * 100,
        'range_so_far_pct': high_so_far_pct + low_so_far_pct,
        'gap_pct': gap_pct,
        'prev_range_pct': prev_range_pct,
        'prev_return': prev_return,
        'volatility_5d': volatility_5d
    }
```

## Code Location

### Feature Calculation (Training)
```
ml/train_shrinking_range_model.py:172-193
```

### Feature Calculation (Server)
```
ml/predict_server.py:758-790 (predict_shrinking_range function)
```

---

## Genius Enhancement Section

### Feature Optimization Reference: FEAT-SR

### Current Feature Gaps (CRITICAL)

| Gap ID | Missing Feature | Why It Matters | Priority |
|--------|----------------|----------------|----------|
| FEAT-SR-G1 | **Simulated data only** | Real intraday data would be much better | Critical |
| FEAT-SR-G2 | No real-time volume | Volume confirms direction commitment | Critical |
| FEAT-SR-G3 | No VWAP reference | Mean reversion target | High |
| FEAT-SR-G4 | No order flow | Shows where price is likely to go | High |
| FEAT-SR-G5 | Static time decay | Non-linear decay would be more accurate | Medium |

### Feature Enhancement Table

| Current Feature | Enhancement ID | Enhancement | Expected Impact |
|-----------------|---------------|-------------|-----------------|
| `time_remaining` | FEAT-SR-01 | Add `time_decay_factor` (non-linear curve) | +2% capture |
| `time_remaining` | FEAT-SR-02 | Add `is_lunch_hour`, `is_power_hour` flags | +2% capture |
| `high_so_far_pct` | FEAT-SR-03 | Add `high_velocity` (how fast high was reached) | +1% capture |
| `low_so_far_pct` | FEAT-SR-04 | Add `low_velocity` (how fast low was reached) | +1% capture |
| `current_vs_open_pct` | FEAT-SR-05 | Add `price_momentum_5min` (recent direction) | +2% capture |
| `range_so_far_pct` | FEAT-SR-06 | Add `range_vs_expected` (actual/wide range %) | +2% capture |
| `volatility_5d` | FEAT-SR-07 | Add `intraday_volatility` (current day's vol) | +3% capture |
| - | FEAT-SR-08 | **NEW: `volume_ratio_intraday`** - Current vs expected volume | +3% capture |
| - | FEAT-SR-09 | **NEW: `vwap_distance`** - Price vs VWAP % | +2% capture |
| - | FEAT-SR-10 | **NEW: `vwap_slope`** - VWAP direction | +1% capture |
| - | FEAT-SR-11 | **NEW: `tick_imbalance`** - Net upticks vs downticks | +2% capture |
| - | FEAT-SR-12 | **NEW: `bid_ask_pressure`** - Order flow direction | +3% capture |
| - | FEAT-SR-13 | **NEW: `intraday_rsi`** - 14-period RSI on 5-min bars | +2% capture |
| - | FEAT-SR-14 | **NEW: `intraday_macd`** - MACD on 5-min bars | +1% capture |
| - | FEAT-SR-15 | **NEW: `key_level_distance`** - Distance to support/resistance | +2% capture |
| - | FEAT-SR-16 | **NEW: `options_gamma`** - Gamma exposure at nearby strikes | +2% capture |
| - | FEAT-SR-17 | **NEW: `trend_strength`** - ADX on 5-min bars | +1% capture |
| - | FEAT-SR-18 | **NEW: `breakout_flag`** - Did we break yesterday's high/low? | +2% capture |

### Implementation Priority Matrix

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENHANCEMENT PRIORITY                      │
├──────────────────────┬──────────────────────────────────────────────┤
│  HIGH IMPACT         │  MEDIUM IMPACT                               │
│  EASY TO IMPLEMENT   │  EASY TO IMPLEMENT                           │
│  ═══════════════     │  ════════════════                            │
│  • FEAT-SR-02 (time) │  • FEAT-SR-01 (decay curve)                  │
│  • FEAT-SR-06 (range)│  • FEAT-SR-05 (momentum)                     │
│  • FEAT-SR-18 (break)│  • FEAT-SR-07 (intraday vol)                 │
├──────────────────────┼──────────────────────────────────────────────┤
│  HIGH IMPACT         │  MEDIUM IMPACT                               │
│  HARD TO IMPLEMENT   │  HARD TO IMPLEMENT                           │
│  ═══════════════     │  ════════════════                            │
│  • FEAT-SR-08 (volume)│ • FEAT-SR-09 (VWAP)                         │
│  • FEAT-SR-12 (order)│  • FEAT-SR-13 (intraday RSI)                 │
│  • FEAT-SR-16 (gamma)│  • FEAT-SR-15 (key levels)                   │
└──────────────────────┴──────────────────────────────────────────────┘
```

### Critical: Real Intraday Data

The single biggest improvement is replacing simulated data with real intraday bars:

```python
# CURRENT (simulated):
high_so_far_pct = day_high_pct * sqrt(hours_elapsed / 6.5) * random(0.7, 1.0)

# IMPROVED (real data):
def get_real_intraday_features(ticker, date, time):
    """Fetch actual intraday bars up to specified time"""
    bars = fetch_intraday_bars(ticker, date, '09:30', time)

    return {
        'high_so_far': bars['high'].max(),
        'low_so_far': bars['low'].min(),
        'current_price': bars.iloc[-1]['close'],
        'volume_so_far': bars['volume'].sum(),
        'vwap': (bars['close'] * bars['volume']).sum() / bars['volume'].sum(),
        # ... actual values, not simulations
    }
```

### Code Template: New Features

```python
# FEAT-SR-02: Time Period Flags
def add_time_flags(hours_elapsed):
    return {
        'is_first_hour': 1 if hours_elapsed <= 1.0 else 0,
        'is_lunch_hour': 1 if 2.5 <= hours_elapsed <= 3.5 else 0,
        'is_power_hour': 1 if hours_elapsed >= 5.5 else 0,
        'time_bucket': int(hours_elapsed)  # 0-6 categorical
    }

# FEAT-SR-08: Intraday Volume Ratio
def calculate_intraday_volume_ratio(volume_so_far, hours_elapsed, avg_daily_volume):
    """Compare current volume to expected volume at this time"""
    # Volume typically follows U-shaped curve
    expected_pct = calculate_expected_volume_pct(hours_elapsed)
    expected_volume = avg_daily_volume * expected_pct
    return volume_so_far / expected_volume

def calculate_expected_volume_pct(hours_elapsed):
    """Expected % of daily volume by time"""
    # Empirical U-shaped curve
    volume_curve = {
        0.5: 0.12,   # First 30 min: 12% of volume
        1.0: 0.22,   # First hour: 22%
        1.5: 0.30,
        2.0: 0.38,
        2.5: 0.45,   # Lunch starts
        3.0: 0.52,   # Lunch (slower)
        3.5: 0.58,
        4.0: 0.65,
        4.5: 0.72,
        5.0: 0.80,
        5.5: 0.88,   # Power hour starts
        6.0: 0.96,
    }
    return volume_curve.get(hours_elapsed, hours_elapsed / 6.5)

# FEAT-SR-09: VWAP Distance
def calculate_vwap_features(bars):
    """Calculate VWAP and related features"""
    vwap = (bars['close'] * bars['volume']).cumsum() / bars['volume'].cumsum()
    current_price = bars.iloc[-1]['close']
    current_vwap = vwap.iloc[-1]

    return {
        'vwap_distance': (current_price - current_vwap) / current_vwap * 100,
        'vwap_slope': (vwap.iloc[-1] - vwap.iloc[-5]) / vwap.iloc[-5] * 100 if len(vwap) >= 5 else 0,
        'above_vwap': 1 if current_price > current_vwap else 0
    }

# FEAT-SR-06: Range vs Expected
def calculate_range_vs_expected(range_so_far_pct, wide_range_pct, hours_elapsed):
    """How much of expected range has been used?"""
    # Expected range consumption by time (sqrt approximation)
    expected_consumption = np.sqrt(hours_elapsed / 6.5)
    expected_range = wide_range_pct * expected_consumption

    return {
        'range_vs_expected': range_so_far_pct / expected_range if expected_range > 0 else 1.0,
        'range_used_pct': range_so_far_pct / wide_range_pct if wide_range_pct > 0 else 0,
        'range_remaining_pct': 1 - (range_so_far_pct / wide_range_pct) if wide_range_pct > 0 else 1.0
    }

# FEAT-SR-18: Breakout Detection
def detect_breakout(current_price, today_high, today_low, prev_high, prev_low):
    """Check if price has broken previous day's range"""
    return {
        'broke_prev_high': 1 if today_high > prev_high else 0,
        'broke_prev_low': 1 if today_low < prev_low else 0,
        'breakout_direction': 1 if today_high > prev_high else (-1 if today_low < prev_low else 0)
    }
```

### Non-Linear Time Decay (FEAT-SR-01)

```python
def calculate_time_decay_factor(hours_elapsed):
    """
    Non-linear decay curve based on market microstructure:
    - Opening: High volatility, fast range development
    - Mid-day: Lower volatility, slower development
    - Close: Increased volatility, position squaring
    """
    if hours_elapsed < 1.0:
        # First hour: 40% of range typically established
        return 1.0 - (hours_elapsed * 0.4)

    elif hours_elapsed < 3.5:
        # Mid-morning to lunch: slower decay
        base = 0.6
        progress = (hours_elapsed - 1.0) / 2.5
        return base - (progress * 0.2)  # 20% over 2.5 hours

    elif hours_elapsed < 5.5:
        # Afternoon: moderate decay
        base = 0.4
        progress = (hours_elapsed - 3.5) / 2.0
        return base - (progress * 0.2)

    else:
        # Power hour: fast final decay
        base = 0.2
        progress = (hours_elapsed - 5.5) / 1.0
        return base - (progress * 0.15)
```

---

Last Verified: December 8, 2025
