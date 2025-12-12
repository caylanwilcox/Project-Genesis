# Wide Range (High/Low) - Feature Documentation

## Overview
The wide range model uses 29 features to predict the full day's high and low prices as percentages from the open.

## Feature List

### Previous Day Metrics (5)

| Feature | Description | Calculation | File Reference |
|---------|-------------|-------------|----------------|
| `prev_close` | Previous close price | `Close.shift(1)` | `ml/train_highlow_model_v2.py:68` |
| `prev_high` | Previous high price | `High.shift(1)` | `ml/train_highlow_model_v2.py:69` |
| `prev_low` | Previous low price | `Low.shift(1)` | `ml/train_highlow_model_v2.py:70` |
| `prev_open` | Previous open price | `Open.shift(1)` | `ml/train_highlow_model_v2.py:71` |
| `gap_pct` | Overnight gap | `(Open - prev_close) / prev_close * 100` | `ml/train_highlow_model_v2.py:74` |

### Previous Day Range (4)

| Feature | Description | Calculation | File Reference |
|---------|-------------|-------------|----------------|
| `prev_range_pct` | Previous day's full range | `(prev_high - prev_low) / prev_close * 100` | `ml/train_highlow_model_v2.py:77` |
| `prev_high_pct` | Previous high from open | `(prev_high - prev_open) / prev_open * 100` | `ml/train_highlow_model_v2.py:78` |
| `prev_low_pct` | Previous low from open | `(prev_open - prev_low) / prev_open * 100` | `ml/train_highlow_model_v2.py:79` |
| `prev_close_pct` | Previous close from open | `(prev_close - prev_open) / prev_open * 100` | `ml/train_highlow_model_v2.py:80` |

### Return Features (3)

| Feature | Description | Calculation | File Reference |
|---------|-------------|-------------|----------------|
| `prev_return` | Previous day return | `Close.pct_change().shift(1) * 100` | `ml/train_highlow_model_v2.py:86` |
| `prev_2_return` | 2-day ago return | `Close.pct_change().shift(2) * 100` | `ml/train_highlow_model_v2.py:87` |
| `prev_3_return` | 3-day ago return | `Close.pct_change().shift(3) * 100` | `ml/train_highlow_model_v2.py:88` |

### Momentum Features (2)

| Feature | Description | Calculation | File Reference |
|---------|-------------|-------------|----------------|
| `momentum_3d` | 3-day momentum | `rolling(3).sum()` of returns | `ml/train_highlow_model_v2.py:91` |
| `momentum_5d` | 5-day momentum | `rolling(5).sum()` of returns | `ml/train_highlow_model_v2.py:92` |

### Volatility Features (3)

| Feature | Description | Calculation | File Reference |
|---------|-------------|-------------|----------------|
| `volatility_5d` | 5-day std dev | `std(returns, 5).shift(1)` | `ml/train_highlow_model_v2.py:95` |
| `volatility_10d` | 10-day std dev | `std(returns, 10).shift(1)` | `ml/train_highlow_model_v2.py:96` |
| `volatility_20d` | 20-day std dev | `std(returns, 20).shift(1)` | `ml/train_highlow_model_v2.py:97` |

### ATR Features (3)

| Feature | Description | Calculation | File Reference |
|---------|-------------|-------------|----------------|
| `atr_5` | 5-period ATR % | `ATR(5) / prev_close * 100` | `ml/train_highlow_model_v2.py:100-105` |
| `atr_10` | 10-period ATR % | `ATR(10) / prev_close * 100` | `ml/train_highlow_model_v2.py:106` |
| `atr_14` | 14-period ATR % | `ATR(14) / prev_close * 100` | `ml/train_highlow_model_v2.py:107` |

### Historical Pattern Features (6)

| Feature | Description | Calculation | File Reference |
|---------|-------------|-------------|----------------|
| `avg_high_5d` | Avg high % (5 days) | `rolling(5).mean()` of high_pct | `ml/train_highlow_model_v2.py:110` |
| `avg_low_5d` | Avg low % (5 days) | `rolling(5).mean()` of low_pct | `ml/train_highlow_model_v2.py:111` |
| `avg_high_10d` | Avg high % (10 days) | `rolling(10).mean()` of high_pct | `ml/train_highlow_model_v2.py:112` |
| `avg_low_10d` | Avg low % (10 days) | `rolling(10).mean()` of low_pct | `ml/train_highlow_model_v2.py:113` |
| `max_high_5d` | Max high % (5 days) | `rolling(5).max()` of high_pct | `ml/train_highlow_model_v2.py:114` |
| `max_low_5d` | Max low % (5 days) | `rolling(5).max()` of low_pct | `ml/train_highlow_model_v2.py:115` |

### Technical Indicators (3)

| Feature | Description | Calculation | File Reference |
|---------|-------------|-------------|----------------|
| `rsi_14` | 14-period RSI | Standard RSI, shifted | `ml/train_highlow_model_v2.py:118-123` |
| `macd_histogram` | MACD histogram | `MACD - Signal` | `ml/train_highlow_model_v2.py:126-130` |
| `price_vs_sma20` | Price vs SMA20 | `(close - SMA20) / SMA20 * 100` | `ml/train_highlow_model_v2.py:133-134` |
| `price_vs_sma50` | Price vs SMA50 | `(close - SMA50) / SMA50 * 100` | `ml/train_highlow_model_v2.py:135-136` |

### Other Features (2)

| Feature | Description | Calculation | File Reference |
|---------|-------------|-------------|----------------|
| `day_of_week` | Day of week (0-4) | `index.dayofweek` | `ml/train_highlow_model_v2.py:139` |
| `volume_ratio` | Volume vs average | `Volume.shift(1) / SMA(Volume, 20)` | `ml/train_highlow_model_v2.py:142` |
| `consec_up` | Consecutive up days | `rolling(5).sum()` of up_day flag | `ml/train_highlow_model_v2.py:145-146` |

## Feature Importance

Most predictive features for high/low prediction:

1. **atr_14** - ATR is the best predictor of range
2. **avg_high_5d / avg_low_5d** - Recent pattern persistence
3. **volatility_5d** - Short-term volatility
4. **prev_range_pct** - Previous day's range
5. **momentum_5d** - Directional bias affects high vs low

## Code Location

### Feature Calculation Function
```
ml/train_highlow_model_v2.py:60-147 (calculate_features)
```

### Server Feature Calculation
```
ml/predict_server.py:665-715 (calculate_highlow_features)
```

---

## Genius Enhancement Section

### Feature Optimization Reference: FEAT-HL

### Current Feature Gaps

| Gap ID | Missing Feature | Why It Matters | Priority |
|--------|----------------|----------------|----------|
| FEAT-HL-G1 | No VIX for range scaling | VIX directly predicts range width | Critical |
| FEAT-HL-G2 | No options implied move | Market's expected move is priced in options | High |
| FEAT-HL-G3 | No event flags | Earnings/FOMC cause range blow-outs | High |
| FEAT-HL-G4 | No intraday update | First 30-min data can refine prediction | Medium |
| FEAT-HL-G5 | No asymmetric features | Different features for up vs down bias | Medium |

### Feature Enhancement Table

| Current Feature | Enhancement ID | Enhancement | Expected Impact |
|-----------------|---------------|-------------|-----------------|
| `atr_14` | FEAT-HL-01 | Add `atr_percentile` (rank vs last 60 days) | -5% range width |
| `atr_14` | FEAT-HL-02 | Add `atr_ratio` (ATR5/ATR20 for vol trend) | -3% range width |
| `volatility_5d` | FEAT-HL-03 | Add `vol_expansion` (today's vol vs yesterday) | +1% capture |
| `prev_range_pct` | FEAT-HL-04 | Add `range_percentile` (rank vs 20 days) | -3% range width |
| `gap_pct` | FEAT-HL-05 | Add `gap_direction_aligned` (gap matches expected direction) | -2% range width |
| `gap_pct` | FEAT-HL-06 | Add `gap_fill_expectation` (historical gap fill %) | -2% range width |
| `day_of_week` | FEAT-HL-07 | Add `is_triple_witching`, `is_month_end` | Avoid outliers |
| `rsi_14` | FEAT-HL-08 | Add `rsi_extreme` (>70 or <30 flag) | +1% capture |
| - | FEAT-HL-09 | **NEW: `vix_level`** - Scale range by VIX | -15% range width |
| - | FEAT-HL-10 | **NEW: `vix_percentile`** - VIX rank vs 30 days | -5% range width |
| - | FEAT-HL-11 | **NEW: `implied_move`** - ATM straddle % | Best range estimate |
| - | FEAT-HL-12 | **NEW: `is_earnings_week`** - Earnings calendar flag | Avoid blow-outs |
| - | FEAT-HL-13 | **NEW: `is_fomc_day`** - FOMC calendar flag | Avoid blow-outs |
| - | FEAT-HL-14 | **NEW: `is_cpi_day`** - CPI/economic data flag | Avoid blow-outs |
| - | FEAT-HL-15 | **NEW: `overnight_range`** - Pre-market high-low % | -5% range width |
| - | FEAT-HL-16 | **NEW: `first_30min_range`** - Opening range | -10% range width |
| - | FEAT-HL-17 | **NEW: `direction_bias`** - From direction model | Asymmetric ranges |
| - | FEAT-HL-18 | **NEW: `spy_qqq_corr`** - 20-day correlation | Diversification effect |

### Implementation Priority Matrix

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENHANCEMENT PRIORITY                      │
├──────────────────────┬──────────────────────────────────────────────┤
│  HIGH IMPACT         │  MEDIUM IMPACT                               │
│  EASY TO IMPLEMENT   │  EASY TO IMPLEMENT                           │
│  ═══════════════     │  ════════════════                            │
│  • FEAT-HL-09 (VIX)  │  • FEAT-HL-01 (ATR percentile)               │
│  • FEAT-HL-12 (earnings)│ • FEAT-HL-04 (range percentile)           │
│  • FEAT-HL-13 (FOMC) │  • FEAT-HL-07 (special days)                 │
│  • FEAT-HL-17 (direction)│ • FEAT-HL-08 (RSI extreme)               │
├──────────────────────┼──────────────────────────────────────────────┤
│  HIGH IMPACT         │  MEDIUM IMPACT                               │
│  HARD TO IMPLEMENT   │  HARD TO IMPLEMENT                           │
│  ═══════════════     │  ════════════════                            │
│  • FEAT-HL-11 (implied)│ • FEAT-HL-15 (overnight range)             │
│  • FEAT-HL-16 (30min)│  • FEAT-HL-02 (ATR ratio)                    │
│                      │  • FEAT-HL-18 (correlation)                  │
└──────────────────────┴──────────────────────────────────────────────┘
```

### Code Template: New Features

```python
# FEAT-HL-09: VIX-Based Range Scaling
def add_vix_features(df, vix_df):
    df = df.merge(vix_df[['date', 'close']], on='date', how='left')
    df = df.rename(columns={'close': 'vix_level'})

    # VIX percentile for dynamic scaling
    df['vix_percentile'] = df['vix_level'].rolling(30).apply(
        lambda x: (x.iloc[-1] <= x).mean() * 100
    )

    # VIX regime for categorical feature
    df['vix_regime'] = pd.cut(df['vix_level'],
                              bins=[0, 15, 20, 25, 35, 100],
                              labels=['low', 'normal', 'elevated', 'high', 'extreme'])
    return df

# FEAT-HL-12, FEAT-HL-13, FEAT-HL-14: Event Flags
def add_event_flags(df, events_df):
    """
    events_df should have columns: date, event_type
    event_type: 'earnings', 'fomc', 'cpi', 'jobs', etc.
    """
    df['is_earnings_week'] = df['date'].isin(events_df[events_df['event_type'] == 'earnings']['date'])
    df['is_fomc_day'] = df['date'].isin(events_df[events_df['event_type'] == 'fomc']['date'])
    df['is_cpi_day'] = df['date'].isin(events_df[events_df['event_type'] == 'cpi']['date'])
    df['is_high_impact_day'] = df['is_fomc_day'] | df['is_cpi_day']
    return df

# FEAT-HL-01: ATR Percentile
def add_atr_percentile(df):
    df['atr_percentile'] = df['atr_14'].rolling(60).apply(
        lambda x: (x.iloc[-1] <= x).mean() * 100
    )
    return df

# FEAT-HL-17: Direction Bias Integration
def add_direction_bias(df, direction_probs):
    """
    direction_probs: dict {date: bullish_probability}
    """
    df['direction_bias'] = df['date'].map(direction_probs)
    df['upside_weight'] = df['direction_bias'].apply(
        lambda p: 1.2 if p > 0.6 else (0.8 if p < 0.4 else 1.0)
    )
    df['downside_weight'] = df['direction_bias'].apply(
        lambda p: 0.8 if p > 0.6 else (1.2 if p < 0.4 else 1.0)
    )
    return df

# FEAT-HL-16: First 30-Min Range (requires intraday data)
def calculate_first_30min_range(ticker, date):
    """Fetch and calculate opening range"""
    # Fetch 5-min bars from 9:30-10:00 AM
    bars = fetch_intraday_bars(ticker, date, '09:30', '10:00')
    if bars is None:
        return None

    open_price = bars.iloc[0]['open']
    high_30min = bars['high'].max()
    low_30min = bars['low'].min()

    return {
        'first_30min_high_pct': (high_30min - open_price) / open_price * 100,
        'first_30min_low_pct': (open_price - low_30min) / open_price * 100,
        'first_30min_range_pct': (high_30min - low_30min) / open_price * 100
    }
```

### Dynamic Range Adjustment

```python
def adjust_range_by_features(base_high, base_low, features):
    """Adjust predicted range based on special conditions"""

    high_mult = 1.0
    low_mult = 1.0

    # VIX adjustment
    if features['vix_level'] > 25:
        high_mult *= 1.3
        low_mult *= 1.3
    elif features['vix_level'] < 15:
        high_mult *= 0.8
        low_mult *= 0.8

    # Event day adjustment
    if features['is_fomc_day'] or features['is_cpi_day']:
        high_mult *= 1.5
        low_mult *= 1.5

    # Direction bias adjustment
    if features['direction_bias'] > 0.65:
        high_mult *= 1.15
        low_mult *= 0.85
    elif features['direction_bias'] < 0.35:
        high_mult *= 0.85
        low_mult *= 1.15

    return base_high * high_mult, base_low * low_mult
```

---

Last Verified: December 8, 2025
