# Daily Direction Prediction - Feature Documentation

## Overview
The daily direction model uses 21 features to predict whether the market will close bullish (up) or bearish (down).

All features use **shifted data** (previous day values) to prevent lookahead bias.

## Feature List

### Price Movement Features (6)

| Feature | Description | Calculation | File Reference |
|---------|-------------|-------------|----------------|
| `prev_return` | Previous day's return | `(Close[-1] - Close[-2]) / Close[-2] * 100` | `ml/predict_server.py:609` |
| `prev_2_return` | Return from 2 days ago | `Close.pct_change().shift(2) * 100` | `ml/predict_server.py:610` |
| `prev_3_return` | Return from 3 days ago | `Close.pct_change().shift(3) * 100` | `ml/predict_server.py:611` |
| `momentum_3d` | 3-day momentum | `sum(prev_return over 3 days)` | `ml/predict_server.py:614` |
| `momentum_5d` | 5-day momentum | `sum(prev_return over 5 days)` | `ml/predict_server.py:615` |
| `gap_pct` | Overnight gap | `(Open - prev_Close) / prev_Close * 100` | Calculated at inference |

### Volatility Features (4)

| Feature | Description | Calculation | File Reference |
|---------|-------------|-------------|----------------|
| `volatility_5d` | 5-day volatility | `std(daily_return over 5 days)` | `ml/predict_server.py:618` |
| `volatility_10d` | 10-day volatility | `std(daily_return over 10 days)` | `ml/predict_server.py:619` |
| `prev_range` | Previous day's range | `(High[-1] - Low[-1]) / Close[-1] * 100` | `ml/predict_server.py:623` |
| `avg_range_5d` | Average range (5 days) | `mean(daily_range over 5 days)` | `ml/predict_server.py:624` |

### Technical Indicators (7)

| Feature | Description | Calculation | File Reference |
|---------|-------------|-------------|----------------|
| `rsi_14` | 14-period RSI | Standard RSI formula, shifted | `ml/predict_server.py:627-632` |
| `macd` | MACD line | `EMA(12) - EMA(26)`, shifted | `ml/predict_server.py:635-637` |
| `macd_signal` | MACD signal | `EMA(9) of MACD` | `ml/predict_server.py:638` |
| `macd_histogram` | MACD histogram | `MACD - Signal`, shifted | `ml/predict_server.py:639` |
| `prev_sma_20_dist` | Distance to SMA20 | `(Close - SMA20) / SMA20 * 100` | `ml/predict_server.py:642-643` |
| `prev_sma_50_dist` | Distance to SMA50 | `(Close - SMA50) / SMA50 * 100` | `ml/predict_server.py:644-645` |
| `prev_atr_pct` | ATR as % of price | `ATR(14) / Close * 100` | `ml/predict_server.py:652` |

### Volume Features (2)

| Feature | Description | Calculation | File Reference |
|---------|-------------|-------------|----------------|
| `prev_volume_ratio` | Volume vs average | `Volume[-1] / SMA(Volume, 20)` | `ml/predict_server.py:655-656` |
| `volume_trend` | Volume momentum | Derived from volume ratio trend | Calculated at inference |

### Calendar Features (2)

| Feature | Description | Calculation | File Reference |
|---------|-------------|-------------|----------------|
| `day_of_week` | Day (0=Mon, 4=Fri) | `datetime.dayofweek` | `ml/predict_server.py:659` |
| `is_monday` | Monday flag | `1 if Monday else 0` | `ml/predict_server.py:660` |
| `is_friday` | Friday flag | `1 if Friday else 0` | `ml/predict_server.py:661` |

## Feature Importance

Based on model analysis, the most predictive features are:

1. **momentum_5d** - 5-day momentum captures trend strength
2. **rsi_14** - Overbought/oversold conditions
3. **volatility_5d** - Recent volatility predicts range
4. **macd_histogram** - Momentum confirmation
5. **prev_return** - Previous day's direction often continues

## Feature Scaling

All features are scaled using `StandardScaler`:
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

The scaler is saved with each model for consistent inference.

## Code Location

### Feature Calculation Function
```
ml/predict_server.py:597-663 (calculate_daily_features)
```

### Training Script
```
ml/daily_prediction_model.py:45-120
```

---

## Genius Enhancement Section

### Feature Optimization Reference: FEAT-DIR

### Current Feature Gaps

| Gap ID | Missing Feature | Why It Matters | Priority |
|--------|----------------|----------------|----------|
| FEAT-DIR-G1 | No VIX/volatility regime | High VIX changes direction patterns | Critical |
| FEAT-DIR-G2 | No inter-market signals | Bonds/DXY affect equity direction | High |
| FEAT-DIR-G3 | No sentiment data | Put/Call ratio, fear/greed | High |
| FEAT-DIR-G4 | No pre-market context | Futures gap predicts cash direction | Medium |
| FEAT-DIR-G5 | No seasonality | Monthly patterns (Jan effect, etc.) | Low |

### Feature Enhancement Table

| Current Feature | Enhancement ID | Enhancement | Expected Impact |
|-----------------|---------------|-------------|-----------------|
| `prev_return` | FEAT-DIR-01 | Add `weighted_return` (recent days weighted more) | +1% accuracy |
| `momentum_5d` | FEAT-DIR-02 | Add `momentum_10d`, `momentum_20d` for multi-scale | +1-2% accuracy |
| `rsi_14` | FEAT-DIR-03 | Add `rsi_divergence` (price vs RSI direction) | +2% accuracy |
| `rsi_14` | FEAT-DIR-04 | Add `rsi_7` for faster signal | +1% accuracy |
| `macd_histogram` | FEAT-DIR-05 | Add `macd_crossover_days` (days since cross) | +1% accuracy |
| `volatility_5d` | FEAT-DIR-06 | Add `vol_regime` categorical (low/med/high/extreme) | +2% accuracy |
| `prev_sma_20_dist` | FEAT-DIR-07 | Add `sma_20_slope` (trend direction) | +1% accuracy |
| `prev_volume_ratio` | FEAT-DIR-08 | Add `volume_trend_3d` (volume momentum) | +1% accuracy |
| `day_of_week` | FEAT-DIR-09 | Add `month_of_year`, `is_month_end` | +0.5% accuracy |
| - | FEAT-DIR-10 | **NEW: `vix_level`** - Current VIX reading | +3% accuracy |
| - | FEAT-DIR-11 | **NEW: `vix_change`** - VIX daily change % | +2% accuracy |
| - | FEAT-DIR-12 | **NEW: `put_call_ratio`** - Options sentiment | +2% accuracy |
| - | FEAT-DIR-13 | **NEW: `futures_gap`** - ES/NQ overnight gap | +3% accuracy |
| - | FEAT-DIR-14 | **NEW: `dxy_change`** - Dollar index movement | +1% accuracy |
| - | FEAT-DIR-15 | **NEW: `yield_change`** - 10Y Treasury change | +1% accuracy |
| - | FEAT-DIR-16 | **NEW: `sector_rotation`** - XLK/XLF relative strength | +2% accuracy |
| - | FEAT-DIR-17 | **NEW: `consecutive_days`** - Streak of up/down | +1% accuracy |
| - | FEAT-DIR-18 | **NEW: `gap_fill_prob`** - Historical gap fill rate | +1% accuracy |

### Implementation Priority Matrix

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENHANCEMENT PRIORITY                      │
├──────────────────────┬──────────────────────────────────────────────┤
│  HIGH IMPACT         │  MEDIUM IMPACT                               │
│  EASY TO IMPLEMENT   │  EASY TO IMPLEMENT                           │
│  ═══════════════     │  ════════════════                            │
│  • FEAT-DIR-10 (VIX) │  • FEAT-DIR-02 (multi-momentum)              │
│  • FEAT-DIR-17 (streak)│ • FEAT-DIR-04 (RSI 7)                      │
│  • FEAT-DIR-06 (vol) │  • FEAT-DIR-09 (seasonality)                 │
├──────────────────────┼──────────────────────────────────────────────┤
│  HIGH IMPACT         │  MEDIUM IMPACT                               │
│  HARD TO IMPLEMENT   │  HARD TO IMPLEMENT                           │
│  ═══════════════     │  ════════════════                            │
│  • FEAT-DIR-13 (futures)│ • FEAT-DIR-03 (RSI divergence)            │
│  • FEAT-DIR-12 (P/C) │  • FEAT-DIR-07 (SMA slope)                   │
│  • FEAT-DIR-16 (sector)│ • FEAT-DIR-18 (gap fill)                   │
└──────────────────────┴──────────────────────────────────────────────┘
```

### Code Template: New Features

```python
# FEAT-DIR-10, FEAT-DIR-11: VIX Features
def add_vix_features(df, vix_df):
    df = df.merge(vix_df[['date', 'close']], on='date', how='left')
    df = df.rename(columns={'close': 'vix_level'})
    df['vix_change'] = df['vix_level'].pct_change() * 100
    df['vix_zone'] = pd.cut(df['vix_level'],
                            bins=[0, 15, 20, 25, 35, 100],
                            labels=[0, 1, 2, 3, 4])
    return df

# FEAT-DIR-17: Consecutive Days
def add_streak_feature(df):
    df['up_day'] = (df['Close'] > df['Open']).astype(int)
    df['streak'] = 0
    streak = 0
    prev_direction = None

    for i, row in df.iterrows():
        current_dir = row['up_day']
        if current_dir == prev_direction:
            streak += 1 if current_dir == 1 else -1
        else:
            streak = 1 if current_dir == 1 else -1
        df.at[i, 'streak'] = streak
        prev_direction = current_dir

    return df

# FEAT-DIR-03: RSI Divergence
def add_rsi_divergence(df):
    # Bullish divergence: price lower low, RSI higher low
    df['price_lower_low'] = df['Low'] < df['Low'].shift(1)
    df['rsi_higher_low'] = df['rsi_14'] > df['rsi_14'].shift(1)
    df['bullish_divergence'] = (df['price_lower_low'] & df['rsi_higher_low']).astype(int)

    # Bearish divergence: price higher high, RSI lower high
    df['price_higher_high'] = df['High'] > df['High'].shift(1)
    df['rsi_lower_high'] = df['rsi_14'] < df['rsi_14'].shift(1)
    df['bearish_divergence'] = (df['price_higher_high'] & df['rsi_lower_high']).astype(int)

    df['rsi_divergence'] = df['bullish_divergence'] - df['bearish_divergence']
    return df
```

---

Last Verified: December 8, 2025
