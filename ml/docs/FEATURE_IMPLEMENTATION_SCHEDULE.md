# ML Feature Implementation Schedule

A structured plan to add all new technical indicators and features to strengthen the prediction models.

---

## Phase 1: Core Volatility & Momentum (Days 1-2)

### Day 1: TTM Squeeze + KDJ Indicator

**TTM Squeeze** - Predicts volatility breakouts

| Feature | Description |
|---------|-------------|
| `ttm_squeeze_on` | Binary: Bollinger Bands inside Keltner Channels |
| `ttm_squeeze_off` | Binary: Squeeze just released |
| `ttm_squeeze_fired` | Binary: First bar after squeeze release |
| `ttm_squeeze_momentum` | Direction of momentum when squeeze fires |
| `ttm_squeeze_bars` | Count of consecutive squeeze bars |

```python
# TTM Squeeze Implementation
# Keltner Channels
kc_middle = df['Close'].ewm(span=20).mean()
kc_atr = atr_14  # Use existing ATR
kc_upper = kc_middle + 1.5 * kc_atr
kc_lower = kc_middle - 1.5 * kc_atr

# Squeeze Detection (BB inside KC = low volatility)
df['ttm_squeeze_on'] = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(int)
df['ttm_squeeze_off'] = (~df['ttm_squeeze_on'].astype(bool)).astype(int)
df['ttm_squeeze_fired'] = ((df['ttm_squeeze_on'].shift(1) == 1) & (df['ttm_squeeze_on'] == 0)).astype(int)

# Momentum (Donchian midline deviation)
donchian_mid = (df['High'].rolling(20).max() + df['Low'].rolling(20).min()) / 2
df['ttm_squeeze_momentum'] = df['Close'] - (donchian_mid + sma_20) / 2

# Count consecutive squeeze bars
df['ttm_squeeze_bars'] = df['ttm_squeeze_on'].groupby(
    (df['ttm_squeeze_on'] != df['ttm_squeeze_on'].shift()).cumsum()
).cumcount() + 1
df.loc[df['ttm_squeeze_on'] == 0, 'ttm_squeeze_bars'] = 0
```

**KDJ Indicator (9,3,3)** - Enhanced Stochastic with J-line

| Feature | Description |
|---------|-------------|
| `kdj_k` | %K line (smoothed RSV) |
| `kdj_d` | %D line (smoothed %K) |
| `kdj_j` | J line = 3K - 2D (amplified momentum) |
| `kdj_golden_cross` | K crosses above D |
| `kdj_death_cross` | K crosses below D |
| `kdj_j_overbought` | J > 100 |
| `kdj_j_oversold` | J < 0 |

```python
# KDJ (9,3,3) Implementation
low_9 = df['Low'].rolling(9).min()
high_9 = df['High'].rolling(9).max()
rsv = (df['Close'] - low_9) / (high_9 - low_9 + 0.001) * 100

df['kdj_k'] = rsv.ewm(com=2, adjust=False).mean()  # 3-period EMA
df['kdj_d'] = df['kdj_k'].ewm(com=2, adjust=False).mean()
df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

# Crossovers
df['kdj_golden_cross'] = ((df['kdj_k'] > df['kdj_d']) &
                          (df['kdj_k'].shift(1) <= df['kdj_d'].shift(1))).astype(int)
df['kdj_death_cross'] = ((df['kdj_k'] < df['kdj_d']) &
                         (df['kdj_k'].shift(1) >= df['kdj_d'].shift(1))).astype(int)

# Extremes
df['kdj_j_overbought'] = (df['kdj_j'] > 100).astype(int)
df['kdj_j_oversold'] = (df['kdj_j'] < 0).astype(int)
```

**Checklist Day 1:**
- [ ] Add TTM Squeeze to `calculate_daily_features()`
- [ ] Add KDJ to `calculate_daily_features()`
- [ ] Update feature documentation
- [ ] Test with sample data
- [ ] Verify no NaN values in output

---

### Day 2: Volatility Regime Detection

**Historical Volatility & Regime**

| Feature | Description |
|---------|-------------|
| `hv_10` | 10-day historical volatility (annualized %) |
| `hv_20` | 20-day historical volatility (annualized %) |
| `hv_ratio` | HV10 / HV20 (vol expansion/contraction) |
| `vol_regime` | Categorical: low/normal/elevated/high |
| `vol_percentile` | Current vol vs 252-day history |
| `vol_expanding` | Binary: HV10 > HV20 * 1.2 |
| `vol_contracting` | Binary: HV10 < HV20 * 0.8 |

```python
# Historical Volatility
returns = df['Close'].pct_change()
df['hv_10'] = returns.rolling(10).std() * np.sqrt(252) * 100
df['hv_20'] = returns.rolling(20).std() * np.sqrt(252) * 100
df['hv_ratio'] = df['hv_10'] / (df['hv_20'] + 0.001)

# Volatility Regime
df['vol_percentile'] = df['hv_20'].rolling(252).rank(pct=True)
df['vol_regime'] = pd.cut(df['vol_percentile'],
                          bins=[0, 0.25, 0.5, 0.75, 1.0],
                          labels=[0, 1, 2, 3])  # low, normal, elevated, high

# Expansion/Contraction
df['vol_expanding'] = (df['hv_10'] > df['hv_20'] * 1.2).astype(int)
df['vol_contracting'] = (df['hv_10'] < df['hv_20'] * 0.8).astype(int)
```

**Checklist Day 2:**
- [ ] Add volatility regime features
- [ ] Test regime classification
- [ ] Update documentation

---

## Phase 2: Momentum Divergences & Price Action (Days 3-4)

### Day 3: Divergence Detection

**RSI Divergences**

| Feature | Description |
|---------|-------------|
| `rsi_bullish_div` | Price new low, RSI higher low |
| `rsi_bearish_div` | Price new high, RSI lower high |
| `rsi_div_strength` | Magnitude of divergence |

**MACD Divergences**

| Feature | Description |
|---------|-------------|
| `macd_bullish_div` | Price new low, MACD histogram higher |
| `macd_bearish_div` | Price new high, MACD histogram lower |

**OBV Divergences**

| Feature | Description |
|---------|-------------|
| `obv_bullish_div` | Price new low, OBV higher |
| `obv_bearish_div` | Price new high, OBV lower |

```python
# RSI Divergence Detection
lookback = 20

# Price extremes
price_new_high = df['Close'] == df['Close'].rolling(lookback).max()
price_new_low = df['Close'] == df['Close'].rolling(lookback).min()

# RSI not confirming
rsi_not_high = df['rsi_14'] < df['rsi_14'].rolling(lookback).max()
rsi_not_low = df['rsi_14'] > df['rsi_14'].rolling(lookback).min()

df['rsi_bearish_div'] = (price_new_high & rsi_not_high).astype(int)
df['rsi_bullish_div'] = (price_new_low & rsi_not_low).astype(int)

# MACD Divergence
macd_not_high = df['macd_histogram'] < df['macd_histogram'].rolling(lookback).max()
macd_not_low = df['macd_histogram'] > df['macd_histogram'].rolling(lookback).min()

df['macd_bearish_div'] = (price_new_high & macd_not_high).astype(int)
df['macd_bullish_div'] = (price_new_low & macd_not_low).astype(int)

# OBV Divergence
obv = (np.sign(df['Close'].pct_change()) * df['Volume']).cumsum()
obv_not_high = obv < obv.rolling(lookback).max()
obv_not_low = obv > obv.rolling(lookback).min()

df['obv_bearish_div'] = (price_new_high & obv_not_high).astype(int)
df['obv_bullish_div'] = (price_new_low & obv_not_low).astype(int)

# Combined divergence score
df['bullish_div_count'] = df['rsi_bullish_div'] + df['macd_bullish_div'] + df['obv_bullish_div']
df['bearish_div_count'] = df['rsi_bearish_div'] + df['macd_bearish_div'] + df['obv_bearish_div']
```

**Checklist Day 3:**
- [ ] Add RSI divergence detection
- [ ] Add MACD divergence detection
- [ ] Add OBV divergence detection
- [ ] Create combined divergence score
- [ ] Test on historical reversals

---

### Day 4: Price Action Patterns

**Bar Patterns**

| Feature | Description |
|---------|-------------|
| `inside_bar` | High < prev high AND low > prev low |
| `outside_bar` | High > prev high AND low < prev low |
| `narrow_range_4` | Smallest range in 4 bars |
| `narrow_range_7` | Smallest range in 7 bars |
| `wide_range_bar` | Range > 2x average range |

**Trend Structure**

| Feature | Description |
|---------|-------------|
| `higher_high` | Today's high > yesterday's high |
| `lower_low` | Today's low < yesterday's low |
| `higher_low` | Today's low > yesterday's low |
| `lower_high` | Today's high < yesterday's high |
| `trend_structure_3` | HH/HL count minus LH/LL count (3 bars) |

**Pivot Points**

| Feature | Description |
|---------|-------------|
| `pivot` | (H + L + C) / 3 |
| `r1`, `r2`, `r3` | Resistance levels |
| `s1`, `s2`, `s3` | Support levels |
| `dist_to_pivot` | % distance to pivot |
| `dist_to_r1`, `dist_to_s1` | % distance to key levels |

```python
# Bar Patterns
df['inside_bar'] = ((df['High'] < df['High'].shift(1)) &
                    (df['Low'] > df['Low'].shift(1))).astype(int)
df['outside_bar'] = ((df['High'] > df['High'].shift(1)) &
                     (df['Low'] < df['Low'].shift(1))).astype(int)

daily_range = df['High'] - df['Low']
df['narrow_range_4'] = (daily_range == daily_range.rolling(4).min()).astype(int)
df['narrow_range_7'] = (daily_range == daily_range.rolling(7).min()).astype(int)
df['wide_range_bar'] = (daily_range > daily_range.rolling(20).mean() * 2).astype(int)

# Trend Structure
df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
df['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
df['higher_low'] = (df['Low'] > df['Low'].shift(1)).astype(int)
df['lower_high'] = (df['High'] < df['High'].shift(1)).astype(int)

df['trend_structure_3'] = (
    (df['higher_high'] + df['higher_low']).rolling(3).sum() -
    (df['lower_high'] + df['lower_low']).rolling(3).sum()
)

# Pivot Points (using previous day)
df['pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
df['r1'] = 2 * df['pivot'] - df['Low'].shift(1)
df['s1'] = 2 * df['pivot'] - df['High'].shift(1)
df['r2'] = df['pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
df['s2'] = df['pivot'] - (df['High'].shift(1) - df['Low'].shift(1))

df['dist_to_pivot'] = (df['Close'] - df['pivot']) / df['pivot'] * 100
df['dist_to_r1'] = (df['r1'] - df['Close']) / df['Close'] * 100
df['dist_to_s1'] = (df['Close'] - df['s1']) / df['Close'] * 100
```

**Checklist Day 4:**
- [ ] Add bar pattern features
- [ ] Add trend structure features
- [ ] Add pivot point features
- [ ] Test pattern detection accuracy

---

## Phase 3: Cross-Asset & Range Features (Days 5-6)

### Day 5: Cross-Asset Correlation

**Relative Strength**

| Feature | Description |
|---------|-------------|
| `qqq_rel_strength` | QQQ/SPY ratio (tech leadership) |
| `iwm_rel_strength` | IWM/SPY ratio (small cap leadership) |
| `qqq_rs_momentum` | 5-day change in QQQ relative strength |
| `iwm_rs_momentum` | 5-day change in IWM relative strength |

**Correlation Features**

| Feature | Description |
|---------|-------------|
| `spy_qqq_corr_20` | 20-day rolling correlation |
| `spy_iwm_corr_20` | 20-day rolling correlation |
| `corr_breakdown` | Correlation < 0.7 (regime change signal) |

**Spread Features**

| Feature | Description |
|---------|-------------|
| `spy_qqq_spread` | Daily return difference |
| `spy_iwm_spread` | Daily return difference |
| `spread_extreme` | Spread > 2 std |

```python
# Note: Requires fetching all three tickers
# This would be calculated in a separate function that has access to all tickers

def calculate_cross_asset_features(spy_df, qqq_df, iwm_df):
    # Relative Strength
    spy_df['qqq_rel_strength'] = qqq_df['Close'] / spy_df['Close']
    spy_df['iwm_rel_strength'] = iwm_df['Close'] / spy_df['Close']

    spy_df['qqq_rs_momentum'] = spy_df['qqq_rel_strength'].pct_change(5) * 100
    spy_df['iwm_rs_momentum'] = spy_df['iwm_rel_strength'].pct_change(5) * 100

    # Correlations
    spy_ret = spy_df['Close'].pct_change()
    qqq_ret = qqq_df['Close'].pct_change()
    iwm_ret = iwm_df['Close'].pct_change()

    spy_df['spy_qqq_corr_20'] = spy_ret.rolling(20).corr(qqq_ret)
    spy_df['spy_iwm_corr_20'] = spy_ret.rolling(20).corr(iwm_ret)
    spy_df['corr_breakdown'] = (spy_df['spy_qqq_corr_20'] < 0.7).astype(int)

    # Spreads
    spy_df['spy_qqq_spread'] = spy_ret - qqq_ret
    spy_df['spy_iwm_spread'] = spy_ret - iwm_ret

    spread_std = spy_df['spy_qqq_spread'].rolling(20).std()
    spy_df['spread_extreme'] = (abs(spy_df['spy_qqq_spread']) > 2 * spread_std).astype(int)

    return spy_df
```

**Checklist Day 5:**
- [ ] Create cross-asset feature function
- [ ] Implement relative strength
- [ ] Implement correlation features
- [ ] Test with historical data

---

### Day 6: Range Pattern Features

**Range Analysis**

| Feature | Description |
|---------|-------------|
| `avg_range_10` | 10-day average range % |
| `avg_range_20` | 20-day average range % |
| `range_vs_avg` | Today's range / average |
| `range_expansion` | Range > 1.5x average |
| `range_contraction` | Range < 0.5x average |

**Opening Range Features** (for intraday)

| Feature | Description |
|---------|-------------|
| `gap_size` | Open vs prev close % |
| `gap_filled_prev` | Did yesterday's gap fill? |
| `gap_reversal` | Gap opposite to prev day close |

**Historical Range Patterns**

| Feature | Description |
|---------|-------------|
| `range_rank_20` | Percentile rank of today's range |
| `consec_narrow` | Consecutive narrow range days |
| `breakout_potential` | Low range + high volume setup |

```python
# Range Analysis
range_pct = (df['High'] - df['Low']) / df['Close'] * 100
df['avg_range_10'] = range_pct.rolling(10).mean()
df['avg_range_20'] = range_pct.rolling(20).mean()
df['range_vs_avg'] = range_pct / df['avg_range_20']
df['range_expansion'] = (df['range_vs_avg'] > 1.5).astype(int)
df['range_contraction'] = (df['range_vs_avg'] < 0.5).astype(int)

# Range percentile
df['range_rank_20'] = range_pct.rolling(20).rank(pct=True)

# Consecutive narrow range
narrow = (df['range_vs_avg'] < 0.7).astype(int)
df['consec_narrow'] = narrow.groupby((narrow != narrow.shift()).cumsum()).cumcount() + 1
df.loc[narrow == 0, 'consec_narrow'] = 0

# Breakout potential (narrow range + increasing volume)
df['breakout_potential'] = (
    (df['consec_narrow'] >= 2) &
    (df['prev_volume_ratio'] > 1.2)
).astype(int)
```

**Checklist Day 6:**
- [ ] Add range analysis features
- [ ] Add historical range patterns
- [ ] Create breakout potential signal
- [ ] Test feature correlations

---

## Phase 4: Advanced Features (Days 7-8)

### Day 7: Fibonacci & Support/Resistance

**Fibonacci Retracements**

| Feature | Description |
|---------|-------------|
| `fib_236` | 23.6% retracement level |
| `fib_382` | 38.2% retracement level |
| `fib_500` | 50% retracement level |
| `fib_618` | 61.8% retracement level |
| `near_fib_level` | Within 0.5% of any fib |
| `fib_support` | Price bounced off fib level |

**Dynamic Support/Resistance**

| Feature | Description |
|---------|-------------|
| `swing_high_20` | 20-day swing high |
| `swing_low_20` | 20-day swing low |
| `dist_to_resistance` | % to swing high |
| `dist_to_support` | % to swing low |
| `at_resistance` | Within 0.5% of swing high |
| `at_support` | Within 0.5% of swing low |

```python
# Fibonacci Levels (from 20-day swing)
swing_high = df['High'].rolling(20).max()
swing_low = df['Low'].rolling(20).min()
swing_range = swing_high - swing_low

df['fib_236'] = swing_high - 0.236 * swing_range
df['fib_382'] = swing_high - 0.382 * swing_range
df['fib_500'] = swing_high - 0.500 * swing_range
df['fib_618'] = swing_high - 0.618 * swing_range

# Distance to fib levels
df['dist_to_fib_382'] = abs(df['Close'] - df['fib_382']) / df['Close'] * 100
df['dist_to_fib_618'] = abs(df['Close'] - df['fib_618']) / df['Close'] * 100

# Near any fib level
df['near_fib_level'] = (
    (df['dist_to_fib_382'] < 0.5) |
    (df['dist_to_fib_618'] < 0.5)
).astype(int)

# Support/Resistance
df['swing_high_20'] = swing_high
df['swing_low_20'] = swing_low
df['dist_to_resistance'] = (swing_high - df['Close']) / df['Close'] * 100
df['dist_to_support'] = (df['Close'] - swing_low) / df['Close'] * 100
df['at_resistance'] = (df['dist_to_resistance'] < 0.5).astype(int)
df['at_support'] = (df['dist_to_support'] < 0.5).astype(int)
```

**Checklist Day 7:**
- [ ] Add Fibonacci features
- [ ] Add support/resistance features
- [ ] Test level detection
- [ ] Validate with chart analysis

---

### Day 8: Calendar & Order Flow Proxies

**Enhanced Calendar Features**

| Feature | Description |
|---------|-------------|
| `is_opex_week` | Options expiration week |
| `days_to_opex` | Days until monthly opex |
| `is_quad_witching` | Quarterly opex (Mar/Jun/Sep/Dec) |
| `is_fomc_week` | FOMC meeting week |
| `is_earnings_season` | Jan/Apr/Jul/Oct |

**Order Flow Proxies** (estimated from OHLCV)

| Feature | Description |
|---------|-------------|
| `buying_pressure` | (Close - Low) / Range |
| `selling_pressure` | (High - Close) / Range |
| `net_pressure` | Buying - Selling pressure |
| `accumulation` | Buying pressure * Volume |
| `distribution` | Selling pressure * Volume |
| `acc_dist_ratio_5` | 5-day accumulation / distribution |

```python
# Calendar Features
df['month'] = df.index.month
df['is_earnings_season'] = df['month'].isin([1, 4, 7, 10]).astype(int)

# Options expiration (3rd Friday of month)
# This is simplified - would need actual calendar lookup for accuracy
df['day_of_month'] = df.index.day
df['is_opex_week'] = ((df['day_of_month'] >= 15) & (df['day_of_month'] <= 21)).astype(int)
df['is_quad_witching'] = (df['is_opex_week'] & df['month'].isin([3, 6, 9, 12])).astype(int)

# Order Flow Proxies
range_size = df['High'] - df['Low'] + 0.001
df['buying_pressure'] = (df['Close'] - df['Low']) / range_size
df['selling_pressure'] = (df['High'] - df['Close']) / range_size
df['net_pressure'] = df['buying_pressure'] - df['selling_pressure']

df['accumulation'] = df['buying_pressure'] * df['Volume']
df['distribution'] = df['selling_pressure'] * df['Volume']
df['acc_dist_ratio_5'] = (
    df['accumulation'].rolling(5).sum() /
    (df['distribution'].rolling(5).sum() + 1)
)

# Money Flow Index style
df['money_flow'] = df['net_pressure'] * df['Volume']
df['money_flow_5'] = df['money_flow'].rolling(5).sum()
df['money_flow_10'] = df['money_flow'].rolling(10).sum()
```

**Checklist Day 8:**
- [ ] Add calendar features
- [ ] Add order flow proxy features
- [ ] Validate money flow calculations
- [ ] Test feature importance

---

## Phase 5: Integration & Testing (Days 9-10)

### Day 9: Integration into predict_server.py

**Tasks:**
- [ ] Add all new features to `calculate_daily_features()`
- [ ] Add cross-asset features (separate function)
- [ ] Update feature lists in model predictions
- [ ] Handle NaN values appropriately
- [ ] Add feature validation logging

**Code Structure:**
```python
def calculate_daily_features(df):
    # Existing features...

    # ========== NEW FEATURES ==========

    # TTM Squeeze (Phase 1)
    df = add_ttm_squeeze_features(df)

    # KDJ (Phase 1)
    df = add_kdj_features(df)

    # Volatility Regime (Phase 1)
    df = add_volatility_regime_features(df)

    # Divergences (Phase 2)
    df = add_divergence_features(df)

    # Price Action (Phase 2)
    df = add_price_action_features(df)

    # Range Patterns (Phase 3)
    df = add_range_features(df)

    # Fibonacci/S&R (Phase 4)
    df = add_fib_sr_features(df)

    # Calendar (Phase 4)
    df = add_calendar_features(df)

    # Order Flow (Phase 4)
    df = add_order_flow_features(df)

    return df
```

---

### Day 10: Testing & Documentation

**Testing Tasks:**
- [ ] Run feature calculation on 5 years of data
- [ ] Verify no NaN/inf values in production range
- [ ] Check feature correlations (remove highly correlated)
- [ ] Benchmark prediction accuracy with new features
- [ ] A/B test: old features vs new features

**Documentation Tasks:**
- [ ] Update INTRADAY_MODEL_FEATURES.md with all new features
- [ ] Create feature importance analysis
- [ ] Document any removed/modified features
- [ ] Add example calculations for each new feature

---

## Summary: Feature Count

| Phase | Features Added | Cumulative |
|-------|---------------|------------|
| Phase 1 (Days 1-2) | 18 | 18 |
| Phase 2 (Days 3-4) | 22 | 40 |
| Phase 3 (Days 5-6) | 18 | 58 |
| Phase 4 (Days 7-8) | 20 | 78 |
| **Total New Features** | **78** | - |

**Current Features:** ~80
**After Implementation:** ~158

---

## Priority If Time-Constrained

If you can only implement some features, prioritize in this order:

1. **TTM Squeeze** - Best breakout predictor
2. **KDJ (9,3,3)** - Enhanced momentum timing
3. **Volatility Regime** - Adapts predictions to market conditions
4. **Divergences** - Powerful reversal signals
5. **Price Action Patterns** - Direct high/low predictors
6. **Range Features** - Volatility expansion setup
7. **Fibonacci/S&R** - Key price levels
8. **Cross-Asset** - Market regime context
9. **Calendar Features** - Event-based adjustments
10. **Order Flow** - Volume-based confirmation

---

## Execution Command

To start implementation:
```bash
cd /Users/it/Documents/mvp_coder_starter_kit\ \(2\)/mvp-trading-app/ml
# Back up current predict_server.py
cp predict_server.py predict_server.py.backup

# Begin Phase 1 implementation
# Edit predict_server.py to add TTM Squeeze and KDJ
```
