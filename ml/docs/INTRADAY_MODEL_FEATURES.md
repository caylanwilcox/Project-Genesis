# Intraday Model Features Reference

This document provides a complete reference of all 16 features used in the Intraday Direction Prediction Model.

---

## Feature Summary

| # | Feature Name | Category | Data Type | Range |
|---|-------------|----------|-----------|-------|
| 1 | `time_pct` | Time-Based | Float | 0.0 - 1.0 |
| 2 | `time_remaining` | Time-Based | Float | 0.0 - 1.0 |
| 3 | `gap` | Gap Analysis | Float | Typically -0.05 to 0.05 |
| 4 | `gap_direction` | Gap Analysis | Int | -1, 0, 1 |
| 5 | `gap_size` | Gap Analysis | Float | 0.0 - 0.10+ |
| 6 | `prev_return` | Previous Day | Float | Typically -0.05 to 0.05 |
| 7 | `prev_range` | Previous Day | Float | Typically 0.005 to 0.05 |
| 8 | `current_vs_open` | Current Session | Float | Typically -0.03 to 0.03 |
| 9 | `current_vs_open_direction` | Current Session | Int | -1, 0, 1 |
| 10 | `position_in_range` | Current Session | Float | 0.0 - 1.0 |
| 11 | `range_so_far_pct` | Current Session | Float | 0.0 - 0.05+ |
| 12 | `high_so_far_pct` | Current Session | Float | 0.0 - 0.03+ |
| 13 | `low_so_far_pct` | Current Session | Float | 0.0 - 0.03+ |
| 14 | `above_open` | Momentum | Binary | 0, 1 |
| 15 | `near_high` | Momentum | Binary | 0, 1 |
| 16 | `gap_filled` | Gap Analysis | Binary | 0, 1 |

---

## Detailed Feature Descriptions

### Time-Based Features

#### 1. `time_pct`
- **Description:** Progress through the trading session as a decimal
- **Calculation:** `(current_time - market_open) / (market_close - market_open)`
- **Values:**
  - `0.0` = Market open (9:30 AM ET)
  - `0.5` = Mid-session (12:15 PM ET)
  - `1.0` = Market close (4:00 PM ET)
- **Importance:** High - model accuracy increases with time_pct

#### 2. `time_remaining`
- **Description:** Fraction of the trading session remaining
- **Calculation:** `1.0 - time_pct`
- **Values:**
  - `1.0` = Full session remaining (at open)
  - `0.0` = Session ending (at close)
- **Importance:** Indicates how much time is left for price reversal

---

### Gap Analysis Features

#### 3. `gap`
- **Description:** Overnight gap between today's open and yesterday's close
- **Calculation:** `(today_open - prev_close) / prev_close`
- **Values:**
  - Positive = Gap up (bullish overnight sentiment)
  - Negative = Gap down (bearish overnight sentiment)
  - Zero = Flat open
- **Typical Range:** -5% to +5%
- **Importance:** Strong predictor, especially early in session

#### 4. `gap_direction`
- **Description:** Categorical direction of the overnight gap
- **Calculation:**
  - `+1` if gap > 0 (gap up)
  - `-1` if gap < 0 (gap down)
  - `0` if gap = 0 (flat)
- **Importance:** Simplifies gap signal for model interpretation

#### 5. `gap_size`
- **Description:** Absolute magnitude of the overnight gap
- **Calculation:** `abs(gap)`
- **Values:** Always >= 0
- **Typical Range:** 0% to 5%
- **Importance:** Larger gaps often have higher reversal probability

#### 16. `gap_filled`
- **Description:** Whether the price has filled (closed) the overnight gap
- **Calculation:**
  - For gap up: `1` if low_so_far <= prev_close
  - For gap down: `1` if high_so_far >= prev_close
- **Values:** Binary (0 or 1)
- **Importance:** Gap fill patterns have predictive value for close direction

---

### Previous Day Context Features

#### 6. `prev_return`
- **Description:** Yesterday's daily return
- **Calculation:** `(prev_close - prev_prev_close) / prev_prev_close`
- **Values:**
  - Positive = Yesterday was an up day
  - Negative = Yesterday was a down day
- **Typical Range:** -5% to +5%
- **Importance:** Captures momentum/mean-reversion tendencies

#### 7. `prev_range`
- **Description:** Yesterday's trading range as a percentage
- **Calculation:** `(prev_high - prev_low) / prev_close`
- **Values:** Always >= 0
- **Typical Range:** 0.5% to 5%
- **Importance:** Indicates recent volatility level

---

### Current Session Features

#### 8. `current_vs_open`
- **Description:** Current price change from today's open
- **Calculation:** `(current_price - today_open) / today_open`
- **Values:**
  - Positive = Currently trading above open
  - Negative = Currently trading below open
- **Typical Range:** -3% to +3%
- **Importance:** Primary intraday momentum indicator

#### 9. `current_vs_open_direction`
- **Description:** Categorical direction relative to today's open
- **Calculation:**
  - `+1` if current_price > open (currently up)
  - `-1` if current_price < open (currently down)
  - `0` if current_price = open
- **Importance:** Simplifies current direction for model

#### 10. `position_in_range`
- **Description:** Current price position within today's high-low range
- **Calculation:** `(current_price - low_so_far) / (high_so_far - low_so_far)`
- **Values:**
  - `0.0` = At the day's low
  - `0.5` = Middle of the range
  - `1.0` = At the day's high
- **Importance:** Indicates whether price is near support or resistance

#### 11. `range_so_far_pct`
- **Description:** Today's trading range (so far) as a percentage
- **Calculation:** `(high_so_far - low_so_far) / today_open`
- **Values:** Always >= 0
- **Typical Range:** 0% to 5%
- **Importance:** Indicates today's volatility level

#### 12. `high_so_far_pct`
- **Description:** Distance from open to today's high (so far)
- **Calculation:** `(high_so_far - today_open) / today_open`
- **Values:** Always >= 0
- **Typical Range:** 0% to 3%
- **Importance:** Measures bullish extension from open

#### 13. `low_so_far_pct`
- **Description:** Distance from open to today's low (so far)
- **Calculation:** `(today_open - low_so_far) / today_open`
- **Values:** Always >= 0
- **Typical Range:** 0% to 3%
- **Importance:** Measures bearish extension from open

---

### Momentum Indicator Features

#### 14. `above_open`
- **Description:** Binary flag indicating if currently above today's open
- **Calculation:** `1 if current_price > today_open else 0`
- **Values:** Binary (0 or 1)
- **Importance:** Simple directional confirmation

#### 15. `near_high`
- **Description:** Whether price is closer to today's high than low
- **Calculation:** `1 if (high_so_far - current_price) < (current_price - low_so_far) else 0`
- **Values:** Binary (0 or 1)
- **Importance:** Indicates bullish vs bearish positioning within range

---

## Feature Categories by Predictive Power

### High Importance (Early Session)
1. `gap` - Strong signal at open
2. `gap_direction` - Directional context
3. `gap_size` - Magnitude matters
4. `prev_return` - Recent momentum

### High Importance (Mid-Session+)
1. `current_vs_open` - Primary momentum
2. `position_in_range` - Support/resistance proximity
3. `time_pct` - Accuracy increases with time
4. `above_open` - Direction confirmation

### Moderate Importance
1. `gap_filled` - Gap fill patterns
2. `range_so_far_pct` - Volatility context
3. `high_so_far_pct` / `low_so_far_pct` - Extension levels
4. `prev_range` - Historical volatility

### Context Features
1. `time_remaining` - Reversal window
2. `near_high` - Range position
3. `current_vs_open_direction` - Simplified direction

---

## Feature Calculation Example

```python
# Example: SPY at 11:00 AM ET
# Previous day: Close $580.00, High $582.50, Low $578.00
# Today: Open $582.00, Current $583.50, High $584.00, Low $581.00

time_pct = 0.23  # (11:00 - 9:30) / (16:00 - 9:30)
time_remaining = 0.77  # 1.0 - 0.23

gap = (582.00 - 580.00) / 580.00  # = 0.00345 (+0.34%)
gap_direction = 1  # Gap up
gap_size = 0.00345

prev_return = -0.005  # Yesterday was -0.5%
prev_range = (582.50 - 578.00) / 580.00  # = 0.00776

current_vs_open = (583.50 - 582.00) / 582.00  # = 0.00258 (+0.26%)
current_vs_open_direction = 1  # Above open
position_in_range = (583.50 - 581.00) / (584.00 - 581.00)  # = 0.833
range_so_far_pct = (584.00 - 581.00) / 582.00  # = 0.00515
high_so_far_pct = (584.00 - 582.00) / 582.00  # = 0.00344
low_so_far_pct = (582.00 - 581.00) / 582.00  # = 0.00172

above_open = 1  # Current > Open
near_high = 1  # Closer to high than low
gap_filled = 0  # Low (581) hasn't reached prev_close (580)
```

---

## Model Targets

The intraday model predicts two targets:

### Target A: `target` (Close > Open)
- **Question:** Will today close higher than today's open?
- **Use Case:** Classic "bullish day" prediction

### Target B: `target_close_above_current`
- **Question:** Will today close higher than the CURRENT price?
- **Use Case:** Real-time directional signal for trading decisions
- **Note:** This is the primary target used in production

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Period | 2003-2023 (~5,000 days/ticker) |
| Test Period | 2024-2025 (out-of-sample) |
| Time Slices | 0%, 10%, 25%, 50%, 75%, 90% |
| Samples/Day | 6 (one per time slice) |
| Total Samples | ~30,000 per ticker |
| Feature Scaling | RobustScaler |
| Ensemble Models | XGBoost, GradientBoosting, RandomForest, ExtraTrees |

---

## Source Code Reference

- **Training Script:** `ml/train_intraday_model.py`
- **Feature Extraction:** `create_intraday_features()` function
- **Session Simulation:** `create_session_snapshots()` function
- **Production Models:** `ml/models/{ticker}_intraday_model.pkl`

---

# Additional Technical Indicators (Daily Models)

The production prediction server (`predict_server.py`) calculates many additional technical indicators for the daily prediction models. These are computed from historical OHLCV data and used by various models (FVG, High/Low, Daily Direction, etc.).

---

## Moving Averages

### Simple Moving Averages (SMA)

| Feature | Period | Description |
|---------|--------|-------------|
| `sma_5` | 5 days | Short-term trend |
| `sma_10` | 10 days | Short-term trend |
| `sma_20` | 20 days | Medium-term trend (1 month) |
| `sma_50` | 50 days | Intermediate trend (2.5 months) |

### Exponential Moving Averages (EMA)

| Feature | Period | Description |
|---------|--------|-------------|
| `ema_9` | 9 days | Fast EMA for crossover signals |
| `ema_12` | 12 days | MACD fast line component |
| `ema_21` | 21 days | Slow EMA for crossover signals |
| `ema_26` | 26 days | MACD slow line component |

### Price vs Moving Average Features

| Feature | Calculation | Description |
|---------|-------------|-------------|
| `price_vs_sma5` | `(close - sma_5) / sma_5 * 100` | % distance from 5-day SMA |
| `price_vs_sma10` | `(close - sma_10) / sma_10 * 100` | % distance from 10-day SMA |
| `price_vs_sma20` | `(close - sma_20) / sma_20 * 100` | % distance from 20-day SMA |
| `price_vs_sma50` | `(close - sma_50) / sma_50 * 100` | % distance from 50-day SMA |
| `price_vs_ema9` | `(close - ema_9) / ema_9 * 100` | % distance from 9-day EMA |
| `price_vs_ema21` | `(close - ema_21) / ema_21 * 100` | % distance from 21-day EMA |

### Moving Average Crossover Features

| Feature | Calculation | Description |
|---------|-------------|-------------|
| `sma5_vs_sma20` | `(sma_5 - sma_20) / sma_20 * 100` | Short vs medium-term trend |
| `sma10_vs_sma50` | `(sma_10 - sma_50) / sma_50 * 100` | Medium vs long-term trend |
| `ema9_vs_ema21` | `(ema_9 - ema_21) / ema_21 * 100` | EMA crossover signal |
| `ema_cross` | Alias for `ema9_vs_ema21` | Used by improved model |

### Trend Alignment

| Feature | Calculation | Range |
|---------|-------------|-------|
| `trend_alignment` | Sum of 5 trend conditions / 5 | 0.0 - 1.0 |
| `trend_strength` | Alias for `trend_alignment` | 0.0 - 1.0 |

**Conditions checked:**
1. Price > SMA 5
2. Price > SMA 20
3. Price > SMA 50
4. SMA 5 > SMA 20
5. SMA 20 > SMA 50

---

## RSI (Relative Strength Index)

| Feature | Description |
|---------|-------------|
| `rsi_14` | 14-period RSI (0-100) |
| `rsi_9` | 9-period RSI (faster, more sensitive) |
| `prev_rsi` | Previous day's RSI 14 |
| `rsi_oversold` | Binary: RSI < 30 |
| `rsi_overbought` | Binary: RSI > 70 |
| `rsi_momentum` | RSI change over 2 periods |
| `rsi_change` | Day-over-day RSI change |

### RSI Interpretation
- **< 30:** Oversold (potential bounce)
- **30-40:** Low, approaching oversold
- **40-60:** Neutral zone
- **60-70:** Elevated, approaching overbought
- **> 70:** Overbought (potential pullback)

---

## MACD (Moving Average Convergence Divergence)

| Feature | Calculation | Description |
|---------|-------------|-------------|
| `macd` | `ema_12 - ema_26` | MACD line |
| `macd_signal` | `ema(macd, 9)` | Signal line |
| `macd_histogram` | `macd - macd_signal` | Histogram (momentum) |
| `prev_macd_hist` | Previous histogram value | For trend analysis |
| `macd_crossover` | Signal line crossover | +1 bullish, -1 bearish |
| `macd_cross` | Histogram sign change | Binary crossover |
| `macd_divergence` | Histogram change | Momentum acceleration |
| `macd_hist_change` | Alias for divergence | Used by improved model |

---

## Bollinger Bands

| Feature | Calculation | Description |
|---------|-------------|-------------|
| `bb_middle` | 20-day SMA | Center line |
| `bb_upper` | `middle + 2 * std` | Upper band (+2σ) |
| `bb_lower` | `middle - 2 * std` | Lower band (-2σ) |
| `bb_width` | `(upper - lower) / middle * 100` | Band width % |
| `bb_position` | `(close - lower) / (upper - lower)` | Position in bands (0-1) |
| `bb_squeeze` | Width < 20th percentile | Volatility contraction |

---

## Stochastic Oscillator

| Feature | Calculation | Range |
|---------|-------------|-------|
| `stoch_k` | `(close - low_14) / (high_14 - low_14) * 100` | 0-100 |
| `stoch_d` | 3-period SMA of %K | 0-100 |
| `stoch_crossover` | %K vs %D crossover | -1, 0, +1 |

---

## Williams %R

| Feature | Calculation | Range |
|---------|-------------|-------|
| `williams_r` | `(high_14 - close) / (high_14 - low_14) * -100` | -100 to 0 |

**Interpretation:**
- **-80 to -100:** Oversold
- **0 to -20:** Overbought

---

## ADX (Average Directional Index)

| Feature | Description | Range |
|---------|-------------|-------|
| `adx` | Trend strength indicator | 0-100 |
| `di_diff` | +DI minus -DI | -100 to +100 |

**ADX Interpretation:**
- **< 20:** Weak/no trend
- **20-40:** Developing trend
- **40-60:** Strong trend
- **> 60:** Very strong trend

---

## ATR (Average True Range)

| Feature | Calculation | Description |
|---------|-------------|-------------|
| `atr_14` | 14-period ATR | Volatility measure |
| `atr_pct` | `atr_14 / close * 100` | ATR as % of price |
| `atr_normalized` | `atr_14 / sma_20 * 100` | Normalized ATR |

---

## Volume Indicators

| Feature | Calculation | Description |
|---------|-------------|-------------|
| `volume_sma_20` | 20-day volume SMA | Average volume |
| `prev_volume_ratio` | `volume / volume_sma_20` | Relative volume |
| `volume_trend` | `vol_5 / vol_20` | Volume momentum |
| `volume_price_trend` | `return * volume_ratio` | Price-volume correlation |
| `vol_price_corr` | 10-day correlation | Volume-price relationship |
| `obv_trend` | On-Balance Volume trend | Cumulative volume flow |

---

## Z-Score Features

| Feature | Period | Description |
|---------|--------|-------------|
| `zscore_5` | 5-day | Distance from MA in std units |
| `zscore_10` | 10-day | Distance from MA in std units |
| `zscore_20` | 20-day | Distance from MA in std units |
| `zscore_50` | 50-day | Distance from MA in std units |

**Interpretation:**
- **> +2:** Significantly above average (overbought)
- **+1 to +2:** Moderately above average
- **-1 to +1:** Normal range
- **-2 to -1:** Moderately below average
- **< -2:** Significantly below average (oversold)

---

## Distance Features

| Feature | Description |
|---------|-------------|
| `dist_20d_high` | % from 20-day high |
| `dist_20d_low` | % from 20-day low |
| `dist_52w_high` | % from 52-week high |
| `dist_52w_low` | % from 52-week low |

---

## Candlestick Features

| Feature | Calculation | Description |
|---------|-------------|-------------|
| `upper_wick` | `(high - max(open,close)) / range` | Upper shadow ratio |
| `lower_wick` | `(min(open,close) - low) / range` | Lower shadow ratio |
| `body_ratio` | `abs(close - open) / range` | Body size ratio |

---

## Consecutive Day Features (Mean Reversion)

| Feature | Description |
|---------|-------------|
| `consec_up` | Consecutive up days count |
| `consec_down` | Consecutive down days count |
| `prev_consec_up` | Previous day's up streak |
| `prev_consec_down` | Previous day's down streak |
| `streak` | `consec_up - consec_down` |
| `streak_extreme` | Binary: \|streak\| >= 3 |

---

## Calendar Features

| Feature | Description |
|---------|-------------|
| `day_of_week` | 0=Monday, 4=Friday |
| `is_monday` | Binary |
| `is_friday` | Binary |
| `day_of_month` | 1-31 |
| `is_month_start` | Day <= 3 |
| `is_month_end` | Day >= 27 |

---

## Feature Calculation Code Reference

```python
# RSI Calculation
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / (loss + 0.001)
rsi_14 = 100 - (100 / (1 + rs))

# MACD Calculation
ema_12 = df['Close'].ewm(span=12).mean()
ema_26 = df['Close'].ewm(span=26).mean()
macd = ema_12 - ema_26
macd_signal = macd.ewm(span=9).mean()
macd_histogram = macd - macd_signal

# Bollinger Bands
bb_middle = df['Close'].rolling(20).mean()
bb_std = df['Close'].rolling(20).std()
bb_upper = bb_middle + 2 * bb_std
bb_lower = bb_middle - 2 * bb_std

# ATR Calculation
high_low = df['High'] - df['Low']
high_close = abs(df['High'] - df['Close'].shift(1))
low_close = abs(df['Low'] - df['Close'].shift(1))
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
atr_14 = tr.rolling(14).mean()
```

---

---

## TTM Squeeze Indicator

The TTM Squeeze detects volatility compression when Bollinger Bands contract inside Keltner Channels, signaling potential explosive moves.

| Feature | Description | Range |
|---------|-------------|-------|
| `ttm_squeeze_on` | BB inside KC (compression active) | Binary 0/1 |
| `ttm_squeeze_off` | BB outside KC (volatility expanding) | Binary 0/1 |
| `ttm_squeeze_fired` | Transition from squeeze ON to OFF | Binary 0/1 |
| `ttm_squeeze_momentum` | Price - midline (direction indicator) | Continuous |
| `ttm_squeeze_momentum_rising` | Momentum increasing | Binary 0/1 |
| `ttm_squeeze_bars` | Consecutive bars in squeeze | Integer 0+ |

**Calculation:**
- Bollinger Bands: 20-period SMA ± 2 standard deviations
- Keltner Channels: 20-period EMA ± 1.5 ATR
- Squeeze ON: BB lower > KC lower AND BB upper < KC upper

**Trading Signals:**
- Squeeze ON → Low volatility, consolidation phase
- Squeeze FIRED → Breakout imminent, enter on momentum direction
- Rising momentum + squeeze fired → Strong directional move expected

---

## KDJ Indicator (9,3,3)

KDJ is an enhanced Stochastic oscillator with a J line that provides extreme overbought/oversold readings.

| Feature | Description | Range |
|---------|-------------|-------|
| `kdj_k` | Fast stochastic line | 0-100 |
| `kdj_d` | Signal line (smoothed K) | 0-100 |
| `kdj_j` | Extreme indicator (3K - 2D) | -50 to 150+ |
| `kdj_golden_cross` | K crosses above D (bullish) | Binary 0/1 |
| `kdj_death_cross` | K crosses below D (bearish) | Binary 0/1 |
| `kdj_j_overbought` | J > 100 (extremely overbought) | Binary 0/1 |
| `kdj_j_oversold` | J < 0 (extremely oversold) | Binary 0/1 |
| `kdj_zone` | Overbought (+1), Oversold (-1), Neutral (0) | -1, 0, +1 |

**Calculation:**
```python
RSV = (Close - Lowest_Low_9) / (Highest_High_9 - Lowest_Low_9) * 100
K = EMA(RSV, 3)
D = EMA(K, 3)
J = 3 * K - 2 * D
```

**Trading Signals:**
- J > 100: Extremely overbought, reversal likely
- J < 0: Extremely oversold, bounce likely
- Golden Cross + J rising from below 20: Strong buy
- Death Cross + J falling from above 80: Strong sell

---

## Volatility Regime

Classifies the market into volatility regimes to adjust trading strategies.

| Feature | Description | Range |
|---------|-------------|-------|
| `hv_10` | 10-day historical volatility (annualized %) | 5-100+ |
| `hv_20` | 20-day historical volatility (annualized %) | 5-100+ |
| `hv_ratio` | HV10 / HV20 ratio | 0.5-2.0 |
| `vol_regime` | 0=Low, 1=Normal, 2=High volatility | 0, 1, 2 |
| `vol_percentile` | 1-year volatility percentile | 0-100 |
| `vol_expanding` | HV10 > HV20 (volatility rising) | Binary 0/1 |
| `vol_contracting` | HV10 < HV20 * 0.8 (volatility falling) | Binary 0/1 |
| `atr_pct` | ATR as % of price | 0.5-5.0 |
| `atr_regime` | ATR-based regime (0=Low, 1=Normal, 2=High) | 0, 1, 2 |

**Regime Classification:**
- **Low Volatility (0):** HV20 < 20th percentile of past year
- **Normal Volatility (1):** HV20 between 20th-80th percentile
- **High Volatility (2):** HV20 > 80th percentile of past year

**Trading Applications:**
- Low vol regime → Tighter stops, smaller targets
- High vol regime → Wider stops, larger targets
- Vol expanding → Momentum strategies work better
- Vol contracting → Mean reversion strategies work better

---

## Divergence Detection

Divergences occur when price makes a new extreme but momentum indicators don't confirm, signaling potential reversals.

### RSI Divergence

| Feature | Description | Range |
|---------|-------------|-------|
| `rsi_bullish_div` | Price new low, RSI higher low | Binary 0/1 |
| `rsi_bearish_div` | Price new high, RSI lower high | Binary 0/1 |
| `rsi_div_strength` | How far RSI is from confirming | 0-50+ |

### MACD Divergence

| Feature | Description | Range |
|---------|-------------|-------|
| `macd_bullish_div` | Price new low, MACD histogram higher | Binary 0/1 |
| `macd_bearish_div` | Price new high, MACD histogram lower | Binary 0/1 |

### OBV Divergence

| Feature | Description | Range |
|---------|-------------|-------|
| `obv_bullish_div` | Price new low, OBV higher | Binary 0/1 |
| `obv_bearish_div` | Price new high, OBV lower | Binary 0/1 |

### Combined Divergence Scores

| Feature | Description | Range |
|---------|-------------|-------|
| `bullish_div_count` | Sum of bullish divergences (RSI+MACD+OBV) | 0-3 |
| `bearish_div_count` | Sum of bearish divergences (RSI+MACD+OBV) | 0-3 |
| `div_signal` | Bullish - Bearish count | -3 to +3 |

**Trading Signals:**
- `bullish_div_count >= 2` → Strong reversal potential from bottom
- `bearish_div_count >= 2` → Strong reversal potential from top
- Divergences are most reliable after extended trends

---

## Bar Patterns

Candlestick and bar patterns that indicate volatility and trend changes.

| Feature | Description | Range |
|---------|-------------|-------|
| `inside_bar` | Range contained within previous bar | Binary 0/1 |
| `outside_bar` | Range exceeds previous bar (engulfing) | Binary 0/1 |
| `narrow_range_4` | Smallest range in 4 bars (NR4) | Binary 0/1 |
| `narrow_range_7` | Smallest range in 7 bars (NR7) | Binary 0/1 |
| `wide_range_bar` | Range > 2x average range | Binary 0/1 |

**Trading Signals:**
- Inside bar → Consolidation, breakout imminent
- NR4/NR7 → Extreme compression, explosive move likely
- Wide range bar → Institutional activity, trend continuation

---

## Trend Structure

Higher highs/higher lows (uptrend) vs lower highs/lower lows (downtrend) analysis.

| Feature | Description | Range |
|---------|-------------|-------|
| `higher_high` | Today's high > yesterday's high | Binary 0/1 |
| `lower_low` | Today's low < yesterday's low | Binary 0/1 |
| `higher_low` | Today's low > yesterday's low | Binary 0/1 |
| `lower_high` | Today's high < yesterday's high | Binary 0/1 |
| `trend_structure_3` | HH+HL - LH-LL over 3 bars | -6 to +6 |

**Interpretation:**
- `trend_structure_3 > 2` → Strong bullish structure
- `trend_structure_3 < -2` → Strong bearish structure
- `trend_structure_3 ≈ 0` → Choppy/range-bound

---

## Pivot Points

Classic floor trader pivot levels for support/resistance.

| Feature | Description |
|---------|-------------|
| `pivot` | (H + L + C) / 3 from previous day |
| `pivot_r1` | First resistance: 2 × Pivot - Low |
| `pivot_s1` | First support: 2 × Pivot - High |
| `pivot_r2` | Second resistance: Pivot + Range |
| `pivot_s2` | Second support: Pivot - Range |
| `dist_to_pivot` | % distance from pivot |
| `dist_to_r1` | % distance to R1 |
| `dist_to_s1` | % distance to S1 |
| `above_pivot` | Price above pivot (1) or below (0) |

**Trading Applications:**
- Price above pivot → Bullish bias for the day
- R1/R2 → Take profit levels for longs
- S1/S2 → Take profit levels for shorts
- Pivot often acts as magnet during consolidation

---

## Range Pattern Features

Analyze daily range patterns to detect volatility expansion/contraction cycles.

| Feature | Description | Range |
|---------|-------------|-------|
| `avg_range_10` | 10-day average range % | 0.5-5.0% |
| `avg_range_20` | 20-day average range % | 0.5-5.0% |
| `range_vs_avg` | Today's range / 20-day avg | 0.2-3.0 |
| `range_expansion` | Range > 1.5x average | Binary 0/1 |
| `range_contraction` | Range < 0.5x average | Binary 0/1 |
| `range_rank_20` | Percentile rank of today's range | 0-1 |
| `consec_narrow` | Consecutive narrow range days | 0-10+ |
| `breakout_potential` | Narrow range + high volume | Binary 0/1 |

**Trading Signals:**
- `consec_narrow >= 3` → Breakout imminent
- `breakout_potential = 1` → High probability breakout setup
- `range_expansion = 1` → Trend day likely

---

## Fibonacci Retracements

Classic Fibonacci levels based on 20-day swing range.

| Feature | Description |
|---------|-------------|
| `fib_236` | 23.6% retracement level |
| `fib_382` | 38.2% retracement level |
| `fib_500` | 50% retracement level |
| `fib_618` | 61.8% retracement level |
| `dist_to_fib_382` | % distance to 38.2% level |
| `dist_to_fib_618` | % distance to 61.8% level |
| `near_fib_level` | Within 0.5% of key fib | Binary 0/1 |

**Trading Applications:**
- 38.2% retracement → Shallow pullback in strong trend
- 50% retracement → Standard pullback
- 61.8% retracement → Deep pullback, last support before trend break

---

## Swing High/Low Support & Resistance

Dynamic support/resistance based on 20-day swing points.

| Feature | Description | Range |
|---------|-------------|-------|
| `swing_high_20` | 20-day swing high | Price |
| `swing_low_20` | 20-day swing low | Price |
| `dist_to_resistance` | % to swing high | 0-10%+ |
| `dist_to_support` | % to swing low | 0-10%+ |
| `at_resistance` | Within 0.5% of swing high | Binary 0/1 |
| `at_support` | Within 0.5% of swing low | Binary 0/1 |
| `swing_position` | Position in swing range | 0.0-1.0 |

**Interpretation:**
- `swing_position > 0.8` → Near resistance, caution for longs
- `swing_position < 0.2` → Near support, caution for shorts
- `at_resistance` or `at_support` → Key decision points

---

## Enhanced Calendar Features

Seasonal and calendar-based patterns.

| Feature | Description | Range |
|---------|-------------|-------|
| `day_of_week` | Monday=0 to Friday=4 | 0-4 |
| `is_monday` | Monday trading | Binary 0/1 |
| `is_friday` | Friday trading | Binary 0/1 |
| `day_of_month` | Day number | 1-31 |
| `week_of_month` | Week number | 1-5 |
| `is_month_start` | Days 1-3 | Binary 0/1 |
| `is_month_end` | Days 27+ | Binary 0/1 |
| `is_quarter_end` | Last week of Q | Binary 0/1 |
| `is_quarter_start` | First week of Q | Binary 0/1 |
| `is_opex_week` | Options expiration week | Binary 0/1 |
| `is_first_5_days` | First 5 days of month | Binary 0/1 |
| `is_last_5_days` | Last 5 days of month | Binary 0/1 |

**Trading Patterns:**
- Monday → Often weaker, gap fills common
- Friday → Position squaring, lower volume
- Month end → Window dressing, potential volatility
- OPEX week → Increased volatility, pinning effects

---

## Source Files

| File | Features |
|------|----------|
| `ml/predict_server.py` | All technical indicators (production) |
| `ml/train_intraday_model.py` | Intraday session features |
| `ml/train_fvg_model.py` | FVG-specific features |
| `ml/train_highlow_model.py` | High/Low prediction features |
