# V6 Intraday Trading Model - Official Documentation

**Version:** 6.0 (Time-Split)
**Last Updated:** December 23, 2025
**Training Period:** January 1, 2022 - January 1, 2025 (3 years)
**Test Period:** January 2, 2025 - December 19, 2025 (out-of-sample)

---

## Overview

The V6 model is a time-split ensemble machine learning system that predicts intraday price direction for SPY, QQQ, and IWM ETFs. It uses separate models for early session (9:30 AM - 12 PM) and late session (12 PM - 4 PM) to maximize accuracy.

### Key Innovation: Target B

The model's primary signal is **Target B** - predicting whether the closing price will be higher or lower than the 11 AM price. This target was chosen because:

1. By 12 PM, 2.5 hours of price action provides strong predictive features
2. The 11 AM reference point captures the morning volatility settling
3. Out-of-sample accuracy reaches 80%+ at high confidence levels

---

## Historical Win Rates

### Combined (All Tickers) - Target B at 12 PM

| Confidence | Win Rate | Trades | Recommendation |
|------------|----------|--------|----------------|
| **90%+**   | 100.0%   | 6      | Very Strong    |
| **85-90%** | 79.4%    | 63     | Strong Signal  |
| **80-85%** | 79.6%    | 108    | Strong Signal  |
| **75-80%** | 75.9%    | 108    | Good Signal    |
| **70-75%** | 65.1%    | 106    | Moderate       |
| **65-70%** | 59.6%    | 114    | Weak Signal    |
| **60-65%** | 59.8%    | 82     | Weak Signal    |
| **55-60%** | 53.6%    | 84     | Coin Flip      |
| **50-55%** | 53.4%    | 58     | Coin Flip      |

**Key Insight:** Only take trades with 75%+ confidence for reliable edge.

---

### SPY Win Rates by Confidence

| Confidence | Win Rate | Trades |
|------------|----------|--------|
| 90%+       | 100.0%   | 4      |
| 85-90%     | 78.6%    | 14     |
| 80-85%     | 84.2%    | 38     |
| 75-80%     | 69.4%    | 36     |
| 70-75%     | 70.7%    | 41     |
| 65-70%     | 57.9%    | 38     |
| 60-65%     | 66.7%    | 27     |
| 55-60%     | 60.0%    | 30     |
| 50-55%     | 53.3%    | 15     |

**SPY Overall:** 68.7% (167/243 trades)
**Best Range:** 80-85% confidence = 84.2% accuracy

---

### QQQ Win Rates by Confidence

| Confidence | Win Rate | Trades |
|------------|----------|--------|
| 90%+       | N/A      | 0      |
| 85-90%     | 77.8%    | 27     |
| 80-85%     | 73.7%    | 38     |
| 75-80%     | 78.1%    | 32     |
| 70-75%     | 64.0%    | 25     |
| 65-70%     | 71.9%    | 32     |
| 60-65%     | 54.5%    | 33     |
| 55-60%     | 51.4%    | 35     |
| 50-55%     | 52.4%    | 21     |

**QQQ Overall:** 65.8% (160/243 trades)
**Best Range:** 75-80% confidence = 78.1% accuracy

---

### IWM Win Rates by Confidence

| Confidence | Win Rate | Trades |
|------------|----------|--------|
| 90%+       | 100.0%   | 2      |
| 85-90%     | 81.8%    | 22     |
| 80-85%     | 81.2%    | 32     |
| 75-80%     | 80.0%    | 40     |
| 70-75%     | 60.0%    | 40     |
| 65-70%     | 52.3%    | 44     |
| 60-65%     | 59.1%    | 22     |
| 55-60%     | 47.4%    | 19     |
| 50-55%     | 54.5%    | 22     |

**IWM Overall:** 65.4% (159/243 trades)
**Best Range:** 75-80% confidence = 80.0% accuracy

---

## Feature List (29 Features)

### Time Features
| # | Feature | Description |
|---|---------|-------------|
| 1 | `time_pct` | Percentage of trading day elapsed (0-1) |
| 28 | `is_monday` | Binary: Is it Monday? |
| 29 | `is_friday` | Binary: Is it Friday? |

### Gap Features
| # | Feature | Description |
|---|---------|-------------|
| 2 | `gap` | Overnight gap: (Open - PrevClose) / PrevClose |
| 3 | `gap_size` | Absolute value of gap |
| 4 | `gap_direction` | 1 = gap up, -1 = gap down, 0 = flat |
| 15 | `gap_filled` | Binary: Has the gap been filled? |

### Previous Day Features
| # | Feature | Description |
|---|---------|-------------|
| 5 | `prev_return` | Previous day's return |
| 6 | `prev_range` | Previous day's range as % of close |
| 7 | `prev_body` | Previous day's body (Close-Open)/Open |
| 8 | `prev_bullish` | Binary: Was previous day bullish? |

### Current Session Features
| # | Feature | Description |
|---|---------|-------------|
| 9 | `current_vs_open` | Current price vs today's open |
| 10 | `current_vs_open_direction` | Direction of current vs open |
| 11 | `position_in_range` | Where price is in today's range (0-1) |
| 12 | `range_so_far_pct` | Today's range as % of open |
| 13 | `above_open` | Binary: Is price above open? |
| 14 | `near_high` | Binary: Is price nearer to high than low? |
| 16 | `morning_reversal` | Binary: Has morning reversed the gap? |
| 17 | `last_hour_return` | Return in the last hour |
| 18 | `bullish_bar_ratio` | Ratio of bullish bars today |
| 19 | `first_hour_return` | Return in first hour of trading |

### Multi-Day Features
| # | Feature | Description |
|---|---------|-------------|
| 20 | `return_3d` | 3-day return |
| 21 | `return_5d` | 5-day return |
| 22 | `volatility_5d` | 5-day volatility (std of returns) |
| 23 | `mean_reversion_signal` | Negative of previous return |
| 24 | `consecutive_up` | Count of consecutive up days |
| 25 | `consecutive_down` | Count of consecutive down days |

### Target B Specific Features
| # | Feature | Description |
|---|---------|-------------|
| 26 | `current_vs_11am` | Current price vs 11 AM price |
| 27 | `above_11am` | Binary: Is price above 11 AM? |

---

## Model Architecture

### Ensemble Components

The model uses a weighted ensemble of 4 algorithms:

1. **XGBoost** (40% weight) - Gradient boosted trees
2. **Random Forest** (25% weight) - Bagged decision trees
3. **Gradient Boosting** (20% weight) - Sequential boosting
4. **Extra Trees** (15% weight) - Extremely randomized trees

### Session Split

| Session | Time | Target |
|---------|------|--------|
| Early | 9:30 AM - 12:00 PM | Target A (Close > Open) |
| Late | 12:00 PM - 4:00 PM | Target B (Close > 11 AM) |

### Model Files

```
ml/models/
├── spy_intraday_v6.pkl
├── qqq_intraday_v6.pkl
└── iwm_intraday_v6.pkl
```

Each `.pkl` file contains:
- `scaler_early` / `scaler_late` - RobustScaler for feature normalization
- `models_early` - Early session ensemble
- `models_late_a` - Late session Target A ensemble
- `models_late_b` - Late session Target B ensemble
- `weights_*` - Ensemble weights
- `feature_cols` - Ordered list of 29 features

---

## Trading Rules

### Entry Rules

1. **Time:** Only trade Target B signals after 12:00 PM ET
2. **Confidence:** Minimum 70% confidence recommended, 75%+ preferred
3. **Direction:**
   - If `prob_b > 0.5` → LONG (price expected to close above 11 AM)
   - If `prob_b < 0.5` → SHORT (price expected to close below 11 AM)

### Position Sizing

Based on confidence level:
- 90%+ confidence: Full position (40% of capital max)
- 80-89%: 75% position
- 75-79%: 50% position
- 70-74%: 25% position
- Below 70%: No trade or minimal position

### Risk Management

- **Stop Loss:** 0.25% from entry
- **Take Profit:** 0.50% from entry (2:1 reward/risk)
- **Time Stop:** Close all positions by 3:55 PM ET

---

## Interpreting Signals

### Probability Display

The dashboard shows `prob_b` as a percentage. To interpret:

| prob_b | Meaning | Action |
|--------|---------|--------|
| 83% | 83% chance Close > 11 AM | LONG (83% confidence) |
| 17% | 17% chance Close > 11 AM | SHORT (83% confidence) |
| 55% | 55% chance Close > 11 AM | LONG (55% confidence - weak) |
| 45% | 45% chance Close > 11 AM | SHORT (55% confidence - weak) |
| 50% | Equal chance | NO TRADE |

### Confidence Calculation

```
confidence = max(prob_b, 1 - prob_b) * 100
```

Example: prob_b = 0.23 → confidence = max(0.23, 0.77) = 77%

---

## Performance Summary

| Metric | Value |
|--------|-------|
| Training Period | 2022-01-01 to 2025-01-01 |
| Test Period | 2025-01-02 to 2025-12-19 |
| Total Test Trades | 729 (243 per ticker) |
| Overall Accuracy | 66.7% |
| High Confidence (75%+) Accuracy | 78.4% |
| Best Single Bucket | 80-85% @ 79.6% |

---

## Files Reference

| File | Purpose |
|------|---------|
| `ml/train_time_split.py` | Training script |
| `ml/predict_server.py` | Flask API server |
| `ml/backtest_target_b.py` | Backtest script |
| `app/api/v2/trading-directions/route.ts` | Next.js API proxy |
| `app/dashboard/page.tsx` | Frontend dashboard |
