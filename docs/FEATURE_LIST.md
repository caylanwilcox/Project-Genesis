# V6 Model Feature List

**Total Features:** 29
**Model Version:** V6 Time-Split
**Last Updated:** December 23, 2025

---

## Feature Categories

| Category | Count | Features |
|----------|-------|----------|
| Time | 3 | time_pct, is_monday, is_friday |
| Gap | 4 | gap, gap_size, gap_direction, gap_filled |
| Previous Day | 4 | prev_return, prev_range, prev_body, prev_bullish |
| Current Session | 11 | current_vs_open, position_in_range, etc. |
| Multi-Day | 6 | return_3d, return_5d, volatility_5d, etc. |
| Target B Specific | 2 | current_vs_11am, above_11am |

---

## Complete Feature Reference

### 1. time_pct
- **Type:** Float (0.0 - 1.0)
- **Description:** Percentage of trading day elapsed
- **Calculation:** `(current_hour - 9.5) / 6.5`
- **Example:** At 12:00 PM = 0.385, At 3:00 PM = 0.846

### 2. gap
- **Type:** Float (typically -0.05 to 0.05)
- **Description:** Overnight gap as percentage
- **Calculation:** `(today_open - prev_close) / prev_close`
- **Example:** Open at $595, prev close $590 → gap = 0.0085 (0.85%)

### 3. gap_size
- **Type:** Float (0.0+)
- **Description:** Absolute size of gap
- **Calculation:** `abs(gap)`
- **Example:** gap = -0.012 → gap_size = 0.012

### 4. gap_direction
- **Type:** Integer (-1, 0, 1)
- **Description:** Direction of the gap
- **Calculation:**
  - 1 = gap up
  - -1 = gap down
  - 0 = flat (gap < 0.001)

### 5. prev_return
- **Type:** Float
- **Description:** Previous day's return
- **Calculation:** `(prev_close - prev_prev_close) / prev_prev_close`
- **Example:** Yesterday closed +1.2% → prev_return = 0.012

### 6. prev_range
- **Type:** Float
- **Description:** Previous day's trading range as % of close
- **Calculation:** `(prev_high - prev_low) / prev_close`
- **Example:** High $600, Low $595, Close $598 → 0.0084

### 7. prev_body
- **Type:** Float
- **Description:** Previous day's candle body as % of open
- **Calculation:** `(prev_close - prev_open) / prev_open`
- **Example:** Open $595, Close $598 → 0.005 (bullish)

### 8. prev_bullish
- **Type:** Binary (0, 1)
- **Description:** Was previous day bullish?
- **Calculation:** `1 if prev_close > prev_open else 0`

### 9. current_vs_open
- **Type:** Float
- **Description:** Current price vs today's open
- **Calculation:** `(current_price - today_open) / today_open`
- **Example:** Open $595, Current $597 → 0.0034

### 10. current_vs_open_direction
- **Type:** Integer (-1, 0, 1)
- **Description:** Direction of current vs open
- **Calculation:**
  - 1 = above open
  - -1 = below open
  - 0 = at open

### 11. position_in_range
- **Type:** Float (0.0 - 1.0)
- **Description:** Where price is in today's range
- **Calculation:** `(current_price - low_so_far) / (high_so_far - low_so_far)`
- **Example:** 0.0 = at low, 1.0 = at high, 0.5 = middle

### 12. range_so_far_pct
- **Type:** Float
- **Description:** Today's range as % of open
- **Calculation:** `(high_so_far - low_so_far) / today_open`
- **Example:** Range of $3 on $600 open → 0.005

### 13. above_open
- **Type:** Binary (0, 1)
- **Description:** Is current price above open?
- **Calculation:** `1 if current_price > today_open else 0`

### 14. near_high
- **Type:** Binary (0, 1)
- **Description:** Is price closer to high than low?
- **Calculation:** `1 if (high - current) < (current - low) else 0`

### 15. gap_filled
- **Type:** Binary (0, 1)
- **Description:** Has the overnight gap been filled?
- **Calculation:**
  - If gap up: `1 if low_so_far <= prev_close else 0`
  - If gap down: `1 if high_so_far >= prev_close else 0`

### 16. morning_reversal
- **Type:** Binary (0, 1)
- **Description:** Has morning reversed the gap direction?
- **Calculation:**
  - If gap up: `1 if current_price < today_open else 0`
  - If gap down: `1 if current_price > today_open else 0`

### 17. last_hour_return
- **Type:** Float
- **Description:** Return in the last hour
- **Calculation:** `(current_price - price_1hr_ago) / price_1hr_ago`

### 18. bullish_bar_ratio
- **Type:** Float (0.0 - 1.0)
- **Description:** Ratio of bullish hourly bars today
- **Calculation:** `count(close > open) / total_bars`
- **Example:** 3 bullish out of 4 bars → 0.75

### 19. first_hour_return
- **Type:** Float
- **Description:** Return in first hour of trading
- **Calculation:** `(close_10am - today_open) / today_open`

### 20. return_3d
- **Type:** Float
- **Description:** 3-day cumulative return
- **Calculation:** `(close_today - close_3d_ago) / close_3d_ago`

### 21. return_5d
- **Type:** Float
- **Description:** 5-day cumulative return
- **Calculation:** `(close_today - close_5d_ago) / close_5d_ago`

### 22. volatility_5d
- **Type:** Float
- **Description:** 5-day volatility (standard deviation of returns)
- **Calculation:** `std([daily_return_1, daily_return_2, ..., daily_return_5])`

### 23. mean_reversion_signal
- **Type:** Float
- **Description:** Contrarian signal based on previous return
- **Calculation:** `-prev_return`
- **Intuition:** Big up day → expect down, big down day → expect up

### 24. consecutive_up
- **Type:** Integer (0-5)
- **Description:** Count of consecutive up days
- **Calculation:** Count days where close > open, stop at first down day

### 25. consecutive_down
- **Type:** Integer (0-5)
- **Description:** Count of consecutive down days
- **Calculation:** Count days where close < open, stop at first up day

### 26. current_vs_11am
- **Type:** Float
- **Description:** Current price vs 11 AM price (Target B reference)
- **Calculation:** `(current_price - price_11am) / price_11am`
- **Note:** Only used in late session (12 PM+)

### 27. above_11am
- **Type:** Binary (0, 1)
- **Description:** Is current price above 11 AM price?
- **Calculation:** `1 if current_price > price_11am else 0`
- **Note:** Only used in late session (12 PM+)

### 28. is_monday
- **Type:** Binary (0, 1)
- **Description:** Is today Monday?
- **Calculation:** `1 if day_of_week == 0 else 0`
- **Intuition:** Mondays often have different patterns (weekend news)

### 29. is_friday
- **Type:** Binary (0, 1)
- **Description:** Is today Friday?
- **Calculation:** `1 if day_of_week == 4 else 0`
- **Intuition:** Fridays often have position squaring before weekend

---

## Feature Importance (Approximate)

Based on model analysis, features ranked by importance:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | current_vs_11am | High |
| 2 | position_in_range | High |
| 3 | current_vs_open | High |
| 4 | gap | Medium-High |
| 5 | bullish_bar_ratio | Medium-High |
| 6 | prev_return | Medium |
| 7 | first_hour_return | Medium |
| 8 | last_hour_return | Medium |
| 9 | gap_filled | Medium |
| 10 | morning_reversal | Medium |
| 11-29 | Others | Low-Medium |

---

## Data Sources

| Feature | Data Source | Update Frequency |
|---------|-------------|------------------|
| Price features | Polygon.io Hourly Bars | Real-time |
| Previous day | Polygon.io Daily Bars | Daily |
| Multi-day | Polygon.io Daily Bars | Daily |
| Time/Day | System clock | Real-time |

---

## Feature Scaling

All features are scaled using **RobustScaler** before model input:
- Centers on median (not mean)
- Scales by IQR (interquartile range)
- Robust to outliers

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```
