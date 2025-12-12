# API Response Formats

## Direction Probabilities

All direction predictions use a probability scale:

| Probability | Direction | Confidence Tier |
|------------|-----------|-----------------|
| >= 70% | BULLISH | HIGH |
| 60-70% | BULLISH | MEDIUM |
| 55-60% | BULLISH | LOW |
| 45-55% | NEUTRAL | - |
| 40-45% | BEARISH | LOW |
| 30-40% | BEARISH | MEDIUM |
| < 30% | BEARISH | HIGH |

## FVG Recommendations

Based on direction probability:

| Probability | FVG Recommendation |
|------------|-------------------|
| >= 55% | BULLISH FVGs |
| 45-55% | EITHER (low conviction) |
| <= 45% | BEARISH FVGs |

## Range Prediction Format

### Wide Range (Full Day)
```json
{
  "wide": {
    "low": 674.50,   // Open - predicted_low_pct
    "high": 691.00   // Open + predicted_high_pct
  },
  "wide_capture_rate": 100.0  // % of test days range contained close
}
```

### Shrinking Range (Time-Adjusted)
```json
{
  "shrinking": {
    "low": 680.00,   // Current - predicted_remaining_down
    "high": 687.00   // Current + predicted_remaining_up
  },
  "shrinking_capture_rate": 91.9,  // % of time slices range contained close
  "time_remaining_pct": 65.4       // % of trading day remaining
}
```

## Confidence Calculations

```python
# Direction confidence (0-1 scale)
confidence = abs(bullish_prob - 0.5) * 2

# Confidence tiers
if confidence >= 0.5:
    tier = "HIGH"
elif confidence >= 0.25:
    tier = "MEDIUM"
else:
    tier = "LOW"
```

## Emoji Mapping

| Direction | Emoji |
|-----------|-------|
| BULLISH | 🟢 |
| NEUTRAL | 🟡 |
| BEARISH | 🔴 |

## Time Remaining Calculation

```python
# Market hours: 9:30 AM - 4:00 PM (6.5 hours)
now = datetime.now(eastern_tz)
market_open = now.replace(hour=9, minute=30)
market_close = now.replace(hour=16, minute=0)

# Minutes elapsed and remaining
minutes_elapsed = (now - market_open).total_seconds() / 60
total_minutes = 6.5 * 60  # 390 minutes

time_remaining_pct = max(0, (1 - minutes_elapsed / total_minutes)) * 100
```

**File Reference:** `ml/predict_server.py:758-790`

---

Last Verified: December 8, 2025
