# High/Low Range Model - Technical Documentation

## Model Overview

| Property | Value |
|----------|-------|
| **Purpose** | Predict daily HIGH and LOW prices as % from open |
| **Type** | Regression (Ensemble) |
| **Output** | High % and Low % from open price |
| **Features** | 72 (enhanced with VIX + inter-market data) |
| **Training Data** | 2011-2024 (~13 years with VIX proxy) |
| **Test Data** | 2025 (234 days) |

---

## What the Model Predicts

The model predicts the **expected HIGH and LOW of the day** as a percentage from the opening price.

### Example

If **SPY opens at $600.00** and the model predicts:
- **High: +0.80%** → Expected high = $604.80
- **Low: -0.50%** → Expected low = $597.00

This gives you a **predicted trading range** of $597.00 - $604.80 for the day.

### Accuracy Metric: Range Capture Rate

The key metric is: **Does the predicted range contain the end-of-day closing price?**

- If SPY closes at $602.50 and the predicted range was $597 - $604.80 → ✓ Captured
- If SPY closes at $606.00 and the predicted range was $597 - $604.80 → ✗ Missed

---

## Model Performance (2025 Test Data)

### Prediction Accuracy (MAE = Mean Absolute Error)

| Ticker | High MAE | Low MAE | Dollar Error (at $600) |
|--------|----------|---------|------------------------|
| **SPY** | **0.349%** | **0.446%** | ~$2.09 / ~$2.68 |
| QQQ | 0.456% | 0.569% | ~$2.28 / ~$2.85 |
| IWM | 0.555% | 0.554% | ~$2.78 / ~$2.77 |

### Range Capture Rate

| Ticker | Capture Rate | Avg Range Width | Buffer |
|--------|--------------|-----------------|--------|
| **SPY** | **90.6%** | 2.13% | +0.46% |
| QQQ | 85.5% | 2.51% | +0.48% |
| IWM | 81.6% | 2.75% | +0.48% |

---

## Improvement from Enhanced Features

### Before vs After (High MAE)

| Ticker | BEFORE (42 features) | AFTER (72 features) | Improvement |
|--------|---------------------|---------------------|-------------|
| **SPY** | 1.407% | **0.349%** | **-75%** |
| QQQ | 0.447% | 0.456% | ~same |
| **IWM** | 4.040% | **0.555%** | **-86%** |

### Dollar Error Example (SPY at $600)

| Metric | BEFORE | AFTER |
|--------|--------|-------|
| High Error | ~$8.44 | **~$2.09** |
| Low Error | ~$3.17 | **~$2.68** |

---

## Model Architecture

### Ensemble Design

```
┌─────────────────────────────────────────────────────────────┐
│                  HIGH/LOW RANGE PREDICTOR                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               HIGH PREDICTION ENSEMBLE               │   │
│  │  ┌─────────┐  ┌─────────┐  ┌──────────────┐        │   │
│  │  │ XGBoost │  │Gradient │  │ RandomForest │        │   │
│  │  │  40%    │  │Boosting │  │    30%       │        │   │
│  │  │         │  │  30%    │  │              │        │   │
│  │  └─────────┘  └─────────┘  └──────────────┘        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               LOW PREDICTION ENSEMBLE                │   │
│  │  ┌─────────┐  ┌─────────┐  ┌──────────────┐        │   │
│  │  │ XGBoost │  │Gradient │  │ RandomForest │        │   │
│  │  │  40%    │  │Boosting │  │    30%       │        │   │
│  │  │         │  │  30%    │  │              │        │   │
│  │  └─────────┘  └─────────┘  └──────────────┘        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│              ↓ apply buffer for safety margin ↓             │
│                                                             │
│         Final Range: [open - low%, open + high%]            │
└─────────────────────────────────────────────────────────────┘
```

### Algorithm Configuration

#### XGBoost (40% weight)
```python
XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0
)
```

#### Gradient Boosting (30% weight)
```python
GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.03,
    min_samples_leaf=10
)
```

#### Random Forest (30% weight)
```python
RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=10
)
```

---

## Feature Categories (72 total)

### 1. Gap Features (4)
- `gap_pct` - Gap from previous close
- `abs_gap` - Absolute gap size
- `gap_up` - Gap direction
- `large_gap` - Gap > 0.5%

### 2. Historical High/Low Patterns (12) - MOST IMPORTANT
- `avg_high_5d`, `avg_high_10d`, `avg_high_20d`
- `avg_low_5d`, `avg_low_10d`, `avg_low_20d`
- `max_high_5d`, `max_high_10d`
- `max_low_5d`, `max_low_10d`
- `min_high_5d`, `min_low_5d`

### 3. Volatility/ATR (10)
- `volatility_5d`, `volatility_10d`, `volatility_20d`
- `vol_ratio_5_20`, `vol_percentile`
- `atr_5`, `atr_10`, `atr_14`, `atr_20`
- `atr_ratio_5_20`, `atr_percentile`

### 4. Range Patterns (4)
- `avg_range_5d`, `avg_range_10d`, `avg_range_20d`
- `range_expanding`

### 5. VIX Features (6) - NEW
- `vix_close` - Previous VIX level
- `vix_high` - VIX > 25
- `vix_extreme` - VIX > 35
- `vix_low` - VIX < 15
- `vix_change` - VIX daily change
- `vix_trend` - VIX vs 5-day average

### 6. Inter-Market Correlations (4) - NEW
- `spy_return`, `spy_range` (for QQQ/IWM)
- `qqq_return`, `qqq_range` (for SPY/IWM)
- `iwm_return`, `iwm_range` (for SPY/QQQ)

### 7. Technical Indicators (12)
- RSI, MACD histogram
- Bollinger Band position/width/squeeze
- Price vs SMA20/50/200
- Trend strength

### 8. Calendar Effects (6)
- Day of week, is_monday, is_friday
- Month end/start, quarter end

### 9. Other (14)
- Returns, momentum, volume, consecutive patterns

---

## Top 15 Features by Importance

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `avg_low_5d` | Average low % over last 5 days |
| 2 | `avg_range_5d` | Average range over last 5 days |
| 3 | `price_vs_sma50` | Price relative to 50-day MA |
| 4 | `avg_low_10d` | Average low % over last 10 days |
| 5 | `trend_strength` | Combined MA position score |
| 6 | `atr_5` | 5-day ATR as % of price |
| 7 | `is_friday` | Friday effect |
| 8 | `price_vs_sma20` | Price relative to 20-day MA |
| 9 | `atr_10` | 10-day ATR |
| 10 | `gap_pct` | Overnight gap size |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Period | 2011 - 2024 |
| Training Samples | ~3,329 per ticker |
| Test Period | Jan 2025 - Dec 2025 |
| Test Samples | 234 days |
| Feature Scaling | StandardScaler |
| Buffer Optimization | Grid search for 85%+ capture |

---

## Model Files

```
ml/models/spy_highlow_model.pkl
ml/models/qqq_highlow_model.pkl
ml/models/iwm_highlow_model.pkl
```

### File Structure
```python
{
    'high_models': {
        'xgb': XGBRegressor,
        'gb': GradientBoostingRegressor,
        'rf': RandomForestRegressor
    },
    'low_models': {
        'xgb': XGBRegressor,
        'gb': GradientBoostingRegressor,
        'rf': RandomForestRegressor
    },
    'weights': {'xgb': 0.4, 'gb': 0.3, 'rf': 0.3},
    'scaler': StandardScaler,
    'feature_cols': [...],  # 72 features
    'buffer': float,  # Safety buffer
    'ticker': str,
    'version': 'enhanced_highlow_v1',
    'train_period': '2000-01-01 to 2024-12-31',
    'test_period': '2025-01-01 to 2025-12-08',
    'metrics': {
        'capture_rate': float,
        'high_mae': float,
        'low_mae': float,
        'avg_range': float,
        'buffer': float,
        'train_samples': int,
        'test_samples': int
    }
}
```

---

## Code References

### Training Script
```
ml/train_enhanced_highlow_model.py:1-450
```

### Feature Calculation
```
ml/train_enhanced_highlow_model.py:70-250
```

### Model Loading
```
ml/predict_server.py:102-114
```

### Prediction Function
```
ml/predict_server.py:717-755
```

---

## Usage Example

### API Response
```json
{
  "SPY": {
    "predicted_high": 605.20,
    "predicted_low": 597.80,
    "high_pct": 0.87,
    "low_pct": 0.37,
    "range_width": "1.24%"
  }
}
```

### Interpretation
- Open: $600.00
- Predicted High: $605.20 (+0.87%)
- Predicted Low: $597.80 (-0.37%)
- Use this range for stop-loss and take-profit levels

---

Last Verified: December 8, 2025
