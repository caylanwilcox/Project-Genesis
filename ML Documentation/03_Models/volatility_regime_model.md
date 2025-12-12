# Volatility Regime Model - Technical Documentation

## Model Overview

| Property | Value |
|----------|-------|
| **Purpose** | Detect market volatility regime and use regime-specific models |
| **Type** | Classification (Regime) + Ensemble (Direction & High/Low) |
| **Regimes** | LOW, NORMAL, HIGH |
| **Output** | Volatility score, regime classification, regime-optimized predictions |
| **Training Data** | 2003-2024 (~21 years) |
| **Test Data** | 2025 (234 days) |

---

## What the Model Does

The Volatility Regime system has two parts:

### 1. Regime Detection
Automatically classifies current market conditions into three volatility regimes:

| Regime | Percentile | Color | Characteristics |
|--------|------------|-------|-----------------|
| **LOW** | 0-30% | Green | Calm markets, tight ranges, predictable |
| **NORMAL** | 30-70% | Yellow | Average volatility, standard behavior |
| **HIGH** | 70-100% | Red | Volatile markets, wide swings, trending |

### 2. Regime-Specific Predictions
Uses models trained specifically on data from each volatility regime for more accurate predictions.

---

## Why Regime-Specific Models?

Different market conditions require different prediction approaches:

| Regime | Best For | Why |
|--------|----------|-----|
| **LOW** | High/Low Range | Prices move predictably, tight ranges are reliable |
| **HIGH** | Direction | Strong trends develop, momentum is more predictable |
| **NORMAL** | Both | Standard market behavior, balanced accuracy |

---

## Model Performance by Regime (2025 Test Data)

### LOW Volatility Regime

| Ticker | Direction Acc | High-Conf Acc | High MAE | Low MAE |
|--------|---------------|---------------|----------|---------|
| SPY | 66.7% | 73.2% | **0.205%** | **0.285%** |
| QQQ | 64.3% | 70.0% | **0.308%** | **0.418%** |
| IWM | 61.5% | 66.7% | **0.361%** | **0.402%** |

**Key Insight**: LOW volatility has the best High/Low accuracy (MAE as low as 0.205%)

### NORMAL Volatility Regime

| Ticker | Direction Acc | High-Conf Acc | High MAE | Low MAE |
|--------|---------------|---------------|----------|---------|
| SPY | 64.4% | 68.1% | 0.262% | 0.463% |
| QQQ | 58.8% | 75.0% | 0.413% | 0.649% |
| IWM | 65.3% | 71.7% | 0.506% | 0.487% |

### HIGH Volatility Regime

| Ticker | Direction Acc | High-Conf Acc | High MAE | Low MAE |
|--------|---------------|---------------|----------|---------|
| SPY | 71.7% | 75.4% | 0.561% | 1.603% |
| QQQ | 69.2% | 78.6% | 0.589% | 0.892% |
| IWM | **73.1%** | **84.3%** | 0.708% | 0.658% |

**Key Insight**: HIGH volatility has the best Direction accuracy (up to 84.3% high-confidence)

---

## Volatility Score Calculation

The volatility score combines two metrics:

### 1. ATR Percentile (50% weight)
- Calculate 14-day ATR
- Rank against last 252 trading days
- Returns percentile (0-1)

### 2. Return Volatility Percentile (50% weight)
- Calculate 20-day rolling standard deviation of returns
- Rank against last 252 trading days
- Returns percentile (0-1)

### Combined Score
```
volatility_score = (ATR_percentile + Return_volatility_percentile) / 2
```

### Regime Thresholds
```
if volatility_score < 0.30:
    regime = "LOW"
elif volatility_score > 0.70:
    regime = "HIGH"
else:
    regime = "NORMAL"
```

---

## Model Architecture

### Per-Regime Models

Each regime has its own set of trained models:

```
┌─────────────────────────────────────────────────────────────┐
│                  VOLATILITY REGIME SYSTEM                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  LOW REGIME  │  │NORMAL REGIME │  │ HIGH REGIME  │       │
│  │   Models     │  │   Models     │  │   Models     │       │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤       │
│  │ Direction:   │  │ Direction:   │  │ Direction:   │       │
│  │  - XGBoost   │  │  - XGBoost   │  │  - XGBoost   │       │
│  │  - GradBoost │  │  - GradBoost │  │  - GradBoost │       │
│  │  - RandForest│  │  - RandForest│  │  - RandForest│       │
│  │  - LogReg    │  │  - LogReg    │  │  - LogReg    │       │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤       │
│  │ High/Low:    │  │ High/Low:    │  │ High/Low:    │       │
│  │  - XGBoost   │  │  - XGBoost   │  │  - XGBoost   │       │
│  │  - GradBoost │  │  - GradBoost │  │  - GradBoost │       │
│  │  - RandForest│  │  - RandForest│  │  - RandForest│       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│                    ↓ Runtime Flow ↓                          │
│                                                              │
│  1. Calculate volatility_score from current data             │
│  2. Determine regime (LOW/NORMAL/HIGH)                       │
│  3. Select regime-specific models                            │
│  4. Generate predictions optimized for current conditions    │
└─────────────────────────────────────────────────────────────┘
```

---

## Trading Guidance by Regime

### LOW Volatility (Green)
```
Trading Approach:
- Use tighter stop-losses and take-profits
- High/Low predictions are most reliable
- Mean-reversion strategies work well
- Expected daily range: 0.2-0.5%
```

### NORMAL Volatility (Yellow)
```
Trading Approach:
- Standard risk management applies
- Both direction and range predictions reliable
- Balance between momentum and mean-reversion
- Expected daily range: 0.5-1.0%
```

### HIGH Volatility (Red)
```
Trading Approach:
- Use wider stops to avoid noise
- Direction predictions are most reliable
- Momentum/breakout strategies work well
- Expected daily range: 1.0-2.0%+
```

---

## API Endpoints

### GET /volatility_meter

Returns current volatility regime for each ticker.

**Response:**
```json
{
  "date": "2025-12-09",
  "market_volatility": "NORMAL",
  "market_volatility_score": 0.645,
  "trading_guidance": "Normal volatility environment...",
  "tickers": {
    "SPY": {
      "regime": "NORMAL",
      "regime_label": "Normal Volatility",
      "regime_color": "yellow",
      "volatility_score": 0.595,
      "volatility_percentile": 59.5,
      "current_atr_pct": 1.219,
      "current_daily_vol": 0.871,
      "expected_range": "0.5-1.0%",
      "regime_model_stats": {
        "direction_accuracy": 64.4,
        "high_conf_accuracy": 68.1,
        "high_mae": 0.262,
        "low_mae": 0.463
      }
    }
  }
}
```

### GET /regime_prediction

Returns predictions using regime-specific models.

**Response:**
```json
{
  "date": "2025-12-09",
  "tickers": {
    "IWM": {
      "regime": "HIGH",
      "volatility_score": 0.706,
      "signal": "BUY",
      "strength": "STRONG",
      "bullish_probability": 0.70,
      "predicted_high": 252.78,
      "predicted_high_pct": 0.762,
      "predicted_low": 249.39,
      "predicted_low_pct": 0.591,
      "current_price": 250.87,
      "model_accuracy": {
        "direction": 73.1,
        "high_conf": 84.3,
        "high_mae": 0.708,
        "low_mae": 0.658
      }
    }
  }
}
```

---

## Model Files

```
ml/models/spy_regime_model.pkl
ml/models/qqq_regime_model.pkl
ml/models/iwm_regime_model.pkl
```

### File Structure
```python
{
    'regime_models': {
        'LOW': {
            'direction': {
                'models': {...},
                'scaler': StandardScaler,
                'weights': {...},
                'accuracy': float,
                'high_conf_accuracy': float
            },
            'highlow': {
                'high_models': {...},
                'low_models': {...},
                'scaler': StandardScaler,
                'high_mae': float,
                'low_mae': float
            }
        },
        'NORMAL': {...},
        'HIGH': {...},
        'ALL': {...}  # Fallback model
    },
    'feature_cols': [...],
    'ticker': str,
    'version': 'regime_v1'
}
```

---

## Signal Thresholds

| Probability | Signal | Strength |
|-------------|--------|----------|
| >= 70% | BUY | STRONG |
| 65-70% | BUY | MODERATE |
| 35-65% | HOLD | NEUTRAL |
| 30-35% | SELL | MODERATE |
| <= 30% | SELL | STRONG |

---

## Code References

### Training Script
```
ml/train_volatility_regime_models.py:1-400
```

### Volatility Meter Endpoint
```
ml/predict_server.py:1468-1589
```

### Regime Prediction Endpoint
```
ml/predict_server.py:1592-1750
```

---

## Example Use Case

**Current Market (Dec 9, 2025):**

| Ticker | Vol Score | Regime | Signal | Best Strategy |
|--------|-----------|--------|--------|---------------|
| SPY | 59.5% | NORMAL | HOLD | Standard approach |
| QQQ | 63.4% | NORMAL | HOLD | Standard approach |
| IWM | 70.6% | HIGH | STRONG BUY | Momentum, wider stops |

**Interpretation:**
- IWM is in HIGH volatility regime with 70% bullish probability
- Use HIGH regime model which has 84.3% accuracy on high-confidence signals
- Expect wider daily range (1.0-2.0%+)
- Consider momentum strategies over mean-reversion

---

Last Verified: December 9, 2025
