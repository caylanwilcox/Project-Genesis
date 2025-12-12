# Daily Direction Model - Technical Documentation

## Model Overview

| Property | Value |
|----------|-------|
| **Purpose** | Predict bullish/bearish daily direction |
| **Type** | Binary Classification (Ensemble) |
| **Output** | Probability (0-100%) of bullish close |
| **Features** | 57 |
| **Training Data** | 21+ years (Sept 2003 - Dec 2024) |
| **Test Data** | 2025 (Jan - Dec) |

---

## What the Model Predicts

The model predicts **whether the price will close HIGHER or LOWER than it opened today**.

| Probability | Meaning |
|-------------|---------|
| **70%** | 70% chance price closes **HIGHER** than it opened |
| **30%** | 30% chance higher = 70% chance price closes **LOWER** |

### What It Does NOT Predict
- How much price will go up/down (that's the high/low model)
- Exact price targets
- Intraday movements

---

## BUY/SELL Signal Thresholds

| Bullish Probability | Signal | Strength | Action |
|---------------------|--------|----------|--------|
| ≥70% | **STRONG BUY** | STRONG | High confidence price goes UP |
| 60-70% | **BUY** | MODERATE | Moderate confidence price goes UP |
| 55-60% | **BUY** | WEAK | Low confidence price goes UP |
| 45-55% | **HOLD** | NEUTRAL | No clear direction - stay flat |
| 40-45% | **SELL** | WEAK | Low confidence price goes DOWN |
| 30-40% | **SELL** | MODERATE | Moderate confidence price goes DOWN |
| ≤30% | **STRONG SELL** | STRONG | High confidence price goes DOWN |

### Example

If SPY opens at $600 and the model outputs **72% bullish**:
- **Signal:** STRONG BUY
- **Meaning:** 72% chance SPY closes above $600 today
- **Action:** Consider going long

---

## Model Performance (2025 Backtest)

### Signal Accuracy

| Signal | Win Rate | Trades | Total P/L |
|--------|----------|--------|-----------|
| **BUY** | **87.5%** | 32 | +27.03% |
| SELL | 48.9% | 45 | +2.16% |
| **STRONG SELL** | **70.7%** | 41 | +24.05% |

### Per-Ticker Results

| Ticker | Win Rate | P/L | Profit Factor | BUY Accuracy |
|--------|----------|-----|---------------|--------------|
| SPY | 63.9% | +10.42% | 2.62 | **90.0%** |
| QQQ | 71.4% | +20.44% | 3.56 | **86.7%** |
| IWM | 65.0% | +22.38% | 3.19 | **85.7%** |

### High-Confidence Accuracy (2025 Test)

| Confidence | SPY | QQQ | IWM |
|------------|-----|-----|-----|
| ≥70% | 79.6% | 78.5% | 82.1% |
| ≥75% | 80.9% | 81.2% | **87.3%** |

---

## Model Architecture

### Ensemble Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ENSEMBLE PREDICTOR                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐ │
│  │  XGBoost  │  │  Random   │  │ Gradient  │  │ Logistic │ │
│  │           │  │  Forest   │  │ Boosting  │  │Regression│ │
│  │ ~25%      │  │  ~25%     │  │  ~25%     │  │  ~25%    │ │
│  └───────────┘  └───────────┘  └───────────┘  └──────────┘ │
│                                                             │
│              ↓ weighted average probability ↓               │
│                                                             │
│                   Final Prediction                          │
└─────────────────────────────────────────────────────────────┘
```

### Component Details

#### XGBoost
```python
XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0
)
```

#### Random Forest
```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=10
)
```

#### Gradient Boosting
```python
GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.03,
    min_samples_leaf=10
)
```

#### Logistic Regression
```python
LogisticRegression(
    max_iter=2000,
    C=0.5
)
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Period | Sept 2003 - Dec 2024 |
| Training Samples | ~5,300 per ticker |
| Test Period | Jan 2025 - Dec 2025 |
| Test Samples | 234 days |
| Feature Scaling | StandardScaler |
| Features | 57 technical indicators |

---

## Top Features (by importance)

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `gap` | Overnight gap from previous close |
| 2 | `rsi_overbought` | RSI > 70 indicator |
| 3 | `price_vs_ema21` | Price relative to 21-day EMA |
| 4 | `volatility_20d` | 20-day volatility |
| 5 | `ema9_vs_ema21` | EMA crossover signal |
| 6 | `roc_10` | 10-day rate of change |
| 7 | `prev_atr_pct` | ATR as % of price |
| 8 | `price_vs_sma50` | Price relative to 50-day SMA |
| 9 | `prev_volume_ratio` | Volume vs 20-day average |
| 10 | `streak` | Consecutive up/down days |

---

## Model Files

```
ml/models/spy_daily_model.pkl
ml/models/qqq_daily_model.pkl
ml/models/iwm_daily_model.pkl
```

### File Structure
```python
{
    'models': {
        'xgb': XGBClassifier,
        'rf': RandomForestClassifier,
        'gb': GradientBoostingClassifier,
        'lr': LogisticRegression
    },
    'weights': {'xgb': 0.25, 'rf': 0.25, 'gb': 0.25, 'lr': 0.25},
    'scaler': StandardScaler,
    'feature_cols': [...],  # 57 features
    'ticker': str,
    'trained_at': str,
    'version': 'longterm_v1',
    'train_period': '2000-01-01 to 2024-12-31',
    'test_period': '2025-01-01 to 2025-12-08',
    'metrics': {
        'accuracy': float,
        'train_samples': int,
        'test_samples': int
    }
}
```

---

## Code References

### Training Script
```
ml/train_longterm_model.py:1-450
```

### Model Loading
```
ml/predict_server.py:88-100
```

### Prediction Endpoint
```
ml/predict_server.py:933-1152  (/daily_signals)
```

### Feature Calculation
```
ml/predict_server.py:563-728
```

---

## API Usage

### Endpoint
```
GET /daily_signals
```

### Response
```json
{
  "SPY": {
    "signal": "STRONG_BUY",
    "strength": "STRONG",
    "probability": 0.72,
    "entry": 600.50,
    "target": 605.20,
    "stop_loss": 597.80,
    "risk_reward": "1:1.74"
  },
  "market_summary": "BULLISH",
  "best_trade": {
    "ticker": "SPY",
    "signal": "STRONG_BUY"
  }
}
```

---

Last Verified: December 8, 2025
