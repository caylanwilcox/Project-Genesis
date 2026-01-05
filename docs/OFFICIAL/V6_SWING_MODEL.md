# V6.1 SWING Model - Multi-Day Direction Predictions

**Document Type:** Model Specification
**Last Updated:** 2026-01-04
**Model Version:** v6.1_swing (upgraded from v6)

---

## Overview

The V6.1 SWING model predicts **multi-day price direction** for swing trades, complementing the intraday V6 model. It provides 5-day and 10-day forecasts using daily and weekly market data.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     V6 MODEL FAMILY                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   INTRADAY V6                    SWING V6.1                         │
│   ───────────                    ─────────                          │
│   • Same-day predictions         • Multi-day predictions            │
│   • Target A: Close > Open       • 5-Day: Price up in 1 week        │
│   • Target B: Close > 11AM       • 10-Day: Price up in 2 weeks      │
│   • Hourly data                  • Daily + Weekly data              │
│   • 89-92% late session acc      • 79.2% (SPY) 5-day accuracy       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Model Performance

### V6.1 Accuracy (2025 Test Set)

| Ticker | 5-Day Accuracy | 10-Day Accuracy | Train Samples | Test Samples |
|--------|----------------|-----------------|---------------|--------------|
| **SPY** | **79.2%** | 71.7% | 5,333 | 240 |
| **QQQ** | **75.4%** | 68.3% | 3,745 | 240 |
| **IWM** | **67.5%** | 57.9% | 5,333 | 240 |

### V6 vs V6.1 Comparison

| Ticker | Metric | V6 | V6.1 | Change |
|--------|--------|-----|------|--------|
| **SPY** | 5-Day | 77.5% | **79.2%** | **+1.7%** |
| SPY | 10-Day | 70.0% | **71.7%** | **+1.7%** |
| QQQ | 5-Day | 76.2% | 75.4% | -0.8% |
| QQQ | 10-Day | 67.9% | **68.3%** | +0.4% |
| IWM | 5-Day | 67.5% | 67.5% | 0.0% |
| IWM | 10-Day | 57.1% | **57.9%** | +0.8% |

### Individual Model Performance (SPY V6.1 5-Day)

| Model | Accuracy |
|-------|----------|
| ExtraTrees | 76.2% |
| XGBoost | 75.4% |
| LightGBM | 77.9% |
| **CatBoost** | **75.0%** |
| LogisticReg | 76.2% |
| **Ensemble** | **79.2%** |

---

## Walk-Forward Backtest Results (2020-2025)

### Overall Performance by Year (SPY)

| Year | Test Samples | 5-Day Accuracy | High Conf Signals |
|------|-------------|----------------|-------------------|
| 2020 | 248 | 71.0% | 99 |
| 2021 | 247 | 71.3% | 99 |
| 2022 | 246 | 62.6% | 92 |
| 2023 | 245 | 74.3% | 86 |
| 2024 | 247 | 70.9% | 93 |
| 2025 | 245 | 77.1% | 84 |
| **AVG** | **1478** | **71.2%** | **553** |

### Accuracy by Confidence Level (2020-2025)

| Confidence | Signals | Accuracy | Recommendation |
|------------|---------|----------|----------------|
| **>85%** | 176 | **94.3%** | STRONG TRADE |
| **80-85%** | 111 | **87.4%** | STRONG TRADE |
| 75-80% | 140 | 77.1% | Moderate trade |
| 70-75% | 134 | 67.9% | Caution |
| 60-70% | 289 | 63.0% | Low edge |
| 50-60% | 228 | 57.5% | NO TRADE |
| 40-50% | 141 | 53.9% | NO TRADE |
| 30-40% | 86 | 65.1% | Low edge |
| 25-30% | 47 | 66.0% | Caution |
| 20-25% | 38 | 78.9% | Moderate trade |
| 15-20% | 29 | 89.7% | STRONG TRADE |
| **<15%** | 59 | **98.3%** | STRONG TRADE |

### High Confidence Only (>80% or <20%)

| Year | Bull >80% | Bear <20% | Combined |
|------|-----------|-----------|----------|
| 2020 | 46 @ 91% | 18 @ 89% | 64 @ 91% |
| 2021 | 55 @ 89% | 11 @ 91% | 66 @ 89% |
| 2022 | 43 @ 84% | 21 @ 95% | 64 @ 88% |
| 2023 | 53 @ 94% | 11 @ 100% | 64 @ 95% |
| 2024 | 45 @ 93% | 10 @ 100% | 55 @ 95% |
| 2025 | 45 @ 98% | 17 @ 100% | 62 @ 98% |
| **TOTAL** | **287 @ 92%** | **88 @ 95%** | **375 @ 93%** |

**Key Finding:** When filtering to high confidence signals (>80% or <20%), the model achieves **92.5% accuracy** across 375 signals over 6 years.

---

## Architecture

### V6.1 Ensemble Composition (5 models)

The V6.1 model adds **CatBoost** to the ensemble:

```python
models = {
    'et': ExtraTreesClassifier(n_estimators=300, max_depth=10),
    'xgb': XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03),
    'lgbm': LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.03),
    'catboost': CatBoostClassifier(iterations=300, depth=6, learning_rate=0.03),  # NEW
    'lr': LogisticRegression(C=0.5)
}
```

Weights are assigned based on individual model accuracy:
```
weight[model] = accuracy[model] / sum(all_accuracies)
```

### Prediction Targets

| Target | Definition | Horizon |
|--------|------------|---------|
| **5-Day** | Will close price be higher in 5 trading days? | 1 week |
| **10-Day** | Will close price be higher in 10 trading days? | 2 weeks |

---

## Feature Engineering

### Feature Count: 52 features (V6.1)

#### V6.1 New Features (8)

| Feature | Description | Category |
|---------|-------------|----------|
| **vix_relative** | Current VIX / 20-day VIX MA | Volatility Regime |
| **vix_elevated** | VIX > 120% of 20-day MA | Volatility Regime |
| **vix_low** | VIX < 80% of 20-day MA | Volatility Regime |
| **vol_regime** | 20d vol / 60d vol ratio | Volatility Regime |
| **vol_expanding** | Volatility expanding? | Volatility Regime |
| **vol_contracting** | Volatility contracting? | Volatility Regime |
| **cross_asset_momentum** | Avg return of other ETFs | Cross-Asset |
| **cross_asset_aligned** | All ETFs moving same direction? | Cross-Asset |
| **relative_strength** | Ticker return vs cross-asset avg | Cross-Asset |
| **price_vol_divergence** | Price/volume diverging? | Divergence |
| **trend_persistence** | % of up days in last 20 | Momentum |

#### Removed Low-Importance Features (V6 → V6.1)

| Feature | V6 Importance | Action |
|---------|---------------|--------|
| return_1d | 0.29% | REMOVED |
| atr_pct | 0.32% | REMOVED |
| volatility_20d | 0.33% | REMOVED |

#### Daily Features (25)

| Category | Features |
|----------|----------|
| **Returns** | return_3d, return_5d, return_10d, return_20d |
| **Moving Averages** | dist_from_sma_5/10/20/50, above_sma_5/10/20/50, sma_alignment |
| **Volatility** | volatility_10d, atr_14 |
| **Momentum** | rsi_14, higher_high, higher_low, lower_high, lower_low |
| **Volume** | volume_ratio, volume_trend |
| **Candles** | body_to_range, is_bullish, upper_wick, lower_wick, consec_up, consec_down |
| **Mean Reversion** | mean_reversion |

#### Weekly Features (7)

| Feature | Description | Importance |
|---------|-------------|------------|
| **weekly_bullish** | Last week closed higher than opened | **26.2%** |
| **weekly_above_sma_4** | Above 4-week SMA? | **10.3%** |
| weekly_return_1w | 1-week return | 4.8% |
| weekly_return_2w | 2-week return | 3.6% |
| weekly_return_4w | 4-week return | ~2% |
| weekly_dist_from_sma_4 | Distance from 4-week SMA | 4.1% |
| weekly_rsi | Weekly RSI | ~2% |

#### Time Features (5)

| Feature | Description |
|---------|-------------|
| day_of_week | 0-4 (Mon-Fri) |
| is_monday | Trade on Monday? |
| is_friday | Trade on Friday? |
| month | 1-12 |
| week_of_year | 1-52 |

---

## Key Insight: Weekly Trend Dominance

The `weekly_bullish` feature remains the **single most predictive signal** (26.2% importance):

```
Simple Rule: "If last week was bullish, next 5 days will be bullish"
  - SPY: 69.0% accuracy
  - QQQ: 68.6% accuracy
  - IWM: 60.0% accuracy
```

The ensemble adds **~10%** more accuracy through:
- Multi-timeframe momentum alignment
- **NEW: VIX/volatility regime detection**
- **NEW: Cross-asset momentum confirmation**
- Mean reversion signals at extremes

---

## Why 2022 Had Lower Accuracy

2022 was a bear market (-20% SPY). Analysis:

| Year | Return | Volatility | 5d Bullish Rate |
|------|--------|------------|-----------------|
| 2020 | +15.1% | 33.6% | 61.7% |
| 2021 | +28.8% | 13.0% | 63.5% |
| **2022** | **-19.9%** | **24.3%** | **44.6%** |
| 2023 | +24.8% | 13.2% | 62.0% |
| 2024 | +24.0% | 12.6% | 63.5% |
| 2025 | +16.6% | 19.5% | 62.0% |

**Root Cause:** Low base rate (44.6% bullish in 2022 vs 62%+ normally). Model trained on mostly bullish years, so "default bullish" bias hurt performance.

**Key:** Even in 2022, high confidence signals achieved 84-88% accuracy. The model's confidence calibration held up.

---

## Training Configuration

```python
TRAIN_START = '2000-01-01'  # 25 years of data
TRAIN_END = '2024-12-31'
TEST_START = '2025-01-01'   # Full 2025 for validation
TEST_END = '2025-12-31'

SHORT_SWING_DAYS = 5   # 1 week trading days
MEDIUM_SWING_DAYS = 10 # 2 weeks trading days
```

---

## Signal Thresholds (Updated for V6.1)

| Probability | Signal | Confidence | Historical Accuracy | Recommendation |
|-------------|--------|------------|---------------------|----------------|
| > 0.80 | BULLISH | STRONG | 92% | TRADE |
| 0.60 - 0.80 | BULLISH | MODERATE | 65-77% | Caution |
| 0.40 - 0.60 | NEUTRAL | WEAK | 56% | NO TRADE |
| 0.20 - 0.40 | BEARISH | MODERATE | 65-79% | Caution |
| < 0.20 | BEARISH | STRONG | 95% | TRADE |

**Trading Rule:** Only trade when confidence is >80% or <20%. This filters to 375 high-confidence signals over 6 years with 92.5% accuracy.

---

## API Integration

### Endpoint: `/api/v2/analysis/swing`

Returns swing predictions for all tickers:

```json
{
  "analysis_type": "SWING",
  "timeframes": ["DAILY", "WEEKLY"],
  "tickers": {
    "SPY": {
      "v6_swing": {
        "prob_5d_up": 0.82,
        "prob_10d_up": 0.75,
        "signal_5d": "BULLISH",
        "signal_10d": "BULLISH",
        "confidence": "STRONG"
      },
      "current_price": 593.50
    }
  }
}
```

### Endpoint: `/api/v2/analysis/mtf`

Multi-timeframe analysis combining intraday + swing:

```json
{
  "analysis_type": "MULTI_TIMEFRAME",
  "tickers": {
    "SPY": {
      "intraday": { ... },
      "swing": {
        "v6_swing": {
          "prob_5d_up": 0.82,
          "signal_5d": "BULLISH",
          "confidence": "STRONG"
        }
      },
      "alignment": {
        "status": "ALIGNED",
        "direction": "LONG",
        "confidence": "HIGH"
      }
    }
  }
}
```

---

## Usage Guidelines

### When to Use SWING Model

1. **Position Trades**: Holding 3-10 days
2. **Trend Confirmation**: Align intraday bias with swing direction
3. **Size Adjustment**: Increase size when confidence >80% and intraday agrees

### Confidence-Based Sizing

| Confidence | Signal Strength | Position Size |
|------------|-----------------|---------------|
| >80% or <20% | STRONG | 100% |
| 75-80% or 20-25% | MODERATE | 50% |
| 60-75% or 25-40% | WEAK | 25% |
| 40-60% | NEUTRAL | 0% (NO TRADE) |

### Alignment Check

```
Intraday: LONG  +  Swing >80%: BULLISH  →  HIGH CONFIDENCE (100% size)
Intraday: LONG  +  Swing 60%: BULLISH   →  MEDIUM CONFIDENCE (50% size)
Intraday: LONG  +  Swing <40%: BEARISH  →  CONFLICT (0% size)
Intraday: LONG  +  Swing 50%: NEUTRAL   →  LOW CONFIDENCE (25% size)
```

---

## Model Files

| File | Location |
|------|----------|
| SPY Model (V6.1) | `ml/v6_models/spy_swing_v6_1.pkl` |
| QQQ Model (V6.1) | `ml/v6_models/qqq_swing_v6_1.pkl` |
| IWM Model (V6.1) | `ml/v6_models/iwm_swing_v6_1.pkl` |
| Training Script (V6.1) | `ml/train_swing_v6_1.py` |
| Training Script (V6) | `ml/train_swing_v6.py` |
| Prediction Logic | `ml/server/v6/swing_predictions.py` |
| Model Loader | `ml/server/models/loader.py` |

---

## V6.1 Changelog

| Change | Impact |
|--------|--------|
| Added CatBoost to ensemble | +0.5% accuracy, more robust |
| Added VIX/volatility regime features | Better bear market detection |
| Added cross-asset momentum | SPY-QQQ-IWM alignment signals |
| Removed low-importance features | Reduced overfitting |
| Increased n_estimators (200→300) | More stable predictions |
| Reduced learning_rate (0.05→0.03) | Less overfitting |

---

*This document describes the V6.1 SWING model architecture. For intraday model details, see V6_MODEL_DATA_SOURCES.md*
