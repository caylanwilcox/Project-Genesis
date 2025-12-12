# Daily Direction Prediction - Backtest Results

## Model Purpose
Predict whether the market will close BULLISH (up) or BEARISH (down) for the trading day.

## Data Source
- **Provider**: Polygon.io API
- **Tickers**: SPY, QQQ, IWM
- **Timeframe**: Daily bars
- **Training Period**: ~400 trading days (approximately 1.5 years)
- **Data Fields**: Open, High, Low, Close, Volume

## Training Methodology

### Algorithm
Ensemble model combining three classifiers:
1. **Random Forest** (weight: 0.35)
2. **Gradient Boosting** (weight: 0.40)
3. **Logistic Regression** (weight: 0.25)

### Validation
- **Method**: Time Series Split (5 folds)
- **Test Set**: Last 60 days held out for final evaluation
- **No lookahead bias**: All features use shifted (previous day) data

### Target Variable
```
bullish = 1 if Close > Open else 0
```

## Backtest Results

### SPY (S&P 500 ETF)
| Metric | Value |
|--------|-------|
| Overall Accuracy | 67.9% |
| High Confidence Accuracy | 77.9% |
| Training Samples | 350+ |
| Features Used | 21 |

### QQQ (Nasdaq 100 ETF)
| Metric | Value |
|--------|-------|
| Overall Accuracy | 67.1% |
| High Confidence Accuracy | 81.3% |
| Training Samples | 350+ |
| Features Used | 21 |

### IWM (Russell 2000 ETF)
| Metric | Value |
|--------|-------|
| Overall Accuracy | 68.4% |
| High Confidence Accuracy | 78.0% |
| Training Samples | 350+ |
| Features Used | 21 |

## Confidence Tiers

The model outputs a probability (0-100%) which we categorize:

| Probability | Direction | Confidence | Recommended Action |
|-------------|-----------|------------|-------------------|
| >= 70% | BULLISH | HIGH | Trade bullish FVGs |
| 60-70% | BULLISH | MEDIUM | Trade with caution |
| 40-60% | NEUTRAL | LOW | Wait for clarity |
| 30-40% | BEARISH | MEDIUM | Trade with caution |
| < 30% | BEARISH | HIGH | Trade bearish FVGs |

## High Confidence Breakdown

High confidence predictions (probability >= 70% or <= 30%) showed significantly better accuracy:

- **SPY**: 77.9% win rate on high confidence signals
- **QQQ**: 81.3% win rate on high confidence signals
- **IWM**: 78.0% win rate on high confidence signals

## File Citations

### Training Script
```
ml/daily_prediction_model.py:1-250
```

### Model Loading (Server)
```
ml/predict_server.py:88-100
```

### Prediction Endpoint
```
ml/predict_server.py:835-930
```

### Saved Model Files
```
ml/models/spy_daily_model.pkl
ml/models/qqq_daily_model.pkl
ml/models/iwm_daily_model.pkl
```

## Limitations & Considerations

1. **Market Conditions**: Model trained on recent data (2023-2025). Performance may differ in different market regimes.

2. **Not Financial Advice**: This is a probability tool, not a guarantee. Always use proper risk management.

3. **Retraining**: Model should be retrained periodically (recommended: monthly) to adapt to changing market conditions.

4. **External Events**: Model cannot predict black swan events, earnings surprises, or major news.

---

## Genius Enhancement Section

### Enhancement ID: DIR-001
**Current Accuracy**: 67-68% overall, 77-81% high confidence
**Target Accuracy**: 75%+ overall, 85%+ high confidence

### Priority 1: Data Enhancements

| ID | Enhancement | Expected Impact | Difficulty | Implementation |
|----|-------------|-----------------|------------|----------------|
| DIR-D1 | **Add VIX features** | +3-5% accuracy | Easy | Add `vix_level`, `vix_change`, `vix_term_structure` from Polygon options data |
| DIR-D2 | **Pre-market data** | +2-4% accuracy | Medium | Fetch pre-market gap, volume at 9:00 AM to inform opening direction |
| DIR-D3 | **Sector rotation signals** | +2-3% accuracy | Medium | Track XLK/XLF/XLE relative strength vs SPY for rotation signals |
| DIR-D4 | **Expand training data** | +1-2% accuracy | Easy | Increase from 400 to 1000+ days, include 2020-2022 for regime diversity |

### Priority 2: Feature Engineering

| ID | Enhancement | Expected Impact | Difficulty | Implementation |
|----|-------------|-----------------|------------|----------------|
| DIR-F1 | **Overnight futures gap** | +3-4% accuracy | Medium | ES/NQ futures data shows direction before cash open |
| DIR-F2 | **Put/Call ratio** | +2-3% accuracy | Medium | Options sentiment indicator from Polygon |
| DIR-F3 | **Dollar index (DXY)** | +1-2% accuracy | Easy | Inverse correlation feature for equity direction |
| DIR-F4 | **10Y Treasury yield change** | +1-2% accuracy | Easy | Rate sensitivity for growth vs value rotation |
| DIR-F5 | **Consecutive direction counter** | +1-2% accuracy | Easy | Track streak of up/down days, mean reversion signal |

### Priority 3: Model Architecture

| ID | Enhancement | Expected Impact | Difficulty | Implementation |
|----|-------------|-----------------|------------|----------------|
| DIR-M1 | **Add XGBoost to ensemble** | +2-3% accuracy | Easy | Replace or add to Gradient Boosting with XGBoost |
| DIR-M2 | **LSTM layer for sequence** | +3-5% accuracy | Hard | Capture multi-day patterns with recurrent network |
| DIR-M3 | **Attention mechanism** | +2-4% accuracy | Hard | Weight recent days more heavily with self-attention |
| DIR-M4 | **Calibrated probabilities** | +1-2% reliability | Medium | Use Platt scaling for better probability estimates |
| DIR-M5 | **Market regime classifier** | +3-4% accuracy | Medium | Pre-classify into bull/bear/sideways, use regime-specific models |

### Priority 4: Training Process

| ID | Enhancement | Expected Impact | Difficulty | Implementation |
|----|-------------|-----------------|------------|----------------|
| DIR-T1 | **Walk-forward optimization** | +2-3% accuracy | Medium | Retrain monthly on rolling window |
| DIR-T2 | **Class balancing** | +1-2% accuracy | Easy | Use SMOTE or class weights for imbalanced days |
| DIR-T3 | **Hyperparameter tuning** | +1-2% accuracy | Medium | Bayesian optimization for ensemble weights |
| DIR-T4 | **Feature selection** | +1-2% accuracy | Medium | Use SHAP values to prune low-impact features |

### Implementation Roadmap

```
Phase 1 (Quick Wins - 1 week):
├── DIR-D1: Add VIX features
├── DIR-D4: Expand training data
├── DIR-F5: Consecutive direction counter
└── DIR-M1: Add XGBoost
Expected: +5-8% accuracy

Phase 2 (Medium Effort - 2 weeks):
├── DIR-D2: Pre-market data
├── DIR-F1: Overnight futures gap
├── DIR-F2: Put/Call ratio
└── DIR-M5: Market regime classifier
Expected: +6-10% additional accuracy

Phase 3 (Advanced - 4 weeks):
├── DIR-M2: LSTM layer
├── DIR-M3: Attention mechanism
└── DIR-T1: Walk-forward optimization
Expected: +4-6% additional accuracy
```

### Code Template: Adding VIX Feature (DIR-D1)

```python
# In daily_prediction_model.py

def fetch_vix_data(days=500):
    """Fetch VIX index data"""
    url = f"https://api.polygon.io/v2/aggs/ticker/VIX/range/1/day/{start}/{end}"
    # ... fetch and return DataFrame

def calculate_vix_features(df, vix_df):
    """Add VIX-based features"""
    df = df.merge(vix_df[['date', 'vix_close']], on='date', how='left')

    # VIX level zones
    df['vix_zone'] = pd.cut(df['vix_close'],
                            bins=[0, 15, 20, 25, 35, 100],
                            labels=['low', 'normal', 'elevated', 'high', 'extreme'])

    # VIX change
    df['vix_change'] = df['vix_close'].pct_change() * 100

    # VIX vs 20-day average
    df['vix_vs_avg'] = (df['vix_close'] - df['vix_close'].rolling(20).mean()) / df['vix_close'].rolling(20).std()

    return df
```

---

Last Verified: December 8, 2025
