# ML Documentation Index

## Overview
This documentation provides a complete audit trail of our machine learning system for FVG (Fair Value Gap) trading predictions. Each model is backtested on historical data with documented accuracy metrics.

## Document Structure

### 01_Backtesting_Results/
Detailed backtesting reports showing what we tested and the results achieved.
- `direction_prediction_backtest.md` - Daily direction model results
- `wide_range_backtest.md` - Full-day high/low prediction results
- `shrinking_range_backtest.md` - Time-decay range prediction results

### 02_Features/
Documentation of each ML feature, how it's calculated, and why it matters.
- `direction_features.md` - 21 features for daily direction prediction
- `highlow_features.md` - 29 features for wide range prediction
- `shrinking_features.md` - 10 features for shrinking range prediction

### 03_Models/
Technical documentation of each ML model architecture and training process.
- `daily_direction_model.md` - Ensemble (RF + GB + LR) for bullish/bearish prediction
- `wide_range_model.md` - Gradient Boosting for full-day high/low
- `shrinking_range_model.md` - Gradient Boosting for remaining range
- `volatility_regime_model.md` - Regime-specific models for LOW/NORMAL/HIGH volatility

### 04_API_Endpoints/
Documentation of server endpoints and response formats.
- `endpoints.md` - All available API endpoints
- `response_formats.md` - JSON response structures

### 05_Frontend_Integration/
How ML predictions are displayed to users.
- `morning_briefing_widget.md` - Dashboard ML component
- `chart_fvg_colors.md` - Chart FVG probability coloring

---

## Quick Reference: Model Accuracy Summary

| Model | Accuracy Metric | SPY | QQQ | IWM |
|-------|----------------|-----|-----|-----|
| Daily Direction | Win Rate | 67.1% | 67.9% | 67.1% |
| Daily Direction (High Conf) | Win Rate | 77.9% | 81.3% | 78.0% |
| Wide Range | Capture Rate | 90.6% | 85.5% | 81.6% |
| Wide Range | High MAE | 0.349% | 0.456% | 0.555% |
| Shrinking Range | Capture Rate | 91.9% | 93.3% | 93.5% |
| **Regime: LOW** | High MAE | **0.205%** | 0.308% | 0.361% |
| **Regime: HIGH** | High-Conf Acc | 75.4% | 78.6% | **84.3%** |

---

## File Paths Reference

### Training Scripts
- `ml/train_enhanced_highlow_model.py` - Wide range model training (72 features)
- `ml/train_shrinking_range_model.py` - Shrinking range model training
- `ml/train_longterm_model.py` - Direction prediction training (21+ years)
- `ml/train_volatility_regime_models.py` - Regime-specific model training

### Saved Models
- `ml/models/spy_highlow_model.pkl`
- `ml/models/qqq_highlow_model.pkl`
- `ml/models/iwm_highlow_model.pkl`
- `ml/models/spy_shrinking_model.pkl`
- `ml/models/qqq_shrinking_model.pkl`
- `ml/models/iwm_shrinking_model.pkl`
- `ml/models/spy_daily_model.pkl`
- `ml/models/qqq_daily_model.pkl`
- `ml/models/iwm_daily_model.pkl`
- `ml/models/spy_regime_model.pkl`
- `ml/models/qqq_regime_model.pkl`
- `ml/models/iwm_regime_model.pkl`

### Server
- `ml/predict_server.py` - Flask API server with all endpoints

### Frontend
- `src/components/MLMorningBriefing.tsx` - Dashboard widget

---

Last Updated: December 9, 2025
