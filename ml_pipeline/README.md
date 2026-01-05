# SPY Daily Range Plan Prediction System

ML-driven intraday trading system that generates calibrated probability estimates for SPY price targets.

## Overview

This system uses 20 years of historical SPY intraday data to train LightGBM models that predict:
- **Touch probabilities** for 8 price levels (T1/T2/T3 long/short + stop losses)
- **First touch prediction** (which level gets hit first)
- **Calibrated probabilities** using isotonic regression

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Daily Range Plan System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Data    │───▶│ Feature  │───▶│  Label   │───▶│  Train   │  │
│  │  Loader  │    │ Engineer │    │Generator │    │ Models   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │                                               │          │
│       ▼                                               ▼          │
│  ┌──────────┐                                   ┌──────────┐    │
│  │ Session  │                                   │Calibrate │    │
│  │ Builder  │                                   │  Probs   │    │
│  └──────────┘                                   └──────────┘    │
│                                                       │          │
│                                                       ▼          │
│                                                 ┌──────────┐    │
│                                                 │ Generate │    │
│                                                 │   Plan   │    │
│                                                 └──────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Price Level System

### Anchor: VWAP
The session VWAP (calculated after Opening Range) serves as the anchor point.

### Unit Calculation
```
unit = max(OR_15_range, intraday_ATR, 0.5 × daily_ATR_14)
```

### Target Levels
| Level | Long | Short |
|-------|------|-------|
| T1 | VWAP + 0.5u | VWAP - 0.5u |
| T2 | VWAP + 1.0u | VWAP - 1.0u |
| T3 | VWAP + 1.5u | VWAP - 1.5u |
| SL | VWAP - 1.25u | VWAP + 1.25u |

## Installation

```bash
cd ml_pipeline
pip install -r requirements.txt
```

## Usage

### 1. Build Dataset

```bash
# Set your Polygon API key
export POLYGON_API_KEY=your_key_here

# Build full dataset (takes ~30 min first time)
python build_dataset.py --start-year 2004 --end-year 2024
```

### 2. Train Models (Walk-Forward)

```bash
# Train with walk-forward validation
python train_walkforward.py --test-years 2015-2024

# This trains:
# - 8 binary models (touch_t1_long, touch_t2_long, etc.)
# - 1 multiclass model (first_touch)
# For each test year, trains on all prior data
```

### 3. Calibrate Probabilities

```bash
# Fit isotonic regression calibrators on OOS predictions
python calibrate_models.py
```

### 4. Evaluate Performance

```bash
# Generate evaluation report
python evaluate.py
```

### 5. Generate Today's Plan

```bash
# Get today's daily range plan
python infer_today.py
```

## Features (28 total)

### Price Action (5)
- `open_to_vwap_pct`: Open price vs VWAP
- `or_high_to_vwap_pct`: Opening Range high vs VWAP
- `or_low_to_vwap_pct`: Opening Range low vs VWAP
- `or_range_pct`: Opening Range as % of price
- `prev_close_to_open_gap_pct`: Overnight gap

### Volatility (6)
- `atr_14`: 14-day ATR
- `atr_5`: 5-day ATR
- `atr_ratio_5_14`: Short vs long-term ATR
- `or_atr_ratio`: OR range vs ATR
- `prev_day_range_pct`: Previous day's range
- `prev_day_body_pct`: Previous day's body

### Momentum (5)
- `rsi_14`: Relative Strength Index
- `price_vs_sma_20`: Price vs 20-day SMA
- `price_vs_sma_50`: Price vs 50-day SMA
- `macd_hist`: MACD histogram
- `adx_14`: Average Directional Index

### Volume (2)
- `volume_ratio_vs_20d_avg`: Volume vs 20-day average
- `or_volume_ratio`: OR volume vs average

### Time (5)
- `day_of_week`: 0-4 (Mon-Fri)
- `month`: 1-12
- `days_since_month_start`: Day of month
- `is_opex_week`: Options expiration week
- `is_fomc_day`: Federal Reserve meeting day

### Market Regime (4)
- `vix_level`: VIX index level
- `vix_change_1d`: VIX daily change
- `spy_20d_return`: 20-day SPY return
- `spy_5d_return`: 5-day SPY return

## Output Format

```
═══════════════════════════════════════════════════════════════
  SPY DAILY RANGE PLAN - 2024-01-15
═══════════════════════════════════════════════════════════════

  ANCHOR: VWAP @ $475.50
  UNIT: $3.20 (OR-driven)

  ─────────────────────────────────────────────────────────────
  LONG SCENARIO (Buy at VWAP)
  ─────────────────────────────────────────────────────────────
    T1 (0.5u): $477.10  →  65% probability
    T2 (1.0u): $478.70  →  45% probability
    T3 (1.5u): $480.30  →  25% probability
    SL (1.25u): $471.50  →  20% probability

  ─────────────────────────────────────────────────────────────
  SHORT SCENARIO (Sell at VWAP)
  ─────────────────────────────────────────────────────────────
    T1 (0.5u): $473.90  →  55% probability
    T2 (1.0u): $472.30  →  35% probability
    T3 (1.5u): $470.70  →  15% probability
    SL (1.25u): $479.50  →  25% probability

  ─────────────────────────────────────────────────────────────
  FIRST TOUCH PROBABILITIES
  ─────────────────────────────────────────────────────────────
    t1_long: 28.5%
    t1_short: 22.3%
    none: 18.2%
    t2_long: 12.1%
    sl_long: 8.4%

  ─────────────────────────────────────────────────────────────
  RECOMMENDATION: LONG (Confidence: 70%)
  ─────────────────────────────────────────────────────────────
  Long EV (0.18) exceeds short EV (0.05). T1 long touch
  probability: 65%. Low stop-loss risk.
═══════════════════════════════════════════════════════════════
```

## Model Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| AUC-ROC | >0.65 | Ranking quality |
| Brier Score | <0.22 | Probability accuracy |
| ECE | <0.08 | Calibration error |
| Stability | >0.75 | Consistency across years |

## Files

| File | Description |
|------|-------------|
| `config.py` | Configuration and constants |
| `data_loader.py` | Polygon API data fetching and sessionization |
| `features.py` | Feature engineering |
| `labels.py` | Label generation (touch events, MFE/MAE) |
| `train.py` | Walk-forward model training |
| `calibrate.py` | Isotonic regression calibration |
| `inference.py` | Daily plan generation |
| `evaluate.py` | Model evaluation and metrics |
| `synthetic_data.py` | Synthetic data for testing |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## API Integration

The system integrates with your trading app via the `inference.py` module:

```python
from ml_pipeline.inference import PlanGenerator

generator = PlanGenerator(model_dir="ml_pipeline/models")
plan = generator.generate_plan(
    features=current_features,
    vwap=current_vwap,
    unit=calculated_unit,
    date=today,
    symbol="SPY"
)

# Use in your trading UI
print(plan.summary())
print(plan.to_json())
```

## License

Internal use only.
