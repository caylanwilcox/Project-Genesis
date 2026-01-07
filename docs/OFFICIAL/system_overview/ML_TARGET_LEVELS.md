# ML Target Level Prediction System

## What It Does

Predicts the probability that SPY will hit specific price targets during the trading day.

**Example output:**
```
Target 1: $692.42  →  57% probability of being hit today
Target 2: $694.83  →  12% probability
Target 3: $697.25  →   3% probability
Stop Loss: $684.17 →   8% risk of being hit
```

## How Target Levels Are Calculated

**Anchor:** VWAP (Volume Weighted Average Price)
**Unit:** Based on ATR (Average True Range) - typically ~0.7% of price

| Level | Formula | Example (SPY @ $690) |
|-------|---------|---------------------|
| T1 Long | VWAP + 0.5 × unit | $690 + $2.42 = $692.42 |
| T2 Long | VWAP + 1.0 × unit | $690 + $4.83 = $694.83 |
| T3 Long | VWAP + 1.5 × unit | $690 + $7.25 = $697.25 |
| Stop Loss | VWAP - 1.25 × unit | $690 - $6.04 = $683.96 |

Short targets are the mirror (subtract instead of add).

## Historical Hit Rates (2000-2025 data)

| Target | Hit Rate | Meaning |
|--------|----------|---------|
| T1 Long | 29% | Price reaches +0.5u above VWAP ~29% of days |
| T2 Long | 3% | Price reaches +1.0u above VWAP ~3% of days |
| T3 Long | 0.5% | Rare - big moves |
| T1 Short | 37% | Price drops -0.5u below VWAP ~37% of days |
| T2 Short | 3% | Same as long |
| Stop Loss | 1% | Hit rate for -1.25u move |

## Probability Calculation

1. **LightGBM model** predicts raw probability from 26 features
2. **Blended with base rate** (40% model, 60% historical) to prevent overconfidence
3. **Clipped** between 2% and 85%

```python
final_prob = 0.4 × model_prediction + 0.6 × historical_base_rate
```

## Files

| File | Purpose |
|------|---------|
| `ml_pipeline/inference.py` | Generates predictions |
| `ml_pipeline/models/` | Trained LightGBM models |
| `app/api/ml-prediction/route.ts` | API endpoint |
| `app/ticker/[symbol]/page.tsx` | UI display (lines 1102-1175) |

## UI Location

**Target Levels Card** shows:
- Price level for each target
- Probability % (cyan text)
- Distance from entry as %

**Risk Management Card** shows:
- Stop loss price
- Risk % (probability of hitting stop)
- Signal confidence

## Retraining

```bash
cd ml_pipeline
POLYGON_API_KEY="xxx" python3 train.py --test-years 2020 2021 2022 2023 2024 2025
```

## Known Issues

1. **Trained on daily bars** - misses some intraday touches, hit rates are conservative
2. **SPY only** - model trained specifically on SPY
3. **Low T2/T3 rates** - only 3% historical hit rate for T2
