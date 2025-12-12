# ML Server API Endpoints

## Server Information

| Property | Value |
|----------|-------|
| **Framework** | Flask |
| **Deployment** | Railway |
| **CORS** | Enabled for all origins |
| **Base URL** | `https://your-railway-url.railway.app` |

## Endpoints

### 1. Health Check

```
GET /health
```

Returns server status and loaded models count.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 12
}
```

**File Reference:** `ml/predict_server.py:300-310`

---

### 2. Morning Briefing

```
GET /morning_briefing
```

Returns comprehensive morning market analysis for all tickers (SPY, QQQ, IWM).

**Response Structure:**
```json
{
  "generated_at": "2025-12-08T09:30:00",
  "market_day": "Monday, December 08, 2025",
  "tickers": {
    "SPY": {
      "direction": "BULLISH",
      "emoji": "🟢",
      "bullish_probability": 0.72,
      "confidence": 0.44,
      "fvg_recommendation": "BULLISH",
      "current_price": 682.50,
      "today_high": 684.00,
      "today_low": 681.00,
      "today_open": 682.00,
      "predicted_range": {
        "wide": {"low": 674.50, "high": 691.00},
        "wide_capture_rate": 100.0,
        "shrinking": {"low": 680.00, "high": 687.00},
        "shrinking_capture_rate": 91.9,
        "time_remaining_pct": 65.4,
        "ml_predicted": true
      },
      "model_accuracy": 0.679
    },
    "QQQ": { ... },
    "IWM": { ... }
  },
  "overall_bias": "BULLISH",
  "overall_emoji": "🟢",
  "best_opportunity": "SPY"
}
```

**Key Fields:**
- `predicted_range.wide` - Full-day range prediction (from open)
- `predicted_range.shrinking` - Current time-adjusted range
- `time_remaining_pct` - Trading day remaining (%)
- `ml_predicted` - True if ML model used, false if ATR fallback

**File Reference:** `ml/predict_server.py:933-1100`

---

### 3. Daily Direction Prediction

```
GET /daily_prediction?ticker=SPY
```

Returns direction prediction for a single ticker.

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `ticker` | string | SPY | Stock symbol (SPY, QQQ, IWM) |

**Response:**
```json
{
  "ticker": "SPY",
  "direction": "BULLISH",
  "direction_emoji": "🟢",
  "bullish_probability": 0.72,
  "bearish_probability": 0.28,
  "confidence": 0.44,
  "confidence_tier": "MEDIUM",
  "fvg_recommendation": "BULLISH FVGs",
  "fvg_avoid": "bearish setups",
  "current_price": 682.50,
  "predicted_range": {
    "low": 674.50,
    "high": 691.00
  },
  "model_accuracy": 0.679,
  "high_conf_accuracy": 0.779,
  "model_version": "daily_v1",
  "generated_at": "2025-12-08T09:30:00"
}
```

**File Reference:** `ml/predict_server.py:835-930`

---

### 4. FVG Prediction (Batch)

```
POST /predict
```

Predict fill probability for multiple FVGs.

**Request Body:**
```json
{
  "fvgs": [
    {
      "ticker": "SPY",
      "fvg_type": "bullish",
      "gap_size_pct": 0.25,
      "atr_14": 8.5,
      "rsi_14": 55,
      "volume_ratio": 1.2,
      "market_structure": "bullish",
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "fill_probability": 0.78,
      "recommendation": "TRADE",
      "confidence_level": "HIGH",
      "model_used": "SPY"
    }
  ]
}
```

**File Reference:** `ml/predict_server.py:400-580`

---

### 5. Single FVG Prediction

```
POST /predict_single
```

Predict fill probability for a single FVG.

**Request Body:**
```json
{
  "ticker": "SPY",
  "fvg_type": "bullish",
  "gap_size_pct": 0.25,
  ...
}
```

**Response:**
```json
{
  "fill_probability": 0.78,
  "recommendation": "TRADE",
  "confidence": "HIGH",
  "model_info": {
    "ticker": "SPY",
    "version": "improved_v2",
    "accuracy": 0.72
  }
}
```

**File Reference:** `ml/predict_server.py:320-400`

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "error": "Error message description"
}
```

Common error codes:
- `404` - Model not found for ticker
- `500` - Internal server error (data fetch failed, etc.)

---

### 6. Volatility Meter

```
GET /volatility_meter
```

Returns current volatility regime classification for all tickers.

**Response:**
```json
{
  "date": "2025-12-09",
  "market_volatility": "NORMAL",
  "market_volatility_score": 0.645,
  "trading_guidance": "Normal volatility environment. Standard risk management applies.",
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
    },
    "QQQ": { ... },
    "IWM": { ... }
  }
}
```

**Key Fields:**
- `volatility_score` - Percentile (0-1) combining ATR and return volatility
- `regime` - LOW (green), NORMAL (yellow), or HIGH (red)
- `regime_model_stats` - Accuracy of regime-specific models
- `trading_guidance` - Recommended trading approach

**File Reference:** `ml/predict_server.py:1468-1589`

---

### 7. Regime-Based Prediction

```
GET /regime_prediction
```

Returns predictions using volatility regime-specific models for optimal accuracy.

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
    },
    "QQQ": { ... },
    "SPY": { ... }
  }
}
```

**Key Fields:**
- `regime` - Current volatility regime used for prediction
- `signal` - BUY/SELL/HOLD based on bullish probability
- `strength` - STRONG/MODERATE/NEUTRAL
- `model_accuracy` - Performance metrics for the regime-specific model used

**Signal Thresholds:**
| Probability | Signal | Strength |
|-------------|--------|----------|
| >= 70% | BUY | STRONG |
| 65-70% | BUY | MODERATE |
| 35-65% | HOLD | NEUTRAL |
| 30-35% | SELL | MODERATE |
| <= 30% | SELL | STRONG |

**File Reference:** `ml/predict_server.py:1592-1750`

---

## Rate Limiting

No rate limiting is currently implemented. Consider adding if public-facing.

## Authentication

No authentication required. For production, consider adding API keys.

---

Last Verified: December 9, 2025
