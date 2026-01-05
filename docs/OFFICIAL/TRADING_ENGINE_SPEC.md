# Trading Engine Specification - Official Rules

## Related Files

### RPE (Reality Proof Engine) - 5-Phase Pipeline

| File | Purpose |
|------|---------|
| [ml/rpe/rpe_engine.py](../../ml/rpe/rpe_engine.py) | Core RPE pipeline implementation |
| [ml/rpe/acceptance.py](../../ml/rpe/acceptance.py) | Phase 1: Acceptance logic |
| [ml/rpe/auction_state.py](../../ml/rpe/auction_state.py) | Phase 1: Auction state detection |
| [ml/rpe/levels.py](../../ml/rpe/levels.py) | Phase 1: Key price levels |
| [ml/rpe/vwap.py](../../ml/rpe/vwap.py) | Phase 1: VWAP calculations |
| [ml/rpe/failures.py](../../ml/rpe/failures.py) | Phase 2: Signal health / failures |
| [ml/rpe/beware.py](../../ml/rpe/beware.py) | Phase 3: Risk/beware conditions |
| [ml/rpe/compute.py](../../ml/rpe/compute.py) | Phase 4: Execution computations |
| [ml/rpe/northstar_pipeline.py](../../ml/rpe/northstar_pipeline.py) | Northstar integration |
| [app/api/v2/rpe/route.ts](../../app/api/v2/rpe/route.ts) | RPE API endpoint (Next.js) |

### V6 Time-Split Model (Separate ML Engine)

| File | Purpose |
|------|---------|
| [ml/v6_models/spy_intraday_v6.pkl](../../ml/v6_models/spy_intraday_v6.pkl) | V6 model for SPY |
| [ml/v6_models/qqq_intraday_v6.pkl](../../ml/v6_models/qqq_intraday_v6.pkl) | V6 model for QQQ |
| [ml/v6_models/iwm_intraday_v6.pkl](../../ml/v6_models/iwm_intraday_v6.pkl) | V6 model for IWM |
| [ml/train_time_split.py](../../ml/train_time_split.py) | V6 model training script |
| [ml/backtest_v6.py](../../ml/backtest_v6.py) | V6 backtesting script |
| [ml/predict_server.py](../../ml/predict_server.py) | Prediction server (serves both RPE + V6) |

**Note:** The V6 model is a separate ML engine from the RPE. The RPE handles market structure analysis (Phases 1-4), while V6 provides ML-based probability predictions. They work together but are independent systems.

## Engine Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RPE (Reality Proof Engine)                    │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: TRUTH         │ Market structure observation          │
│  Phase 2: SIGNAL_HEALTH │ Data integrity check                  │
│  Phase 3: SIGNAL_DENSITY│ Cooldown, budget tracking             │
│  Phase 4: EXECUTION     │ Bias, play type, confidence           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  V6 Time-Split Model (Separate)                  │
├─────────────────────────────────────────────────────────────────┤
│  Target A: Close > Open     (9:30 AM regular market open)       │
│  Target B: Close > 11 AM    (11 AM hourly bar close)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRADING ACTION OUTPUT                         │
├─────────────────────────────────────────────────────────────────┤
│  BUY CALL  │  BUY PUT  │  NO TRADE  │  CLOSE POSITION           │
└─────────────────────────────────────────────────────────────────┘
```

---

## DO / DON'T Rules

### Data Sources

| Rule | DO | DON'T |
|------|-----|-------|
| Open Price | Use `daily_bars[-1]['o']` (9:30 AM) | Use `hourly_bars[0]['o']` (4 AM pre-market) |
| Today's Data | Fetch with `limit=50000` | Use default `limit=50` |
| Time Zone | Always use `America/New_York` | Use UTC or local time |
| Features | Use all 29 V6 features | Skip any features |
| Feature Names | Use `prev_return` | Use `prev_day_return` |

### Session Rules

| Rule | DO | DON'T |
|------|-----|-------|
| Early Session (before 11 AM) | Use Target A probability | Use Target B probability |
| Late Session (11 AM - 4 PM) | Use Target B probability | Use Target A probability |
| Pre-market (before 9:30 AM) | Output NO_TRADE | Generate any trade signal |
| After-hours (after 4 PM) | Output NO_TRADE | Generate any trade signal |

### Signal Generation

| Rule | DO | DON'T |
|------|-----|-------|
| Neutral Zone (25-75%) | Output NO_TRADE | Force a direction |
| High Confidence (>75%) | Allow full position size | Cap position arbitrarily |
| Low Confidence (<75%) | Reduce position size | Trade at full size |
| Market Closed | Show "Market Closed" message | Show stale predictions as live |

---

## IF/THEN Signal Rules

### Primary Trading Logic

```
IF market_closed:
    THEN action = "NO_TRADE"
    THEN reason = "Market is closed"

IF current_hour < 11:
    THEN session = "early"
    THEN action_prob = probability_a (Target A: Close > Open)
ELSE:
    THEN session = "late"
    THEN action_prob = probability_b (Target B: Close > 11 AM)

IF action_prob >= 0.25 AND action_prob <= 0.75:
    THEN action = "NO_TRADE"
    THEN reason = "Neutral probability zone (model accuracy unreliable)"

IF action_prob > 0.75:
    THEN action = "BUY_CALL" (bullish)
    THEN direction = "LONG"

IF action_prob < 0.25:
    THEN action = "BUY_PUT" (bearish)
    THEN direction = "SHORT"
```

### Confidence-Based Sizing

```
IF action_prob >= 0.90 OR action_prob <= 0.10:
    THEN bucket = "very_strong"
    THEN size_mult = 1.0 (100% of base)

IF action_prob >= 0.85 OR action_prob <= 0.15:
    THEN bucket = "strong"
    THEN size_mult = 0.75 (75% of base)

IF action_prob >= 0.80 OR action_prob <= 0.20:
    THEN bucket = "moderate"
    THEN size_mult = 0.50 (50% of base)

IF action_prob >= 0.75 OR action_prob <= 0.25:
    THEN bucket = "weak"
    THEN size_mult = 0.25 (25% of base)
```

### Time-Based Multipliers

```
IF current_hour == 13 OR current_hour == 14 OR current_hour == 15:
    THEN time_mult = 1.0 (peak accuracy hours)

IF current_hour == 11 OR current_hour == 12:
    THEN time_mult = 0.8 (good accuracy)

IF current_hour == 10:
    THEN time_mult = 0.6 (moderate accuracy)

IF current_hour < 10:
    THEN time_mult = 0.4 (early session, lower confidence)
```

### Signal Agreement Multiplier

```
IF probability_a > 0.5 AND probability_b > 0.5:
    THEN agreement = "aligned_bullish"
    THEN agreement_mult = 1.2

IF probability_a < 0.5 AND probability_b < 0.5:
    THEN agreement = "aligned_bearish"
    THEN agreement_mult = 1.2

IF probability_a > 0.5 AND probability_b < 0.5:
    THEN agreement = "conflicting"
    THEN agreement_mult = 0.6

IF probability_a < 0.5 AND probability_b > 0.5:
    THEN agreement = "conflicting"
    THEN agreement_mult = 0.6
```

---

## Trading Action Outputs

### Clear Action Names

| Internal Action | User-Facing Output | Meaning |
|----------------|-------------------|---------|
| `LONG` | **BUY CALL** | Bullish - price expected to rise |
| `SHORT` | **BUY PUT** | Bearish - price expected to fall |
| `NO_TRADE` | **NO TRADE** | Stay out - no edge |
| `CLOSE_LONG` | **SELL CALL** | Exit bullish position |
| `CLOSE_SHORT` | **SELL PUT** | Exit bearish position |

### Complete Output Structure

```json
{
  "ticker": "SPY",
  "action": "BUY_CALL",
  "direction": "LONG",
  "confidence": 78,
  "probability_a": 0.72,
  "probability_b": 0.81,
  "session": "late",
  "bucket": "strong_bull",
  "entry": {
    "price": 595.50,
    "type": "MARKET"
  },
  "exit": {
    "take_profit": 597.00,
    "stop_loss": 593.50,
    "time_limit": "EOD"
  },
  "position_pct": 25.5,
  "reason": "Strong Bull - Target B (vs 11AM) at 81%"
}
```

---

## IF/THEN Entry/Exit Rules

### Entry Rules

```
IF action == "BUY_CALL" AND bucket in ["very_strong", "strong"]:
    THEN entry_type = "MARKET"
    THEN entry_price = current_price

IF action == "BUY_CALL" AND bucket in ["moderate", "weak"]:
    THEN entry_type = "LIMIT"
    THEN entry_price = current_price - (ATR * 0.25)

IF action == "BUY_PUT" AND bucket in ["very_strong", "strong"]:
    THEN entry_type = "MARKET"
    THEN entry_price = current_price

IF action == "BUY_PUT" AND bucket in ["moderate", "weak"]:
    THEN entry_type = "LIMIT"
    THEN entry_price = current_price + (ATR * 0.25)
```

### Exit Rules (Take Profit / Stop Loss)

```
# SPY Targets (based on 2025 median moves)
IF ticker == "SPY":
    THEN take_profit_pct = 0.25%
    THEN stop_loss_pct = 0.33%

# QQQ Targets (higher volatility)
IF ticker == "QQQ":
    THEN take_profit_pct = 0.34%
    THEN stop_loss_pct = 0.45%

# IWM Targets (highest volatility)
IF ticker == "IWM":
    THEN take_profit_pct = 0.45%
    THEN stop_loss_pct = 0.60%

# Apply to position
IF action == "BUY_CALL":
    THEN take_profit = entry_price * (1 + take_profit_pct)
    THEN stop_loss = entry_price * (1 - stop_loss_pct)

IF action == "BUY_PUT":
    THEN take_profit = entry_price * (1 - take_profit_pct)
    THEN stop_loss = entry_price * (1 + stop_loss_pct)
```

### Time Exit Rules

```
IF current_hour >= 15 AND current_minute >= 45:
    THEN action = "CLOSE_POSITION"
    THEN reason = "End of day - close before market close"

IF holding_time > expected_hold_bars:
    THEN review_position = TRUE
    THEN consider_exit = "Time limit exceeded"
```

---

## Dashboard Display Rules

### Color Coding

```
IF action == "BUY_CALL":
    THEN color = GREEN
    THEN icon = "↑" or "📈"

IF action == "BUY_PUT":
    THEN color = RED
    THEN icon = "↓" or "📉"

IF action == "NO_TRADE":
    THEN color = GRAY
    THEN icon = "—" or "⏸"

IF confidence >= 70:
    THEN intensity = BRIGHT
IF confidence >= 50:
    THEN intensity = NORMAL
IF confidence < 50:
    THEN intensity = FADED
```

### Status Messages

```
IF action == "BUY_CALL" AND confidence >= 70:
    THEN message = "STRONG BUY CALL - High confidence bullish"

IF action == "BUY_CALL" AND confidence >= 55:
    THEN message = "BUY CALL - Moderate bullish signal"

IF action == "BUY_PUT" AND confidence >= 70:
    THEN message = "STRONG BUY PUT - High confidence bearish"

IF action == "BUY_PUT" AND confidence >= 55:
    THEN message = "BUY PUT - Moderate bearish signal"

IF action == "NO_TRADE":
    THEN message = "NO TRADE - Wait for clearer signal"
```

---

## Model Accuracy Reference

| Session | Target | Accuracy | Best Hours |
|---------|--------|----------|------------|
| Late (11AM-4PM) | Target A | 89-92% | 1-3 PM |
| Late (11AM-4PM) | Target B | 79-82% | 1-3 PM |
| Early (9:30-11AM) | Target A | 65-70% | 10-11 AM |

---

## Validation Checklist

Before generating any signal, verify:

- [ ] Market is open (9:30 AM - 4:00 PM ET)
- [ ] Using daily_bars[-1]['o'] for open price (NOT hourly_bars[0]['o'])
- [ ] All 29 features computed correctly
- [ ] Session determined correctly (early vs late)
- [ ] Correct probability used (A for early, B for late)
- [ ] Neutral zone check applied (45-55% = NO_TRADE)
- [ ] Position size calculated with all multipliers
- [ ] Stop loss and take profit set correctly

---

*Last Updated: January 3, 2026*
