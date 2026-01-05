# Policy & Risk Engine - System Connections

## The Body Metaphor

The Policy Engine is the **spinal cord and peripheral nerves** - the system that translates brain decisions into precise muscle movements. The brain (RPE + V6) decides "reach for the cup"; the spinal cord translates that into specific finger positions, grip strength, and arm trajectory.

Similarly, the Policy Engine takes "LONG at 78% confidence" and translates it into "25% position, entry at $595.50, stop at $593.50, target at $597.00."

---

## Upstream Connections

### What Policy Engine Receives

| Source | Data | Usage |
|--------|------|-------|
| **V6 Model** | prob_a, prob_b, session | Action determination |
| **Phase 4** | risk_state | Size multiplier |
| **Phase 1** | current_price | Entry/exit calculation |
| **Time Governance** | current_hour | Time multiplier |
| **Config** | base_size, ticker | Sizing and targets |

### Interface Contracts

**V6 → Policy Engine**
```
Input:
  prob_a: 0.78
  prob_b: 0.82
  session: "late"

Processing:
  action_prob = prob_b  # late session uses Target B
  action = LONG  # prob_b > 0.55
  bucket = "strong"  # 0.70 <= prob_b < 0.90

Output:
  action: "BUY_CALL"
  confidence_mult: 0.75
```

**Phase 4 → Policy Engine**
```
Input:
  risk_state: NORMAL | REDUCED | DEFENSIVE

Processing:
  IF risk_state == NORMAL: risk_mult = 1.0
  IF risk_state == REDUCED: risk_mult = 0.5
  IF risk_state == DEFENSIVE: risk_mult = 0.0

Output:
  final_size *= risk_mult
```

---

## Downstream Connections

### What Policy Engine Enables

| Consumer | What It Receives | How It Uses It |
|----------|------------------|----------------|
| **Predict Server** | Full TradingDirections response | API response packaging |
| **Dashboard UI** | action, sizing, targets | Visual display |
| **Order System** | entry_price, stop_loss, take_profit | Order placement |
| **Position Manager** | position_pct | Account allocation |

### Interface Contracts

**Policy Engine → Predict Server**
```
Output:
{
    "ticker": "SPY",
    "action": "BUY_CALL",
    "direction": "LONG",
    "probability_a": 0.78,
    "probability_b": 0.82,
    "session": "late",
    "bucket": "strong",
    "position_pct": 22.5,
    "entry": {"price": 595.50},
    "exit": {"take_profit": 597.00, "stop_loss": 593.50}
}
```

**Policy Engine → Order System**
```
Output:
{
    "action": "BUY",
    "symbol": "SPY",
    "quantity": 100,  # Calculated from position_pct
    "order_type": "MARKET",
    "take_profit": 597.00,
    "stop_loss": 593.50
}
```

---

## Data Flow Position

```
┌─────────────────────────────────────────────────────────────┐
│              PHASES 1-4: RPE PIPELINE                        │
│              (Market structure, health, permission)          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 5: V6 ML PREDICTION                       │
│              (prob_a, prob_b, session)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃              POLICY & RISK ENGINE                            ┃
┃                                                              ┃
┃   prob → action (BUY_CALL / BUY_PUT / NO_TRADE)             ┃
┃   prob → bucket → confidence_mult                            ┃
┃   hour → time_mult                                           ┃
┃   prob_a, prob_b → agreement_mult                            ┃
┃   risk_state → risk_mult                                     ┃
┃   ticker → take_profit, stop_loss                            ┃
┃                                                              ┃
┃   final_size = base × conf × time × agree × risk            ┃
┗━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   PREDICT   │   │  DASHBOARD  │   │    ORDER    │
│   SERVER    │   │     UI      │   │   SYSTEM    │
└─────────────┘   └─────────────┘   └─────────────┘
```

---

## Decision Flow

### From Probability to Trade

```
                    ┌─────────────────────────┐
                    │      prob_b = 0.82      │
                    └───────────┬─────────────┘
                                │
                                ▼
              ┌─────────────────────────────────┐
              │  prob_b > 0.55?                 │
              └───────────┬─────────────────────┘
                    YES   │
                          ▼
              ┌─────────────────────────────────┐
              │  action = BUY_CALL              │
              └───────────┬─────────────────────┘
                          │
                          ▼
              ┌─────────────────────────────────┐
              │  0.70 <= 0.82 < 0.90            │
              │  bucket = "strong"              │
              │  confidence_mult = 0.75         │
              └───────────┬─────────────────────┘
                          │
                          ▼
              ┌─────────────────────────────────┐
              │  hour = 14 (2 PM)               │
              │  time_mult = 1.0 (peak)         │
              └───────────┬─────────────────────┘
                          │
                          ▼
              ┌─────────────────────────────────┐
              │  prob_a > 0.5 AND prob_b > 0.5  │
              │  agreement = aligned_bullish    │
              │  agreement_mult = 1.2           │
              └───────────┬─────────────────────┘
                          │
                          ▼
              ┌─────────────────────────────────┐
              │  risk_state = NORMAL            │
              │  risk_mult = 1.0                │
              └───────────┬─────────────────────┘
                          │
                          ▼
              ┌─────────────────────────────────┐
              │  final = 0.75 × 1.0 × 1.2 × 1.0 │
              │  position_pct = 22.5%           │
              └───────────────────────────────────┘
```

---

## Invariant Enforcement

| Invariant | Enforcement |
|-----------|-------------|
| Neutral zone exclusive | `if NEUTRAL_ZONE_LOW <= prob <= NEUTRAL_ZONE_HIGH: return NO_TRADE` |
| Sizing is multiplicative | All multipliers are floats, product used |
| Targets are ticker-specific | Lookup table by ticker symbol |
| Session determines target | `if hour < 11: use prob_a else: use prob_b` |
| DEFENSIVE = no trade | `if risk_state == DEFENSIVE: return 0` |

---

## System Health Indicators

### When Policy Engine Is Healthy
- Actions match probability direction
- Sizing is within valid range (0-100%)
- Targets are reasonable for ticker
- All multipliers are applied

### When Policy Engine Signals Distress
- Actions contradict probability (bullish action on bearish prob)
- Sizing is 0 when NORMAL risk_state
- Missing multipliers (None values)
- Targets are unreasonable (e.g., 50% stop loss)

---

*The Policy Engine is the final translator from signals to specific trading instructions.*
