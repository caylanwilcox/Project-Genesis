# System Vision

**Document Type:** North Star / Immutable Purpose
**Last Updated:** 2026-01-04
**Read Requirement:** EVERY architect loop, EVERY upgrade, EVERY change

---

## THE GOAL

**Connect high-probability machine learning tested signals to front-end code and tickers to accurately decipher the market, simplify trades, and provide a high options trading win percentage.**

This is the only reason this system exists.

---

## What This Means

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│   MARKET DATA  ──►  ML SIGNALS  ──►  FRONTEND  ──►  USER    │
│                                                              │
│   (Polygon)        (Tested,         (Simple,      (Wins)    │
│                    Governed)         Clear)                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1. High-Probability ML Signals
- Predictions must have **statistical edge** (not random)
- Every signal must be **tested** before production
- Accuracy must be **measured and tracked**
- Models must **fail closed** (NO_TRADE when uncertain)

### 2. Tested
- Every behavior has a **spec rule** (TRADING_ENGINE_SPEC.md)
- Every spec rule has a **test** (SPEC_TEST_TRACE.md)
- No untested code reaches production
- Tests catch drift before users do

### 3. Connected to Frontend and Tickers
- SPY, QQQ, IWM - the liquid options markets
- Dashboard displays signals **clearly**
- User sees: direction, confidence, entry, exit
- No ambiguity. No confusion.

### 4. Accurately Decipher the Market
- Phase 1-4: Observe market structure (deterministic truth)
- Phase 5: Predict direction (probabilistic intelligence)
- Policy: Convert signals to actions (safe execution)
- The system **understands** before it **acts**

### 5. Simplify Trades
- One signal per ticker per hour
- Clear action: BUY_CALL, BUY_PUT, or NO_TRADE
- Entry price, exit targets, stop loss
- User doesn't interpret - user executes

### 6. High Options Trading Win %
- This is the **only metric that matters**
- Late session Target B: 79-82% accuracy
- Late session Target A: 89-92% accuracy
- Every improvement must move this number **up**

---

## The Equation

```
Win Rate = f(Signal Quality × Execution Simplicity × Test Coverage)
```

- **Signal Quality**: ML models that predict better than random
- **Execution Simplicity**: User can act without interpretation
- **Test Coverage**: System behaves as specified, always

All three must be high. If any is low, win rate suffers.

---

## What Every Change Must Ask

Before ANY modification to this system:

1. **Does this improve signal quality?**
   - Better accuracy? Better calibration? Fewer false signals?

2. **Does this simplify execution?**
   - Clearer display? Faster response? Less confusion?

3. **Does this maintain test coverage?**
   - Is there a spec rule? Is there a test? Does it pass?

4. **Does this move win rate UP?**
   - If you can't answer yes, don't make the change.

---

## What This System Is NOT

| NOT This | This Instead |
|----------|--------------|
| Research platform | Production trading system |
| Backtesting sandbox | Live signal generator |
| Experimental playground | Spec-governed engine |
| Complex analytics | Simple actionable signals |
| Academic exercise | Money-making machine |

---

## The User Experience

```
User opens dashboard at 2:00 PM

┌─────────────────────────────────────────────────────────────────────┐
│  MODEL CAROUSEL                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                          │
│  │  TODAY   │  │  5-DAY   │  │  10-DAY  │  ← Timeframe Selector    │
│  │ (active) │  │          │  │          │                          │
│  └──────────┘  └──────────┘  └──────────┘                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  SPY: BUY_CALL                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  82% BULLISH                                                 │   │
│  │  Close will be ABOVE $593.50 (11 AM price)                  │   │
│  │                                                              │   │
│  │  Entry: $593.50  │  Stop: $592.00  │  Target: $595.20       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  ALIGNMENT: ✓ ALIGNED                                        │   │
│  │  Intraday: LONG  +  5-Day: BULLISH  =  HIGH CONFIDENCE      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

User sees TODAY (intraday), NEXT (5-day swing), OUTLOOK (10-day swing).
When signals ALIGN across timeframes → increased confidence.
When signals CONFLICT → reduce size or wait.

User buys call option.
Market closes above entry.
User profits.

THIS IS THE GOAL.
```

---

## Success Metrics

### Intraday Model (V6) - Same Day Predictions

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Late Session Accuracy (Target A) | 89% | 92%+ | ✅ Strong |
| Late Session Accuracy (Target B) | 79% | 85%+ | 🔄 Improving |
| Early Session Accuracy | 67% | 75%+ | 🔄 Improving |

### Swing Model (V6 SWING) - Multi-Day Predictions

| Metric | SPY | QQQ | IWM | Target | Status |
|--------|-----|-----|-----|--------|--------|
| 5-Day Accuracy | **77.5%** | 76.2% | 67.5% | 80%+ | 🔄 Improving |
| 10-Day Accuracy | 70.0% | 67.9% | 57.1% | 75%+ | 🔄 Improving |

### System Health

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Test Coverage | 100% | 100% | ✅ Complete |
| Signal Clarity | High | High | ✅ Complete |
| User Win Rate | Tracking | 75%+ | 📊 Measuring |

---

## Non-Negotiable Principles

1. **Probability before action** - ML signal required for trades; guardrails can still force NO_TRADE (market closed, missing data, neutral zone)
2. **Test before deploy** - Never ship untested code
3. **Simple before complex** - User clarity over system sophistication
4. **Fail closed** - When uncertain, NO_TRADE
5. **Measure everything** - Can't improve what you can't measure

**Note:** Vision chooses direction (what work is worth doing). Spec chooses safety (how the system behaves). When in conflict, Spec wins for runtime behavior.

---

## The Compounding Loop

```
Better Models
     │
     └──► Higher Win Rate
              │
              └──► More Confidence
                       │
                       └──► More Trades
                                │
                                └──► More Data
                                         │
                                         └──► Better Models
```

The system gets smarter. The user wins more. The cycle continues.

---

## Final Word

Every line of code, every test, every spec rule, every model, every feature serves ONE purpose:

**Help the user win options trades.**

If it doesn't serve this purpose, it doesn't belong in this system.

---

*This vision is immutable. Read it before every upgrade. Let it guide every decision.*
