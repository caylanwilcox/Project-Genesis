# Market Structure Layer (Phase 1) - System Connections

## The Body Metaphor

Phase 1 is the **sensory receptors** - the eyes, ears, and touch of the trading system. It observes raw market stimuli without interpretation. Just as sensory neurons faithfully report what they detect without deciding what it means, Phase 1 reports structural truth without trading bias.

---

## Upstream Connections

### What Phase 1 Enables

| Consumer | What It Receives | How It Uses It |
|----------|------------------|----------------|
| **Phase 2 (Health)** | Acceptance states, failure flags | Computes structural_integrity score |
| **Phase 3 (Density)** | Range state, rotation_complete | Filters spam in balanced markets |
| **Phase 4 (Posture)** | Direction, confidence_band | Sets bias and execution_mode |
| **V6 ML Model** | Session label, swing alignment | Context for feature weighting |
| **Dashboard UI** | Levels, auction state | Visual market structure display |
| **Replay Mode** | Historical contexts | Time-travel debugging |

### Interface Contracts

**Phase 1 → Phase 2**
```
Input:  acceptance.accepted, acceptance.strength, failure.present
Output: Phase 2 structural_integrity dimension (0-100)
```

**Phase 1 → Phase 4**
```
Input:  direction, confidence_band, range.state
Output: Phase 4 bias (LONG/SHORT/NEUTRAL), execution_mode
```

**Phase 1 → Dashboard**
```
Input:  levels.set[], auction.state, current_price
Output: Visual level rendering with acceptance status
```

---

## Downstream Protection

### What Phase 1 Protects

| Downstream System | Protection Provided |
|-------------------|---------------------|
| **Phase 2-4** | Ensures they operate on valid market structure |
| **V6 ML Model** | Provides session/alignment context |
| **Policy Engine** | Prevents trades against structural truth |
| **Position Manager** | Invalidation levels for stop placement |

### Failure Modes and Impact

| Phase 1 Failure | System Impact |
|-----------------|---------------|
| Missing bars_1m | Cannot compute intraday context, Phase 2-4 skip |
| Stale swing context | MTF alignment may be wrong |
| VWAP calculation error | Level significance incorrect |
| Acceptance logic bug | False structural confidence |

---

## Data Flow Position

```
┌─────────────────────────────────────────────────────────────┐
│                     External Data                           │
│         (Polygon API: 1m bars, daily bars, quotes)          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃              PHASE 1: MARKET STRUCTURE                       ┃
┃                                                              ┃
┃   bars_1m ──► VWAP ──► Levels ──► Acceptance ──► Auction    ┃
┃                                                              ┃
┃   daily_bars ──────────► Swing Context ─────────────────────┃
┗━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   PHASE 2   │   │   PHASE 3   │   │   PHASE 4   │
│   (Health)  │   │  (Density)  │   │  (Posture)  │
└─────────────┘   └─────────────┘   └─────────────┘
```

---

## Module Breakdown

### Input Modules

| Module | Input | Output |
|--------|-------|--------|
| `vwap.py` | bars_1m | VWAP float |
| `levels.py` | bars_1m, daily_bars | Level set |
| `acceptance.py` | levels, bars_5m | Acceptance states |
| `auction_state.py` | acceptance, levels | Auction classification |

### Output Module

| Module | Input | Output |
|--------|-------|--------|
| `compute.py` | All above | IntradayContext, SwingContext |

---

## Invariant Enforcement

### What Must Always Be True

| Invariant | Enforcement |
|-----------|-------------|
| Phase 1 outputs are immutable | No setter methods on context dicts |
| Same inputs = same outputs | Deterministic computation, cached IDs |
| No future data leakage | Timestamp validation in all calculations |
| Levels are price-ordered | Sorting in levels.py |
| Acceptance requires closes, not wicks | `close > level`, not `high > level` |

---

## System Health Indicators

### When Phase 1 Is Healthy
- Intraday context resolves with clear direction
- Swing context has valid HTF levels
- Acceptance states are deterministic
- No missing or stale data

### When Phase 1 Signals Distress
- bars_1m is empty or incomplete
- daily_bars missing for swing context
- Auction state cannot resolve (conflicting signals)
- Multiple levels in TESTING state (uncertainty)

### System Response to Distress
When Phase 1 cannot produce valid context:
1. Phase 2-4 receive null input, output DEFENSIVE state
2. V6 predictions are suppressed
3. Dashboard shows "Insufficient data" warning
4. No trading signals generated

---

*Phase 1 is the foundation of market observation. Its truth enables all downstream decisions.*
