# Market Structure Layer (Phase 1) - Future Development

## Planned Improvements

### 1. Enhanced Volume Profile Analysis

**Current Gap**: VWAP is single-dimensional; no volume-at-price analysis.

**Improvement**:
```python
# Volume profile zones
{
    'high_volume_nodes': [595.00, 598.50],
    'low_volume_nodes': [596.50, 597.00],
    'point_of_control': 595.50,
    'value_area_high': 598.00,
    'value_area_low': 593.00
}
```

**Impact**: Better level significance weighting.

---

### 2. Dynamic Acceptance Thresholds

**Current Gap**: Fixed thresholds (3/5/9 closes) regardless of volatility.

**Improvement**: Scale thresholds by ATR regime.

| ATR Regime | Weak Threshold | Strong Threshold |
|------------|----------------|------------------|
| Low (<0.3%) | 5 closes | 12 closes |
| Normal | 3 closes | 9 closes |
| High (>0.7%) | 2 closes | 6 closes |

**Impact**: Prevents false acceptances in low-vol regimes.

---

### 3. Multi-Timeframe Hierarchy

**Current Gap**: Swing and intraday contexts computed separately; alignment is simple binary.

**Improvement**: Weighted alignment scoring.

```python
{
    'mtf_alignment_score': 78,
    'alignment_breakdown': {
        'yearly': {'aligned': True, 'weight': 0.30},
        'monthly': {'aligned': True, 'weight': 0.25},
        'weekly': {'aligned': False, 'weight': 0.25},
        'daily': {'aligned': True, 'weight': 0.20}
    }
}
```

**Impact**: Nuanced conflict detection vs. simple ALIGNED/CONFLICT.

---

### 4. Real-Time Level Refresh

**Current Gap**: Swing context computed once per session.

**Improvement**:
- Stream daily bar updates
- Detect intraday-to-swing level interactions
- Alert when intraday action invalidates swing bias

---

### 5. Failure Pattern Library

**Current Gap**: Limited to high/low breakout failures.

**Improvement**: Expand pattern detection:

| Pattern | Description |
|---------|-------------|
| `FAILED_HIGH_BREAKOUT` | Exists |
| `FAILED_LOW_BREAKOUT` | Exists |
| `REVERSAL_AT_OPEN` | New: Gap fill reversal |
| `VWAP_REJECTION` | New: Multiple VWAP tests |
| `LATE_DAY_TRAP` | New: 2PM+ failed expansion |
| `OVERNIGHT_TRAP` | New: Pre-market high/low trap |

---

## Priority Matrix

| Improvement | Complexity | Impact | Priority |
|-------------|------------|--------|----------|
| Volume profile | High | High | P2 |
| Dynamic thresholds | Medium | Medium | P1 |
| MTF hierarchy | Medium | High | P1 |
| Real-time refresh | High | Medium | P3 |
| Failure patterns | Medium | Medium | P2 |

---

## Dependencies

| Improvement | Requires |
|-------------|----------|
| Volume profile | 1-minute volume data with full history |
| Dynamic thresholds | ATR calculation from Phase 2 |
| MTF hierarchy | Refactored swing_link structure |
| Real-time refresh | WebSocket data feed |
| Failure patterns | Pattern recognition framework |

---

*Phase 1 improvements focus on richer market structure detection without introducing ML dependencies.*
