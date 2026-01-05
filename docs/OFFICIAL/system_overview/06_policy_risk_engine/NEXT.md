# Policy & Risk Engine - Future Development

## Planned Improvements

### 1. Dynamic Target Adjustment

**Current Gap**: Fixed TP/SL percentages regardless of volatility.

**Improvement**: Scale targets by ATR regime.

| ATR Regime | SPY TP | SPY SL |
|------------|--------|--------|
| Low (<0.3%) | 0.15% | 0.20% |
| Normal | 0.25% | 0.33% |
| High (>0.7%) | 0.40% | 0.50% |

**Impact**: Better risk/reward in different conditions.

---

### 2. Trailing Stop Logic

**Current Gap**: Static stop loss; no profit protection.

**Improvement**:
```python
{
    'stop_loss': {
        'initial': 593.50,
        'trailing': True,
        'trail_pct': 0.15,
        'lock_profit_after': 0.10  # Lock profit after 10bp move
    }
}
```

**Impact**: Protect profits in trending conditions.

---

### 3. Time-Based Exit Rules

**Current Gap**: No automatic exit timing.

**Improvement**:
```python
{
    'time_exit': {
        'warn_at': '15:30',  # 30 min before close
        'force_close_at': '15:45',  # 15 min before close
        'max_hold_hours': 4
    }
}
```

**Impact**: Avoid holding into close; manage time decay.

---

### 4. Confidence-Weighted Targets

**Current Gap**: Same targets regardless of confidence.

**Improvement**:

| Bucket | TP Multiplier | SL Multiplier |
|--------|---------------|---------------|
| very_strong | 1.5× | 1.0× |
| strong | 1.2× | 1.0× |
| moderate | 1.0× | 0.8× |
| weak | 0.8× | 0.6× |

**Rationale**: Higher confidence = more room to run; lower confidence = tighter stops.

---

### 5. Position Aggregation

**Current Gap**: No concept of aggregate position risk.

**Improvement**:
```python
{
    'aggregate_limits': {
        'max_single_position': 25,  # % of account
        'max_total_exposure': 75,   # % of account
        'max_correlation_exposure': 50,  # Correlated positions
        'max_same_direction': 50   # All longs or all shorts
    }
}
```

**Impact**: Portfolio-level risk management.

---

### 6. Scaling In/Out

**Current Gap**: All-or-nothing position entry/exit.

**Improvement**:
```python
{
    'scaling': {
        'entry_tranches': 2,  # Split entry into 2 parts
        'exit_tranches': 3,   # Take profits in 3 parts
        'first_exit_at': 0.15,  # First partial at 15bp
        'second_exit_at': 0.30  # Second partial at 30bp
    }
}
```

**Impact**: Better average entry/exit prices; reduced timing risk.

---

## Priority Matrix

| Improvement | Complexity | Impact | Priority |
|-------------|------------|--------|----------|
| Dynamic targets | Medium | High | P1 |
| Trailing stops | Medium | High | P1 |
| Time-based exits | Low | Medium | P2 |
| Confidence-weighted targets | Low | Medium | P2 |
| Position aggregation | High | High | P1 |
| Scaling in/out | High | Medium | P3 |

---

## Dependencies

| Improvement | Requires |
|-------------|----------|
| Dynamic targets | ATR calculation from data layer |
| Trailing stops | Real-time price tracking |
| Time-based exits | Time governance integration |
| Confidence-weighted targets | None |
| Position aggregation | Position state tracking |
| Scaling in/out | Order management system |

---

## Risk Model Enhancements

### Value at Risk (VaR) Integration

```python
{
    'risk_metrics': {
        'position_var_1d': 0.5,  # 1-day VaR as % of position
        'account_var_1d': 1.2,   # Account-level VaR
        'max_var_limit': 2.0     # Hard limit
    }
}
```

### Correlation-Based Sizing

```python
{
    'correlation_adjustment': {
        'spy_qqq_correlation': 0.85,
        'if_both_long': 'reduce_size_by_25%'
    }
}
```

---

*Policy Engine improvements focus on dynamic risk management and portfolio-level controls.*
