# Data Source & Feature Integrity Layer - System Connections

## The Body Metaphor

The Data Source & Feature Integrity Layer is the **digestive system** of the trading platform. Just as the digestive system breaks down food into nutrients the body can use, this layer breaks down raw market data into features the ML model can consume.

Like the digestive system filters out toxins and pathogens, this layer filters out bad data, missing values, and invalid records.

---

## Upstream Connections

### What Data Integrity Enables

| Consumer | What It Receives | How It Uses It |
|----------|------------------|----------------|
| **V6 Model** | 29 features, scaled | ML prediction |
| **Phase 1** | hourly_bars, daily_bars | Market structure analysis |
| **VWAP Calculator** | 1m bars | VWAP computation |
| **Level Detection** | daily_bars | Prior day levels |
| **Dashboard UI** | current_price, status | Display |

### Interface Contracts

**Data → V6 Model**
```
Input:
  hourly_bars: List[Dict] with o, h, l, c, v, t
  daily_bars: List[Dict] with o, h, l, c, v, t

Processing:
  today_open = daily_bars[-1]['o']  # 9:30 AM open
  features = build_v6_features(hourly_bars, daily_bars, today_open, current_hour)

Output:
  X = np.array([[features.get(col, 0) for col in V6_FEATURE_COLS]])
  X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
```

**Data → Phase 1**
```
Input:
  hourly_bars: Today's bars for RTH
  daily_bars: 20+ days for swing analysis

Output:
  intraday_context: From today's bars
  swing_context: From daily bars
```

---

## Downstream Protection

### What Data Integrity Protects

| Downstream System | Protection Provided |
|-------------------|---------------------|
| **V6 Model** | Valid features, no NaN |
| **Phase 1** | Valid bar data for structure |
| **Policy Engine** | Valid current_price for targets |
| **User Interface** | Valid data for display |

### Failure Modes and Impact

| Failure | System Impact |
|---------|---------------|
| Polygon API down | No data, NO_TRADE |
| Missing bars | Feature gaps, degraded accuracy |
| Wrong today_open | Training/serving skew |
| NaN in features | Model crashes or bad predictions |
| Stale data | Outdated predictions |

---

## Data Flow Position

```
┌─────────────────────────────────────────────────────────────┐
│                    POLYGON API                               │
│              (hourly, daily, quotes)                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           DATA SOURCE & FEATURE INTEGRITY                    ┃
┃                                                              ┃
┃   fetch_hourly_bars() ──► validate() ──► hourly_bars        ┃
┃   fetch_daily_bars() ───► validate() ──► daily_bars         ┃
┃                                                              ┃
┃   bars ──► build_v6_features() ──► 29 features              ┃
┃                                                              ┃
┃   features ──► nan_to_num() ──► clean features              ┃
┗━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   PHASE 1   │   │  V6 MODEL   │   │   PREDICT   │
│  (bars)     │   │ (features)  │   │   SERVER    │
└─────────────┘   └─────────────┘   └─────────────┘
```

---

## Data Pipeline Stages

### Stage 1: Fetch

```
Polygon API ──► Raw JSON ──► Parse ──► List[Dict]

Error Handling:
- API timeout: Retry 3 times
- Rate limit: Wait and retry
- Invalid response: Log and skip
```

### Stage 2: Validate

```
List[Dict] ──► Validate each bar ──► Filter invalid ──► Valid bars

Validation Rules:
- Required fields: t, o, h, l, c, v
- h >= l (high >= low)
- All prices > 0
- t is valid timestamp
```

### Stage 3: Transform

```
Valid bars ──► build_v6_features() ──► 29 features

Transformations:
- Calculate ratios (open_to_current, etc.)
- Calculate momentum (hourly, morning, etc.)
- Calculate structure (body_to_range, wicks, etc.)
```

### Stage 4: Clean

```
Features dict ──► Order by V6_FEATURE_COLS ──► nan_to_num ──► Array

Cleaning:
- Order features to match training
- Replace NaN with 0
- Replace inf with 0
```

---

## Critical Data Source Rules

### Today's Open (DS-1)

```
# CRITICAL: This is the most common source of skew

# CORRECT: Daily bar open (9:30 AM regular market)
today_open = daily_bars[-1]['o']

# WRONG: Hourly bar 0 is 4 AM pre-market
today_open = hourly_bars[0]['o']  # DON'T USE

# WRONG: Quote data may be pre-market
today_open = quote['o']  # DON'T USE
```

### Previous Day (DS-3)

```
# Previous day = daily_bars[-2], NOT daily_bars[-1]

# CORRECT
prev_day = daily_bars[-2]
prev_close = prev_day['c']

# WRONG: daily_bars[-1] is TODAY
prev_close = daily_bars[-1]['c']  # DON'T USE
```

---

## Invariant Enforcement

| Invariant | Enforcement |
|-----------|-------------|
| 29 features exactly | V6_FEATURE_COLS constant |
| Feature order matches training | Iterate V6_FEATURE_COLS |
| Today's open from daily bar | Explicit daily_bars[-1]['o'] |
| No NaN in features | np.nan_to_num |
| Valid bar data | Validation before use |

---

## System Health Indicators

### When Data Integrity Is Healthy
- Polygon API responding
- All bars validated successfully
- Features built without NaN
- Today's open extracted correctly

### When Data Integrity Signals Distress
- API timeouts or errors
- High rate of invalid bars
- NaN values detected
- Missing critical bars (9:30 AM daily)

### System Response to Distress
When data integrity fails:
1. Return NO_TRADE with reason
2. Log data quality issue
3. Dashboard shows data warning
4. Alert for extended failures

---

*Data Integrity is the immune system protecting prediction accuracy.*
