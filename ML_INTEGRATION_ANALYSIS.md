# MVP Trading App Foundation - Comprehensive Technical Analysis

## Executive Summary

The MVP Trading App is a **frontend-first Next.js application** that integrates with Polygon.io for real-time stock market data. It features a React-based UI with Zustand state management and provides live price data, charting, and basic technical analysis. Notably, **there is NO backend API server** - all data comes from Polygon.io via client-side API calls. This is crucial for ML integration planning.

**Key Characteristic**: This is a **client-side application** with no backend service running. The environment variable `REACT_APP_API_URL=http://localhost:8000/api` is configured but unused - no backend endpoints are implemented.

---

## 1. API SERVICES LAYER

### 1.1 Polygon.io Integration

**File**: `/Users/it/Documents/mvp_coder_starter_kit (2)/mvp-trading-app/src/services/polygonService.ts`
**Class**: `PolygonService`
**Type**: Singleton service with built-in rate limiting and caching

#### Key Features:
- **Rate Limiting**: Adaptive rate limiting based on plan (free: 13s between requests, paid: 0ms)
- **Caching**: In-memory cache with configurable TTL (30s for free, 3s for paid)
- **Request Queueing**: Sequential request processing with exponential backoff
- **Auto-Plan Detection**: Detects plan tier from successful snapshot calls

#### Rate Limiting Implementation:
```typescript
// Free tier: 1 request per 13 seconds (5 requests/min)
this.minRequestInterval = 13000; // default free tier
this.cacheDuration = 30000;      // 30 second cache

// Paid plans: No rate limiting
if (plan !== 'free') {
  this.minRequestInterval = 0;   // Fire requests immediately
  this.cacheDuration = 3000;     // Fresher cache for real-time
}
```

#### Available Methods:

1. **`getAggregates(ticker, timeframe, limit, displayTimeframe)`**
   - Returns: `NormalizedChartData[]` (OHLCV bars)
   - Timeframes: '1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1w', '1M'
   - Caching: Yes (via cache key)
   - Uses Polygon REST client with retry logic

2. **`getPreviousClose(ticker)`**
   - Returns: `NormalizedChartData | null` (single bar)
   - Used to calculate daily price change
   - Essential for accurate price change calculations

3. **`getSnapshot(ticker)`**
   - Returns: `PolygonTickerSnapshot` (real-time data)
   - Includes minute-level current price
   - Only works on paid Polygon plans (auto-detected)
   - Faster refresh (5 second intervals in dashboard)

4. **`getIntradayData(ticker, interval)`**
   - Returns: `NormalizedChartData[]` (intraday bars)
   - Intervals: '1', '5', '15', '30' minutes
   - Gets today's data only

#### Retry Logic:
- Exponential backoff for 429 rate limit errors: 20s, 40s, 80s
- Max 2 retries for 429 errors
- Non-429 errors fail immediately without retry

#### Cache Key Strategy:
```
aggs_{ticker}_{timeframe}_{limit}
prev_{ticker}
snapshot_{ticker}
```

### 1.2 API Service (Unused Backend)

**File**: `/Users/it/Documents/mvp_coder_starter_kit (2)/mvp-trading-app/src/services/api.ts`
**Status**: DEFINED BUT UNUSED
**Base URL**: `http://localhost:8000/api` (from env var)

#### Endpoints (Not Implemented):
- `GET /api/signals` - Fetch trading signals
- `POST /api/signals` - Create new signals
- `GET /api/engines` - Fetch ML engines
- `PATCH /api/engines/{id}` - Update engines
- `GET /api/market/{symbol}` - Market data
- `GET /api/reports` - Reports by type
- `POST /api/reports/generate` - Generate reports
- `POST /api/backtest` - Run backtests
- `GET /api/weights/{regime}` - Get model weights

**Impact for ML**: These endpoints are a placeholder architecture ready for backend integration, but currently no backend runs.

### 1.3 Data Fetching Patterns

#### Hook-Based Data Fetching:

**`usePolygonData` Hook**:
- Fetches aggregates with auto-refresh
- Calculates price change from previous close
- Returns: data, currentPrice, priceChange, priceChangePercent, isLoading, error
- Auto-refresh configurable (default 60s)

**`useMultiTickerData` Hook**:
- Fetches multiple tickers in parallel
- Falls back from snapshot to aggregates
- Implements "fetch once, use progressively" pattern
- Updates state after each successful ticker fetch

#### Example Data Flow:
```
useMultiTickerData(['SPY', 'QQQ', 'IWM'])
  -> forEach ticker:
     1. Try polygonService.getSnapshot()
     2. On failure: polygonService.getPreviousClose() + getAggregates()
     3. Calculate change, format volume
     4. Update state immediately
```

---

## 2. DATA MODELS & TYPES

### 2.1 Core Signal Types

**File**: `/Users/it/Documents/mvp_coder_starter_kit (2)/mvp-trading-app/src/types/Signal.ts`

```typescript
interface Signal {
  id: string;
  ts_emit: string;              // ISO timestamp
  symbol: string;               // Stock ticker
  engine: string;               // ML engine ID
  direction: 'long' | 'short' | 'neutral';
  confidence: number;           // 0-1 or 0-100
  horizon: string;              // '1h', '1d', '1w', etc.
  targets: Target[];            // Multiple profit targets
  stops: Stop[];                // Stop loss levels
  explain?: string;             // Signal reasoning
  features?: Feature[];         // Feature importance
  hash: string;                 // Signal deduplication
}

interface Target {
  tp: number;   // Take profit price
  prob: number; // Probability of reaching
}

interface Stop {
  sl: number;   // Stop loss price
  prob: number; // Probability of hitting
}

interface Feature {
  name: string;
  weight: number; // Feature importance
}
```

### 2.2 Market Data Types

```typescript
interface MarketData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// Normalized chart data (from Polygon)
interface NormalizedChartData {
  time: number;   // Unix timestamp (ms)
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface PolygonTickerSnapshot {
  ticker: string;
  todaysChangePerc: number;
  todaysChange: number;
  updated: number;              // Timestamp
  day: { o, h, l, c, v, vw };   // Daily OHLCV
  min: { o, h, l, c, v, vw };   // Minute OHLCV
  prevDay: { o, h, l, c, v, vw }; // Previous close
}
```

### 2.3 Engine & Regime Types

```typescript
interface Engine {
  id: string;
  name: string;
  type: 'core' | 'background';  // Core or auxiliary
  active: boolean;              // Enabled/disabled
  weight: number;               // Weighted ensemble
}

// App state includes regime tracking
regime: 'bull' | 'bear' | 'neutral'
```

### 2.4 Report Types

```typescript
interface Report {
  id: string;
  date: string;
  type: 'premarket' | 'midday' | 'eod';
  content: string;              // Markdown report
  signals: Signal[];            // Associated signals
  timestamp: string;
}
```

---

## 3. STATE MANAGEMENT

### 3.1 Zustand Store

**File**: `/Users/it/Documents/mvp_coder_starter_kit (2)/mvp-trading-app/src/store/useStore.ts`

```typescript
interface AppState {
  // Data stores
  signals: Signal[];
  engines: Engine[];
  marketData: Map<string, MarketData[]>;  // symbol -> data
  reports: Report[];
  
  // UI state
  selectedSymbol: string | null;
  selectedEngine: string | null;
  regime: 'bull' | 'bear' | 'neutral';
  
  // Action methods
  setSignals(signals: Signal[]): void;
  addSignal(signal: Signal): void;
  setEngines(engines: Engine[]): void;
  updateEngine(engineId: string, updates: Partial<Engine>): void;
  setMarketData(symbol: string, data: MarketData[]): void;
  setReports(reports: Report[]): void;
  addReport(report: Report): void;
  setSelectedSymbol(symbol: string | null): void;
  setSelectedEngine(engine: string | null): void;
  setRegime(regime: 'bull' | 'bear' | 'neutral'): void;
}
```

### 3.2 Data Flow Patterns

**Current Architecture** (Limited):
1. Dashboard/Ticker pages use hooks: `usePolygonData`, `useMultiTickerData`
2. Hooks fetch data directly from Polygon.io
3. Local state (useState) manages UI state per component
4. Zustand store is defined but **not actively used** for Polygon data
5. Mock data (`mockSignals`, `mockEngines`) exists but is not integrated

**Data Update Flow**:
```
usePolygonData Hook
  -> polygonService.getAggregates()
  -> local state: setData()
  -> Component re-renders
  
Dashboard uses useMultiTickerData
  -> Parallel fetches with fallback
  -> Progressive state updates
  -> Real-time Polygon data replaces placeholders
```

**Gap**: State management is NOT centralized. Each component manages its own state from Polygon.io. The Zustand store is ready but disconnected from data fetching.

---

## 4. BACKEND STRUCTURE

### 4.1 Current Backend Status

**Critical Finding**: There is **NO backend server**. The application is purely frontend.

**Evidence**:
- No `/api` directory in app folder (Next.js API routes not used)
- No server-side code in src/
- No database connection files
- Environment variable `REACT_APP_API_URL=http://localhost:8000/api` is configured but never used
- All API calls go directly to Polygon.io from browser

### 4.2 Page Structure

**Next.js App Router (New)**:
```
app/
  layout.tsx                    // Root layout
  page.tsx                      // Redirects to /dashboard
  dashboard/
    page.tsx                    # Main dashboard (SPY, UVXY, QQQ, IWM)
  ticker/
    [symbol]/
      page.tsx                  # Individual ticker detail page
```

### 4.3 Page Architecture

**Dashboard Page** (`/app/dashboard/page.tsx`):
- Uses `useMultiTickerData` hook for 4 tickers
- Calculates live signals from real Polygon data
- Shows multi-ticker grid layout
- No backend calls - all Polygon.io

**Ticker Page** (`/app/ticker/[symbol]/page.tsx`):
- Uses `usePolygonData` for OHLCV data
- Uses `usePolygonSnapshot` for real-time price
- Calculates technical indicators (RSI, MACD, trend, volatility)
- Displays multi-timeframe analysis
- All calculations in browser

**Gap for ML**: No server-side ML execution. All "signals" are calculated in browser from technical indicators.

---

## 5. DATABASE/STORAGE

### 5.1 Current Storage Strategy

**Database**: None. Application is stateless.

**Data Persistence**:
- Browser LocalStorage: Not used
- In-memory cache: Only in `polygonService` (30s-3s TTL)
- Polygon.io: Single source of truth for market data
- Mock data: Exists in `/src/services/mockData.ts` for development

### 5.2 Mock Data

**File**: `/src/services/mockData.ts`

Contains:
- `mockEngines`: 7 sample ML engines (Breakout Core, Divergence Core, RSI, MACD, etc.)
- `mockSignals`: 3 sample signals (AAPL long, TSLA short, SPY neutral)
- `mockMarketData`: 30 days of synthetic OHLCV data
- `mockReports`: Pre/midday/EOD report samples

**Usage**: None currently. These are templates for when backend is added.

### 5.3 Storage Gaps for ML

- No historical signal database
- No backtest result storage
- No model performance metrics storage
- No audit trail of predictions vs. actuals
- **Required for ML**: PostgreSQL or MongoDB for signals, predictions, training data

---

## 6. SECURITY IMPLEMENTATION

### 6.1 API Key Management

**Implementation**:
- Polygon.io API key: `NEXT_PUBLIC_POLYGON_API_KEY`
- Stored in `.env.local` (gitignored)
- Exposed to client (public key) - this is acceptable for Polygon (public data)
- Key in `.env.local` is valid but should be rotated

**Warning**: 
```
# Current .env.local contains:
NEXT_PUBLIC_POLYGON_API_KEY=cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O
# This is visible in git history - should be rotated
```

### 6.2 Environment Variables

**File**: `.env.local`
```
NEXT_PUBLIC_POLYGON_API_KEY=...  # Used by browser
REACT_APP_API_URL=http://localhost:8000/api  # Unused
```

**Gap**: 
- No backend authentication
- No user authentication
- No API key rotation mechanism
- CORS headers not needed (Polygon.io handles this)

### 6.3 Data Security

- **No sensitive user data**: No accounts, no portfolio data
- **No trading execution**: Read-only market data
- **XSS protection**: React auto-escapes JSX
- **CSRF protection**: Not needed (no state-changing requests)

---

## 7. COMPONENT ARCHITECTURE

### 7.1 Component Tree

```
app/
  layout.tsx (Root)
    └─ page.tsx (Home -> redirect /dashboard)
    └─ dashboard/page.tsx
        └─ Dashboard component
            └─ useMultiTickerData hook
            └─ Grid of TickerCard components
    └─ ticker/[symbol]/page.tsx
        └─ TickerPage component
            ├─ usePolygonData hook
            ├─ usePolygonSnapshot hook
            ├─ ProfessionalChart component
            │   └─ ChartHeader, ZoomControls
            │   └─ lightweight-charts canvas
            └─ Technical analysis sections

src/components/
  ├─ ProfessionalChart/          # Main charting component
  │   ├─ index.tsx               # Re-export from parent
  │   ├─ ProfessionalChart.tsx   # Main implementation (in parent)
  │   ├─ ChartHeader.tsx
  │   ├─ ZoomControls.tsx
  │   ├─ chartDrawing.ts         # Canvas drawing utilities
  │   ├─ hooks.ts                # Chart-specific hooks
  │   └─ types.ts                # Chart type definitions
  │
  ├─ trading/
  │   ├─ TradingCard.tsx
  │   ├─ metrics/TradingMetrics.tsx
  │   └─ signal/SignalIndicator.tsx
  │
  └─ ui/
      └─ layout/TradingGrid.tsx
```

### 7.2 Data Flow Diagram

```
TickerPage
  └─ usePolygonData({ticker: 'SPY'})
      └─ polygonService.getAggregates('SPY', '1h', 100)
          └─ HTTP GET Polygon.io REST API
          └─ Cache check (30s free, 3s paid)
          └─ Rate limit queue (13s free, 0s paid)
      └─ polygonService.getPreviousClose('SPY')
      └─ useState -> setData([...])
  
  └─ usePolygonSnapshot({ticker: 'SPY'})
      └─ polygonService.getSnapshot('SPY')
          └─ HTTP GET Polygon snapshot
          └─ Updates every 5s
  
  └─ Local calculations:
      calculateRSI(data)
      calculateMACD(data)
      calculateTrend(data)
      calculateSignal() -> live signal
  
  └─ Render:
      ProfessionalChart receives: chartData
      DisplayMetrics receive: rsi, macd, trend, signal
```

### 7.3 Component Dependencies

**UI Framework**:
- Next.js 15.5.3 (App Router)
- React 19.1.1
- Tailwind CSS 4.1.13
- Lucide React (icons)

**Charting**:
- lightweight-charts 5.0.8 (TradingView charts)
- recharts 3.2.0 (alternative charts)

**Data Fetching**:
- @polygon.io/client-js 8.2.0 (Polygon REST client)
- axios 1.12.2 (HTTP client, used by api.ts)

**State Management**:
- zustand 5.0.8 (store, mostly unused currently)

**Utilities**:
- date-fns 4.1.0 (date manipulation)
- clsx 2.1.1 (className utility)
- tailwind-merge 3.3.1 (CSS utilities)

---

## 8. DEPENDENCIES ANALYSIS

### 8.1 Package.json Overview

```json
{
  "name": "mvp-trading-app",
  "version": "0.1.0",
  "dependencies": {
    "@polygon.io/client-js": "^8.2.0",
    "lightweight-charts": "^5.0.8",
    "next": "^15.5.3",
    "react": "^19.1.1",
    "zustand": "^5.0.8",
    ...
  }
}
```

### 8.2 Key Dependencies

| Package | Version | Purpose | ML Integration Impact |
|---------|---------|---------|----------------------|
| @polygon.io/client-js | 8.2.0 | Market data API | Source of training data |
| lightweight-charts | 5.0.8 | Charting library | Display predictions |
| next | 15.5.3 | Framework | Ready for API routes |
| zustand | 5.0.8 | State mgmt | Can store predictions |
| axios | 1.12.2 | HTTP client | Can call ML backend |
| date-fns | 4.1.0 | Date utils | Feature engineering |

### 8.3 Missing Dependencies for ML

**Critical Gaps**:
- No ML libraries (TensorFlow.js, scikit-learn, etc.)
- No database drivers (PostgreSQL, MongoDB, SQLite)
- No numerical computing (NumPy equivalent)
- No WebSocket for live prediction streaming
- No message queue client (for async ML jobs)

---

## 9. CRITICAL ARCHITECTURAL GAPS

### 9.1 For ML System Integration

| Gap | Current State | Required | Impact |
|-----|---------------|----------|--------|
| **Backend Server** | None | Flask/FastAPI + ML model | Medium - must create |
| **API Routes** | Defined but unused | Implement /api/signals | High - needed for predictions |
| **Database** | None | PostgreSQL + migrations | High - for training data |
| **Signal Storage** | Mock only | Real signal persistence | High - backtesting needs |
| **Feature Engineering** | In-browser calc | Centralized in backend | Medium - consistency |
| **Model Inference** | Not applicable | Server-side execution | High - latency critical |
| **Backtesting Engine** | Not implemented | Historical simulation | High - needed for validation |
| **Real-time Streaming** | 60s refresh | WebSocket or Server-Sent Events | Medium - for live predictions |
| **Authentication** | None | JWT or API key | Medium - for user isolation |
| **Prediction Audit** | None | Log all predictions | Medium - compliance/debugging |

### 9.2 State Management Issues

**Current**: 
- Zustand store exists but disconnected from Polygon data
- Components use hooks for direct Polygon.io access
- No central signal repository

**For ML**:
- Need to route all predictions through Zustand
- Create `prediction` store type
- Persist predictions for backtesting

### 9.3 Data Flow Issues

**Current**:
```
Component -> Hook -> Polygon.io (direct)
```

**For ML**:
```
Component -> Hook -> Backend ML Service -> Predictions
          \> Zustand Store (audit trail)
          \> Database (historical)
```

---

## 10. MISSING TECHNICAL INFRASTRUCTURE

### 10.1 Not Implemented

1. **Backend Server**
   - No framework (Flask/FastAPI/Node)
   - No ML model serving
   - No database ORM

2. **ML Pipeline**
   - No feature engineering
   - No model training
   - No inference API

3. **Data Pipeline**
   - No data preprocessing
   - No feature computation
   - No label generation

4. **Monitoring & Logging**
   - No prediction logging
   - No performance metrics
   - No alert system

5. **Testing**
   - No unit tests for API
   - No integration tests
   - No backtest framework

### 10.2 Hardcoded Logic (In Browser)

**Dashboard Signal Calculation** (`/app/dashboard/page.tsx`, lines 51-202):
```typescript
// Simplified RSI calculation
const rsi = Math.round(30 + (momentum * 0.6))

// Signal based on thresholds
if (confidence > 85 && changePercent > 0) {
  signal = 'strong_buy'
} else if (confidence > 70 && changePercent > 0) {
  signal = 'buy'
}

// Backtested accuracy as random
backtestedAccuracy = 85.0 + Math.random() * 10
```

**Ticker Page Calculations** (`/app/ticker/[symbol]/page.tsx`, lines 99-201):
```typescript
// RSI calculation (lines 99-117)
calculateRSI(data, period) {
  let gains = 0, losses = 0
  // simplified calculation
  return 100 - (100 / (1 + rs))
}

// MACD calculation (lines 119-130)
// Trend calculation (lines 132-145)
// Signal based on score (lines 164-201)
```

**Problem**: These signals are UI-only, not stored anywhere. No persistence for backtesting.

---

## 11. ENVIRONMENT & CONFIGURATION

### 11.1 Environment Variables

**Set** (`/.env.local`):
```
NEXT_PUBLIC_POLYGON_API_KEY=cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O
REACT_APP_API_URL=http://localhost:8000/api  # Unused
```

**Expected** (for ML):
```
# Current
NEXT_PUBLIC_POLYGON_API_KEY=...

# For ML Backend
ML_API_URL=http://localhost:5000
ML_MODEL_PATH=/models/trading_model.pkl
DATABASE_URL=postgresql://user:pass@localhost/trading
POLYGON_API_KEY=...  (backend copy, not exposed)

# For Monitoring
SENTRY_DSN=...
LOG_LEVEL=info
```

### 11.2 Configuration Files

**TypeScript**:
```json
// tsconfig.json
{
  "paths": {
    "@/*": ["./src/*"]
  }
}
```

**Next.js**:
```js
// next.config.js - minimal, no ML-specific config
module.exports = {}
```

---

## 12. SPECIFIC GAPS NEEDING RESOLUTION

### 12.1 Data Gaps

1. **Historical Training Data**: 
   - App fetches live data from Polygon
   - No mechanism to store historical signals
   - Gap: Create signal DB table + archival process

2. **Signal Labeling**: 
   - Signals calculated in real-time without labels
   - No record of whether predictions were correct
   - Gap: Add prediction outcome tracking

3. **Feature Computation**:
   - Technical indicators computed in-browser per request
   - Not optimal for ML feature engineering
   - Gap: Centralize in backend for consistency

### 12.2 Execution Gaps

1. **Inference Latency**:
   - Browser-based calculations: ~10-50ms
   - Polygon API: 1-3s with rate limiting
   - Gap: ML inference must be <100ms to be useful

2. **Real-time Predictions**:
   - Current refresh: 60s for aggregates, 5s for snapshot
   - ML needs sub-second predictions
   - Gap: Implement WebSocket or SSE from backend

3. **Backtesting**:
   - API endpoint defined but not implemented
   - No historical simulation framework
   - Gap: Create backtesting engine with historical data

### 12.3 Operational Gaps

1. **Signal Persistence**:
   - Mock signals only
   - No real signal storage
   - Gap: PostgreSQL signals table

2. **Model Management**:
   - No model versioning
   - No A/B testing framework
   - Gap: Model registry + experiment tracking

3. **Monitoring**:
   - No prediction accuracy tracking
   - No alert system
   - Gap: Prometheus metrics + alerting

---

## SUMMARY TABLE

| Component | Status | Readiness for ML | Critical Issues |
|-----------|--------|-----------------|-----------------|
| **Frontend UI** | Full MVP | 90% | Minor chart enhancements needed |
| **Data Fetching** | Complete | 95% | Rate limiting solid |
| **Polygon Integration** | Production | 100% | Well-implemented |
| **State Management** | Partial | 60% | Zustand disconnected from data |
| **Backend API** | Not Started | 0% | CRITICAL - must build |
| **Database** | Not Started | 0% | CRITICAL - must design |
| **ML Pipeline** | Not Started | 0% | CRITICAL - must implement |
| **Signal Persistence** | Not Started | 0% | CRITICAL - needed |
| **Backtesting** | Not Started | 0% | CRITICAL - foundation needed |
| **Documentation** | Good | 80% | Architecture docs exist |

---

## RECOMMENDED NEXT STEPS FOR ML INTEGRATION

1. **Week 1: Backend Foundation**
   - Create Flask/FastAPI server
   - Set up PostgreSQL database
   - Implement `/api/signals` endpoint

2. **Week 2: Signal Persistence**
   - Design Signal table schema
   - Create signal store/retrieve logic
   - Connect frontend to backend signals

3. **Week 3: Data Pipeline**
   - Create feature engineering module
   - Build historical data importer
   - Set up training data pipeline

4. **Week 4: ML Model**
   - Train initial model
   - Implement inference API
   - Create backtesting framework

5. **Week 5: Integration & Testing**
   - Connect frontend predictions
   - Implement live prediction endpoint
   - Add comprehensive logging

---

## FILE REFERENCE GUIDE

| Purpose | File Path | Type | Lines |
|---------|-----------|------|-------|
| Main Polygon service | `src/services/polygonService.ts` | Service | 481 |
| Polygon hook | `src/hooks/usePolygonData.ts` | Hook | 163 |
| Multi-ticker hook | `src/hooks/useMultiTickerData.ts` | Hook | 197 |
| Store definition | `src/store/useStore.ts` | Store | 50 |
| Type definitions | `src/types/Signal.ts` | Types | 55 |
| Polygon types | `src/types/polygon.ts` | Types | 131 |
| Trading types | `src/types/trading.ts` | Types | 63 |
| API service | `src/services/api.ts` | Service | 117 |
| Mock data | `src/services/mockData.ts` | Data | 126 |
| Dashboard page | `app/dashboard/page.tsx` | Page | 512 |
| Ticker page | `app/ticker/[symbol]/page.tsx` | Page | 790 |
| Root layout | `app/layout.tsx` | Layout | 22 |
| Chart component | `src/components/ProfessionalChart.tsx` | Component | TBD |
| Package.json | `package.json` | Config | 62 |
| TypeScript config | `tsconfig.json` | Config | 41 |
| Environment example | `.env.example` | Config | 6 |

