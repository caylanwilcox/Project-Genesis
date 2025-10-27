# MVP Trading App - Comprehensive System Analysis & Security Audit

**Document Version:** 1.0
**Date:** 2025-10-24
**Status:** Current Production System Analysis

---

## Executive Summary

This document provides an in-depth analysis of the MVP Trading App, a Next.js-based ETF/stock trading platform that integrates real-time market data from Polygon.io. The system is currently in MVP stage with a React/TypeScript frontend, client-side state management via Zustand, and plans to implement advanced ML-based trading algorithms across multiple time horizons (1m to 1M timeframes).

**Key Findings:**
- **Architecture:** Client-side heavy architecture with no backend implementation yet
- **Security Posture:** High risk - API keys exposed, no authentication, vulnerable to attacks
- **Data Layer:** Direct API calls from client to Polygon.io, with basic caching
- **Scalability:** Limited by free-tier rate limits, no server-side infrastructure
- **Foundation Quality:** Solid frontend structure but critical backend infrastructure missing

---

## 1. System Architecture Overview

### 1.1 Technology Stack

**Frontend Framework:**
- Next.js 15.5.3 (React 19.1.1)
- TypeScript 4.9.5
- Tailwind CSS 4.1.13
- Client-side rendering with 'use client' directives

**State Management:**
- Zustand 5.0.8 (lightweight global state)
- React Hooks for local component state

**Chart Libraries:**
- lightweight-charts 5.0.8 (TradingView-style charts)
- recharts 3.2.0 (secondary charting)

**Data Sources:**
- Polygon.io REST API (@polygon.io/client-js 8.2.0)
- WebSocket support via react-use-websocket 4.13.0 (not yet implemented)

**HTTP Client:**
- Axios 1.12.2

**Build & Dev Tools:**
- ESLint, PostCSS, Autoprefixer

### 1.2 Current Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      CLIENT BROWSER                          │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Next.js Frontend Application                 │ │
│  │                                                          │ │
│  │  ┌──────────────┐    ┌────────────────┐               │ │
│  │  │   UI Layer   │    │  State (Zustand)│               │ │
│  │  │  Components  │◄───┤   - signals     │               │ │
│  │  │              │    │   - engines     │               │ │
│  │  └──────┬───────┘    │   - marketData  │               │ │
│  │         │            └────────┬────────┘               │ │
│  │         │                     │                         │ │
│  │         ▼                     ▼                         │ │
│  │  ┌──────────────────────────────────┐                 │ │
│  │  │      Custom React Hooks          │                 │ │
│  │  │  - usePolygonData                │                 │ │
│  │  │  - useMultiTickerData            │                 │ │
│  │  └──────────┬───────────────────────┘                 │ │
│  │             │                                           │ │
│  │             ▼                                           │ │
│  │  ┌──────────────────────────────────┐                 │ │
│  │  │      Service Layer               │                 │ │
│  │  │  - polygonService (w/ cache)     │                 │ │
│  │  │  - apiService (localhost:8000)   │                 │ │
│  │  └──────────┬───────────────────────┘                 │ │
│  │             │                                           │ │
│  └─────────────┼───────────────────────────────────────┘ │
│                │                                            │
└────────────────┼────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌─────────┐ ┌─────────┐ ┌──────────────┐
│Polygon  │ │ Backend │ │  .env.local  │
│  .io    │ │  API    │ │  (API Keys)  │
│ (Live)  │ │ (Mock)  │ └──────────────┘
└─────────┘ └─────────┘
  Real API    Not Implemented
  (Exposed    (Returns [])
   Key!)
```

### 1.3 Data Flow Analysis

**Primary Data Flow (Current Implementation):**

1. **User Interaction** → Component (e.g., ProfessionalChart)
2. **Component** → Custom Hook (usePolygonData)
3. **Hook** → polygonService.getAggregates()
4. **Service** → Check cache (30s TTL for free tier)
5. **Service** → Queue request (rate limiter: 13s between requests)
6. **Service** → Polygon.io REST API (HTTPS)
7. **API Response** → Cache → Normalize → Hook → Component → Render

**Secondary Data Flow (Planned but Not Implemented):**

1. Component → apiService (localhost:8000)
2. apiService → Backend API (signals, engines, reports, backtesting)
3. **Issue:** Backend returns empty arrays/null - not functional

---

## 2. Component Architecture & Structure

### 2.1 Key Components

**Page Components:**
- `app/page.tsx` - Landing/home page
- `app/dashboard/page.tsx` - Trading dashboard
- `app/ticker/[symbol]/page.tsx` - Individual ticker view
- `app/layout.tsx` - Root layout wrapper

**Trading UI Components:**
- `TradingView.tsx` - Main trading interface orchestrator
- `ProfessionalChart.tsx` - Advanced candlestick chart with indicators
- `TradingHeader.tsx` - Ticker info, price display
- `TradingForm.tsx` - Order entry form (non-functional)
- `OrderBook.tsx` - Mock order book display
- `TradeHistory.tsx` - Mock trade history
- `BottomTabs.tsx` - Orders/assets tab panel

**Data Visualization:**
- `PriceChart.tsx` - Basic price chart wrapper
- `TradingChart.tsx` - Alternative chart implementation
- `ChartView.tsx` - Chart container
- `ProfessionalChart/` - Modularized chart system
  - `chartDrawing.ts` - Canvas rendering logic
  - `hooks.ts` - Chart-specific hooks
  - `types.ts` - Type definitions
  - `ZoomControls.tsx`, `ChartHeader.tsx` - Chart UI

**Signal & Engine Management:**
- `SignalList.tsx` - Display trading signals
- `EnginePanel.tsx` - Manage ML engines
- `ReportViewer.tsx` - View trading reports
- `Dashboard.tsx` - Dashboard layout

### 2.2 State Management Structure

**Global State (Zustand Store - [useStore.ts](src/store/useStore.ts)):**

```typescript
interface AppState {
  signals: Signal[]           // Trading signals from ML engines
  engines: Engine[]           // ML algorithm engines
  marketData: Map<string, MarketData[]>  // Historical price data
  reports: Report[]           // Pre-market, midday, EOD reports
  selectedSymbol: string | null
  selectedEngine: string | null
  regime: 'bull' | 'bear' | 'neutral'  // Market regime

  // Actions
  setSignals, addSignal
  setEngines, updateEngine
  setMarketData
  setReports, addReport
  setSelectedSymbol, setSelectedEngine
  setRegime
}
```

**Local Component State:**
- Chart data (via usePolygonData hook)
- Form inputs (trading form, search)
- UI state (tabs, modals, dropdowns)
- Loading/error states

### 2.3 Type System

**Core Types ([types/Signal.ts](src/types/Signal.ts)):**

```typescript
Signal {
  id, ts_emit, symbol, engine
  direction: 'long' | 'short' | 'neutral'
  confidence: number (0-1)
  horizon: string (1m, 5m, 1h, 1d, etc.)
  targets: Target[]  // Take-profit levels
  stops: Stop[]      // Stop-loss levels
  explain?: string
  features?: Feature[]
  hash: string
}

Engine {
  id, name
  type: 'core' | 'background'
  active: boolean
  weight: number
}

Report {
  id, date
  type: 'premarket' | 'midday' | 'eod'
  content: string
  signals: Signal[]
  timestamp: string
}
```

**Trading Types ([types/trading.ts](src/types/trading.ts)):**
- TradingPair, OrderBook, Trade, Order, Asset, ChartData

**Polygon Types ([types/polygon.ts](src/types/polygon.ts)):**
- NormalizedChartData, PolygonAggregatesResponse, Timeframe configs

---

## 3. Data Layer & API Integration

### 3.1 Polygon.io Service ([services/polygonService.ts](src/services/polygonService.ts))

**Purpose:** Real-time and historical market data fetching with rate limiting and caching.

**Key Features:**

1. **Rate Limiting:**
   - Free tier: 13-second minimum interval between requests
   - Starter/Developer: Parallel requests (0ms interval)
   - Request queue with automatic processing

2. **Caching:**
   - Free tier: 30-second cache TTL
   - Paid tier: 3-second cache TTL
   - In-memory Map-based cache with timestamps

3. **Auto-Retry Logic:**
   - Handles 429 (rate limit) errors
   - Exponential backoff: 20s, 40s, 80s
   - Max 2 retries

4. **API Methods:**
   - `getAggregates(ticker, timeframe, limit)` - OHLCV bars
   - `getPreviousClose(ticker)` - Previous day's close
   - `getSnapshot(ticker)` - Real-time snapshot (paid plans)
   - `getIntradayData(ticker, interval)` - Intraday minute bars

5. **Plan Detection:**
   - Checks `NEXT_PUBLIC_POLYGON_PLAN` env var
   - Auto-upgrades from 'free' if snapshot endpoint works

**Rate Limiting Implementation:**

```typescript
private requestQueue: Array<() => Promise<any>> = []
private minRequestInterval: number = 13000 // free tier
private lastRequestTime: number = 0

private async processQueue() {
  while (requestQueue.length > 0) {
    const timeSinceLastRequest = Date.now() - lastRequestTime
    if (timeSinceLastRequest < minRequestInterval) {
      await sleep(minRequestInterval - timeSinceLastRequest)
    }
    const request = requestQueue.shift()
    lastRequestTime = Date.now()
    await request()
  }
}
```

### 3.2 Backend API Service ([services/api.ts](src/services/api.ts))

**Purpose:** Interface to backend ML engine API (currently non-functional).

**Planned Endpoints:**
- `GET /api/signals` - Fetch trading signals
- `POST /api/signals` - Create new signal
- `GET /api/engines` - List ML engines
- `PATCH /api/engines/:id` - Update engine config
- `GET /api/market/:symbol` - Market data
- `GET /api/reports` - Fetch reports
- `POST /api/reports/generate` - Generate report
- `POST /api/backtest` - Run backtest
- `GET /api/weights/:regime` - Get regime weights

**Current Status:**
- Configured to `http://localhost:8000/api`
- All methods return empty arrays or null
- Error handling via console.error (silently fails)
- No authentication or authorization

**Critical Issue:** Backend does not exist - API calls always fail gracefully.

### 3.3 Custom React Hooks

**usePolygonData ([hooks/usePolygonData.ts](src/hooks/usePolygonData.ts)):**

```typescript
usePolygonData({
  ticker: 'AAPL',
  timeframe: '1h',
  limit: 100,
  autoRefresh: true,
  refreshInterval: 60000,
  displayTimeframe: 'YTD'
})

Returns: {
  data: NormalizedChartData[]
  currentPrice, priceChange, priceChangePercent
  isLoading, error
  refetch: () => Promise<void>
}
```

**Features:**
- Automatic refetching on interval
- Error handling with user-friendly messages
- API key validation
- Previous close calculation for accurate daily change

**useMultiTickerData ([hooks/useMultiTickerData.ts](src/hooks/useMultiTickerData.ts)):**

```typescript
useMultiTickerData(['AAPL', 'SPY', 'QQQ'], autoRefresh, refreshInterval)

Returns: {
  tickers: Map<string, TickerSnapshot>
  isLoading, error
  refetch: () => Promise<void>
}
```

**Features:**
- Parallel ticker fetching (service handles rate limiting)
- Snapshot API with fallback to aggregates
- Progressive UI updates (updates after each ticker loads)
- Volume formatting (1.2M, 345.6K)

---

## 4. Security Vulnerabilities & Critical Issues

### 4.1 🔴 CRITICAL: API Key Exposure

**Issue:** Polygon.io API key is stored in `.env.local` and exposed to client-side code.

**File:** [.env.local](.env.local:2)
```
NEXT_PUBLIC_POLYGON_API_KEY=cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O
```

**Why This Is Critical:**
- The `NEXT_PUBLIC_` prefix makes this variable available to browser JavaScript
- Anyone can inspect network requests and extract the API key
- Attackers can use the key for their own purposes, exhausting rate limits
- API key abuse can lead to account suspension or charges

**Impact:** HIGH - Active credential leak in production

**Remediation:**
1. **Immediate:** Rotate the exposed API key via Polygon.io dashboard
2. **Short-term:** Move API calls to a Next.js API route (server-side)
3. **Long-term:** Implement proper backend with authentication

### 4.2 🔴 CRITICAL: No Authentication System

**Issue:** Application has no user authentication or authorization.

**Implications:**
- No user accounts or session management
- Cannot restrict access to features
- Cannot track user activity
- Cannot implement rate limiting per user
- Cannot store user-specific data (portfolios, preferences)

**Impact:** HIGH - Cannot deploy to production safely

**Remediation:**
- Implement NextAuth.js or similar authentication
- Add JWT-based session management
- Implement role-based access control (RBAC)
- Add API key authentication for backend calls

### 4.3 🟡 HIGH: No Backend Infrastructure

**Issue:** Backend API at `localhost:8000` does not exist.

**Missing Components:**
- ML engine infrastructure
- Signal generation pipeline
- Backtesting engine
- Database for storing signals, trades, user data
- Report generation system
- Real-time WebSocket server

**Impact:** MEDIUM - Application cannot execute core functionality

**Current State:** Frontend is a "demo UI" with mock data

**Remediation:** Build backend infrastructure (see Section 6)

### 4.4 🟡 HIGH: Client-Side Rate Limiting

**Issue:** Rate limiting is implemented client-side, which can be bypassed.

**Code:** [polygonService.ts](src/services/polygonService.ts:111-137)

**Why This Is Unsafe:**
- JavaScript can be modified in browser dev tools
- Multiple browser tabs bypass the queue
- Malicious users can remove rate limiting code
- No server-side enforcement

**Impact:** MEDIUM - API abuse potential, account suspension risk

**Remediation:**
- Implement server-side rate limiting
- Use Redis for distributed rate limiting
- Monitor API usage on backend
- Implement circuit breakers

### 4.5 🟡 MEDIUM: In-Memory Cache

**Issue:** Cache is stored in-memory (Map) in the browser.

**Problems:**
- Cache cleared on page refresh
- No cache sharing between users
- No cache persistence
- Memory leaks possible with long sessions

**Impact:** LOW-MEDIUM - Performance and scalability limitations

**Remediation:**
- Implement Redis cache on backend
- Use browser localStorage for client-side cache (with size limits)
- Add cache invalidation strategies
- Implement ETags for HTTP caching

### 4.6 🟡 MEDIUM: No Input Validation

**Issue:** No validation on user inputs or API responses.

**Examples:**
- Ticker symbols not validated (can cause API errors)
- Timeframe inputs not validated
- API response structure not validated (runtime errors possible)
- No sanitization of user-generated content

**Impact:** MEDIUM - Potential for errors, XSS, injection attacks

**Remediation:**
- Implement Zod or Yup schema validation
- Validate all inputs before API calls
- Use TypeScript strict mode
- Sanitize all rendered content
- Validate API response schemas

### 4.7 🟢 LOW: Error Handling

**Issue:** Errors are logged to console but not properly reported or monitored.

**Code Pattern:**
```typescript
catch (error) {
  console.error('Error fetching signals:', error)
  return []
}
```

**Problems:**
- Silent failures in production
- No error tracking or alerting
- Users see no error messages
- Cannot debug production issues

**Impact:** LOW - Operational visibility issues

**Remediation:**
- Integrate Sentry or similar error tracking
- Implement user-facing error messages
- Add retry logic for transient errors
- Log errors to backend for analysis

### 4.8 🟢 LOW: No Request Signing or HMAC

**Issue:** API requests to backend (when implemented) have no signature verification.

**Impact:** LOW (backend doesn't exist yet) - Future request forgery risk

**Remediation:**
- Implement HMAC request signing
- Add timestamp-based replay protection
- Use HTTPS with certificate pinning
- Implement request ID tracking

---

## 5. Data Flow & Dependencies

### 5.1 Frontend Dependencies

**Production Dependencies (31 packages):**

**Core Framework:**
- next@15.5.3
- react@19.1.1, react-dom@19.1.1
- typescript@4.9.5

**Data & API:**
- @polygon.io/client-js@8.2.0
- axios@1.12.2
- react-use-websocket@4.13.0 (unused)

**State Management:**
- zustand@5.0.8

**UI & Styling:**
- tailwindcss@4.1.13
- lucide-react@0.544.0 (icons)
- clsx@2.1.1, tailwind-merge@3.3.1

**Charts:**
- lightweight-charts@5.0.8
- recharts@3.2.0

**Utilities:**
- date-fns@4.1.0
- web-vitals@2.1.4

**Testing:**
- @testing-library/react@16.3.0
- @testing-library/jest-dom@6.8.0
- @testing-library/user-event@13.5.0

### 5.2 Environment Variables

**Current Configuration:**

```bash
# Public (exposed to browser)
NEXT_PUBLIC_POLYGON_API_KEY=cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O  # 🔴 LEAKED
NEXT_PUBLIC_POLYGON_PLAN=free  # Optional: free|starter|developer

# Private (server-side only) - NOT IN USE
REACT_APP_API_URL=http://localhost:8000/api
```

**Missing Critical Variables:**
```bash
# Database
DATABASE_URL=postgresql://...
REDIS_URL=redis://...

# Authentication
NEXTAUTH_SECRET=...
NEXTAUTH_URL=...

# API Keys (server-side)
POLYGON_API_KEY=...  # Should be private
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...

# Services
SENTRY_DSN=...
LOGFLARE_API_KEY=...

# Trading
PAPER_TRADING_MODE=true
RISK_LIMIT_PERCENT=2
```

### 5.3 Build & Deployment

**Build Configuration ([next.config.js](next.config.js)):**
```javascript
{
  reactStrictMode: true,
  typescript: { ignoreBuildErrors: false },
  eslint: { ignoreDuringBuilds: false }
}
```

**Scripts:**
- `npm run dev` - Development server (Next.js)
- `npm run build` - Production build
- `npm run start` - Production server
- `npm run lint` - ESLint checks

**No CI/CD Configuration Found:**
- No GitHub Actions workflows
- No Docker configuration
- No deployment scripts
- No environment-specific configs

---

## 6. Foundation Assessment & Improvement Roadmap

### 6.1 Current Foundation Strengths

✅ **Strong Frontend Architecture:**
- Modern React/Next.js 15 with App Router
- TypeScript for type safety
- Component modularity and reusability
- Clean separation of concerns (services, hooks, components)

✅ **Chart & Visualization:**
- Professional-grade charting with lightweight-charts
- Multiple chart implementations for flexibility
- Real-time data updates

✅ **State Management:**
- Zustand provides clean, performant state
- Well-defined data models and types

✅ **Data Fetching Layer:**
- Custom hooks for data management
- Rate limiting and caching (client-side)
- Error handling and retries

### 6.2 Critical Foundation Gaps

❌ **No Backend Infrastructure:**
- No server to process ML algorithms
- No database for persistence
- No API for signal generation
- No WebSocket server for real-time updates

❌ **No Security Layer:**
- No authentication system
- No authorization controls
- API keys exposed to client
- No request validation or sanitization

❌ **No ML Pipeline:**
- Algorithms documented but not implemented
- No training infrastructure
- No model serving layer
- No backtesting engine

❌ **No Trading Execution:**
- No broker integration (Alpaca, Interactive Brokers)
- No order management system
- No position tracking
- No risk management

### 6.3 Architecture Improvements Needed

**1. Backend Infrastructure (Priority: CRITICAL)**

```
Required Components:
┌─────────────────────────────────────────────────┐
│           Backend API Server (FastAPI/Node)     │
│                                                  │
│  ┌──────────────────┐  ┌────────────────────┐ │
│  │  Authentication  │  │   API Gateway       │ │
│  │   - JWT tokens   │  │   - Rate limiting   │ │
│  │   - Sessions     │  │   - Request signing │ │
│  └──────────────────┘  └────────────────────┘ │
│                                                  │
│  ┌──────────────────┐  ┌────────────────────┐ │
│  │  Signal Service  │  │  Market Data Service│ │
│  │  - ML engines    │  │  - Polygon proxy    │ │
│  │  - Backtesting   │  │  - Data aggregation │ │
│  └──────────────────┘  └────────────────────┘ │
│                                                  │
│  ┌──────────────────────────────────────────┐ │
│  │    Database Layer (PostgreSQL + Redis)   │ │
│  │    - Users, Signals, Trades, Portfolios  │ │
│  └──────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

**Technologies:**
- FastAPI (Python) or Express/NestJS (Node.js)
- PostgreSQL for relational data
- Redis for caching and rate limiting
- Celery/Bull for background jobs
- Docker for containerization

**2. ML Pipeline Infrastructure (Priority: CRITICAL)**

```
ML Pipeline Architecture:
┌───────────────────────────────────────────────────┐
│             ML Training & Inference               │
│                                                    │
│  ┌──────────────┐   ┌──────────────────────────┐│
│  │ Data Pipeline│   │   Feature Engineering    ││
│  │ - Ingestion  │──▶│   - Technical indicators ││
│  │ - Cleaning   │   │   - Normalization        ││
│  └──────────────┘   └─────────┬──────��─────────┘│
│                                │                  │
│                                ▼                  │
│  ┌──────────────────────────────────────────────┐│
│  │         Model Training (Offline)             ││
│  │  - XGBoost, LSTM, Transformers               ││
│  │  - Hyperparameter tuning                     ││
│  │  - Cross-validation                          ││
│  └─────────────┬────────────────────────────────┘│
│                │                                  │
│                ▼                                  │
│  ┌──────────────────────────────────────────────┐│
│  │         Model Serving (Real-time)            ││
│  │  - TorchServe / TensorFlow Serving           ││
│  │  - Model versioning                          ││
│  │  - A/B testing                               ││
│  └─────────────┬────────────────────────────────┘│
│                │                                  │
│                ▼                                  │
│  ┌──────────────────────────────────────────────┐│
│  │         Signal Generation                    ││
│  │  - Multi-timeframe analysis                  ││
│  │  - Ensemble predictions                      ││
│  │  - Confidence scoring                        ││
│  └──────────────────────────────────────────────┘│
└───────────────────────────────────────────────────┘
```

**Technologies:**
- Python 3.11+ with scikit-learn, XGBoost, PyTorch
- MLflow for experiment tracking
- Weights & Biases for monitoring
- ONNX for model optimization
- Kubeflow or Airflow for pipeline orchestration

**3. Security Enhancements (Priority: HIGH)**

**Immediate Actions:**
1. Rotate exposed Polygon.io API key
2. Move API key to server-side environment variable (no `NEXT_PUBLIC_`)
3. Create Next.js API routes to proxy Polygon.io requests
4. Implement NextAuth.js with JWT

**Short-term (1-2 weeks):**
1. Add email/password authentication
2. Implement OAuth (Google, GitHub)
3. Add API key authentication for backend
4. Implement CORS policies
5. Add rate limiting with Redis
6. Input validation with Zod

**Long-term (1-3 months):**
1. Implement RBAC (roles: free, premium, admin)
2. Add 2FA authentication
3. Implement audit logging
4. Add request signing (HMAC-SHA256)
5. Security headers (CSP, HSTS)
6. Penetration testing

**4. Database Schema (Priority: HIGH)**

```sql
-- Users & Authentication
CREATE TABLE users (
  id UUID PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255),
  role VARCHAR(50) DEFAULT 'free',
  created_at TIMESTAMP DEFAULT NOW(),
  last_login TIMESTAMP
);

CREATE TABLE sessions (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  token VARCHAR(500) NOT NULL,
  expires_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Trading Signals
CREATE TABLE signals (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  ts_emit TIMESTAMP NOT NULL,
  symbol VARCHAR(10) NOT NULL,
  engine VARCHAR(100) NOT NULL,
  direction VARCHAR(10) CHECK (direction IN ('long', 'short', 'neutral')),
  confidence DECIMAL(3,2) CHECK (confidence >= 0 AND confidence <= 1),
  horizon VARCHAR(10) NOT NULL,
  targets JSONB,
  stops JSONB,
  explain TEXT,
  features JSONB,
  hash VARCHAR(64) NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_signals_symbol ON signals(symbol);
CREATE INDEX idx_signals_ts_emit ON signals(ts_emit DESC);
CREATE INDEX idx_signals_engine ON signals(engine);

-- ML Engines
CREATE TABLE engines (
  id UUID PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  type VARCHAR(50) CHECK (type IN ('core', 'background')),
  active BOOLEAN DEFAULT true,
  weight DECIMAL(3,2),
  config JSONB,
  performance_metrics JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Market Data Cache
CREATE TABLE market_data (
  symbol VARCHAR(10) NOT NULL,
  timeframe VARCHAR(10) NOT NULL,
  ts TIMESTAMP NOT NULL,
  open DECIMAL(12,4),
  high DECIMAL(12,4),
  low DECIMAL(12,4),
  close DECIMAL(12,4),
  volume BIGINT,
  PRIMARY KEY (symbol, timeframe, ts)
);

CREATE INDEX idx_market_data_symbol_ts ON market_data(symbol, ts DESC);

-- Reports
CREATE TABLE reports (
  id UUID PRIMARY KEY,
  date DATE NOT NULL,
  type VARCHAR(20) CHECK (type IN ('premarket', 'midday', 'eod')),
  content TEXT,
  signal_ids UUID[],
  created_at TIMESTAMP DEFAULT NOW()
);

-- Backtests
CREATE TABLE backtests (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  engine_id UUID REFERENCES engines(id),
  symbol VARCHAR(10) NOT NULL,
  start_date DATE,
  end_date DATE,
  initial_capital DECIMAL(12,2),
  final_capital DECIMAL(12,2),
  total_return DECIMAL(5,2),
  sharpe_ratio DECIMAL(5,2),
  max_drawdown DECIMAL(5,2),
  win_rate DECIMAL(5,2),
  results JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);

-- User Portfolios
CREATE TABLE portfolios (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  name VARCHAR(255),
  positions JSONB,
  cash DECIMAL(12,2),
  total_value DECIMAL(12,2),
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Trades
CREATE TABLE trades (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  portfolio_id UUID REFERENCES portfolios(id),
  signal_id UUID REFERENCES signals(id),
  symbol VARCHAR(10) NOT NULL,
  side VARCHAR(10) CHECK (side IN ('buy', 'sell')),
  quantity DECIMAL(12,4),
  price DECIMAL(12,4),
  status VARCHAR(20) CHECK (status IN ('pending', 'filled', 'canceled', 'rejected')),
  broker_order_id VARCHAR(255),
  executed_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW()
);
```

**5. API Architecture (Priority: HIGH)**

**Endpoint Structure:**

```
/api/v1/
├── auth/
│   ├── POST /register
│   ├── POST /login
│   ├── POST /logout
│   ├── POST /refresh
│   └── GET /me
├── market/
│   ├── GET /ticker/:symbol
│   ├── GET /ticker/:symbol/aggregates
│   ├── GET /ticker/:symbol/snapshot
│   └── GET /search
├── signals/
│   ├── GET /signals
│   ├── GET /signals/:id
│   ├── POST /signals (admin only)
│   └── DELETE /signals/:id (admin only)
├── engines/
│   ├── GET /engines
│   ├── GET /engines/:id
│   ├── PATCH /engines/:id (admin only)
│   └── POST /engines/:id/backtest
├── reports/
│   ├── GET /reports
│   ├── GET /reports/:id
│   └── POST /reports/generate (admin only)
├── portfolio/
│   ├── GET /portfolios
│   ├── GET /portfolios/:id
│   ├── POST /portfolios
│   ├── PATCH /portfolios/:id
│   └── DELETE /portfolios/:id
└── trades/
    ├── GET /trades
    ├── POST /trades
    └── GET /trades/:id
```

**6. Real-Time Infrastructure (Priority: MEDIUM)**

**WebSocket Server:**
```typescript
// WebSocket events
ws://api.tradingapp.com/ws

Events:
- ticker.update.{symbol} - Real-time price updates
- signal.new - New trading signal generated
- trade.executed - Trade execution notification
- portfolio.update - Portfolio value change
- market.status - Market open/close status
```

**Technologies:**
- Socket.io or native WebSockets
- Redis Pub/Sub for message distribution
- Connection pooling and load balancing
- Heartbeat and reconnection logic

---

## 7. Algorithm Implementation Strategy

Based on [algorithms.md](src/components/ProfessionalChart/algorithms.md), the system plans to implement:

**Algorithms by Timeframe:**

| Timeframe | Algorithms | Status |
|-----------|-----------|--------|
| 1m-5m | XGBoost + LSTM | ❌ Not implemented |
| 10m-15m | LightGBM + CNN | ❌ Not implemented |
| 30m-1h | LSTM + Transformer | ❌ Not implemented |
| 4h-10h | Transformer + XGBoost | ❌ Not implemented |
| 1d | LSTM + Gradient Boosting + Meta-Ensemble | ❌ Not implemented |
| 7d-14d | Transformer + Random Forest | ❌ Not implemented |
| 1M | Reinforcement Learning + Transformer | ❌ Not implemented |

**Expected Improvements:**
- Accuracy: 60-76% (vs 50% baseline)
- Improvement vs traditional signals: +10% to +25%

**Implementation Requirements:**

1. **Data Pipeline:**
   - Historical data ingestion (1min to 1M bars)
   - Feature engineering (50+ technical indicators)
   - Data normalization and cleaning
   - Train/validation/test split

2. **Training Infrastructure:**
   - GPU servers for deep learning (LSTM, Transformers)
   - Distributed training for large datasets
   - Hyperparameter optimization (Optuna, Ray Tune)
   - Model versioning and artifact storage

3. **Feature Engineering:**
   - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
   - Volume-based features (VWAP, OBV)
   - Market microstructure (bid-ask spread, depth)
   - Time-based features (day of week, market session)
   - Sentiment analysis (news, social media)
   - Macro indicators (VIX, treasury yields)

4. **Model Training:**
   - Rolling window cross-validation
   - Walk-forward optimization
   - Out-of-sample testing
   - Ensemble methods (stacking, blending)

5. **Model Serving:**
   - Real-time inference (<100ms latency)
   - Batch prediction for backtesting
   - Model monitoring (drift detection)
   - A/B testing framework

6. **Backtesting Engine:**
   - Historical simulation with realistic slippage
   - Transaction cost modeling
   - Position sizing and risk management
   - Performance metrics (Sharpe, Sortino, max drawdown)

**Gap Analysis:**
- **Current State:** 0% implemented
- **Required Effort:** 6-12 months for full implementation
- **Team Required:** 2-3 ML engineers, 1 backend engineer
- **Infrastructure Cost:** $500-2000/month (GPU, cloud services)

---

## 8. Vulnerabilities Summary Table

| # | Severity | Category | Issue | Impact | Remediation Priority |
|---|----------|----------|-------|--------|---------------------|
| 1 | 🔴 CRITICAL | Security | API key exposed client-side | Credential leak, abuse | IMMEDIATE |
| 2 | 🔴 CRITICAL | Security | No authentication system | Cannot deploy to production | HIGH |
| 3 | 🟡 HIGH | Infrastructure | No backend exists | Core functionality missing | HIGH |
| 4 | 🟡 HIGH | Security | Client-side rate limiting | Bypassable, API abuse | MEDIUM |
| 5 | 🟡 MEDIUM | Performance | In-memory cache only | Poor scalability | MEDIUM |
| 6 | 🟡 MEDIUM | Security | No input validation | XSS, injection risks | MEDIUM |
| 7 | 🟡 MEDIUM | Security | No request signing | Request forgery (future) | LOW |
| 8 | 🟢 LOW | Operations | Poor error handling | Monitoring gaps | LOW |
| 9 | 🟢 LOW | Reliability | No retry logic | Transient failures | LOW |
| 10 | 🟢 LOW | Security | No HTTPS enforcement | MITM attacks (deployment) | LOW |

---

## 9. Recommended Next Steps

### Phase 1: Security Hardening (Week 1-2) 🔴 URGENT

1. **Immediate:**
   - [ ] Rotate exposed Polygon.io API key
   - [ ] Remove `NEXT_PUBLIC_` prefix from API key
   - [ ] Create Next.js API route `/api/market/proxy`
   - [ ] Move all Polygon.io calls to server-side

2. **Short-term:**
   - [ ] Implement NextAuth.js with email/password
   - [ ] Add JWT session management
   - [ ] Implement API route authentication
   - [ ] Add input validation (Zod schemas)
   - [ ] Add CORS policies

### Phase 2: Backend Foundation (Week 3-6)

1. **Setup Infrastructure:**
   - [ ] Choose backend framework (FastAPI recommended)
   - [ ] Setup PostgreSQL database
   - [ ] Setup Redis for caching/rate limiting
   - [ ] Create Docker Compose environment
   - [ ] Implement database migrations (Prisma/Alembic)

2. **Build Core API:**
   - [ ] Implement authentication endpoints
   - [ ] Create signal management endpoints
   - [ ] Create engine management endpoints
   - [ ] Implement market data proxy
   - [ ] Add report generation endpoints

3. **Testing & Documentation:**
   - [ ] Write API integration tests
   - [ ] Create OpenAPI/Swagger docs
   - [ ] Setup CI/CD pipeline
   - [ ] Add monitoring (Sentry, DataDog)

### Phase 3: ML Pipeline (Week 7-16)

1. **Data Pipeline:**
   - [ ] Setup data ingestion jobs
   - [ ] Implement feature engineering
   - [ ] Create data validation pipeline
   - [ ] Setup data versioning (DVC)

2. **Model Development:**
   - [ ] Implement baseline models (XGBoost)
   - [ ] Train LSTM models for time series
   - [ ] Experiment with Transformers
   - [ ] Create ensemble meta-models

3. **Model Serving:**
   - [ ] Setup TorchServe/TensorFlow Serving
   - [ ] Implement real-time inference API
   - [ ] Add model monitoring
   - [ ] Create A/B testing framework

### Phase 4: Trading Integration (Week 17-24)

1. **Broker Integration:**
   - [ ] Integrate Alpaca API (paper trading)
   - [ ] Implement order management system
   - [ ] Add position tracking
   - [ ] Implement risk management rules

2. **Portfolio Management:**
   - [ ] Create portfolio tracking system
   - [ ] Implement P&L calculation
   - [ ] Add performance analytics
   - [ ] Create user dashboards

3. **Production Readiness:**
   - [ ] Load testing
   - [ ] Security audit
   - [ ] Penetration testing
   - [ ] Documentation completion

---

## 10. Technology Recommendations

### Backend Stack

**Option 1: Python Stack (Recommended for ML-heavy)**
```
- FastAPI (async, high performance)
- PostgreSQL (relational data)
- Redis (cache, rate limiting, queues)
- Celery (background jobs)
- SQLAlchemy (ORM)
- Alembic (migrations)
- pytest (testing)
```

**Option 2: Node.js Stack (Recommended for speed-to-market)**
```
- NestJS (structured, TypeScript)
- PostgreSQL + Prisma (type-safe ORM)
- Redis (cache, queues)
- Bull (background jobs)
- Jest (testing)
```

### ML Stack

```
Core:
- Python 3.11+
- scikit-learn 1.3+
- XGBoost 2.0+
- LightGBM 4.0+
- PyTorch 2.0+ (LSTM, Transformers)

Experiment Tracking:
- MLflow or Weights & Biases

Model Serving:
- TorchServe or FastAPI
- ONNX Runtime (optimization)

Feature Store:
- Feast or custom solution

Data Processing:
- pandas, polars
- numpy, scipy
- ta-lib (technical analysis)
```

### Infrastructure

```
Containerization:
- Docker
- Docker Compose (local)
- Kubernetes (production)

CI/CD:
- GitHub Actions
- GitLab CI/CD

Monitoring:
- Sentry (errors)
- DataDog or Prometheus + Grafana (metrics)
- LogRocket or FullStory (user sessions)

Hosting:
- Vercel (frontend)
- AWS/GCP (backend, ML)
- Render or Railway (simpler option)
```

---

## 11. Cost Estimation

### Development Phase (6 months)

**Personnel:**
- 2 ML Engineers: $120k-180k/year × 2 = $120k-180k (6 months)
- 1 Backend Engineer: $100k-150k/year = $50k-75k (6 months)
- 1 Frontend Engineer (part-time): $100k-150k/year × 0.5 = $25k-37.5k (6 months)
- **Total Personnel: $195k-292.5k**

**Infrastructure:**
- GPU instances (training): $500-1500/month × 6 = $3k-9k
- Cloud hosting (dev/staging): $200-500/month × 6 = $1.2k-3k
- SaaS tools (MLflow, monitoring): $100-300/month × 6 = $600-1.8k
- **Total Infrastructure: $4.8k-13.8k**

**APIs & Data:**
- Polygon.io Starter Plan: $29/month × 6 = $174
- Testing & data costs: $500
- **Total APIs: $674**

**Grand Total (6 months): $200k-306k**

### Production Phase (Monthly)

**Infrastructure:**
- Frontend hosting (Vercel Pro): $20/month
- Backend servers (2 instances): $100-200/month
- Database (PostgreSQL): $50-100/month
- Redis cache: $30-50/month
- ML inference servers: $200-500/month
- **Subtotal: $400-870/month**

**APIs & Services:**
- Polygon.io: $29-99/month
- Monitoring (Sentry, DataDog): $50-150/month
- Error tracking: $29/month
- **Subtotal: $108-278/month**

**Monthly Operating Cost: $508-1,148/month**

---

## 12. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| API key abuse | HIGH | HIGH | Immediate rotation + server-side proxy |
| Data breach | MEDIUM | CRITICAL | Implement authentication + encryption |
| ML models underperform | MEDIUM | HIGH | Extensive backtesting + gradual rollout |
| Regulatory compliance | LOW | CRITICAL | Consult legal, add disclaimers |
| Broker API downtime | MEDIUM | HIGH | Fallback brokers + circuit breakers |
| Database failure | LOW | CRITICAL | Backups + replication + monitoring |
| Cost overruns | MEDIUM | MEDIUM | Budget tracking + alerts |
| Talent acquisition | MEDIUM | MEDIUM | Competitive salaries + equity |

---

## 13. Conclusion

**Current State:**
The MVP Trading App has a solid frontend foundation with professional-grade charting and a well-structured React/TypeScript codebase. However, critical backend infrastructure is missing, and severe security vulnerabilities exist (exposed API keys, no authentication).

**Key Findings:**
- ✅ Strong UI/UX and data visualization
- ✅ Clean code architecture and type safety
- ❌ No backend or database
- ❌ No authentication or authorization
- ❌ API keys exposed to public
- ❌ ML algorithms documented but not implemented
- ❌ No trading execution capability

**Assessment:**
This is a **proof-of-concept frontend** that demonstrates the UI vision but is **not production-ready**. Approximately **20% of the full system is built** (frontend only).

**Priority Actions:**
1. 🔴 **IMMEDIATE:** Secure API keys and implement server-side proxy
2. 🔴 **URGENT:** Build authentication system
3. 🟡 **HIGH:** Develop backend API and database
4. 🟡 **MEDIUM:** Implement ML training pipeline
5. 🟢 **LOW:** Add broker integration and live trading

**Timeline to Production:**
- Minimum viable security: 2 weeks
- Backend infrastructure: 4-6 weeks
- ML pipeline: 12-16 weeks
- Full production system: 6-9 months

**Recommendation:**
Focus on security hardening and backend infrastructure before investing heavily in ML development. The current system cannot safely handle user data or real trading activity.

---

**Document End**

*For questions or clarifications, contact the development team.*
