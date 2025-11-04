# Week 1: Database & Infrastructure Setup - Implementation Plan

**Importance: 🔴 CRITICAL (10/10)**

**Timeline:** 5 days (can be compressed to 3 days with managed database)

---

## Executive Summary

Week 1 establishes the **foundational data infrastructure** for the ML trading prediction system. This week transforms your current frontend-focused Next.js application into a full-stack ML-ready platform with robust data persistence, time-series optimization, and future trading bot integration capabilities.

**Current State:**
- ✅ Next.js 15 frontend with TypeScript
- ✅ Polygon.io integration for real-time market data
- ✅ React-based charting (lightweight-charts, recharts)
- ✅ Zustand for state management
- ❌ No database (all data is ephemeral)
- ❌ No ML prediction storage
- ❌ No historical data persistence
- ❌ No API for external system integration

**Week 1 Goals:**
- ✅ PostgreSQL + TimescaleDB setup (time-series optimization)
- ✅ Complete database schema for ML system
- ✅ Data persistence layer (ORM/query builder)
- ✅ API routes for ML predictions and trading bot integration
- ✅ Performance benchmarks met (100K inserts <5s, 1yr query <500ms)

---

## Current System Architecture Analysis

### Tech Stack
```
Frontend:
├── Next.js 15.5.3 (App Router)
├── React 19.1.1
├── TypeScript 4.9.5
├── Zustand (state management)
└── Tailwind CSS 4.x

Data Sources:
├── Polygon.io REST API (@polygon.io/client-js)
└── In-memory caching (30s cache, rate limiting)

UI Components:
├── lightweight-charts (candlestick charts)
├── recharts (statistical charts)
└── lucide-react (icons)
```

### Current Data Flow
```
User Request → Polygon Service → API Call → Cache → Component State → UI
                     ↓
              Rate Limiting (13s free tier)
              Exponential Backoff (429 handling)
              In-memory cache (30s TTL)
```

### Key Files
- **`src/services/polygonService.ts`** - Market data fetching, caching, rate limiting
- **`src/types/trading.ts`** - Trading interfaces (Order, Trade, Asset, ChartData)
- **`src/types/polygon.ts`** - Polygon.io data types
- **`app/dashboard/page.tsx`** - Main dashboard
- **`app/ticker/[symbol]/page.tsx`** - Individual ticker views

### Current Limitations (Blockers for ML System)
1. **No Data Persistence** - Cannot store historical data for training
2. **No Feature Storage** - Cannot cache computed technical indicators
3. **No Prediction Tracking** - Cannot measure model accuracy over time
4. **No Model Registry** - Cannot version or deploy trained models
5. **No Trading Bot Interface** - Cannot programmatically access predictions
6. **Performance Constraints** - Polygon.io free tier (5 calls/min) too slow for backtesting

---

## Week 1 Architecture Enhancements

### New System Architecture (Post-Week 1)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                              │
├─────────────────────────────────────────────────────────────────────┤
│  Next.js Frontend  │  Trading Bot API  │  ML Training Scripts       │
└──────────┬──────────┴──────────┬────────┴──────────┬────────────────┘
           │                     │                   │
           ▼                     ▼                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      API LAYER (Next.js API Routes)                  │
├─────────────────────────────────────────────────────────────────────┤
│  /api/market-data   │  /api/predictions   │  /api/trading          │
│  - Historical bars  │  - Get predictions  │  - Execute orders      │
│  - Real-time data   │  - Track accuracy   │  - Get signals         │
│  - Features         │  - Model metadata   │  - Portfolio status    │
└──────────┬──────────┴──────────┬────────┴──────────┬────────────────┘
           │                     │                   │
           ▼                     ▼                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA ACCESS LAYER (Prisma ORM)                    │
├─────────────────────────────────────────────────────────────────────┤
│  marketDataRepository  │  predictionRepository  │  tradingRepository │
└──────────┬─────────────┴────────────┬────────────┴──────────┬────────┘
           │                          │                       │
           ▼                          ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                PostgreSQL + TimescaleDB (Cloud/Local)                │
├─────────────────────────────────────────────────────────────────────┤
│  market_data (hypertable)     │  predictions (hypertable)           │
│  - OHLCV bars (1m-1M)         │  - ML predictions with confidence   │
│  - 2+ years history           │  - Actual outcomes (accuracy)       │
│  - Continuous compression     │  - Model versioning                 │
│                               │                                     │
│  features (hypertable)        │  models                             │
│  - 100+ technical indicators  │  - Model metadata                   │
│  - RSI, MACD, Bollinger, etc. │  - Training metrics                 │
│  - Pre-computed for speed     │  - Deployment status                │
│                               │                                     │
│  trades (normal table)        │  portfolio (normal table)           │
│  - Executed trades            │  - Current positions                │
│  - P&L tracking               │  - Balance history                  │
└─────────────────────────────────────────────────────────────────────┘
           ▲                          ▲
           │                          │
┌──���───────┴──────────┐    ┌──────────┴──────────┐
│  Data Ingestion     │    │  ML Training        │
│  (Polygon.io ETL)   │    │  (Python/Node.js)   │
│  - Backfill history │    │  - XGBoost          │
│  - Real-time stream │    │  - LSTM             │
└─────────────────────┘    └─────────────────────┘
```

### Key Enhancements
1. **TimescaleDB Hypertables** - 10x faster time-series queries
2. **API Layer** - RESTful endpoints for trading bots and ML services
3. **Data Access Layer** - Type-safe database operations with Prisma
4. **Separation of Concerns** - Market data, predictions, trading logic isolated
5. **Extensibility** - Easy to add new data sources or ML models

---

## Database Schema Design

### Core Principles
- **Time-series first** - All market data uses TimescaleDB hypertables
- **Type safety** - Prisma schema ensures compile-time type checking
- **Denormalization** - Pre-compute features for fast inference
- **Audit trail** - Track all predictions for accuracy measurement
- **Flexibility** - Schema supports multiple tickers, timeframes, models

### Schema Overview

```prisma
// FILE: prisma/schema.prisma

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

generator client {
  provider = "prisma-client-js"
}

// ============================================================================
// MARKET DATA TABLES (TimescaleDB Hypertables)
// ============================================================================

model MarketData {
  id         String   @id @default(uuid())
  ticker     String   // SPY, QQQ, IWM, UVXY
  timeframe  String   // 1m, 5m, 15m, 1h, 4h, 1d
  timestamp  DateTime // Bar open time (indexed for time-series)

  // OHLCV data
  open       Float
  high       Float
  low        Float
  close      Float
  volume     Float

  // Metadata
  source     String   @default("polygon") // polygon, alpaca, etc.
  createdAt  DateTime @default(now())

  @@unique([ticker, timeframe, timestamp])
  @@index([ticker, timeframe, timestamp(sort: Desc)])
  @@index([timestamp])
  @@map("market_data")
}

// ============================================================================
// FEATURE ENGINEERING (Pre-computed Technical Indicators)
// ============================================================================

model Feature {
  id         String   @id @default(uuid())
  ticker     String
  timeframe  String
  timestamp  DateTime

  // Moving Averages
  sma_20     Float?
  sma_50     Float?
  sma_200    Float?
  ema_12     Float?
  ema_26     Float?

  // Momentum Indicators
  rsi_14     Float?
  macd       Float?
  macd_signal Float?
  macd_hist  Float?
  stoch_k    Float?
  stoch_d    Float?

  // Volatility Indicators
  bb_upper   Float?
  bb_middle  Float?
  bb_lower   Float?
  atr_14     Float?

  // Volume Indicators
  volume_sma_20 Float?
  obv        Float?

  // Price Action
  high_low_ratio Float?
  close_open_ratio Float?

  // Additional Features (100+ total, expandable)
  features   Json?    // Store additional features as JSON for flexibility

  createdAt  DateTime @default(now())

  @@unique([ticker, timeframe, timestamp])
  @@index([ticker, timeframe, timestamp(sort: Desc)])
  @@map("features")
}

// ============================================================================
// ML PREDICTIONS & TRACKING
// ============================================================================

model Prediction {
  id              String   @id @default(uuid())
  ticker          String
  timeframe       String   // Prediction horizon (1m, 5m, 1h, 1d)
  modelId         String   // Reference to trained model
  modelVersion    String   // v1.0.0, v1.1.0, etc.

  // Prediction Details
  predictionTime  DateTime // When prediction was made
  targetTime      DateTime // When prediction is for (predictionTime + horizon)
  direction       String   // UP, DOWN, NEUTRAL
  confidence      Float    // 0.0 to 1.0
  probability     Float    // Predicted probability of direction

  // Price Predictions
  predictedPrice  Float?
  predictedChange Float?   // Predicted % change

  // Actual Outcome (filled later)
  actualDirection String?  // UP, DOWN, NEUTRAL
  actualPrice     Float?
  actualChange    Float?
  correct         Boolean? // Was prediction correct?

  // Metadata
  features        Json?    // Features used for this prediction
  modelOutput     Json?    // Raw model output for debugging

  createdAt       DateTime @default(now())
  updatedAt       DateTime @updatedAt

  model           Model    @relation(fields: [modelId], references: [id])

  @@index([ticker, timeframe, predictionTime(sort: Desc)])
  @@index([modelId, predictionTime])
  @@index([targetTime]) // For looking up predictions to verify
  @@map("predictions")
}

// ============================================================================
// MODEL REGISTRY
// ============================================================================

model Model {
  id              String   @id @default(uuid())
  name            String   // "SPY_1m_XGBoost_v1"
  ticker          String
  timeframe       String
  algorithm       String   // XGBoost, LSTM, Ensemble
  version         String   // Semantic versioning

  // Training Metadata
  trainedAt       DateTime
  trainingDataFrom DateTime
  trainingDataTo   DateTime
  trainingRows    Int

  // Performance Metrics
  accuracy        Float?   // Test set accuracy
  precision       Float?
  recall          Float?
  f1Score         Float?
  sharpeRatio     Float?   // From backtesting
  maxDrawdown     Float?

  // Hyperparameters
  hyperparameters Json

  // Feature Importance
  featureImportance Json?

  // Deployment
  deployedAt      DateTime?
  isActive        Boolean  @default(false)
  modelPath       String?  // S3/local path to serialized model

  // Relations
  predictions     Prediction[]

  createdAt       DateTime @default(now())
  updatedAt       DateTime @updatedAt

  @@unique([ticker, timeframe, algorithm, version])
  @@index([isActive, ticker, timeframe])
  @@map("models")
}

// ============================================================================
// TRADING & PORTFOLIO (Future Trading Bot Integration)
// ============================================================================

model Trade {
  id              String   @id @default(uuid())
  ticker          String
  side            String   // BUY, SELL
  type            String   // MARKET, LIMIT

  // Execution Details
  quantity        Float
  price           Float
  totalValue      Float
  fees            Float    @default(0)

  // Timestamps
  orderTime       DateTime
  executionTime   DateTime?

  // Status
  status          String   // PENDING, FILLED, CANCELED, FAILED

  // Strategy Context
  predictionId    String?  // Link to ML prediction that triggered trade
  strategy        String?  // "ML_Signal", "Manual", "Hedge", etc.

  // P&L (calculated after exit)
  exitPrice       Float?
  exitTime        DateTime?
  profitLoss      Float?
  profitLossPct   Float?

  createdAt       DateTime @default(now())
  updatedAt       DateTime @updatedAt

  @@index([ticker, orderTime(sort: Desc)])
  @@index([status, orderTime])
  @@map("trades")
}

model Portfolio {
  id              String   @id @default(uuid())
  ticker          String   @unique

  // Position
  quantity        Float    @default(0)
  avgEntryPrice   Float?
  currentPrice    Float?
  marketValue     Float?

  // P&L
  unrealizedPnL   Float?
  unrealizedPnLPct Float?
  realizedPnL     Float    @default(0)

  // Risk
  stopLoss        Float?
  takeProfit      Float?

  updatedAt       DateTime @updatedAt

  @@map("portfolio")
}

// ============================================================================
// SYSTEM METADATA
// ============================================================================

model DataIngestionLog {
  id              String   @id @default(uuid())
  ticker          String
  timeframe       String
  source          String   // polygon, alpaca, etc.

  fromDate        DateTime
  toDate          DateTime
  rowsIngested    Int

  status          String   // SUCCESS, PARTIAL, FAILED
  errorMessage    String?

  createdAt       DateTime @default(now())

  @@index([ticker, timeframe, createdAt(sort: Desc)])
  @@map("data_ingestion_logs")
}

model SystemMetric {
  id              String   @id @default(uuid())
  metricName      String
  metricValue     Float
  metadata        Json?
  timestamp       DateTime @default(now())

  @@index([metricName, timestamp(sort: Desc)])
  @@map("system_metrics")
}
```

### TimescaleDB Hypertable Configuration

After schema creation, we'll convert time-series tables to hypertables:

```sql
-- Convert market_data to hypertable (partitioned by timestamp)
SELECT create_hypertable('market_data', 'timestamp',
  chunk_time_interval => INTERVAL '1 week',
  if_not_exists => TRUE
);

-- Convert features to hypertable
SELECT create_hypertable('features', 'timestamp',
  chunk_time_interval => INTERVAL '1 week',
  if_not_exists => TRUE
);

-- Convert predictions to hypertable
SELECT create_hypertable('predictions', 'predictionTime',
  chunk_time_interval => INTERVAL '1 month',
  if_not_exists => TRUE
);

-- Add compression policy (compress data older than 7 days)
SELECT add_compression_policy('market_data', INTERVAL '7 days');
SELECT add_compression_policy('features', INTERVAL '7 days');

-- Add retention policy (optional - keep last 5 years)
SELECT add_retention_policy('market_data', INTERVAL '5 years');

-- Create continuous aggregates for fast queries
CREATE MATERIALIZED VIEW daily_market_summary
WITH (timescaledb.continuous) AS
SELECT
  ticker,
  time_bucket('1 day', timestamp) AS day,
  first(open, timestamp) AS open,
  max(high) AS high,
  min(low) AS low,
  last(close, timestamp) AS close,
  sum(volume) AS volume
FROM market_data
WHERE timeframe = '1m'
GROUP BY ticker, day;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('daily_market_summary',
  start_offset => INTERVAL '1 month',
  end_offset => INTERVAL '1 hour',
  schedule_interval => INTERVAL '1 hour'
);
```

---

## API Layer Design (Trading Bot Integration)

### API Routes Structure

```
app/api/
├── market-data/
│   ├── route.ts                    # GET /api/market-data?ticker=SPY&timeframe=1h&limit=100
│   ├── bulk/route.ts               # POST /api/market-data/bulk (batch insert)
│   └── features/route.ts           # GET /api/market-data/features
│
├── predictions/
│   ├── route.ts                    # GET /api/predictions?ticker=SPY&timeframe=1h
│   ├── latest/route.ts             # GET /api/predictions/latest (current signals)
│   ├── accuracy/route.ts           # GET /api/predictions/accuracy (rolling metrics)
│   └── webhook/route.ts            # POST /api/predictions/webhook (real-time alerts)
│
├── models/
│   ├── route.ts                    # GET /api/models (list active models)
│   ├── [id]/route.ts               # GET /api/models/:id (model details)
│   └── deploy/route.ts             # POST /api/models/deploy (activate model)
│
├── trading/
│   ├── signals/route.ts            # GET /api/trading/signals (actionable signals)
│   ├── execute/route.ts            # POST /api/trading/execute (place trade)
│   ├── portfolio/route.ts          # GET /api/trading/portfolio
│   └── history/route.ts            # GET /api/trading/history
│
└── health/
    └── route.ts                    # GET /api/health (system status)
```

### Example API Route Implementation

```typescript
// FILE: app/api/predictions/latest/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const ticker = searchParams.get('ticker');
    const timeframe = searchParams.get('timeframe');

    // Get latest active predictions
    const predictions = await prisma.prediction.findMany({
      where: {
        ticker: ticker || undefined,
        timeframe: timeframe || undefined,
        model: {
          isActive: true,
        },
        targetTime: {
          gte: new Date(), // Only future predictions
        },
      },
      include: {
        model: {
          select: {
            name: true,
            algorithm: true,
            accuracy: true,
            version: true,
          },
        },
      },
      orderBy: {
        predictionTime: 'desc',
      },
      take: 20,
    });

    return NextResponse.json({
      success: true,
      count: predictions.length,
      predictions,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error('Error fetching latest predictions:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to fetch predictions' },
      { status: 500 }
    );
  }
}
```

### Trading Bot Integration Example

```typescript
// External trading bot can consume predictions like this:

const response = await fetch('https://your-app.com/api/predictions/latest?ticker=SPY&timeframe=1h', {
  headers: {
    'Authorization': `Bearer ${API_KEY}`,
  },
});

const { predictions } = await response.json();

// Filter high-confidence signals
const strongSignals = predictions.filter(p =>
  p.confidence > 0.75 &&
  p.model.accuracy > 0.65
);

// Execute trades based on signals
for (const signal of strongSignals) {
  if (signal.direction === 'UP') {
    await executeBuyOrder(signal.ticker, signal.confidence);
  }
}
```

---

## Implementation Steps (Day-by-Day Breakdown)

### Day 1: Database Setup & Schema Design
**Duration:** 8 hours (or 2 hours with Supabase)

**Tasks:**
1. ✅ Choose database hosting (Supabase recommended for speed)
2. ✅ Install PostgreSQL + TimescaleDB extension
3. ✅ Set up environment variables (`.env.local`)
4. ✅ Install Prisma: `npm install prisma @prisma/client`
5. ✅ Create Prisma schema (`prisma/schema.prisma`)
6. ✅ Initialize Prisma: `npx prisma init`
7. ✅ Generate Prisma client: `npx prisma generate`
8. ✅ Run migrations: `npx prisma migrate dev --name init`

**Deliverables:**
- ✅ Database running (local or Supabase)
- ✅ Prisma schema created
- ✅ All tables created with proper indexes
- ✅ TimescaleDB hypertables configured

**Decision Point: Database Hosting**

| Option | Pros | Cons | Cost | Setup Time |
|--------|------|------|------|------------|
| **Supabase** (Recommended) | Managed PostgreSQL, built-in TimescaleDB, free tier, auto backups, real-time subscriptions | Vendor lock-in (easy to migrate) | Free (up to 500MB), then $25/mo | 30 min |
| **Railway** | Easy deploy, automatic scaling, good for production | Starts at $5/mo, no free tier | $5-20/mo | 1 hour |
| **Local PostgreSQL** | Full control, free, good for development | Manual setup, no auto backups, not production-ready | Free | 4-6 hours |
| **AWS RDS** | Enterprise-grade, high scalability | Complex setup, expensive | $30+/mo | 4-6 hours |

**Recommendation:** Start with **Supabase** for Week 1-4, migrate to Railway/AWS for production deployment in Week 12.

---

### Day 2: Data Access Layer & ORM Setup
**Duration:** 6 hours

**Tasks:**
1. ✅ Create Prisma client singleton (`lib/prisma.ts`)
2. ✅ Build repository pattern for data access
3. ✅ Create TypeScript interfaces aligned with Prisma schema
4. ✅ Write database utility functions
5. ✅ Set up connection pooling for performance

**File Structure:**
```
lib/
├── prisma.ts              # Prisma client singleton
├── repositories/
│   ├── marketData.ts      # CRUD for market data
│   ├── features.ts        # Feature storage/retrieval
│   ├── predictions.ts     # Prediction tracking
│   ├── models.ts          # Model registry
│   └── trading.ts         # Trades & portfolio
└── utils/
    ├── database.ts        # DB utilities
    └── validation.ts      # Input validation
```

**Example: Market Data Repository**

```typescript
// FILE: lib/repositories/marketData.ts

import { prisma } from '@/lib/prisma';
import { Prisma } from '@prisma/client';

export class MarketDataRepository {
  /**
   * Insert bulk market data (optimized for speed)
   */
  async bulkInsert(data: Prisma.MarketDataCreateManyInput[]) {
    const startTime = Date.now();

    const result = await prisma.marketData.createMany({
      data,
      skipDuplicates: true, // Avoid errors on duplicate timestamps
    });

    const duration = Date.now() - startTime;
    console.log(`[MarketData] Inserted ${result.count} rows in ${duration}ms`);

    return result;
  }

  /**
   * Get historical bars (optimized query)
   */
  async getHistoricalBars(
    ticker: string,
    timeframe: string,
    limit: number = 100,
    endDate?: Date
  ) {
    return prisma.marketData.findMany({
      where: {
        ticker,
        timeframe,
        timestamp: endDate ? { lte: endDate } : undefined,
      },
      orderBy: {
        timestamp: 'desc',
      },
      take: limit,
    });
  }

  /**
   * Get latest bar for a ticker
   */
  async getLatest(ticker: string, timeframe: string) {
    return prisma.marketData.findFirst({
      where: { ticker, timeframe },
      orderBy: { timestamp: 'desc' },
    });
  }

  /**
   * Check data completeness (for validation)
   */
  async getDataGaps(ticker: string, timeframe: string, fromDate: Date, toDate: Date) {
    // Use raw SQL for complex time-series analysis
    return prisma.$queryRaw`
      SELECT
        ticker,
        timeframe,
        COUNT(*) as total_bars,
        MIN(timestamp) as earliest,
        MAX(timestamp) as latest
      FROM market_data
      WHERE ticker = ${ticker}
        AND timeframe = ${timeframe}
        AND timestamp BETWEEN ${fromDate} AND ${toDate}
      GROUP BY ticker, timeframe
    `;
  }
}

export const marketDataRepo = new MarketDataRepository();
```

**Deliverables:**
- ✅ Prisma client configured
- ✅ Repository classes for all tables
- ✅ Type-safe database operations
- ✅ Connection pooling enabled

---

### Day 3: Historical Data Ingestion (ETL Pipeline)
**Duration:** 8 hours

**Tasks:**
1. ✅ Create data ingestion script
2. ✅ Backfill 2 years of historical data for SPY (test ticker)
3. ✅ Validate data completeness (no gaps)
4. ✅ Benchmark insert performance (target: 100K rows <5s)
5. ✅ Set up automated data refresh (cron job or Next.js API route)

**ETL Script Example:**

```typescript
// FILE: scripts/backfill-market-data.ts

import { polygonService } from '@/src/services/polygonService';
import { marketDataRepo } from '@/lib/repositories/marketData';
import { Prisma } from '@prisma/client';

interface BackfillConfig {
  tickers: string[];
  timeframes: string[];
  yearsBack: number;
  batchSize: number;
}

async function backfillMarketData(config: BackfillConfig) {
  const { tickers, timeframes, yearsBack, batchSize } = config;

  for (const ticker of tickers) {
    for (const timeframe of timeframes) {
      console.log(`\n[Backfill] Starting ${ticker} ${timeframe}...`);

      const endDate = new Date();
      const startDate = new Date();
      startDate.setFullYear(startDate.getFullYear() - yearsBack);

      let currentDate = new Date(startDate);
      let totalInserted = 0;

      while (currentDate < endDate) {
        // Fetch data in chunks (Polygon.io limit: 50K per request)
        const nextDate = new Date(currentDate);
        nextDate.setDate(nextDate.getDate() + 30); // 30-day chunks

        try {
          const bars = await polygonService.getAggregates(
            ticker,
            timeframe as any,
            10000 // Large limit to get all data in date range
          );

          // Transform to Prisma format
          const records: Prisma.MarketDataCreateManyInput[] = bars.map(bar => ({
            ticker,
            timeframe,
            timestamp: new Date(bar.time),
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
            source: 'polygon',
          }));

          // Bulk insert
          if (records.length > 0) {
            const result = await marketDataRepo.bulkInsert(records);
            totalInserted += result.count;
            console.log(`[Backfill] Inserted ${result.count} bars (total: ${totalInserted})`);
          }

          currentDate = nextDate;

          // Rate limiting (respect Polygon.io limits)
          await new Promise(resolve => setTimeout(resolve, 13000));

        } catch (error) {
          console.error(`[Backfill] Error fetching ${ticker} ${timeframe}:`, error);
          // Log error and continue
          await prisma.dataIngestionLog.create({
            data: {
              ticker,
              timeframe,
              source: 'polygon',
              fromDate: currentDate,
              toDate: nextDate,
              rowsIngested: 0,
              status: 'FAILED',
              errorMessage: error.message,
            },
          });
        }
      }

      console.log(`[Backfill] Completed ${ticker} ${timeframe}: ${totalInserted} total bars`);
    }
  }
}

// Run backfill
backfillMarketData({
  tickers: ['SPY'], // Start with SPY for Week 3
  timeframes: ['1m', '5m', '15m', '1h', '4h', '1d'],
  yearsBack: 2,
  batchSize: 10000,
}).then(() => {
  console.log('\n✅ Backfill complete!');
  process.exit(0);
}).catch(error => {
  console.error('\n❌ Backfill failed:', error);
  process.exit(1);
});
```

**Performance Optimization:**
```typescript
// Use Prisma's batch insert for maximum speed
await prisma.$transaction(
  records.map(record => prisma.marketData.create({ data: record })),
  { timeout: 60000 }
);

// Alternative: Use raw SQL for even faster inserts
await prisma.$executeRaw`
  INSERT INTO market_data (ticker, timeframe, timestamp, open, high, low, close, volume, source)
  VALUES ${Prisma.join(records.map(r => Prisma.sql`(${r.ticker}, ${r.timeframe}, ${r.timestamp}, ${r.open}, ${r.high}, ${r.low}, ${r.close}, ${r.volume}, ${r.source})`))}
  ON CONFLICT (ticker, timeframe, timestamp) DO NOTHING
`;
```

**Deliverables:**
- ✅ 2+ years of SPY data stored (500K+ bars per timeframe)
- ✅ Data ingestion logs tracked
- ✅ No data gaps validated
- ✅ Insert performance: 100K rows in <5 seconds ✅

---

### Day 4: API Routes & Trading Bot Integration
**Duration:** 6 hours

**Tasks:**
1. ✅ Create Next.js API routes (market-data, predictions, trading)
2. ✅ Implement authentication (API keys for trading bots)
3. ✅ Add rate limiting (protect endpoints)
4. ✅ Write API documentation (OpenAPI/Swagger)
5. ✅ Test endpoints with Postman/cURL

**API Route Example:**

```typescript
// FILE: app/api/market-data/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { marketDataRepo } from '@/lib/repositories/marketData';
import { z } from 'zod';

// Input validation schema
const querySchema = z.object({
  ticker: z.string().min(1).max(10),
  timeframe: z.enum(['1m', '5m', '15m', '1h', '4h', '1d']),
  limit: z.coerce.number().min(1).max(1000).default(100),
  endDate: z.coerce.date().optional(),
});

export async function GET(request: NextRequest) {
  try {
    // Parse query params
    const { searchParams } = new URL(request.url);
    const params = {
      ticker: searchParams.get('ticker'),
      timeframe: searchParams.get('timeframe'),
      limit: searchParams.get('limit'),
      endDate: searchParams.get('endDate'),
    };

    // Validate input
    const validated = querySchema.parse(params);

    // Fetch data
    const data = await marketDataRepo.getHistoricalBars(
      validated.ticker,
      validated.timeframe,
      validated.limit,
      validated.endDate
    );

    return NextResponse.json({
      success: true,
      ticker: validated.ticker,
      timeframe: validated.timeframe,
      count: data.length,
      data,
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { success: false, error: 'Invalid parameters', details: error.errors },
        { status: 400 }
      );
    }

    console.error('[API] Error:', error);
    return NextResponse.json(
      { success: false, error: 'Internal server error' },
      { status: 500 }
    );
  }
}
```

**Authentication Middleware:**

```typescript
// FILE: lib/middleware/auth.ts

import { NextRequest, NextResponse } from 'next/server';

const API_KEYS = new Set([
  process.env.TRADING_BOT_API_KEY,
  process.env.ML_SERVICE_API_KEY,
]);

export function withAuth(handler: (req: NextRequest) => Promise<NextResponse>) {
  return async (req: NextRequest) => {
    const apiKey = req.headers.get('Authorization')?.replace('Bearer ', '');

    if (!apiKey || !API_KEYS.has(apiKey)) {
      return NextResponse.json(
        { success: false, error: 'Unauthorized' },
        { status: 401 }
      );
    }

    return handler(req);
  };
}
```

**Deliverables:**
- ✅ API routes deployed
- ✅ Authentication working
- ✅ Rate limiting configured
- ✅ API documentation created

---

### Day 5: Performance Benchmarking & Validation
**Duration:** 6 hours

**Tasks:**
1. ✅ Run performance benchmarks
2. ✅ Validate success criteria
3. ✅ Set up monitoring (optional: Grafana/Prometheus)
4. ✅ Create Week 1 summary report
5. ✅ Prepare for Week 2 (feature engineering setup)

**Benchmark Script:**

```typescript
// FILE: scripts/benchmark-database.ts

import { marketDataRepo } from '@/lib/repositories/marketData';
import { performance } from 'perf_hooks';

async function runBenchmarks() {
  console.log('Starting database performance benchmarks...\n');

  // Test 1: Bulk insert speed (100K rows)
  console.log('Test 1: Bulk Insert (100K rows)');
  const insertData = Array.from({ length: 100000 }, (_, i) => ({
    ticker: 'BENCH',
    timeframe: '1m',
    timestamp: new Date(Date.now() - i * 60000),
    open: 100 + Math.random() * 10,
    high: 105 + Math.random() * 10,
    low: 95 + Math.random() * 10,
    close: 100 + Math.random() * 10,
    volume: 1000000 + Math.random() * 100000,
    source: 'benchmark',
  }));

  const insertStart = performance.now();
  await marketDataRepo.bulkInsert(insertData);
  const insertDuration = performance.now() - insertStart;
  console.log(`✅ Inserted 100K rows in ${insertDuration.toFixed(0)}ms (${insertDuration < 5000 ? 'PASS' : 'FAIL'})\n`);

  // Test 2: Query speed (1 year of 1m data = ~100K rows)
  console.log('Test 2: Query 1 Year of Data');
  const queryStart = performance.now();
  const oneYearAgo = new Date();
  oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);

  const data = await marketDataRepo.getHistoricalBars('SPY', '1m', 100000, new Date());
  const queryDuration = performance.now() - queryStart;
  console.log(`✅ Queried ${data.length} rows in ${queryDuration.toFixed(0)}ms (${queryDuration < 500 ? 'PASS' : 'FAIL'})\n`);

  // Test 3: Latest bar query (should be <10ms)
  console.log('Test 3: Latest Bar Query');
  const latestStart = performance.now();
  await marketDataRepo.getLatest('SPY', '1m');
  const latestDuration = performance.now() - latestStart;
  console.log(`✅ Latest bar query in ${latestDuration.toFixed(0)}ms (${latestDuration < 10 ? 'PASS' : 'FAIL'})\n`);

  console.log('Benchmarks complete!');
}

runBenchmarks().then(() => process.exit(0));
```

**Success Criteria Checklist:**
- ✅ PostgreSQL + TimescaleDB running
- ✅ All tables created with proper indexes
- ✅ Can insert 100K rows in <5 seconds
- ✅ Can query 1 year of data in <500ms
- ✅ 2+ years of SPY data stored
- ✅ API routes working
- ✅ Trading bot can fetch predictions via API

---

## Environment Variables Setup

Create `.env.local`:

```env
# Database
DATABASE_URL="postgresql://user:password@localhost:5432/trading_ml?schema=public"
# For Supabase: "postgresql://postgres:[password]@db.[project].supabase.co:5432/postgres"

# Polygon.io
NEXT_PUBLIC_POLYGON_API_KEY="your_polygon_api_key"
NEXT_PUBLIC_POLYGON_PLAN="free"  # or "starter", "developer"

# API Authentication
TRADING_BOT_API_KEY="your_secure_random_key_here"
ML_SERVICE_API_KEY="another_secure_random_key"

# Optional: Monitoring
SENTRY_DSN="your_sentry_dsn"
```

---

## Risk Mitigation & Contingency Plans

### Risk 1: TimescaleDB Not Available on Hosting Provider
**Likelihood:** Low (Supabase/Railway support it)
**Impact:** Medium (slower queries, but system still works)
**Mitigation:**
- Use PostgreSQL without TimescaleDB for Week 1
- Manually create indexes for timestamp-based queries
- Add TimescaleDB later when migrating to production

### Risk 2: Polygon.io Rate Limits Too Restrictive
**Likelihood:** High (free tier = 5 calls/min)
**Impact:** High (data ingestion takes 10+ hours)
**Mitigation:**
- Run backfill over multiple days
- Upgrade to Polygon Starter plan ($29/mo, unlimited historical)
- Use alternative data source (Alpha Vantage, Yahoo Finance)

### Risk 3: Database Insert Performance Below Target
**Likelihood:** Medium
**Impact:** Medium (longer ingestion times)
**Mitigation:**
- Use raw SQL instead of Prisma for bulk inserts
- Disable indexes during bulk insert, re-enable after
- Use PostgreSQL `COPY` command for maximum speed

---

## Week 1 Deliverables Summary

### Technical Deliverables
- ✅ PostgreSQL + TimescaleDB database (cloud or local)
- ✅ Prisma schema with all tables
- ✅ Data access layer (repositories)
- ✅ API routes for trading bot integration
- ✅ 2+ years of SPY historical data
- ✅ Performance benchmarks passed

### Documentation Deliverables
- ✅ Database schema documentation
- ✅ API endpoint documentation
- ✅ Environment setup guide
- ✅ Week 1 summary report

### Validation Criteria
- ✅ Can insert 100K rows in <5 seconds
- ✅ Can query 1 year of data in <500ms
- ✅ No data gaps in historical data
- ✅ API returns predictions in <100ms
- ✅ Trading bot can authenticate and fetch data

---

## Next Steps (Week 2 Preview)

Week 2 focuses on **feature engineering** - transforming raw OHLCV data into 100+ technical indicators:

1. **Compute Features:**
   - Moving averages (SMA, EMA)
   - Momentum indicators (RSI, MACD, Stochastic)
   - Volatility indicators (Bollinger Bands, ATR)
   - Volume indicators (OBV, Volume SMA)
   - Price action features

2. **Store Features:**
   - Populate `features` table
   - Optimize for fast retrieval during inference

3. **Data Quality:**
   - Validate feature calculations
   - Handle missing data
   - Normalize/standardize features

**Preparation for Week 2:**
- Install Python (for TA-Lib technical indicators library)
- Review feature engineering best practices
- Identify most predictive features for SPY

---

## Conclusion

Week 1 transforms your application from a **frontend prototype** to a **production-ready ML platform** with:
- ✅ Persistent data storage (PostgreSQL + TimescaleDB)
- ✅ Scalable time-series architecture
- ✅ API layer for trading bot integration
- ✅ 2+ years of historical data
- ✅ Type-safe database operations
- ✅ Performance-optimized queries

This foundation enables **Week 2 (feature engineering)**, **Week 3 (ML training)**, and **Week 11 (feedback loops)** - the critical path to a working ML trading system.

**Estimated Time:** 5 days (40 hours) or 3 days compressed with Supabase

**Recommended Approach:**
1. Use **Supabase** for Week 1-4 (fast setup, managed database)
2. Backfill SPY data first (test case for Week 3)
3. Build API routes early (enables trading bot development in parallel)
4. Run benchmarks daily to catch performance issues early

**Go/No-Go Decision:**
At end of Week 1, you should have:
- ✅ Database with 2+ years of SPY data
- ✅ API returning data in <100ms
- ✅ No data gaps or quality issues

If any criteria fails, **STOP** and debug before Week 2. Week 2-3 depend entirely on Week 1's data quality.

---

**Ready to begin Week 1? Let's start with database selection and setup!**
