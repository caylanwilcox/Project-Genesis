# Day 1 & 2: Database Setup with Prisma

## Overview

Day 1 & 2 establish a production-ready database layer using:
- **Prisma ORM**: Type-safe database access with auto-generated types
- **TimescaleDB**: PostgreSQL extension for time-series optimization
- **Repository Pattern**: Clean separation of data access logic
- **Supabase**: Managed PostgreSQL hosting

## What Was Built

### Day 1: Database Setup
- ✅ Prisma schema matching existing database structure
- ✅ Prisma client configuration
- ✅ Database connection setup
- ⏳ TimescaleDB hypertables (requires database password)
- ⏳ Compression & retention policies (requires database password)

### Day 2: Data Access Layer
- ✅ Repository pattern implementation
- ✅ Market data repository with CRUD operations
- ✅ Features repository for technical indicators
- ✅ Predictions repository for ML models
- ✅ Ingestion log repository for tracking
- ✅ Updated data ingestion service using Prisma
- ✅ New API routes (`/api/v2/*`) using repositories

## Prerequisites

1. **Supabase Database Password**: You need to get this from your Supabase dashboard
   - Go to: https://app.supabase.com/project/yvrfkqggtxmfhmqjzulh
   - Click **Project Settings** → **Database**
   - Copy the **Database password**

2. **TypeScript**: Upgraded to latest version (required by Prisma)

3. **Prisma**: Installed (`prisma` and `@prisma/client`)

## Setup Steps

### Step 1: Configure Database URL

1. Open `.env.local`
2. Find the line with `DATABASE_URL`
3. Replace `[YOUR-DATABASE-PASSWORD]` with your actual Supabase database password

```bash
# Before:
DATABASE_URL="postgresql://postgres:[YOUR-DATABASE-PASSWORD]@db.yvrfkqggtxmfhmqjzulh.supabase.co:5432/postgres?pgbouncer=true&connection_limit=1"

# After:
DATABASE_URL="postgresql://postgres:your-actual-password@db.yvrfkqggtxmfhmqjzulh.supabase.co:5432/postgres?pgbouncer=true&connection_limit=1"
```

### Step 2: Generate Prisma Client

Once the database URL is configured, run:

```bash
npx prisma generate
```

This generates TypeScript types and the Prisma client based on your schema.

### Step 3: Sync Schema with Database

Since the database schema already exists (from Week 1), we need to pull it into Prisma:

```bash
npx prisma db pull
```

This will update the Prisma schema to match your existing database.

Then regenerate the client:

```bash
npx prisma generate
```

### Step 4: Enable TimescaleDB (Optional but Recommended)

TimescaleDB optimizes time-series queries. To enable it:

1. Go to Supabase SQL Editor
2. Run this SQL:

```sql
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Convert market_data to hypertable
SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);

-- Convert features to hypertable
SELECT create_hypertable('features', 'timestamp', if_not_exists => TRUE);

-- Convert predictions to hypertable
SELECT create_hypertable('predictions', 'timestamp', if_not_exists => TRUE);

-- Add compression policy (compress data older than 7 days)
SELECT add_compression_policy('market_data', INTERVAL '7 days');
SELECT add_compression_policy('features', INTERVAL '7 days');
SELECT add_compression_policy('predictions', INTERVAL '7 days');

-- Add retention policy (keep data for 2 years)
SELECT add_retention_policy('market_data', INTERVAL '2 years');
SELECT add_retention_policy('features', INTERVAL '2 years');
SELECT add_retention_policy('predictions', INTERVAL '2 years');
```

### Step 5: Test the Connection

Create a test script:

```typescript
// test-prisma.ts
import { prisma } from './src/lib/prisma'

async function test() {
  try {
    const count = await prisma.marketData.count()
    console.log(`✅ Connection successful! Found ${count} market data records`)
  } catch (error) {
    console.error('❌ Connection failed:', error)
  } finally {
    await prisma.$disconnect()
  }
}

test()
```

Run it:
```bash
npx ts-node test-prisma.ts
```

## Project Structure

```
mvp-trading-app/
├── prisma/
│   └── schema.prisma              # Prisma schema definition
├── src/
│   ├── lib/
│   │   ├── prisma.ts              # Prisma client singleton
│   │   └── supabase.ts            # Legacy Supabase client (still works)
│   ├── repositories/              # Repository pattern (Day 2)
│   │   ├── index.ts               # Barrel exports
│   │   ├── marketDataRepository.ts    # Market data CRUD
│   │   ├── featuresRepository.ts      # Features CRUD
│   │   ├── predictionsRepository.ts   # Predictions CRUD
│   │   └── ingestionLogRepository.ts  # Logging CRUD
│   └── services/
│       ├── dataIngestionService.ts    # Legacy (Supabase)
│       └── dataIngestionService.v2.ts # New (Prisma)
├── app/api/
│   ├── data/                      # Legacy API routes
│   │   ├── ingest/route.ts
│   │   ├── ingest/status/route.ts
│   │   └── market/route.ts
│   └── v2/data/                   # New Prisma-based API routes
│       ├── ingest/route.ts
│       ├── ingest/status/route.ts
│       └── market/route.ts
└── .env.local                     # Environment variables
```

## API Routes

### Legacy Routes (Supabase client)
- `POST /api/data/ingest` - Ingest data
- `GET /api/data/ingest/status` - Check status
- `GET /api/data/market` - Get market data

### New Routes (Prisma)
- `POST /api/v2/data/ingest` - Ingest data (Prisma)
- `GET /api/v2/data/ingest/status` - Check status (Prisma)
- `GET /api/v2/data/market` - Get market data (Prisma)

## Repository Pattern Usage

### Market Data Repository

```typescript
import { marketDataRepo } from '@/repositories'

// Insert/update market data
const data = await marketDataRepo.upsertMany([
  {
    ticker: 'SPY',
    timeframe: '1h',
    timestamp: new Date(),
    open: 450.00,
    high: 451.00,
    low: 449.50,
    close: 450.75,
    volume: 1000000n,
    source: 'polygon'
  }
])

// Find market data
const bars = await marketDataRepo.findMany(
  { ticker: 'SPY', timeframe: '1h' },
  100
)

// Check if data exists
const exists = await marketDataRepo.exists('SPY', '1h')

// Get summary statistics
const summary = await marketDataRepo.getSummary('SPY')

// Get OHLCV for charting
const ohlcv = await marketDataRepo.getOHLCV('SPY', '1h')
```

### Features Repository

```typescript
import { featuresRepo } from '@/repositories'

// Insert features
await featuresRepo.upsertMany([
  {
    ticker: 'SPY',
    timeframe: '1h',
    timestamp: new Date(),
    featureName: 'rsi_14',
    featureValue: 65.5
  }
])

// Get latest features
const latest = await featuresRepo.getLatestFeatures('SPY', '1h')

// Get feature time series
const rsi = await featuresRepo.getFeatureTimeSeries('SPY', '1h', 'rsi_14')

// Get all feature names
const names = await featuresRepo.getFeatureNames('SPY', '1h')
```

### Predictions Repository

```typescript
import { predictionsRepo } from '@/repositories'

// Create prediction
await predictionsRepo.create({
  ticker: 'SPY',
  timeframe: '1h',
  timestamp: new Date(),
  modelName: 'xgboost',
  predictedDirection: 'up',
  predictedChange: 0.5,
  confidence: 0.85
})

// Get model accuracy
const accuracy = await predictionsRepo.getModelAccuracy('xgboost', 'SPY')

// Update with actuals (for backtesting)
await predictionsRepo.updateActuals(predictionId, 'up', 0.6)
```

## Key Differences: Supabase Client vs Prisma

### Supabase Client (Legacy)
```typescript
// Direct SQL-like queries
const { data, error } = await supabase
  .from('market_data')
  .select('*')
  .eq('ticker', 'SPY')
  .order('timestamp', { ascending: false })
  .limit(100)
```

### Prisma (New)
```typescript
// Type-safe ORM queries
const data = await prisma.marketData.findMany({
  where: { ticker: 'SPY' },
  orderBy: { timestamp: 'desc' },
  take: 100
})
```

## Benefits of Prisma

1. **Type Safety**: Auto-generated TypeScript types prevent runtime errors
2. **Better IDE Support**: Autocomplete for all queries
3. **Migration Management**: Schema changes tracked in version control
4. **Cleaner Code**: Less boilerplate, more readable
5. **Performance**: Query optimization and connection pooling
6. **Repository Pattern**: Better separation of concerns

## Testing the New API

### Test Ingestion (Prisma version)

```bash
curl -X POST http://localhost:3001/api/v2/data/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "SPY",
    "timeframe": "1h",
    "daysBack": 7
  }'
```

### Check Status

```bash
curl http://localhost:3001/api/v2/data/ingest/status
```

### Get Market Data

```bash
curl "http://localhost:3001/api/v2/data/market?ticker=SPY&timeframe=1h&limit=100"
```

## Migration Path

You have two options:

### Option 1: Keep Both (Recommended for now)
- Legacy routes (`/api/data/*`) use Supabase client
- New routes (`/api/v2/data/*`) use Prisma
- Gradually migrate features to Prisma

### Option 2: Full Migration
- Replace old routes with new ones
- Update all services to use Prisma
- Remove Supabase client

## Troubleshooting

### Error: "Missing database password"
- Update `DATABASE_URL` in `.env.local` with actual password
- Get password from Supabase dashboard → Project Settings → Database

### Error: "Prisma client not generated"
- Run `npx prisma generate`
- Restart dev server

### Error: "Can't reach database server"
- Check database password is correct
- Verify network connection
- Check Supabase project is running

### Error: "Type errors in repositories"
- Run `npx prisma generate` to regenerate types
- Restart TypeScript server in VSCode

## Database Schema

The Prisma schema defines these models:

- `MarketData` - Historical OHLCV data
- `Feature` - Technical indicators
- `Prediction` - ML model predictions
- `Model` - Model metadata
- `Trade` - Trading history
- `Portfolio` - Portfolio tracking
- `IngestionLog` - Data fetch logs

All models have proper indexes for performance.

## Next Steps

After completing Day 1 & 2:

**Week 2: Feature Engineering**
- Use `featuresRepo` to store RSI, MACD, SMA
- Build feature calculation pipelines
- Store features for ML training

**Week 3: Model Training**
- Use `predictionsRepo` to store model predictions
- Track model accuracy
- Backtest predictions

**Week 11: Trading Bot**
- Use `Trade` and `Portfolio` models
- Track live trading performance
- Monitor portfolio metrics

## Resources

- [Prisma Documentation](https://www.prisma.io/docs)
- [Prisma with Next.js](https://www.prisma.io/docs/guides/other/troubleshooting-orm/help-articles/nextjs-prisma-client-dev-practices)
- [TimescaleDB Docs](https://docs.timescale.com/)
- [Repository Pattern](https://learn.microsoft.com/en-us/dotnet/architecture/microservices/microservice-ddd-cqrs-patterns/infrastructure-persistence-layer-design)

## Completion Checklist

- [ ] Database password added to `.env.local`
- [ ] Prisma client generated (`npx prisma generate`)
- [ ] Schema synced with database (`npx prisma db pull`)
- [ ] TimescaleDB extension enabled (optional)
- [ ] Hypertables created (optional)
- [ ] Test connection successful
- [ ] New API routes (`/api/v2/*`) tested
- [ ] Repository pattern working

Once complete, Day 1 & 2 are done! 🎉
