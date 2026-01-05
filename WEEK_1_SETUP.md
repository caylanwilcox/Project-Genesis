# Week 1: Database & Infrastructure Setup

## 🎯 What We Built

Week 1 establishes the foundation for the ML trading system:
- ✅ **Supabase Integration**: PostgreSQL database with TypeScript types
- ✅ **Data Ingestion Service**: Fetch historical data from Polygon.io and store in database
- ✅ **API Routes**: RESTful endpoints for data access
- ✅ **Database Schema**: Tables for market data, features, predictions, models, trades, and portfolio

## 📋 Prerequisites

1. **Supabase Account**: You already have one!
   - URL: `https://yvrfkqggtxmfhmqjzulh.supabase.co`
   - Anon Key: Already configured in `.env.local`

2. **Polygon.io API Key**: Already configured
   - API Key: `cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O`

## 🚀 Setup Steps

### Step 1: Run Database Schema

1. Go to your Supabase dashboard: https://app.supabase.com/project/yvrfkqggtxmfhmqjzulh
2. Click on **SQL Editor** in the left sidebar
3. Click **New query**
4. Copy the contents of `supabase/schema.sql`
5. Paste into the SQL Editor
6. Click **Run** (or press Cmd/Ctrl + Enter)

This will create all the tables, indexes, and views needed for the system.

### Step 2: Verify Database Tables

After running the schema, verify the tables were created:

1. Go to **Table Editor** in Supabase
2. You should see these tables:
   - `market_data` - Historical OHLCV data
   - `features` - Technical indicators
   - `predictions` - ML predictions
   - `models` - Model metadata
   - `trades` - Trade history (for Week 11)
   - `portfolio` - Portfolio tracking (for Week 11)
   - `ingestion_log` - Data fetch logs

### Step 3: Test the Integration

Start your development server (should already be running):
```bash
npm run dev
```

## 📊 Data Ingestion

### Ingest Historical Data

You can ingest historical data using the API:

**Ingest all tickers (SPY, QQQ, IWM, UVXY) for the past 30 days:**
```bash
curl -X POST http://localhost:3001/api/data/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "daysBack": 30
  }'
```

**Ingest specific ticker and timeframe:**
```bash
curl -X POST http://localhost:3001/api/data/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "SPY",
    "timeframe": "1h",
    "daysBack": 60
  }'
```

**Check ingestion status:**
```bash
curl http://localhost:3001/api/data/ingest/status
```

### Fetch Market Data

**Get market data from database:**
```bash
curl "http://localhost:3001/api/data/market?ticker=SPY&timeframe=1h&limit=100"
```

## 🗂️ Database Schema

### Market Data Table
Stores historical OHLCV data:
```sql
market_data (
  id UUID PRIMARY KEY,
  ticker VARCHAR(10),           -- 'SPY', 'QQQ', etc.
  timeframe VARCHAR(10),        -- '1h', '1d', '1w'
  timestamp TIMESTAMPTZ,        -- Bar timestamp
  open, high, low, close DECIMAL(12,4),
  volume BIGINT,
  source VARCHAR(50),           -- 'polygon'
  created_at TIMESTAMPTZ,
  UNIQUE(ticker, timeframe, timestamp)
)
```

### Features Table
Technical indicators for ML:
```sql
features (
  id UUID PRIMARY KEY,
  ticker VARCHAR(10),
  timeframe VARCHAR(10),
  timestamp TIMESTAMPTZ,
  feature_name VARCHAR(100),    -- 'rsi_14', 'macd', 'sma_20'
  feature_value DECIMAL(12,6),
  UNIQUE(ticker, timeframe, timestamp, feature_name)
)
```

### Predictions Table
ML model predictions:
```sql
predictions (
  id UUID PRIMARY KEY,
  ticker VARCHAR(10),
  timeframe VARCHAR(10),
  timestamp TIMESTAMPTZ,
  model_name VARCHAR(50),       -- 'xgboost', 'lstm'
  predicted_direction VARCHAR(10), -- 'up', 'down', 'neutral'
  predicted_change DECIMAL(8,4),
  confidence DECIMAL(5,4),      -- 0-1
  actual_direction VARCHAR(10),
  actual_change DECIMAL(8,4),
  accuracy DECIMAL(5,4)
)
```

## 📁 Project Structure

```
mvp-trading-app/
├── app/
│   └── api/
│       └── data/
│           ├── ingest/
│           │   └── route.ts         # Ingestion API
│           └── market/
│               └── route.ts         # Market data API
├── src/
│   ├── lib/
│   │   └── supabase.ts             # Supabase client
│   └── services/
│       ├── polygonService.ts       # Polygon API
│       └── dataIngestionService.ts # Data ingestion
├── supabase/
│   └── schema.sql                  # Database schema
└── .env.local                      # Environment variables
```

## 🔧 Configuration

Your `.env.local` file now includes:

```env
# Polygon.io API
NEXT_PUBLIC_POLYGON_API_KEY=cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O

# Supabase
NEXT_PUBLIC_SUPABASE_URL=https://yvrfkqggtxmfhmqjzulh.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGc...
```

## 🧪 Testing

### 1. Test Database Connection

Create a simple test file:

```typescript
// test-supabase.ts
import { supabase } from './src/lib/supabase'

async function test() {
  const { data, error } = await supabase
    .from('market_data')
    .select('count')

  console.log('Connection test:', data, error)
}

test()
```

### 2. Test Data Ingestion

Use the ingestion API to fetch and store data:

```bash
# Ingest SPY 1h data for past week
curl -X POST http://localhost:3001/api/data/ingest \
  -H "Content-Type: application/json" \
  -d '{"ticker": "SPY", "timeframe": "1h", "daysBack": 7}'
```

### 3. Verify Data in Supabase

Go to Supabase Table Editor and check the `market_data` table. You should see rows with:
- Ticker symbols (SPY, QQQ, etc.)
- Timestamps
- OHLCV data

## 📈 Next Steps (Week 2+)

After Week 1 is complete, you'll be ready for:

**Week 2: Feature Engineering**
- Calculate RSI, MACD, Moving Averages
- Store in `features` table
- Build feature pipelines

**Week 3: SPY Model Training**
- Train XGBoost, LSTM models
- Store model metadata in `models` table
- Save predictions to `predictions` table

**Week 11: Trading Bot Integration**
- Connect to trading API
- Execute trades based on predictions
- Track performance in `trades` and `portfolio` tables

## 🐛 Troubleshooting

### Error: "Missing Supabase environment variables"
- Check that `.env.local` exists and has the Supabase credentials
- Restart the dev server after adding environment variables

### Error: "relation 'market_data' does not exist"
- Run the schema.sql file in Supabase SQL Editor
- Check that all tables were created successfully

### Error: 429 (Rate Limited) from Polygon
- Reduce `daysBack` parameter
- Add delays between requests (already implemented - 200ms)
- Check your Polygon.io plan limits

### No Data Inserted
- Check the `ingestion_log` table for error messages
- Verify your Polygon API key is valid
- Check console logs for detailed error information

## 📚 Resources

- [Supabase Documentation](https://supabase.com/docs)
- [Polygon.io API Docs](https://polygon.io/docs)
- [Next.js API Routes](https://nextjs.org/docs/app/building-your-application/routing/route-handlers)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

## ✅ Week 1 Completion Checklist

- [ ] Database schema created in Supabase
- [ ] All tables visible in Table Editor
- [ ] Successfully ingested data for at least one ticker
- [ ] Can fetch data via API routes
- [ ] `ingestion_log` shows successful runs
- [ ] No errors in console/logs

Once all items are checked, Week 1 is complete! 🎉
