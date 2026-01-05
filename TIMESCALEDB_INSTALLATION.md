# TimescaleDB Installation Guide

TimescaleDB is not available on Supabase Free tier. Here are your options:

---

## Option 1: Upgrade Supabase to Pro (Recommended) ✅

**Cost:** $25/month
**Setup Time:** 5 minutes
**Benefits:**
- TimescaleDB automatically available
- Better performance limits
- Daily backups
- Point-in-time recovery
- Better support

**Steps:**
1. Go to: https://app.supabase.com/project/yvrfkqggtxmfhmqjzulh/settings/billing
2. Click **Upgrade to Pro**
3. Complete payment
4. TimescaleDB extension will be available
5. Run our SQL script: `supabase/timescaledb-setup.sql`

---

## Option 2: Self-Hosted TimescaleDB (Free)

**Cost:** Free (or $5-20/month for hosting)
**Setup Time:** 1-2 hours
**Complexity:** High

### A. Local Installation (macOS)

```bash
# Install PostgreSQL with Homebrew
brew install postgresql@15

# Install TimescaleDB
brew tap timescale/tap
brew install timescaledb

# Initialize TimescaleDB
timescaledb-tune --quiet --yes

# Start PostgreSQL
brew services start postgresql@15

# Create database
createdb trading_ml

# Update .env.local
DATABASE_URL="postgresql://your_user:password@localhost:5432/trading_ml"

# Run Prisma migrations
npx prisma db push

# Run TimescaleDB setup
psql trading_ml < supabase/timescaledb-setup.sql
```

### B. Docker Installation (Any OS)

```bash
# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: trading_ml
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: your_password_here
    volumes:
      - timescale_data:/var/lib/postgresql/data

volumes:
  timescale_data:
EOF

# Start container
docker-compose up -d

# Update .env.local
DATABASE_URL="postgresql://postgres:your_password_here@localhost:5432/trading_ml"

# Run Prisma migrations
npx prisma db push

# Run TimescaleDB setup
docker exec -i timescaledb psql -U postgres -d trading_ml < supabase/timescaledb-setup.sql
```

### C. Cloud Hosting Options

**Railway** ($5-20/month)
1. Go to: https://railway.app/
2. Create new project
3. Add PostgreSQL + TimescaleDB template
4. Copy DATABASE_URL
5. Update `.env.local`
6. Run migrations

**Timescale Cloud** (Free tier available)
1. Go to: https://www.timescale.com/
2. Sign up for free account
3. Create new database
4. Copy connection string
5. Update `.env.local`
6. Run migrations

---

## Option 3: Continue Without TimescaleDB (Current Setup)

**Cost:** Free
**Setup Time:** 0 (already done!)
**Performance:** Good (not optimal)

**What you have:**
- ✅ PostgreSQL with proper indexes
- ✅ Prisma ORM optimized queries
- ✅ Working API endpoints
- ✅ 900 bars successfully stored

**What you're missing:**
- ⏳ 10x query speed boost
- ⏳ 70-90% storage compression
- ⏳ Automatic partitioning

**For now, this is fine!** You can:
1. Continue with Day 3 (backfill data)
2. Continue with Week 2 (feature engineering)
3. Continue with Week 3 (ML training)
4. Upgrade to TimescaleDB later when needed

---

## Performance Comparison

| Feature | Without TimescaleDB | With TimescaleDB |
|---------|-------------------|------------------|
| Insert 1K bars | ~200ms | ~100ms |
| Query 1 year | ~500ms | ~50ms (10x faster) |
| Storage (1M bars) | ~500MB | ~50-150MB (3-10x smaller) |
| Daily aggregates | ~2s | ~10ms (200x faster) |

**Bottom Line:** TimescaleDB is nice but **not required** for the ML system to work!

---

## My Recommendation

**For Development (Now):**
✅ **Keep using Supabase Free** - It's working great!

**For Production (Later):**
- If you need performance: **Upgrade to Supabase Pro** ($25/mo)
- If you want free: **Use Timescale Cloud free tier**
- If you want control: **Self-host with Docker**

---

## Decision Matrix

| Priority | Recommendation |
|----------|----------------|
| **Cost sensitive** | Continue without TimescaleDB ✅ |
| **Performance critical** | Upgrade Supabase Pro ($25/mo) |
| **Learning/Experimenting** | Docker local installation |
| **Production ready** | Supabase Pro or Timescale Cloud |

---

## What to Do Right Now?

**My suggestion:**

**Continue without TimescaleDB for now.** Here's why:

1. ✅ Your current setup is **working perfectly**
2. ✅ 900 bars ingested successfully
3. ✅ API responses are fast (<200ms)
4. ✅ No performance bottlenecks yet
5. ✅ Can always add TimescaleDB later

**When to upgrade:**

- When you have 100K+ bars (queries slow down)
- When storage becomes expensive
- When you need real-time aggregations
- Before production deployment

---

## Next Steps

Choose your path:

**Path A: Continue without TimescaleDB** (Recommended for now)
```bash
# No action needed - you're ready for Day 3!
# Skip to: Historical Data Ingestion
```

**Path B: Upgrade Supabase to Pro**
```bash
# 1. Upgrade at: https://app.supabase.com/settings/billing
# 2. Wait for upgrade confirmation
# 3. Run: supabase/timescaledb-setup.sql
# 4. Continue to Day 3
```

**Path C: Install Docker TimescaleDB**
```bash
# See Docker Installation section above
# Takes 1-2 hours
```

---

**What do you want to do?**

I recommend **Path A** - continue without TimescaleDB and add it later when you actually need the performance boost. Your current setup is excellent for development and Week 2-3 work!
