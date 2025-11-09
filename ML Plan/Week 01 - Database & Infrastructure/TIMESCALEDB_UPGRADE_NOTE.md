# TimescaleDB Upgrade - Action Item

**Status:** ⏳ DEFERRED (Not Required for Development)
**Priority:** Medium (Required before production)
**Estimated Cost:** $25/month (Supabase Pro)

---

## Current Situation

✅ **What's Working:**
- PostgreSQL database on Supabase Free tier
- Prisma ORM with optimized queries
- All tables with proper indexes
- 900 bars successfully stored
- API responses fast (<200ms)
- All Day 1 & 2 objectives met

❌ **What's Missing (TimescaleDB):**
- 10x faster time-series queries
- 70-90% storage compression
- Automatic data partitioning (hypertables)
- Continuous aggregates for instant rollups

---

## Why Not Installed Now?

TimescaleDB extension is **not available on Supabase Free tier**.

**Error received:**
```
ERROR: extension "timescaledb" is not available
HINT: The extension must first be installed on the system where PostgreSQL is running.
```

**Available on:**
- Supabase Pro ($25/month)
- Self-hosted PostgreSQL + TimescaleDB
- Timescale Cloud
- Docker local installation

---

## When to Upgrade?

**Upgrade TimescaleDB when:**

1. **Data Volume Grows**
   - [ ] Have 100K+ bars (currently: 900)
   - [ ] Storage costs become significant
   - [ ] Query performance degrades

2. **Performance Requirements**
   - [ ] Need <50ms query times (currently acceptable)
   - [ ] Real-time aggregations needed
   - [ ] Multiple concurrent users

3. **Production Deployment**
   - [ ] Week 11-12: Trading bot deployment
   - [ ] Need production-grade performance
   - [ ] Require automatic backups

**Earliest Recommended:** Before Week 11 (Trading Bot Integration)

---

## How to Upgrade

### Option 1: Supabase Pro (Recommended)

**Steps:**
1. Go to: https://app.supabase.com/project/yvrfkqggtxmfhmqjzulh/settings/billing
2. Click **Upgrade to Pro** ($25/month)
3. Wait for confirmation email
4. Run SQL script: `supabase/timescaledb-setup.sql`
5. Verify: `SELECT * FROM timescaledb_information.hypertables;`

**Time:** 5-10 minutes
**Cost:** $25/month
**Complexity:** Easy

### Option 2: Docker TimescaleDB (Free)

**Steps:**
1. Install Docker Desktop
2. Run: `docker-compose up -d` (using provided docker-compose.yml)
3. Update DATABASE_URL in `.env.local`
4. Run: `npx prisma db push`
5. Run: `supabase/timescaledb-setup.sql`

**Time:** 1-2 hours
**Cost:** Free (or $5-20/mo if hosted)
**Complexity:** Medium

### Option 3: Timescale Cloud

**Steps:**
1. Sign up: https://www.timescale.com/
2. Create free database
3. Copy connection string
4. Update `.env.local`
5. Run migrations and setup script

**Time:** 30 minutes
**Cost:** Free tier available, then $30-50/mo
**Complexity:** Easy

---

## Performance Impact Estimate

Based on industry benchmarks:

| Metric | Current (PostgreSQL) | With TimescaleDB | Improvement |
|--------|---------------------|------------------|-------------|
| Insert 1K bars | ~200ms | ~100ms | 2x faster |
| Query 1 year (100K bars) | ~500ms | ~50ms | **10x faster** |
| Query latest bar | ~5ms | ~2ms | 2-3x faster |
| Storage (1M bars) | ~500MB | ~50-150MB | **3-10x smaller** |
| Daily aggregates | ~2000ms | ~10ms | **200x faster** |

**Bottom Line:** TimescaleDB provides **significant** performance gains but is **not required** for ML training (Weeks 2-10).

---

## Cost-Benefit Analysis

### Without TimescaleDB (Current - Free)

**Pros:**
- ✅ Free
- ✅ Working well now
- ✅ Sufficient for development
- ✅ Good for Weeks 2-10 (ML training)

**Cons:**
- ⏳ Slower queries at scale
- ⏳ More storage costs long-term
- ⏳ Manual aggregation computation

### With TimescaleDB ($25/mo Supabase Pro)

**Pros:**
- ✅ 10x faster queries
- ✅ 70-90% storage reduction
- ✅ Automatic backups
- ✅ Production-ready
- ✅ Point-in-time recovery

**Cons:**
- ❌ $25/month cost
- ❌ Need to migrate (easy though)

**ROI:** Worth it for production, not urgent for development

---

## Decision

**For Week 1-10 (Development & Training):**
✅ **Continue with PostgreSQL (Supabase Free)**
- Current performance is acceptable
- Focus on building ML models
- Save costs during development

**For Week 11-12 (Production Deployment):**
⏳ **Upgrade to TimescaleDB (Supabase Pro)**
- Trading bot needs fast queries
- Production requires reliability
- Cost justified by performance

---

## Action Items

### Now (Week 1-10)
- [x] Document TimescaleDB limitation
- [x] Note upgrade path for later
- [x] Continue with current setup
- [ ] Monitor query performance
- [ ] Track when 100K bars reached

### Before Week 11 (Production)
- [ ] Evaluate query performance at scale
- [ ] Review storage costs
- [ ] Make upgrade decision
- [ ] Budget for Supabase Pro
- [ ] Plan migration if needed

### Migration Checklist (When Upgrading)
- [ ] Back up current database
- [ ] Upgrade Supabase plan (or migrate to new DB)
- [ ] Update DATABASE_URL if needed
- [ ] Run `supabase/timescaledb-setup.sql`
- [ ] Verify hypertables created
- [ ] Test all API endpoints
- [ ] Monitor compression policies
- [ ] Update documentation

---

## Files Ready for Upgrade

When you're ready to upgrade, these files are prepared:

1. **Setup Script:** `supabase/timescaledb-setup.sql`
   - Enables TimescaleDB extension
   - Creates hypertables
   - Adds compression policies
   - Sets up continuous aggregates
   - Includes monitoring functions

2. **Installation Guide:** `TIMESCALEDB_INSTALLATION.md`
   - All installation options documented
   - Step-by-step instructions
   - Troubleshooting guide

3. **Setup Guide:** `TIMESCALEDB_SETUP_GUIDE.md`
   - How to run the script
   - Verification steps
   - Performance testing
   - Monitoring queries

---

## Timeline

```
Week 1-2:   PostgreSQL ✅ (Current)
Week 3-10:  PostgreSQL ✅ (ML Training)
Week 11:    Evaluate upgrade ⏳
Week 12:    TimescaleDB for production 🎯
```

---

## References

- **Installation Options:** [TIMESCALEDB_INSTALLATION.md](../../../TIMESCALEDB_INSTALLATION.md)
- **Setup Guide:** [TIMESCALEDB_SETUP_GUIDE.md](../../../TIMESCALEDB_SETUP_GUIDE.md)
- **Setup SQL:** [supabase/timescaledb-setup.sql](../../../supabase/timescaledb-setup.sql)
- **Supabase Pricing:** https://supabase.com/pricing
- **TimescaleDB Docs:** https://docs.timescale.com/

---

## Recommendation

**Continue with current setup through Week 10.**

Revisit this decision:
- At 100K bars stored
- Before Week 11 (Trading Bot)
- If query performance degrades
- When deploying to production

**Expected upgrade timing:** Week 11 or Week 12

---

**Last Updated:** November 5, 2025
**Next Review:** Week 10 or at 100K bars
**Status:** Documented and deferred ✅
