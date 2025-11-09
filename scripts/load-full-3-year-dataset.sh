#!/bin/bash
# Load Full 3-Year Dataset (2/3 Train, 1/3 Test Split)
# No artificial caps - get ALL available data

set -e

echo "========================================"
echo "Full 3-Year Dataset Ingestion"
echo "2/3 Training (730 days) + 1/3 Testing (365 days)"
echo "========================================"
echo ""

API_URL="http://localhost:3000/api/v2/data/ingest"
TICKERS=("SPY" "QQQ" "IWM" "UVXY")

# Calculate days: 3 years = 1095 days
DAYS_3_YEARS=1095

echo "Step 1: Clear existing data to start fresh"
echo "-----------------------------------"
read -p "Do you want to clear existing market data? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Clearing database..."
    # This will be done via Prisma Studio or SQL
    echo "⚠️  Please run: DELETE FROM market_data; in Supabase SQL Editor"
    echo "Then press Enter to continue..."
    read
fi

echo ""
echo "Step 2: Ingest 3 years of Daily data (1095 days)"
echo "-----------------------------------"
job_num=0
total_daily_jobs=${#TICKERS[@]}

for ticker in "${TICKERS[@]}"; do
    job_num=$((job_num + 1))
    echo "[Daily $job_num/$total_daily_jobs] $ticker (1d) - 1095 days"

    response=$(curl -s -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -d "{\"ticker\": \"$ticker\", \"timeframe\": \"1d\", \"daysBack\": $DAYS_3_YEARS}")

    success=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['success'])" 2>/dev/null || echo "false")

    if [ "$success" = "True" ]; then
        bars=$(echo "$response" | python3 -c "import sys, json; r=json.load(sys.stdin)['results'][0]; print(r['barsInserted'])" 2>/dev/null || echo "0")
        echo "✅ SUCCESS: $bars bars inserted"
    else
        echo "❌ FAILED"
    fi

    echo ""
    sleep 2
done

echo ""
echo "Step 3: Ingest 3 years of Hourly data (1095 days)"
echo "-----------------------------------"
job_num=0
total_hourly_jobs=${#TICKERS[@]}

for ticker in "${TICKERS[@]}"; do
    job_num=$((job_num + 1))
    echo "[Hourly $job_num/$total_hourly_jobs] $ticker (1h) - 1095 days"

    response=$(curl -s -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -d "{\"ticker\": \"$ticker\", \"timeframe\": \"1h\", \"daysBack\": $DAYS_3_YEARS}")

    success=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['success'])" 2>/dev/null || echo "false")

    if [ "$success" = "True" ]; then
        bars=$(echo "$response" | python3 -c "import sys, json; r=json.load(sys.stdin)['results'][0]; print(r['barsInserted'])" 2>/dev/null || echo "0")
        echo "✅ SUCCESS: $bars bars inserted"
    else
        echo "❌ FAILED"
    fi

    echo ""
    sleep 2
done

echo ""
echo "Step 4: Ingest 5-minute data (30 days, uncapped)"
echo "-----------------------------------"
job_num=0
total_5m_jobs=${#TICKERS[@]}

for ticker in "${TICKERS[@]}"; do
    job_num=$((job_num + 1))
    echo "[5m $job_num/$total_5m_jobs] $ticker (5m) - 30 days"

    response=$(curl -s -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -d "{\"ticker\": \"$ticker\", \"timeframe\": \"5m\", \"daysBack\": 30}")

    success=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['success'])" 2>/dev/null || echo "false")

    if [ "$success" = "True" ]; then
        bars=$(echo "$response" | python3 -c "import sys, json; r=json.load(sys.stdin)['results'][0]; print(r['barsInserted'])" 2>/dev/null || echo "0")
        echo "✅ SUCCESS: $bars bars inserted"
    else
        echo "❌ FAILED"
    fi

    echo ""
    sleep 2
done

echo ""
echo "Step 5: Ingest 1-minute data (7 days, uncapped)"
echo "-----------------------------------"
job_num=0
total_1m_jobs=${#TICKERS[@]}

for ticker in "${TICKERS[@]}"; do
    job_num=$((job_num + 1))
    echo "[1m $job_num/$total_1m_jobs] $ticker (1m) - 7 days"

    response=$(curl -s -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -d "{\"ticker\": \"$ticker\", \"timeframe\": \"1m\", \"daysBack\": 7}")

    success=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['success'])" 2>/dev/null || echo "false")

    if [ "$success" = "True" ]; then
        bars=$(echo "$response" | python3 -c "import sys, json; r=json.load(sys.stdin)['results'][0]; print(r['barsInserted'])" 2>/dev/null || echo "0")
        echo "✅ SUCCESS: $bars bars inserted"
    else
        echo "❌ FAILED"
    fi

    echo ""
    sleep 2
done

echo ""
echo "========================================"
echo "Full Dataset Ingestion Complete!"
echo "========================================"
echo ""
echo "Expected Results:"
echo "  - Daily (1d): ~1095 bars per ticker (3 years)"
echo "  - Hourly (1h): ~7,117 bars per ticker (1095 days × 6.5 hours)"
echo "  - 5-minute (5m): ~2,340 bars per ticker (30 days × 78 bars/day)"
echo "  - 1-minute (1m): ~2,730 bars per ticker (7 days × 390 bars/day)"
echo ""
echo "Train/Test Split:"
echo "  - Training: 2/3 = ~730 days (Dec 2022 - Dec 2024)"
echo "  - Testing: 1/3 = ~365 days (Dec 2024 - Dec 2025)"
echo ""
echo "Verify with: curl http://localhost:3000/api/v2/data/ingest/status"
