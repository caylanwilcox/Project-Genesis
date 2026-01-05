#!/bin/bash
# Add All Scalping Timeframes (1m and 5m)
# With starter plan, this should complete in ~5 minutes instead of 2 hours

echo "========================================"
echo "Adding Scalping Timeframes (1m, 5m)"
echo "Using Polygon.io 'starter' plan"
echo "========================================"
echo ""

API_URL="http://localhost:3001/api/v2/data/ingest"
TICKERS=("QQQ" "IWM" "UVXY")  # SPY already done
TIMEFRAMES=("5m" "1m")

total_jobs=$((${#TICKERS[@]} * ${#TIMEFRAMES[@]}))
current_job=0

for ticker in "${TICKERS[@]}"; do
  for timeframe in "${TIMEFRAMES[@]}"; do
    ((current_job++))

    if [ "$timeframe" = "1m" ]; then
      days=7
      desc="1-minute (7 days)"
    else
      days=30
      desc="5-minute (30 days)"
    fi

    echo "[$current_job/$total_jobs] $ticker $desc"
    echo "-----------------------------------"

    response=$(curl -s -X POST "$API_URL" \
      -H "Content-Type: application/json" \
      -d "{\"ticker\":\"$ticker\",\"timeframe\":\"$timeframe\",\"daysBack\":$days}")

    success=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['success'])" 2>/dev/null || echo "false")

    if [ "$success" = "True" ] || [ "$success" = "true" ]; then
      bars=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['summary']['totalBarsInserted'])" 2>/dev/null || echo "0")
      echo "✅ SUCCESS: $bars bars inserted"
    else
      echo "❌ FAILED"
      echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    fi

    echo ""

    # Small delay between requests (respectful, even on paid plan)
    if [ $current_job -lt $total_jobs ]; then
      echo "⏳ Waiting 2s..."
      sleep 2
    fi
  done
done

echo ""
echo "========================================"
echo "Scalping Data Addition Complete!"
echo "========================================"
echo ""
echo "Verifying database..."
curl -s "http://localhost:3002/api/v2/data/ingest/status" | python3 -m json.tool | grep -A 15 "ticker"

echo ""
echo "✅ Done!"
