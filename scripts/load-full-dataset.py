#!/usr/bin/env python3
"""
Load Full 3-Year Dataset with 2/3 Train, 1/3 Test Split
Ingests all timeframes for all tickers with proper error handling
"""

import requests
import time
import json
from typing import List, Dict

API_URL = "http://localhost:3000/api/v2/data/ingest"
TICKERS = ["SPY", "QQQ", "IWM", "UVXY"]

# Full 3 years = 1095 days
DAYS_3_YEARS = 1095

def ingest_data(ticker: str, timeframe: str, days_back: int) -> Dict:
    """Ingest data for a single ticker/timeframe combination"""
    payload = {
        "ticker": ticker,
        "timeframe": timeframe,
        "daysBack": days_back
    }

    print(f"[{ticker} {timeframe}] Ingesting {days_back} days...")

    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        data = response.json()

        if data.get("success"):
            result = data["results"][0]
            bars_inserted = result.get("barsInserted", 0)
            print(f"✅ SUCCESS: {bars_inserted} bars inserted")
            return {"success": True, "bars": bars_inserted}
        else:
            error_msg = data.get("error", "Unknown error")
            print(f"❌ FAILED: {error_msg}")
            return {"success": False, "error": error_msg}

    except requests.exceptions.Timeout:
        print("❌ TIMEOUT: Request took too long")
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return {"success": False, "error": str(e)}

def main():
    print("=" * 60)
    print("Full 3-Year Dataset Ingestion")
    print("2/3 Training (730 days) + 1/3 Testing (365 days)")
    print("=" * 60)
    print()

    total_bars = 0
    total_jobs = len(TICKERS) * 4  # 4 timeframes per ticker
    completed_jobs = 0

    # Step 1: Daily data (3 years)
    print(f"Step 1/4: Ingesting Daily (1d) data - {DAYS_3_YEARS} days")
    print("-" * 60)
    for ticker in TICKERS:
        result = ingest_data(ticker, "1d", DAYS_3_YEARS)
        if result["success"]:
            total_bars += result["bars"]
        completed_jobs += 1
        print(f"Progress: {completed_jobs}/{total_jobs} jobs")
        print()
        time.sleep(2)  # Respectful delay

    # Step 2: Hourly data (3 years)
    print()
    print(f"Step 2/4: Ingesting Hourly (1h) data - {DAYS_3_YEARS} days")
    print("-" * 60)
    for ticker in TICKERS:
        result = ingest_data(ticker, "1h", DAYS_3_YEARS)
        if result["success"]:
            total_bars += result["bars"]
        completed_jobs += 1
        print(f"Progress: {completed_jobs}/{total_jobs} jobs")
        print()
        time.sleep(2)  # Respectful delay

    # Step 3: 5-minute data (30 days, uncapped)
    print()
    print("Step 3/4: Ingesting 5-minute (5m) data - 30 days (uncapped)")
    print("-" * 60)
    for ticker in TICKERS:
        result = ingest_data(ticker, "5m", 30)
        if result["success"]:
            total_bars += result["bars"]
        completed_jobs += 1
        print(f"Progress: {completed_jobs}/{total_jobs} jobs")
        print()
        time.sleep(2)  # Respectful delay

    # Step 4: 1-minute data (7 days, uncapped)
    print()
    print("Step 4/4: Ingesting 1-minute (1m) data - 7 days (uncapped)")
    print("-" * 60)
    for ticker in TICKERS:
        result = ingest_data(ticker, "1m", 7)
        if result["success"]:
            total_bars += result["bars"]
        completed_jobs += 1
        print(f"Progress: {completed_jobs}/{total_jobs} jobs")
        print()
        time.sleep(2)  # Respectful delay

    # Summary
    print()
    print("=" * 60)
    print("Full Dataset Ingestion Complete!")
    print("=" * 60)
    print(f"Total jobs: {completed_jobs}/{total_jobs}")
    print(f"Total bars inserted: {total_bars:,}")
    print()
    print("Expected Results:")
    print("  - Daily (1d): ~1,095 bars per ticker")
    print("  - Hourly (1h): ~7,117 bars per ticker")
    print("  - 5-minute (5m): ~2,340 bars per ticker")
    print("  - 1-minute (1m): ~2,730 bars per ticker")
    print()
    print("Train/Test Split:")
    print("  - Training: 2/3 = ~730 days")
    print("  - Testing: 1/3 = ~365 days")
    print()
    print("Verify with: curl http://localhost:3001/api/v2/data/ingest/status")

if __name__ == "__main__":
    main()
