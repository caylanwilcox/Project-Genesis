"""
Phase 5 Agreement Analysis: V6 Target B + RPE Phase 4

Tests the accuracy when:
- V6 Target B (close > 11 AM) is BULLISH/BEARISH
- RPE Phase 4 bias is LONG/SHORT
- Both AGREE vs CONFLICT

Key Questions:
1. When both agree on direction, what's the win rate?
2. When they conflict, what's the win rate?
3. Is there a "golden signal" when both are high confidence?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rpe'))

import requests
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

# RPE imports
from rpe.northstar_pipeline import NorthstarPipeline

# Import V6 feature creation from existing backtest
from backtest_v6 import create_features as v6_create_features
from backtest_v6 import predict_v6

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v6_models')


def fetch_daily_bars(ticker: str, start_date: str, end_date: str) -> list:
    """Fetch daily bars for date range."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}
    resp = requests.get(url, params=params)
    data = resp.json()

    if 'results' not in data:
        return []

    bars = []
    for r in data['results']:
        ts = datetime.fromtimestamp(r['t'] / 1000)
        bars.append({
            'date': ts.strftime('%Y-%m-%d'),
            'o': r['o'],
            'h': r['h'],
            'l': r['l'],
            'c': r['c'],
            'v': r['v']
        })
    return bars


def fetch_hourly_bars(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly bars for date range."""
    all_data = []
    current_start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    while current_start < end:
        chunk_end = min(current_start + timedelta(days=30), end)
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{current_start.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}"
        params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}
        response = requests.get(url, params=params)
        data = response.json()
        if 'results' in data:
            all_data.extend(data['results'])
        current_start = chunk_end + timedelta(days=1)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df['datetime'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
    df = df.set_index('datetime')
    df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def fetch_1m_bars(ticker: str, date: str) -> pd.DataFrame:
    """Fetch 1-minute bars for a specific date."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}"
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}
    resp = requests.get(url, params=params)
    data = resp.json()

    if 'results' not in data:
        return pd.DataFrame()

    rows = []
    for r in data['results']:
        ts = datetime.fromtimestamp(r['t'] / 1000)
        rows.append({
            'timestamp': ts,
            'open': r['o'],
            'high': r['h'],
            'low': r['l'],
            'close': r['c'],
            'volume': r['v']
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.set_index('timestamp')
    return df


def get_price_at_hour(hourly_df: pd.DataFrame, date, hour: int) -> float:
    """Get price at specific hour from hourly bars."""
    try:
        day_start = pd.Timestamp(date).tz_localize('America/New_York').replace(hour=hour, minute=0)
        day_end = day_start + timedelta(hours=1)
        bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index < day_end)]
        if len(bars) > 0:
            return bars['Close'].iloc[0]
    except:
        pass
    return None


def load_v6_model(ticker: str):
    """Load V6 model for ticker."""
    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_v6.pkl')
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)


# Use predict_v6 from backtest_v6 module - it returns (prob_a, prob_b)


def get_rpe_signals(pipeline: NorthstarPipeline, bars_1m: pd.DataFrame, daily_bars: list, ticker: str) -> dict:
    """Get RPE Phase 1 direction and Phase 4 allowed status.

    Returns:
        dict with:
            - direction: 'UP', 'DOWN', or 'BALANCED' (Phase 1 truth)
            - allowed: True if Phase 4 allows trading
            - confidence_band: 'STRUCTURAL_EDGE', 'CONTEXT_ONLY', 'NO_TRADE'
            - execution_mode: 'TREND_CONTINUATION', 'MEAN_REVERSION', 'SCALP', 'NO_TRADE'
    """
    default = {
        'direction': 'BALANCED',
        'allowed': False,
        'confidence_band': 'NO_TRADE',
        'execution_mode': 'NO_TRADE'
    }

    if bars_1m is None or len(bars_1m) < 30:
        return default

    try:
        # Rename columns to match expected format (lowercase)
        bars_renamed = bars_1m.copy()
        if 'open' not in bars_renamed.columns:
            bars_renamed.columns = [c.lower() for c in bars_renamed.columns]

        # Convert daily bars to DataFrame format expected by pipeline
        if daily_bars:
            daily_df = pd.DataFrame(daily_bars)
            if 'o' in daily_df.columns:
                daily_df = daily_df.rename(columns={
                    'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
                })
        else:
            daily_df = None

        # Run Northstar pipeline
        result = pipeline.run(ticker, bars_renamed, daily_df)

        if result:
            return {
                'direction': result['phase1'].get('direction', 'BALANCED'),
                'allowed': result['phase4'].get('allowed', False),
                'confidence_band': result['phase1'].get('confidence_band', 'NO_TRADE'),
                'execution_mode': result['phase4'].get('execution_mode', 'NO_TRADE')
            }
    except Exception as e:
        # Uncomment to debug: print(f"RPE error: {e}")
        pass

    return default


def classify_agreement(v6_prob: float, rpe_signals: dict) -> dict:
    """
    Classify the agreement between V6 Target B and RPE.

    RPE Phases:
    - Phase 1 (direction): UP/DOWN/BALANCED - structural truth
    - Phase 2 (health): Signal quality/reliability
    - Phase 3 (density): Spam control
    - Phase 4 (allowed): Combined gate - True means high quality setup

    Returns dict with:
        - category: 'ALIGNED_BULLISH', 'ALIGNED_BEARISH', 'CONFLICT', 'NEUTRAL'
        - allowed: Phase 4 allowed (True = high quality confirmation)
        - direction: Phase 1 direction
        - confidence_band: Phase 1 confidence
    """
    v6_bullish = v6_prob > 0.5
    v6_bearish = v6_prob < 0.5

    direction = rpe_signals['direction']
    allowed = rpe_signals['allowed']
    confidence = rpe_signals['confidence_band']

    # Determine category based on Phase 1 direction
    if direction == 'UP' and v6_bullish:
        category = 'ALIGNED_BULLISH'
    elif direction == 'DOWN' and v6_bearish:
        category = 'ALIGNED_BEARISH'
    elif direction == 'BALANCED':
        category = 'NEUTRAL'
    elif direction == 'UP' and v6_bearish:
        category = 'CONFLICT'
    elif direction == 'DOWN' and v6_bullish:
        category = 'CONFLICT'
    else:
        category = 'NEUTRAL'

    return {
        'category': category,
        'allowed': allowed,
        'direction': direction,
        'confidence_band': confidence
    }


def get_confidence_level(v6_prob: float) -> str:
    """Classify V6 confidence level."""
    dist = abs(v6_prob - 0.5)
    if dist >= 0.25:  # >= 75% or <= 25%
        return 'HIGH'
    elif dist >= 0.15:  # >= 65% or <= 35%
        return 'MEDIUM'
    else:
        return 'LOW'


def run_backtest(ticker: str, start_date: str, end_date: str):
    """Run Phase 5 agreement backtest for a ticker."""

    print(f"\n{'='*70}")
    print(f"  PHASE 5 AGREEMENT BACKTEST: {ticker}")
    print(f"  V6 Target B + RPE (Phase 1 Direction + Phase 4 Allowed)")
    print(f"  Period: {start_date} to {end_date}")
    print(f"{'='*70}")

    # Load V6 model
    model_data = load_v6_model(ticker)
    if model_data is None:
        print(f"  No V6 model found for {ticker}")
        return None

    # Initialize RPE pipeline
    pipeline = NorthstarPipeline()

    # Fetch data
    fetch_start = (pd.to_datetime(start_date) - timedelta(days=15)).strftime('%Y-%m-%d')
    print(f"  Fetching hourly data...")
    hourly_df = fetch_hourly_bars(ticker, fetch_start, end_date)
    print(f"    Got {len(hourly_df)} hourly bars")

    print(f"  Fetching daily data...")
    daily_bars = fetch_daily_bars(ticker, fetch_start, end_date)
    print(f"    Got {len(daily_bars)} daily bars")

    # Get trading days to test
    test_start = pd.to_datetime(start_date).date()
    test_end = pd.to_datetime(end_date).date()
    trading_days = sorted(set(hourly_df.index.date))
    trading_days = [d for d in trading_days if test_start <= d <= test_end]

    print(f"  Testing {len(trading_days)} trading days (late session only)")

    # Results tracking - Now track by category AND by Phase 4 allowed
    results = {
        'ALIGNED_BULLISH': {'wins': 0, 'total': 0, 'allowed_wins': 0, 'allowed_total': 0, 'probs': [], 'dates': []},
        'ALIGNED_BEARISH': {'wins': 0, 'total': 0, 'allowed_wins': 0, 'allowed_total': 0, 'probs': [], 'dates': []},
        'CONFLICT': {'wins': 0, 'total': 0, 'allowed_wins': 0, 'allowed_total': 0, 'probs': [], 'dates': []},
        'NEUTRAL': {'wins': 0, 'total': 0, 'allowed_wins': 0, 'allowed_total': 0, 'probs': [], 'dates': []},
    }

    # Track V6 tradeable signals (outside neutral zone)
    v6_tradeable_count = 0
    v6_neutral_count = 0

    # Track RPE direction distribution
    rpe_direction_counts = {'UP': 0, 'DOWN': 0, 'BALANCED': 0}
    rpe_allowed_counts = {'allowed': 0, 'not_allowed': 0}

    # Confidence breakdown
    confidence_results = {
        'HIGH': {'aligned_wins': 0, 'aligned_total': 0, 'conflict_wins': 0, 'conflict_total': 0},
        'MEDIUM': {'aligned_wins': 0, 'aligned_total': 0, 'conflict_wins': 0, 'conflict_total': 0},
        'LOW': {'aligned_wins': 0, 'aligned_total': 0, 'conflict_wins': 0, 'conflict_total': 0},
    }

    for day in trading_days:
        # Find day index in daily bars
        day_str = day.strftime('%Y-%m-%d') if hasattr(day, 'strftime') else str(day)
        day_idx = None
        for i, bar in enumerate(daily_bars):
            if bar['date'] == day_str:
                day_idx = i
                break

        if day_idx is None or day_idx < 5:
            continue

        today = daily_bars[day_idx]
        prev_day_dict = daily_bars[day_idx - 1]
        prev_prev_day_dict = daily_bars[day_idx - 2]

        today_open = today['o']
        today_close = today['c']

        # Get price at 11 AM
        price_11am = get_price_at_hour(hourly_df, day, 11)
        if price_11am is None:
            continue

        # Actual outcome: Target B = close > 11 AM price
        actual_above_11am = today_close > price_11am

        # Get hourly bars for this day up to 2 PM (for late session prediction)
        day_start = pd.Timestamp(day).tz_localize('America/New_York').replace(hour=9, minute=0)
        day_2pm = pd.Timestamp(day).tz_localize('America/New_York').replace(hour=14, minute=0)
        day_bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index <= day_2pm)]

        if len(day_bars) < 3:
            continue

        # Convert prev_day format to match backtest_v6 expectations
        prev_day = {'Open': prev_day_dict['o'], 'High': prev_day_dict['h'],
                    'Low': prev_day_dict['l'], 'Close': prev_day_dict['c']}
        prev_prev_day = {'Close': prev_prev_day_dict['c']}

        # Create daily DataFrame for multi-day features
        daily_df = pd.DataFrame([
            {'Open': b['o'], 'High': b['h'], 'Low': b['l'], 'Close': b['c'], 'Volume': b['v']}
            for b in daily_bars[:day_idx + 1]
        ])

        # Create features using v6 feature creation (includes multi-day features)
        features = v6_create_features(day_bars, today_open, prev_day, prev_prev_day,
                                      daily_df, day_idx, price_11am)

        # Get V6 Target B prediction (late session)
        prob_a, prob_b = predict_v6(model_data, features, 14)  # Use hour 14 for late session
        v6_prob_b = prob_b if prob_b is not None else 0.5

        # Track V6 tradeable vs neutral
        if v6_prob_b > 0.75 or v6_prob_b < 0.25:
            v6_tradeable_count += 1
        else:
            v6_neutral_count += 1

        # Get RPE signals (Phase 1 direction + Phase 4 allowed)
        bars_1m = fetch_1m_bars(ticker, day_str)
        if len(bars_1m) < 30:
            continue

        # Filter to 12:00-14:00 for late session RPE
        bars_1m_filtered = bars_1m[(bars_1m.index.hour >= 12) & (bars_1m.index.hour < 14)]
        if len(bars_1m_filtered) < 30:
            bars_1m_filtered = bars_1m[bars_1m.index.hour < 14]  # Use all morning data

        rpe_signals = get_rpe_signals(pipeline, bars_1m_filtered, daily_bars[:day_idx], ticker)

        # Track RPE distribution
        rpe_direction_counts[rpe_signals['direction']] += 1
        if rpe_signals['allowed']:
            rpe_allowed_counts['allowed'] += 1
        else:
            rpe_allowed_counts['not_allowed'] += 1

        # Classify agreement using Phase 1 direction
        agreement_info = classify_agreement(v6_prob_b, rpe_signals)
        category = agreement_info['category']
        allowed = agreement_info['allowed']
        confidence = get_confidence_level(v6_prob_b)

        # Determine win/loss based on V6 direction
        v6_bullish = v6_prob_b > 0.5
        won = (v6_bullish and actual_above_11am) or (not v6_bullish and not actual_above_11am)

        # Track results
        results[category]['wins'] += 1 if won else 0
        results[category]['total'] += 1
        if category not in ['NEUTRAL']:
            results[category]['probs'].append(v6_prob_b)
            results[category]['dates'].append(day_str)

        # Track when Phase 4 allows trading (high quality setups)
        if allowed:
            results[category]['allowed_wins'] += 1 if won else 0
            results[category]['allowed_total'] += 1

        # Confidence breakdown (for aligned vs conflict)
        if category in ['ALIGNED_BULLISH', 'ALIGNED_BEARISH']:
            confidence_results[confidence]['aligned_wins'] += 1 if won else 0
            confidence_results[confidence]['aligned_total'] += 1
        elif category == 'CONFLICT':
            confidence_results[confidence]['conflict_wins'] += 1 if won else 0
            confidence_results[confidence]['conflict_total'] += 1

    # Print V6 signal distribution
    total_signals = v6_tradeable_count + v6_neutral_count
    print(f"\n  V6 Signal Distribution:")
    print(f"    Tradeable (>75% or <25%): {v6_tradeable_count}/{total_signals} ({v6_tradeable_count/total_signals*100 if total_signals > 0 else 0:.1f}%)")
    print(f"    Neutral (25-75%): {v6_neutral_count}/{total_signals} ({v6_neutral_count/total_signals*100 if total_signals > 0 else 0:.1f}%)")

    # Print RPE distribution
    print(f"\n  RPE Phase 1 Direction Distribution:")
    rpe_total = sum(rpe_direction_counts.values())
    for dir_name, count in rpe_direction_counts.items():
        pct = count/rpe_total*100 if rpe_total > 0 else 0
        print(f"    {dir_name}: {count}/{rpe_total} ({pct:.1f}%)")

    print(f"\n  RPE Phase 4 Allowed Distribution:")
    print(f"    Allowed: {rpe_allowed_counts['allowed']} ({rpe_allowed_counts['allowed']/rpe_total*100 if rpe_total > 0 else 0:.1f}%)")
    print(f"    Not Allowed: {rpe_allowed_counts['not_allowed']} ({rpe_allowed_counts['not_allowed']/rpe_total*100 if rpe_total > 0 else 0:.1f}%)")

    # Print results
    print(f"\n  {'='*60}")
    print(f"  AGREEMENT ANALYSIS RESULTS")
    print(f"  {'='*60}")

    for category, data in results.items():
        if data['total'] > 0:
            win_rate = data['wins'] / data['total'] * 100
            avg_prob = np.mean(data['probs']) if data['probs'] else 0.5
            print(f"\n  {category}:")
            print(f"    All Signals: {data['wins']}/{data['total']} = {win_rate:.1f}%")
            if data['allowed_total'] > 0:
                allowed_rate = data['allowed_wins'] / data['allowed_total'] * 100
                print(f"    When Ph4 Allowed: {data['allowed_wins']}/{data['allowed_total']} = {allowed_rate:.1f}%")
            if data['probs']:
                print(f"    Avg V6 Prob: {avg_prob:.1%}")

    print(f"\n  {'='*60}")
    print(f"  CONFIDENCE BREAKDOWN")
    print(f"  {'='*60}")

    for conf, data in confidence_results.items():
        print(f"\n  {conf} CONFIDENCE (V6 prob distance from 50%):")
        if data['aligned_total'] > 0:
            rate = data['aligned_wins'] / data['aligned_total'] * 100
            print(f"    Aligned: {data['aligned_wins']}/{data['aligned_total']} = {rate:.1f}%")
        if data['conflict_total'] > 0:
            rate = data['conflict_wins'] / data['conflict_total'] * 100
            print(f"    Conflict: {data['conflict_wins']}/{data['conflict_total']} = {rate:.1f}%")

    return results, confidence_results


def main():
    """Run Phase 5 agreement backtest."""
    print("="*70)
    print("  PHASE 5 AGREEMENT ANALYSIS")
    print("  V6 Target B + RPE Phase 4 Bias")
    print("="*70)
    print()
    print("  Question: Does RPE confirmation improve V6 accuracy?")
    print("  - ALIGNED: V6 and RPE agree on direction")
    print("  - CONFLICT: V6 and RPE disagree")
    print()

    # Test period
    end_date = '2025-12-30'
    start_date = '2025-11-01'

    all_results = {}
    all_confidence = {}

    for ticker in ['SPY', 'QQQ', 'IWM']:
        results, confidence = run_backtest(ticker, start_date, end_date)
        if results:
            all_results[ticker] = results
            all_confidence[ticker] = confidence

    # Aggregate results
    print("\n" + "="*70)
    print("  AGGREGATE RESULTS (ALL TICKERS)")
    print("="*70)

    totals = {
        'ALIGNED_BULLISH': {'wins': 0, 'total': 0},
        'ALIGNED_BEARISH': {'wins': 0, 'total': 0},
        'CONFLICT': {'wins': 0, 'total': 0},
        'NEUTRAL': {'wins': 0, 'total': 0},
    }

    for ticker, results in all_results.items():
        for category, data in results.items():
            totals[category]['wins'] += data['wins']
            totals[category]['total'] += data['total']

    print("\n  | Category        | Trades | Wins | Win Rate |")
    print("  |-----------------|--------|------|----------|")

    for category, data in totals.items():
        if data['total'] > 0:
            win_rate = data['wins'] / data['total'] * 100
            print(f"  | {category:15} | {data['total']:6} | {data['wins']:4} | {win_rate:6.1f}% |")

    # Aggregate confidence
    conf_totals = {
        'HIGH': {'aligned_wins': 0, 'aligned_total': 0, 'conflict_wins': 0, 'conflict_total': 0},
        'MEDIUM': {'aligned_wins': 0, 'aligned_total': 0, 'conflict_wins': 0, 'conflict_total': 0},
        'LOW': {'aligned_wins': 0, 'aligned_total': 0, 'conflict_wins': 0, 'conflict_total': 0},
    }

    for ticker, conf in all_confidence.items():
        for level, data in conf.items():
            conf_totals[level]['aligned_wins'] += data['aligned_wins']
            conf_totals[level]['aligned_total'] += data['aligned_total']
            conf_totals[level]['conflict_wins'] += data['conflict_wins']
            conf_totals[level]['conflict_total'] += data['conflict_total']

    print("\n  CONFIDENCE LEVEL BREAKDOWN:")
    print("  | Confidence | Aligned Rate | Conflict Rate | Delta |")
    print("  |------------|--------------|---------------|-------|")

    for level, data in conf_totals.items():
        aligned_rate = data['aligned_wins'] / data['aligned_total'] * 100 if data['aligned_total'] > 0 else 0
        conflict_rate = data['conflict_wins'] / data['conflict_total'] * 100 if data['conflict_total'] > 0 else 0
        delta = aligned_rate - conflict_rate
        print(f"  | {level:10} | {aligned_rate:6.1f}% ({data['aligned_total']:2}) | {conflict_rate:6.1f}% ({data['conflict_total']:2}) | {delta:+5.1f}% |")

    # Conclusion
    print("\n" + "="*70)
    print("  CONCLUSION")
    print("="*70)

    aligned_total = totals['ALIGNED_BULLISH']['total'] + totals['ALIGNED_BEARISH']['total']
    aligned_wins = totals['ALIGNED_BULLISH']['wins'] + totals['ALIGNED_BEARISH']['wins']

    if aligned_total > 0 and totals['CONFLICT']['total'] > 0:
        aligned_rate = aligned_wins / aligned_total * 100
        conflict_rate = totals['CONFLICT']['wins'] / totals['CONFLICT']['total'] * 100

        print(f"\n  When V6 and RPE AGREE: {aligned_rate:.1f}% win rate ({aligned_wins}/{aligned_total})")
        print(f"  When V6 and RPE CONFLICT: {conflict_rate:.1f}% win rate ({totals['CONFLICT']['wins']}/{totals['CONFLICT']['total']})")
        print(f"\n  Agreement Advantage: {aligned_rate - conflict_rate:+.1f}%")

        if aligned_rate > conflict_rate:
            print("\n  RECOMMENDATION: Trade ONLY when V6 and RPE agree!")
        else:
            print("\n  FINDING: Agreement doesn't significantly improve accuracy.")


if __name__ == "__main__":
    main()
