"""
Backtest: V6 Model + Phase 1 Structure Combined
Tests if structural context improves V6 prediction accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

from compute import compute_intraday_context, compute_swing_context

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')


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
            'open': r['o'],
            'high': r['h'],
            'low': r['l'],
            'close': r['c'],
            'volume': r['v']
        })
    return bars


def fetch_1m_bars(ticker: str, date: str) -> list:
    """Fetch 1-minute bars for a specific date."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}"
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}
    resp = requests.get(url, params=params)
    data = resp.json()

    if 'results' not in data:
        return []

    bars = []
    for r in data['results']:
        ts = datetime.fromtimestamp(r['t'] / 1000)
        bars.append({
            'timestamp': ts.strftime('%H:%M'),
            'open': r['o'],
            'high': r['h'],
            'low': r['l'],
            'close': r['c'],
            'volume': r['v']
        })
    return bars


def get_price_at_time(bars_1m: list, target_time: str) -> float:
    """Get price at specific time from 1m bars."""
    for bar in bars_1m:
        if bar['timestamp'] == target_time:
            return bar['close']
    return None


def simulate_v6_signal(daily_bars: list, day_idx: int) -> dict:
    """
    Simulate V6-like signal based on features.
    Returns probability and direction.
    """
    if day_idx < 5:
        return {'prob': 0.5, 'direction': 'NEUTRAL', 'confidence': 50}

    today = daily_bars[day_idx]
    prev = daily_bars[day_idx - 1]

    # Simple feature-based probability (mimics V6 logic)
    score = 0.5

    # Gap effect
    gap = (today['open'] - prev['close']) / prev['close']
    if gap > 0.005:  # Gap up
        score -= 0.05  # Mean reversion tendency
    elif gap < -0.005:  # Gap down
        score += 0.05

    # Previous day effect
    prev_return = (prev['close'] - prev['open']) / prev['open']
    if prev_return > 0.01:
        score -= 0.03  # Mean reversion
    elif prev_return < -0.01:
        score += 0.03

    # 3-day momentum
    if day_idx >= 3:
        ret_3d = (prev['close'] - daily_bars[day_idx-3]['close']) / daily_bars[day_idx-3]['close']
        if ret_3d > 0.02:
            score -= 0.02
        elif ret_3d < -0.02:
            score += 0.02

    # 5-day momentum
    if day_idx >= 5:
        ret_5d = (prev['close'] - daily_bars[day_idx-5]['close']) / daily_bars[day_idx-5]['close']
        if ret_5d > 0.03:
            score -= 0.03
        elif ret_5d < -0.03:
            score += 0.03

    # Clamp probability
    prob = max(0.3, min(0.7, score))

    direction = 'BULLISH' if prob > 0.5 else 'BEARISH'
    confidence = abs(prob - 0.5) * 200  # Convert to 50-100 scale

    return {
        'prob': prob,
        'direction': direction,
        'confidence': 50 + confidence
    }


def get_phase1_filter(bars_1m: list, daily_bars: list, day_idx: int) -> dict:
    """
    Get Phase 1 structural context for filtering.
    Returns structural signals that can confirm/reject V6.
    """
    if not bars_1m or day_idx < 2:
        return {'filter': 'NEUTRAL', 'reason': 'insufficient data'}

    # Get prior day
    prior_day = {
        'high': daily_bars[day_idx - 1]['high'],
        'low': daily_bars[day_idx - 1]['low'],
        'close': daily_bars[day_idx - 1]['close']
    }

    # Compute swing context
    swing_bars = daily_bars[:day_idx]
    swing_ctx = compute_swing_context('TEST', swing_bars) if len(swing_bars) >= 5 else None

    # Get bars up to 12 PM for intraday context
    bars_to_noon = [b for b in bars_1m if b['timestamp'] <= '12:00']
    if len(bars_to_noon) < 30:
        return {'filter': 'NEUTRAL', 'reason': 'not enough intraday data'}

    # Compute intraday context
    intraday_ctx = compute_intraday_context(
        ticker='TEST',
        bars_1m=bars_to_noon,
        prior_day=prior_day,
        swing_context=swing_ctx,
        current_time='12:00'
    )

    if not intraday_ctx:
        return {'filter': 'NEUTRAL', 'reason': 'context computation failed'}

    # Extract structural signals
    auction_state = intraday_ctx['auction']['state']
    auction_direction = intraday_ctx['auction']['resolved_direction']
    has_failure = intraday_ctx['failure']['present']
    failure_types = intraday_ctx['failure']['types']
    risk_level = intraday_ctx['beware']['risk_level']

    # Determine filter
    result = {
        'auction_state': auction_state,
        'auction_direction': auction_direction,
        'has_failure': has_failure,
        'failure_types': failure_types,
        'risk_level': risk_level,
        'filter': 'NEUTRAL',
        'reason': ''
    }

    # CONFIRM: Clean resolved auction in same direction
    if auction_state == 'RESOLVED' and not has_failure:
        if auction_direction == 'UP':
            result['filter'] = 'BULLISH'
            result['reason'] = 'Clean resolved UP, no failures'
        elif auction_direction == 'DOWN':
            result['filter'] = 'BEARISH'
            result['reason'] = 'Clean resolved DOWN, no failures'

    # REJECT: Failed structures
    if has_failure:
        if 'FAILED_BREAKOUT' in failure_types:
            result['filter'] = 'BEARISH'  # Failed breakout = bearish
            result['reason'] = 'Failed breakout detected'
        elif 'FAILED_BREAKDOWN' in failure_types:
            result['filter'] = 'BULLISH'  # Failed breakdown = bullish
            result['reason'] = 'Failed breakdown detected'

    # CAUTION: Balanced market
    if auction_state == 'BALANCED':
        result['filter'] = 'NEUTRAL'
        result['reason'] = 'Balanced/rotational session'

    return result


def run_backtest(ticker: str, start_date: str, end_date: str, sample_days: int = 50):
    """Run backtest comparing V6 alone vs V6 + Phase 1."""

    print(f"\n{'='*60}")
    print(f"BACKTEST: {ticker}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}")

    # Fetch daily data
    print("Fetching daily data...")
    daily_bars = fetch_daily_bars(ticker, start_date, end_date)
    print(f"  Got {len(daily_bars)} daily bars")

    if len(daily_bars) < 20:
        print("  Not enough data")
        return None

    # Results tracking
    results = {
        'v6_only': {'wins': 0, 'losses': 0, 'trades': 0},
        'v6_confirmed': {'wins': 0, 'losses': 0, 'trades': 0},
        'v6_rejected': {'wins': 0, 'losses': 0, 'trades': 0},
        'v6_neutral': {'wins': 0, 'losses': 0, 'trades': 0},
    }

    # Sample random days (to avoid API limits)
    test_indices = list(range(10, min(len(daily_bars), 10 + sample_days)))

    print(f"\nTesting {len(test_indices)} days...")

    for i, day_idx in enumerate(test_indices):
        date = daily_bars[day_idx]['date']

        # Get actual outcome
        today = daily_bars[day_idx]
        actual_bullish = today['close'] > today['open']

        # Get V6 signal
        v6 = simulate_v6_signal(daily_bars, day_idx)
        v6_bullish = v6['direction'] == 'BULLISH'

        # V6 alone result
        v6_correct = (v6_bullish == actual_bullish)
        results['v6_only']['trades'] += 1
        if v6_correct:
            results['v6_only']['wins'] += 1
        else:
            results['v6_only']['losses'] += 1

        # Fetch intraday data for Phase 1
        bars_1m = fetch_1m_bars(ticker, date)

        if bars_1m:
            # Get Phase 1 filter
            p1 = get_phase1_filter(bars_1m, daily_bars, day_idx)

            # Categorize by Phase 1 alignment
            if p1['filter'] == 'NEUTRAL':
                bucket = 'v6_neutral'
            elif (p1['filter'] == 'BULLISH' and v6_bullish) or (p1['filter'] == 'BEARISH' and not v6_bullish):
                bucket = 'v6_confirmed'
            else:
                bucket = 'v6_rejected'

            results[bucket]['trades'] += 1
            if v6_correct:
                results[bucket]['wins'] += 1
            else:
                results[bucket]['losses'] += 1

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_indices)} days...")

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    for name, data in results.items():
        if data['trades'] > 0:
            win_rate = data['wins'] / data['trades'] * 100
            print(f"\n{name.upper().replace('_', ' ')}:")
            print(f"  Trades: {data['trades']}")
            print(f"  Wins: {data['wins']}")
            print(f"  Losses: {data['losses']}")
            print(f"  Win Rate: {win_rate:.1f}%")

    return results


def main():
    """Run combined backtest."""
    print("="*60)
    print("V6 + PHASE 1 COMBINED BACKTEST")
    print("="*60)
    print("\nHypothesis: Phase 1 structural context can filter V6 signals")
    print("- V6 CONFIRMED by Phase 1 → Higher accuracy")
    print("- V6 REJECTED by Phase 1 → Lower accuracy (avoid these)")
    print("- V6 NEUTRAL Phase 1 → Baseline accuracy")

    # Test on recent data (last 2 months)
    end_date = '2025-12-30'
    start_date = '2025-10-01'

    all_results = {}

    for ticker in ['SPY', 'QQQ', 'IWM']:
        results = run_backtest(ticker, start_date, end_date, sample_days=40)
        if results:
            all_results[ticker] = results

    # Aggregate results
    print("\n" + "="*60)
    print("AGGREGATE RESULTS (ALL TICKERS)")
    print("="*60)

    totals = {
        'v6_only': {'wins': 0, 'losses': 0, 'trades': 0},
        'v6_confirmed': {'wins': 0, 'losses': 0, 'trades': 0},
        'v6_rejected': {'wins': 0, 'losses': 0, 'trades': 0},
        'v6_neutral': {'wins': 0, 'losses': 0, 'trades': 0},
    }

    for ticker, results in all_results.items():
        for bucket, data in results.items():
            totals[bucket]['wins'] += data['wins']
            totals[bucket]['losses'] += data['losses']
            totals[bucket]['trades'] += data['trades']

    print("\n| Category | Trades | Wins | Win Rate |")
    print("|----------|--------|------|----------|")

    for name, data in totals.items():
        if data['trades'] > 0:
            win_rate = data['wins'] / data['trades'] * 100
            label = name.replace('_', ' ').title()
            print(f"| {label:20} | {data['trades']:6} | {data['wins']:4} | {win_rate:6.1f}% |")

    # Conclusion
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)

    if totals['v6_confirmed']['trades'] > 0 and totals['v6_rejected']['trades'] > 0:
        confirmed_rate = totals['v6_confirmed']['wins'] / totals['v6_confirmed']['trades'] * 100
        rejected_rate = totals['v6_rejected']['wins'] / totals['v6_rejected']['trades'] * 100
        baseline_rate = totals['v6_only']['wins'] / totals['v6_only']['trades'] * 100

        improvement = confirmed_rate - baseline_rate

        print(f"\nV6 Baseline: {baseline_rate:.1f}%")
        print(f"V6 + Phase 1 Confirmed: {confirmed_rate:.1f}%")
        print(f"V6 + Phase 1 Rejected: {rejected_rate:.1f}%")
        print(f"\nImprovement when confirmed: {improvement:+.1f}%")

        if confirmed_rate > rejected_rate:
            print("\n✓ Phase 1 DOES improve signal quality!")
            print("  → Take V6 signals when Phase 1 confirms")
            print("  → Avoid V6 signals when Phase 1 rejects")
        else:
            print("\n? Results inconclusive - need more data")


if __name__ == "__main__":
    main()
