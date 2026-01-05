"""
Phase 1 Live Runner
Fetches real market data from Polygon and computes Phase 1 context.
"""

import sys
import os
import json
import requests
from datetime import datetime, timedelta

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compute import compute_intraday_context, compute_swing_context

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')


def fetch_1m_bars(ticker: str, date: str) -> list:
    """Fetch 1-minute bars for a specific date."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }

    print(f"  Fetching 1m bars for {ticker} on {date}...")
    response = requests.get(url, params=params)
    data = response.json()

    if 'results' not in data:
        print(f"  No data returned: {data.get('status', 'unknown')}")
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

    print(f"  Got {len(bars)} bars")
    return bars


def fetch_daily_bars(ticker: str, days: int = 60) -> list:
    """Fetch daily bars for swing context."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }

    print(f"  Fetching daily bars for {ticker} ({start_date} to {end_date})...")
    response = requests.get(url, params=params)
    data = response.json()

    if 'results' not in data:
        print(f"  No data returned: {data.get('status', 'unknown')}")
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

    print(f"  Got {len(bars)} daily bars")
    return bars


def get_prior_day(daily_bars: list) -> dict:
    """Get prior day's OHLC from daily bars."""
    if len(daily_bars) < 2:
        return None

    prior = daily_bars[-2]  # Second to last bar
    return {
        'high': prior['high'],
        'low': prior['low'],
        'close': prior['close']
    }


def format_level(level: dict) -> str:
    """Format a level for display."""
    status_icons = {
        'ACCEPTED': '✓',
        'REJECTED': '✗',
        'TESTING': '~',
        'UNTESTED': '-',
        'FAILED_ACCEPTANCE': '!'
    }
    icon = status_icons.get(level['status'], '?')
    strength = f" ({level['strength']})" if level.get('strength') else ""
    return f"  {icon} {level['name']}: ${level['price']:.2f} [{level['status']}{strength}]"


def print_intraday_context(ctx: dict):
    """Pretty print intraday context."""
    print("\n" + "=" * 60)
    print(f"PHASE 1 INTRADAY CONTEXT - {ctx['ticker']}")
    print("=" * 60)

    print(f"\nVersion: {ctx['version']}")
    print(f"As Of: {ctx['as_of']}")
    print(f"Session: {ctx['session']}")
    print(f"Current Price: ${ctx['current_price']:.2f}")
    print(f"Context ID: {ctx['context_id']}")

    # Auction State
    auction = ctx['auction']
    print(f"\n--- AUCTION STATE ---")
    print(f"  State: {auction['state']}")
    print(f"  Direction: {auction['resolved_direction']}")
    print(f"  Rotation Complete: {auction['rotation_complete']}")
    print(f"  Expansion Quality: {auction['expansion_quality']}")

    # Levels
    print(f"\n--- LEVELS ({len(ctx['levels']['set'])}) ---")
    for level in sorted(ctx['levels']['set'], key=lambda x: x['price'], reverse=True):
        print(format_level(level))

    if ctx['levels'].get('nearest_above'):
        na = ctx['levels']['nearest_above']
        print(f"\n  Nearest Above: {na['name']} @ ${na['price']:.2f} ({na['distance_pct']:.3f}%)")
    if ctx['levels'].get('nearest_below'):
        nb = ctx['levels']['nearest_below']
        print(f"  Nearest Below: {nb['name']} @ ${nb['price']:.2f} ({nb['distance_pct']:.3f}%)")

    # Failures
    failure = ctx['failure']
    print(f"\n--- FAILURES ---")
    if failure['present']:
        for f_type in failure['types']:
            print(f"  ⚠️  {f_type}")
        for note in failure.get('notes', []):
            print(f"      {note}")
    else:
        print("  None detected")

    # BEWARE Alerts
    beware = ctx['beware']
    print(f"\n--- BEWARE ALERTS (Risk: {beware['risk_level']}) ---")
    if beware['alerts']:
        for alert in beware['alerts']:
            severity_icon = {'INFO': 'ℹ️', 'WARNING': '⚠️', 'CRITICAL': '🚨'}.get(alert['severity'], '?')
            print(f"  {severity_icon} [{alert['type']}] {alert['message']}")
    else:
        print("  No alerts")

    # Swing Link
    swing = ctx['swing_link']
    print(f"\n--- SWING ALIGNMENT ---")
    print(f"  Alignment: {swing['alignment']}")
    print(f"  Swing Bias: {swing['swing_bias']}")

    print(f"\n--- VALIDITY ---")
    print(f"  Expires: {ctx['validity']['expires_at']}")
    print(f"  Persistence: {ctx['validity']['persistence']}")


def print_swing_context(ctx: dict):
    """Pretty print swing context."""
    print("\n" + "=" * 60)
    print(f"PHASE 1 SWING CONTEXT - {ctx['ticker']}")
    print("=" * 60)

    print(f"\nVersion: {ctx['version']}")
    print(f"As Of Date: {ctx['as_of_date']}")
    print(f"Context ID: {ctx['context_id']}")

    # Auction State
    auction = ctx['auction']
    print(f"\n--- AUCTION STATE ---")
    print(f"  State: {auction['state']}")
    print(f"  Dominant TF: {auction['dominant_tf']}")
    print(f"  Direction: {auction['resolved_direction']}")

    # Bias
    bias = ctx['bias']
    print(f"\n--- BIAS ---")
    print(f"  Context: {bias['context']}")
    print(f"  Strength: {bias['strength']}")
    if bias['invalidation']['level']:
        print(f"  Invalidation: {bias['invalidation']['description']}")

    # Levels
    print(f"\n--- HTF LEVELS ({len(ctx['levels']['set'])}) ---")
    for level in sorted(ctx['levels']['set'], key=lambda x: x['price'], reverse=True):
        print(format_level(level))

    # Failures
    failure = ctx['failure']
    print(f"\n--- FAILURES ---")
    if failure['present']:
        for f_type in failure['types']:
            print(f"  ⚠️  {f_type}")
    else:
        print("  None detected")

    # BEWARE Alerts
    beware = ctx['beware']
    print(f"\n--- BEWARE ALERTS (Risk: {beware['risk_level']}) ---")
    if beware['alerts']:
        for alert in beware['alerts']:
            severity_icon = {'INFO': 'ℹ️', 'WARNING': '⚠️', 'CRITICAL': '🚨'}.get(alert['severity'], '?')
            print(f"  {severity_icon} [{alert['type']}] {alert['message']}")
    else:
        print("  No alerts")


def run_phase1(ticker: str, date: str = None):
    """Run Phase 1 analysis for a ticker."""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    print(f"\n{'#' * 60}")
    print(f"# PHASE 1 STRUCTURE ANALYSIS")
    print(f"# Ticker: {ticker}")
    print(f"# Date: {date}")
    print(f"{'#' * 60}")

    # Fetch data
    print("\n[1/4] Fetching market data...")
    bars_1m = fetch_1m_bars(ticker, date)
    daily_bars = fetch_daily_bars(ticker, days=90)

    if not bars_1m:
        print(f"\n⚠️  No intraday data available for {date}")
        print("    Market may be closed or data not yet available.")
        # Still compute swing context
        if daily_bars:
            print("\n[2/4] Computing swing context...")
            swing_ctx = compute_swing_context(ticker, daily_bars)
            if swing_ctx:
                print_swing_context(swing_ctx)
        return None, None

    prior_day = get_prior_day(daily_bars)

    # Compute swing context first
    print("\n[2/4] Computing swing context...")
    swing_ctx = compute_swing_context(ticker, daily_bars)

    # Get current time from latest bar
    current_time = bars_1m[-1]['timestamp'] if bars_1m else '12:00'

    # Compute intraday context
    print("\n[3/4] Computing intraday context...")
    intraday_ctx = compute_intraday_context(
        ticker=ticker,
        bars_1m=bars_1m,
        prior_day=prior_day,
        swing_context=swing_ctx,
        current_time=current_time
    )

    # Print results
    print("\n[4/4] Results:")

    if swing_ctx:
        print_swing_context(swing_ctx)

    if intraday_ctx:
        print_intraday_context(intraday_ctx)

    # Verify determinism
    print("\n" + "=" * 60)
    print("DETERMINISM CHECK")
    print("=" * 60)

    intraday_ctx2 = compute_intraday_context(
        ticker=ticker,
        bars_1m=bars_1m,
        prior_day=prior_day,
        swing_context=swing_ctx,
        current_time=current_time
    )

    if intraday_ctx and intraday_ctx2:
        # Compare context_ids (should be identical)
        if intraday_ctx['context_id'] == intraday_ctx2['context_id']:
            print(f"✓ Determinism verified - context_id: {intraday_ctx['context_id']}")
        else:
            print(f"✗ DETERMINISM FAILED!")
            print(f"  ID1: {intraday_ctx['context_id']}")
            print(f"  ID2: {intraday_ctx2['context_id']}")

    return intraday_ctx, swing_ctx


def main():
    """Main entry point."""
    tickers = ['SPY', 'QQQ', 'IWM']

    # Use most recent trading day
    today = datetime.now()

    # If weekend, use Friday
    if today.weekday() == 5:  # Saturday
        date = (today - timedelta(days=1)).strftime('%Y-%m-%d')
    elif today.weekday() == 6:  # Sunday
        date = (today - timedelta(days=2)).strftime('%Y-%m-%d')
    else:
        date = today.strftime('%Y-%m-%d')

    print(f"\n{'*' * 60}")
    print(f"* PHASE 1 LIVE MARKET ANALYSIS")
    print(f"* Date: {date}")
    print(f"* Tickers: {', '.join(tickers)}")
    print(f"{'*' * 60}")

    for ticker in tickers:
        run_phase1(ticker, date)

    print("\n" + "=" * 60)
    print("PHASE 1 ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
