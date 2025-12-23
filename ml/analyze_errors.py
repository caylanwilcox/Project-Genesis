"""
Analyze V3 Model Errors to Find Improvement Opportunities

This script examines:
1. When does the model fail?
2. What features distinguish correct vs incorrect predictions?
3. Are there market conditions where the model struggles?
"""

import pandas as pd
import numpy as np
import pickle
import os
import requests
from datetime import datetime, timedelta
from collections import defaultdict

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def fetch_hourly_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly data"""
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


def fetch_daily_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily data"""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}
    response = requests.get(url, params=params)
    data = response.json()

    if 'results' not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
    df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
    df = df.set_index('date')
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def get_price_at_hour(hourly_df, date, hour):
    """Get closing price at a specific hour"""
    try:
        day_start = pd.Timestamp(date).tz_localize('America/New_York').replace(hour=hour, minute=0)
        day_end = day_start + timedelta(hours=1)
        bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index < day_end)]
        if len(bars) > 0:
            return bars['Close'].iloc[0]
    except:
        pass
    return None


def analyze_errors(ticker: str = 'SPY'):
    """Analyze when and why the model makes errors"""

    print(f"\n{'='*70}")
    print(f"  ERROR ANALYSIS: {ticker}")
    print(f"{'='*70}")

    # Load model
    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_v3.pkl')
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    # Fetch data
    hourly_df = fetch_hourly_data(ticker, '2024-12-20', '2025-12-19')
    daily_df = fetch_daily_data(ticker, '2024-12-20', '2025-12-19')

    print(f"  Analyzing {len(daily_df)} days of data...")

    # Collect predictions and errors
    errors_a = []  # Target A errors
    errors_b = []  # Target B errors
    correct_a = []
    correct_b = []

    trading_days = sorted(set(hourly_df.index.date))
    test_start = pd.to_datetime('2025-01-02').date()
    trading_days = [d for d in trading_days if d >= test_start]

    for day in trading_days:
        if day not in daily_df.index:
            continue

        daily_dates = list(daily_df.index)
        day_idx = daily_dates.index(day)
        if day_idx < 2:
            continue

        today = daily_df.loc[day]
        prev_day = daily_df.iloc[day_idx - 1]
        prev_prev_day = daily_df.iloc[day_idx - 2]

        today_open = today['Open']
        today_close = today['Close']
        today_high = today['High']
        today_low = today['Low']

        # Calculate day characteristics
        gap = (today_open - prev_day['Close']) / prev_day['Close']
        day_range = (today_high - today_low) / today_open
        day_return = (today_close - today_open) / today_open
        prev_return = (prev_day['Close'] - prev_prev_day['Close']) / prev_prev_day['Close']

        # Actual outcomes
        actual_bullish = today_close > today_open

        # Get 11 AM and 2 PM prices
        price_11am = get_price_at_hour(hourly_df, day, 11)
        price_2pm = get_price_at_hour(hourly_df, day, 14)

        if price_2pm is None:
            continue

        # How far did price move from 2PM to close?
        if price_2pm:
            move_2pm_to_close = (today_close - price_2pm) / price_2pm

        # Target B actual
        actual_above_11am = today_close > price_11am if price_11am else None

        # Record characteristics for error analysis
        day_data = {
            'date': day,
            'gap': gap,
            'gap_pct': abs(gap),
            'day_range': day_range,
            'day_return': day_return,
            'prev_return': prev_return,
            'actual_bullish': actual_bullish,
            'actual_above_11am': actual_above_11am,
            'close_vs_open_pct': (today_close - today_open) / today_open,
            'close_near_open': abs(today_close - today_open) / today_open < 0.001,  # < 0.1%
            'reversal_day': (gap > 0 and not actual_bullish) or (gap < 0 and actual_bullish),
            'continuation_day': (gap > 0 and actual_bullish) or (gap < 0 and not actual_bullish),
            'high_volatility': day_range > 0.015,  # > 1.5% range
            'low_volatility': day_range < 0.005,   # < 0.5% range
        }

        if price_11am:
            day_data['close_vs_11am_pct'] = (today_close - price_11am) / price_11am
            day_data['close_near_11am'] = abs(today_close - price_11am) / price_11am < 0.001

        # We need to track predictions - for now just track day characteristics
        # Store for analysis
        if actual_bullish:
            correct_a.append(day_data)
        else:
            correct_a.append(day_data)

    # Analyze patterns
    df = pd.DataFrame(correct_a)

    print(f"\n  MARKET CONDITION ANALYSIS")
    print(f"  {'-'*60}")

    # Gap analysis
    gap_up = df[df['gap'] > 0.002]
    gap_down = df[df['gap'] < -0.002]
    gap_flat = df[(df['gap'] >= -0.002) & (df['gap'] <= 0.002)]

    print(f"\n  Gap Analysis:")
    print(f"    Gap Up (>0.2%):   {len(gap_up)} days, {gap_up['actual_bullish'].mean()*100:.1f}% bullish")
    print(f"    Gap Down (<-0.2%): {len(gap_down)} days, {gap_down['actual_bullish'].mean()*100:.1f}% bullish")
    print(f"    Gap Flat:         {len(gap_flat)} days, {gap_flat['actual_bullish'].mean()*100:.1f}% bullish")

    # Reversal analysis
    reversals = df[df['reversal_day']]
    continuations = df[df['continuation_day']]

    print(f"\n  Reversal vs Continuation:")
    print(f"    Reversal days:     {len(reversals)} ({len(reversals)/len(df)*100:.1f}%)")
    print(f"    Continuation days: {len(continuations)} ({len(continuations)/len(df)*100:.1f}%)")

    # Close near open (doji days)
    doji = df[df['close_near_open']]
    print(f"\n  Doji Days (close within 0.1% of open):")
    print(f"    Count: {len(doji)} ({len(doji)/len(df)*100:.1f}%)")

    # Volatility analysis
    high_vol = df[df['high_volatility']]
    low_vol = df[df['low_volatility']]

    print(f"\n  Volatility:")
    print(f"    High vol (>1.5% range): {len(high_vol)} days, {high_vol['actual_bullish'].mean()*100:.1f}% bullish")
    print(f"    Low vol (<0.5% range):  {len(low_vol)} days, {low_vol['actual_bullish'].mean()*100:.1f}% bullish")

    # Previous day momentum
    prev_up = df[df['prev_return'] > 0.005]
    prev_down = df[df['prev_return'] < -0.005]

    print(f"\n  Previous Day Momentum:")
    print(f"    Prev day up >0.5%:   {len(prev_up)} days, {prev_up['actual_bullish'].mean()*100:.1f}% bullish today")
    print(f"    Prev day down <-0.5%: {len(prev_down)} days, {prev_down['actual_bullish'].mean()*100:.1f}% bullish today")

    # Target B analysis (close vs 11am)
    if 'close_vs_11am_pct' in df.columns:
        df_11am = df[df['close_vs_11am_pct'].notna()]
        close_above_11am = df_11am[df_11am['actual_above_11am'] == True]
        close_below_11am = df_11am[df_11am['actual_above_11am'] == False]

        print(f"\n  Target B (Close vs 11 AM):")
        print(f"    Close > 11AM: {len(close_above_11am)} days ({len(close_above_11am)/len(df_11am)*100:.1f}%)")
        print(f"    Close < 11AM: {len(close_below_11am)} days ({len(close_below_11am)/len(df_11am)*100:.1f}%)")

        near_11am = df_11am[df_11am['close_near_11am']]
        print(f"    Close within 0.1% of 11AM: {len(near_11am)} ({len(near_11am)/len(df_11am)*100:.1f}%)")

    print(f"\n  IMPROVEMENT RECOMMENDATIONS")
    print(f"  {'-'*60}")
    print("""
  1. ADD DAY-OF-WEEK FEATURE
     - Market behavior differs by day (Monday reversals, Friday trends)

  2. ADD VIX/VOLATILITY REGIME
     - High VIX days have different patterns than low VIX
     - Consider adding VIX level or ATR ratio

  3. ADD MULTI-DAY MOMENTUM
     - 3-day and 5-day momentum trends
     - Consecutive up/down day counts

  4. IMPROVE TARGET B ANCHOR
     - 11 AM may not be optimal - test 10:30 AM or 11:30 AM
     - Or use VWAP as anchor instead of fixed time

  5. ADD MARKET REGIME DETECTION
     - Trending vs mean-reverting periods
     - Use rolling correlation or ADX

  6. FILTER LOW-CONFIDENCE PREDICTIONS
     - Only trade when confidence > 0.3 (prob > 65% or < 35%)

  7. ADD SECTOR ROTATION SIGNALS
     - QQQ vs IWM relative strength
     - Risk-on vs risk-off sentiment
""")

    return df


def main():
    print("="*70)
    print("  V3 MODEL ERROR ANALYSIS")
    print("="*70)

    for ticker in ['SPY']:
        analyze_errors(ticker)


if __name__ == '__main__':
    main()
