"""
Proper Intraday Model Backtest

Uses REAL intraday (hourly) data - no simulation, no data leakage.
At each hour, we only know:
  - Today's open
  - Price action up to current hour (high/low/close so far)
  - Previous days' data

Tests on recent data (which unfortunately is in training set,
but at least uses real intraday prices with no future leakage).
"""

import pandas as pd
import numpy as np
import pickle
import os
import requests
from datetime import datetime, timedelta

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def fetch_intraday_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch HOURLY intraday data from Polygon"""
    print(f"  Fetching hourly data for {ticker} from {start_date} to {end_date}...")

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{start_date}/{end_date}"
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}

    response = requests.get(url, params=params)
    data = response.json()

    if 'results' not in data:
        print(f"  No results: {data.get('message', 'Unknown error')}")
        return pd.DataFrame()

    df = pd.DataFrame(data['results'])
    df['datetime'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
    df = df.set_index('datetime')

    # Convert to Eastern Time
    df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')

    print(f"  Got {len(df)} hourly bars")
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def fetch_daily_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV data for previous day features"""
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


def load_intraday_model(ticker: str):
    """Load intraday model"""
    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_model.pkl')
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def get_trading_session_hours(date, hourly_df):
    """Get hourly bars for a specific trading day (9:30 AM - 4:00 PM ET)"""
    # Market hours: 9:30 AM - 4:00 PM ET
    # Hourly bars typically at 10:00, 11:00, 12:00, 13:00, 14:00, 15:00, 16:00
    day_start = pd.Timestamp(date).tz_localize('America/New_York').replace(hour=9, minute=0)
    day_end = pd.Timestamp(date).tz_localize('America/New_York').replace(hour=16, minute=30)

    day_bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index <= day_end)]
    return day_bars


def predict_at_hour(model_data, hourly_bars_so_far, today_open, prev_day, prev_prev_day):
    """
    Make prediction using ONLY data available at this hour.

    hourly_bars_so_far: DataFrame of hourly bars from market open to current hour
    today_open: Today's opening price
    prev_day: Previous day's OHLCV
    prev_prev_day: Day before previous (for prev_return)
    """

    feature_cols = model_data['feature_cols']
    scaler = model_data['scaler']
    models = model_data['models']
    weights = model_data['weights']

    # Calculate what we ACTUALLY know at this hour
    current_price = hourly_bars_so_far['Close'].iloc[-1]
    high_so_far = hourly_bars_so_far['High'].max()
    low_so_far = hourly_bars_so_far['Low'].min()

    # Time through day (6.5 hour session)
    hours_elapsed = len(hourly_bars_so_far)
    time_pct = min(hours_elapsed / 7.0, 1.0)  # ~7 hourly bars in a day

    # Previous day features
    prev_close = prev_day['Close']
    prev_high = prev_day['High']
    prev_low = prev_day['Low']

    gap = (today_open - prev_close) / prev_close
    prev_return = (prev_close - prev_prev_day['Close']) / prev_prev_day['Close']
    prev_range = (prev_high - prev_low) / prev_close

    range_so_far = max(high_so_far - low_so_far, 0.0001)

    features = {
        'time_pct': time_pct,
        'time_remaining': 1 - time_pct,
        'gap': gap,
        'gap_direction': 1 if gap > 0 else (-1 if gap < 0 else 0),
        'gap_size': abs(gap),
        'prev_return': prev_return,
        'prev_range': prev_range,
        'current_vs_open': (current_price - today_open) / today_open,
        'current_vs_open_direction': 1 if current_price > today_open else (-1 if current_price < today_open else 0),
        'position_in_range': (current_price - low_so_far) / range_so_far if range_so_far > 0 else 0.5,
        'range_so_far_pct': range_so_far / today_open,
        'high_so_far_pct': (high_so_far - today_open) / today_open,
        'low_so_far_pct': (today_open - low_so_far) / today_open,
        'above_open': 1 if current_price > today_open else 0,
        'near_high': 1 if (high_so_far - current_price) < (current_price - low_so_far) else 0,
        'gap_filled': 1 if (gap > 0 and low_so_far <= prev_close) or (gap <= 0 and high_so_far >= prev_close) else 0,
    }

    X = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}])[feature_cols]
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    X_scaled = scaler.transform(X)

    # Target A: close > open
    prob_open = 0.0
    for model_name, model in models.items():
        prob_open += model.predict_proba(X_scaled)[0][1] * weights.get(model_name, 0.25)

    # Target B: close > current (if model has this)
    prob_current = None
    models_current = model_data.get('models_current')
    weights_current = model_data.get('weights_current')
    if models_current and weights_current:
        prob_current = 0.0
        for model_name, model in models_current.items():
            prob_current += model.predict_proba(X_scaled)[0][1] * weights_current.get(model_name, 0.25)

    return prob_open, prob_current, current_price, time_pct


def backtest_ticker(ticker: str, start_date: str, end_date: str):
    """
    Backtest using REAL hourly data.

    At each hour, we only see data up to that hour - no future leakage.
    """

    print(f"\n{'='*80}")
    print(f"  PROPER BACKTEST: {ticker}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Using REAL hourly intraday data (no simulation)")
    print(f"{'='*80}")

    # Load model
    model_data = load_intraday_model(ticker)
    if not model_data:
        print(f"  No model found for {ticker}")
        return None

    has_current_model = 'models_current' in model_data and model_data['models_current']
    print(f"  Model targets: close>open" + (" AND close>current" if has_current_model else ""))

    # Fetch data
    # Need extra days before start for prev_prev_day
    fetch_start = (pd.to_datetime(start_date) - timedelta(days=10)).strftime('%Y-%m-%d')

    hourly_df = fetch_intraday_data(ticker, fetch_start, end_date)
    daily_df = fetch_daily_data(ticker, fetch_start, end_date)

    if len(hourly_df) < 10 or len(daily_df) < 5:
        print(f"  Not enough data")
        return None

    # Get unique trading days in our test period
    test_start = pd.to_datetime(start_date).date()
    test_end = pd.to_datetime(end_date).date()

    trading_days = sorted(set(hourly_df.index.date))
    trading_days = [d for d in trading_days if test_start <= d <= test_end]

    print(f"  Testing {len(trading_days)} trading days")

    # Results by hour
    results_by_hour = {}  # hour_label -> list of results

    for day in trading_days:
        # Get this day's hourly bars
        day_bars = get_trading_session_hours(day, hourly_df)

        if len(day_bars) < 2:
            continue

        # Get previous days for features
        daily_dates = list(daily_df.index)
        if day not in daily_dates:
            continue

        day_idx = daily_dates.index(day)
        if day_idx < 2:
            continue

        prev_day = daily_df.iloc[day_idx - 1]
        prev_prev_day = daily_df.iloc[day_idx - 2]
        today_daily = daily_df.loc[day]

        today_open = today_daily['Open']
        today_close = today_daily['Close']

        # Test at each hour during the day
        for i in range(1, len(day_bars) + 1):
            bars_so_far = day_bars.iloc[:i]
            current_hour = bars_so_far.index[-1]
            hour_label = current_hour.strftime('%H:00')

            prob_open, prob_current, current_price, time_pct = predict_at_hour(
                model_data, bars_so_far, today_open, prev_day, prev_prev_day
            )

            # Target A: Did close > open?
            actual_bullish = 1 if today_close > today_open else 0
            predicted_bullish = 1 if prob_open > 0.5 else 0
            correct_open = predicted_bullish == actual_bullish
            high_conf_open = prob_open > 0.6 or prob_open < 0.4

            result = {
                'date': day,
                'hour': hour_label,
                'time_pct': time_pct,
                'current_price': current_price,
                'today_open': today_open,
                'today_close': today_close,
                # Target A
                'prob_open': prob_open,
                'actual_bullish': actual_bullish,
                'correct_open': correct_open,
                'high_conf_open': high_conf_open,
            }

            # Target B: Did close > current?
            if prob_current is not None:
                actual_higher = 1 if today_close > current_price else 0
                predicted_higher = 1 if prob_current > 0.5 else 0
                correct_current = predicted_higher == actual_higher
                high_conf_current = prob_current > 0.6 or prob_current < 0.4

                result['prob_current'] = prob_current
                result['actual_higher'] = actual_higher
                result['correct_current'] = correct_current
                result['high_conf_current'] = high_conf_current

            if hour_label not in results_by_hour:
                results_by_hour[hour_label] = []
            results_by_hour[hour_label].append(result)

    # Print results
    print(f"\n  {'='*70}")
    print(f"  TARGET A: Will close > open? (Bullish day prediction)")
    print(f"  {'='*70}")
    print(f"\n  {'Hour':<8} {'Time%':>8} {'Total':>8} {'Correct':>10} {'Accuracy':>10} {'HiConf':>12}")
    print(f"  {'-'*60}")

    summary = {}
    for hour in sorted(results_by_hour.keys()):
        results = results_by_hour[hour]
        df = pd.DataFrame(results)

        total = len(df)
        correct = df['correct_open'].sum()
        accuracy = correct / total if total > 0 else 0

        hc_df = df[df['high_conf_open']]
        hc_total = len(hc_df)
        hc_correct = hc_df['correct_open'].sum() if hc_total > 0 else 0
        hc_accuracy = hc_correct / hc_total if hc_total > 0 else 0

        avg_time_pct = df['time_pct'].mean()

        print(f"  {hour:<8} {avg_time_pct:>7.0%} {total:>8} {correct:>10} {accuracy:>9.1%} {hc_correct}/{hc_total}={hc_accuracy:>5.1%}")

        summary[hour] = {
            'time_pct': avg_time_pct,
            'total': total, 'correct': correct, 'accuracy': accuracy,
            'hc_total': hc_total, 'hc_correct': hc_correct, 'hc_accuracy': hc_accuracy,
        }

    # Target B summary
    if has_current_model and 'prob_current' in results_by_hour.get(list(results_by_hour.keys())[0], [{}])[0]:
        print(f"\n  {'='*70}")
        print(f"  TARGET B: Will close > current price?")
        print(f"  {'='*70}")
        print(f"\n  {'Hour':<8} {'Time%':>8} {'Total':>8} {'Correct':>10} {'Accuracy':>10} {'HiConf':>12}")
        print(f"  {'-'*60}")

        for hour in sorted(results_by_hour.keys()):
            results = results_by_hour[hour]
            df = pd.DataFrame(results)

            if 'correct_current' not in df.columns:
                continue

            total = len(df)
            correct = df['correct_current'].sum()
            accuracy = correct / total if total > 0 else 0

            hc_df = df[df['high_conf_current']]
            hc_total = len(hc_df)
            hc_correct = hc_df['correct_current'].sum() if hc_total > 0 else 0
            hc_accuracy = hc_correct / hc_total if hc_total > 0 else 0

            avg_time_pct = df['time_pct'].mean()

            print(f"  {hour:<8} {avg_time_pct:>7.0%} {total:>8} {correct:>10} {accuracy:>9.1%} {hc_correct}/{hc_total}={hc_accuracy:>5.1%}")

    # Show some example predictions
    print(f"\n  {'='*70}")
    print(f"  SAMPLE PREDICTIONS (11:00 hour)")
    print(f"  {'='*70}")

    if '11:00' in results_by_hour:
        sample_df = pd.DataFrame(results_by_hour['11:00'])
        print(f"\n  {'Date':<12} {'Open':>8} {'@11am':>8} {'Close':>8} {'Prob':>7} {'Pred':>6} {'Actual':>6} {'Result'}")
        print(f"  {'-'*75}")

        for _, r in sample_df.head(10).iterrows():
            pred = 'BULL' if r['prob_open'] > 0.5 else 'BEAR'
            actual = 'BULL' if r['actual_bullish'] else 'BEAR'
            result = '✓' if r['correct_open'] else '✗'
            print(f"  {r['date']}  {r['today_open']:>8.2f} {r['current_price']:>8.2f} {r['today_close']:>8.2f} {r['prob_open']:>6.1%} {pred:>6} {actual:>6} {result}")

    return {
        'ticker': ticker,
        'results_by_hour': results_by_hour,
        'summary': summary,
    }


def main():
    print("="*80)
    print("  PROPER INTRADAY MODEL BACKTEST")
    print("  Using REAL hourly data - NO simulation, NO data leakage")
    print("="*80)
    print()
    print("  ⚠️  NOTE: This tests on Dec 2-20, 2025 which is IN the training set.")
    print("  This validates the methodology but is NOT true out-of-sample testing.")
    print("  For true validation, would need to retrain excluding this period.")
    print()

    # Test last 3 weeks
    start_date = '2025-12-02'
    end_date = '2025-12-20'

    for ticker in ['SPY', 'QQQ', 'IWM']:
        backtest_ticker(ticker, start_date, end_date)

    print("\n" + "="*80)
    print("  INTERPRETATION")
    print("="*80)
    print("""
  - Early hours (10:00-11:00): Lower accuracy expected, less data available
  - Later hours (14:00-15:00): Higher accuracy as more of the day is known
  - The model should show IMPROVING accuracy through the day
  - If accuracy is ~50% throughout, the model has no predictive power
  - If accuracy is very high early, there may still be data leakage

  A realistic expectation:
  - 10:00: ~52-55% (slight edge)
  - 11:00: ~55-60% (moderate edge)
  - 14:00: ~60-70% (good edge)
  - 15:00: ~70-80% (strong edge, but less time to trade)
""")


if __name__ == '__main__':
    main()
