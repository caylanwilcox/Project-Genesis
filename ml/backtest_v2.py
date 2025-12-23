"""
Backtest Intraday Model V2

Tests the new model trained on REAL hourly data.
Target A: Will close > open?
Target B: Will close > 11:00 AM price? (only valid after 11 AM)

Uses TRUE out-of-sample data (test period: 2024-07 to 2025-11).
"""

import pandas as pd
import numpy as np
import pickle
import os
import requests
from datetime import datetime, timedelta

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def fetch_hourly_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly intraday data"""
    print(f"  Fetching hourly data for {ticker}...")

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

    print(f"    Got {len(df)} hourly bars")
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def fetch_daily_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV data"""
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


def load_v2_model(ticker: str):
    """Load V2 intraday model"""
    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_v2.pkl')
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)


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


def predict_at_hour(model_data, hourly_bars_so_far, today_open, prev_day, prev_prev_day, price_11am=None):
    """
    Make prediction using V2 model with real hourly data.
    """
    feature_cols = model_data['feature_cols']
    scaler = model_data['scaler']
    models = model_data['models']
    weights = model_data['weights']

    current_bar = hourly_bars_so_far.iloc[-1]
    current_price = current_bar['Close']
    current_hour = current_bar.name.hour
    high_so_far = hourly_bars_so_far['High'].max()
    low_so_far = hourly_bars_so_far['Low'].min()

    # Time through day
    hours_since_open = (current_bar.name.hour - 9) + (current_bar.name.minute / 60)
    time_pct = min(max(hours_since_open / 6.5, 0), 1)

    # Previous day features
    prev_close = prev_day['Close']
    gap = (today_open - prev_close) / prev_close
    prev_return = (prev_day['Close'] - prev_prev_day['Close']) / prev_prev_day['Close']
    prev_range = (prev_day['High'] - prev_day['Low']) / prev_day['Close']

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
    prob_bullish = 0.0
    for model_name, model in models.items():
        prob_bullish += model.predict_proba(X_scaled)[0][1] * weights.get(model_name, 0.25)

    # Target B: close > 11 AM price (only if we have the model and it's after 11 AM)
    prob_above_11am = None
    if current_hour >= 11 and price_11am is not None and 'models_11am' in model_data:
        models_11am = model_data['models_11am']
        weights_11am = model_data['weights_11am']
        scaler_11am = model_data['scaler_11am']
        feature_cols_11am = model_data['feature_cols_11am']

        # Add current_vs_11am feature
        features['current_vs_11am'] = (current_price - price_11am) / price_11am

        X_b = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols_11am}])[feature_cols_11am]
        X_b = X_b.replace([np.inf, -np.inf], 0).fillna(0)
        X_b_scaled = scaler_11am.transform(X_b)

        prob_above_11am = 0.0
        for model_name, model in models_11am.items():
            prob_above_11am += model.predict_proba(X_b_scaled)[0][1] * weights_11am.get(model_name, 0.25)

    return prob_bullish, prob_above_11am, current_price, time_pct


def backtest_ticker(ticker: str, start_date: str, end_date: str):
    """Backtest V2 model using real hourly data"""

    print(f"\n{'='*70}")
    print(f"  BACKTEST V2: {ticker}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"{'='*70}")

    model_data = load_v2_model(ticker)
    if not model_data:
        print(f"  No V2 model found for {ticker}")
        return None

    has_11am_model = 'models_11am' in model_data
    print(f"  Model version: {model_data.get('version', 'unknown')}")
    print(f"  Has 11 AM model: {has_11am_model}")

    # Fetch data
    fetch_start = (pd.to_datetime(start_date) - timedelta(days=10)).strftime('%Y-%m-%d')
    hourly_df = fetch_hourly_data(ticker, fetch_start, end_date)
    daily_df = fetch_daily_data(ticker, fetch_start, end_date)

    if len(hourly_df) < 10 or len(daily_df) < 5:
        print("  Not enough data")
        return None

    # Get trading days in test period
    test_start = pd.to_datetime(start_date).date()
    test_end = pd.to_datetime(end_date).date()
    trading_days = sorted(set(hourly_df.index.date))
    trading_days = [d for d in trading_days if test_start <= d <= test_end]

    print(f"  Testing {len(trading_days)} trading days")

    results_by_hour = {}

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

        # Get 11 AM price
        price_11am = get_price_at_hour(hourly_df, day, 11)

        # Get hourly bars for this day
        day_start = pd.Timestamp(day).tz_localize('America/New_York').replace(hour=9, minute=0)
        day_end = pd.Timestamp(day).tz_localize('America/New_York').replace(hour=16, minute=30)
        day_bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index <= day_end)]

        if len(day_bars) < 2:
            continue

        # Test at each hour
        for j in range(1, len(day_bars) + 1):
            bars_so_far = day_bars.iloc[:j]
            current_hour = bars_so_far.iloc[-1].name.hour
            hour_label = f"{current_hour:02d}:00"

            prob_bullish, prob_11am, current_price, time_pct = predict_at_hour(
                model_data, bars_so_far, today_open, prev_day, prev_prev_day, price_11am
            )

            # Target A: close > open
            actual_bullish = 1 if today_close > today_open else 0
            predicted_bullish = 1 if prob_bullish > 0.5 else 0
            correct_a = predicted_bullish == actual_bullish
            high_conf_a = prob_bullish > 0.6 or prob_bullish < 0.4

            result = {
                'date': day,
                'hour': hour_label,
                'time_pct': time_pct,
                'current_price': current_price,
                'today_open': today_open,
                'today_close': today_close,
                'prob_bullish': prob_bullish,
                'actual_bullish': actual_bullish,
                'correct_a': correct_a,
                'high_conf_a': high_conf_a,
            }

            # Target B: close > 11 AM price (only after 11 AM)
            if prob_11am is not None and price_11am is not None:
                actual_above_11am = 1 if today_close > price_11am else 0
                predicted_above_11am = 1 if prob_11am > 0.5 else 0
                correct_b = predicted_above_11am == actual_above_11am
                high_conf_b = prob_11am > 0.6 or prob_11am < 0.4

                result['price_11am'] = price_11am
                result['prob_11am'] = prob_11am
                result['actual_above_11am'] = actual_above_11am
                result['correct_b'] = correct_b
                result['high_conf_b'] = high_conf_b

            if hour_label not in results_by_hour:
                results_by_hour[hour_label] = []
            results_by_hour[hour_label].append(result)

    # ============ RESULTS ============
    print(f"\n  {'='*65}")
    print(f"  TARGET A: Will close > open?")
    print(f"  {'='*65}")
    print(f"\n  {'Hour':<8} {'Days':>6} {'Correct':>8} {'Accuracy':>10} {'HiConf':>15}")
    print(f"  {'-'*55}")

    for hour in sorted(results_by_hour.keys()):
        results = results_by_hour[hour]
        df = pd.DataFrame(results)

        total = len(df)
        correct = df['correct_a'].sum()
        accuracy = correct / total if total > 0 else 0

        hc = df[df['high_conf_a']]
        hc_total = len(hc)
        hc_correct = hc['correct_a'].sum() if hc_total > 0 else 0
        hc_accuracy = hc_correct / hc_total if hc_total > 0 else 0

        print(f"  {hour:<8} {total:>6} {correct:>8} {accuracy:>9.1%} {hc_correct}/{hc_total}={hc_accuracy:>5.1%}")

    # Target B results
    if has_11am_model:
        print(f"\n  {'='*65}")
        print(f"  TARGET B: Will close > 11:00 AM price?")
        print(f"  (Only available after 11 AM)")
        print(f"  {'='*65}")
        print(f"\n  {'Hour':<8} {'Days':>6} {'Correct':>8} {'Accuracy':>10} {'HiConf':>15}")
        print(f"  {'-'*55}")

        for hour in sorted(results_by_hour.keys()):
            results = results_by_hour[hour]
            df = pd.DataFrame(results)

            if 'correct_b' not in df.columns or df['correct_b'].isna().all():
                continue

            df_b = df[df['correct_b'].notna()]
            total = len(df_b)
            if total == 0:
                continue

            correct = df_b['correct_b'].sum()
            accuracy = correct / total

            hc = df_b[df_b['high_conf_b']]
            hc_total = len(hc)
            hc_correct = hc['correct_b'].sum() if hc_total > 0 else 0
            hc_accuracy = hc_correct / hc_total if hc_total > 0 else 0

            print(f"  {hour:<8} {total:>6} {correct:>8} {accuracy:>9.1%} {hc_correct}/{hc_total}={hc_accuracy:>5.1%}")

    return results_by_hour


def main():
    print("="*70)
    print("  INTRADAY MODEL V2 BACKTEST")
    print("  Using REAL hourly data - TRUE out-of-sample test")
    print("="*70)
    print()
    print("  Test period: Dec 2-20, 2025")
    print("  (Models trained on 2020-01 to 2024-06, tested on 2024-07 to 2025-11)")
    print()

    # Test on recent data
    for ticker in ['SPY', 'QQQ', 'IWM']:
        backtest_ticker(ticker, '2025-12-02', '2025-12-20')

    print("\n" + "="*70)
    print("  INTERPRETATION")
    print("="*70)
    print("""
  Target A (close > open): Should show ~80-85% on test data
  Target B (close > 11AM): Should show ~75-78% after 11 AM

  If Target B shows consistent accuracy above random (50%), the
  anchored 11 AM approach works better than the old "current price" target.
""")


if __name__ == '__main__':
    main()
