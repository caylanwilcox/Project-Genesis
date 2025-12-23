"""
Backtest: Compare 11 AM vs 12 PM Late Session Start

Tests if we can get reliable Target B predictions starting at 11 AM
instead of waiting until 12 PM.
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
    try:
        day_start = pd.Timestamp(date).tz_localize('America/New_York').replace(hour=hour, minute=0)
        day_end = day_start + timedelta(hours=1)
        bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index < day_end)]
        if len(bars) > 0:
            return bars['Close'].iloc[0]
    except:
        pass
    return None


def get_bars_up_to_hour(hourly_df, date, hour):
    """Get all hourly bars from market open up to specified hour"""
    try:
        day_start = pd.Timestamp(date).tz_localize('America/New_York').replace(hour=9, minute=0)
        day_end = pd.Timestamp(date).tz_localize('America/New_York').replace(hour=hour+1, minute=0)
        bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index < day_end)]
        return bars
    except:
        return pd.DataFrame()


def build_features(bars_so_far, price_11am, daily_df, day_idx):
    """Build feature set for prediction"""
    if len(bars_so_far) < 1:
        return None

    today_open = bars_so_far.iloc[0]['Open']
    current_close = bars_so_far.iloc[-1]['Close']
    current_high = bars_so_far['High'].max()
    current_low = bars_so_far['Low'].min()

    prev_day = daily_df.iloc[day_idx - 1]
    prev_prev_day = daily_df.iloc[day_idx - 2] if day_idx >= 2 else prev_day

    features = {
        'gap': (today_open - prev_day['Close']) / prev_day['Close'],
        'prev_day_return': (prev_day['Close'] - prev_day['Open']) / prev_day['Open'],
        'prev_day_range': (prev_day['High'] - prev_day['Low']) / prev_day['Open'],
        'prev_2day_return': (prev_day['Close'] - prev_prev_day['Close']) / prev_prev_day['Close'],
        'current_vs_open': (current_close - today_open) / today_open,
        'current_vs_open_direction': 1 if current_close > today_open else -1,
        'above_open': 1 if current_close > today_open else 0,
        'position_in_range': (current_close - current_low) / (current_high - current_low + 0.0001),
        'near_high': 1 if current_close >= current_high * 0.995 else 0,
        'time_pct': len(bars_so_far) / 8,
        'first_hour_return': (bars_so_far.iloc[0]['Close'] - today_open) / today_open,
        'last_hour_return': (bars_so_far.iloc[-1]['Close'] - bars_so_far.iloc[-2]['Close']) / bars_so_far.iloc[-2]['Close'] if len(bars_so_far) >= 2 else 0,
        'bullish_bar_ratio': sum(1 for _, b in bars_so_far.iterrows() if b['Close'] > b['Open']) / len(bars_so_far),
        'prev_range': (prev_day['High'] - prev_day['Low']) / prev_day['Close'],
    }

    # 11 AM features
    if price_11am is not None:
        features['current_vs_11am'] = (current_close - price_11am) / price_11am
        features['above_11am'] = 1 if current_close > price_11am else 0
    else:
        features['current_vs_11am'] = 0
        features['above_11am'] = 0

    # Multi-day features
    if day_idx >= 6:
        features['return_3d'] = (daily_df.iloc[day_idx-1]['Close'] - daily_df.iloc[day_idx-4]['Close']) / daily_df.iloc[day_idx-4]['Close']
        features['return_5d'] = (daily_df.iloc[day_idx-1]['Close'] - daily_df.iloc[day_idx-6]['Close']) / daily_df.iloc[day_idx-6]['Close']
        returns = [(daily_df.iloc[i]['Close'] - daily_df.iloc[i-1]['Close']) / daily_df.iloc[i-1]['Close'] for i in range(day_idx-5, day_idx)]
        features['volatility_5d'] = np.std(returns)
    else:
        features['return_3d'] = 0
        features['return_5d'] = 0
        features['volatility_5d'] = 0.01

    # Consecutive days
    features['consecutive_up'] = 0
    features['consecutive_down'] = 0
    for i in range(1, min(4, day_idx)):
        d = daily_df.iloc[day_idx - i]
        if d['Close'] > d['Open']:
            features['consecutive_up'] += 1
        else:
            break
    for i in range(1, min(4, day_idx)):
        d = daily_df.iloc[day_idx - i]
        if d['Close'] < d['Open']:
            features['consecutive_down'] += 1
        else:
            break

    return features


def run_backtest():
    print("=" * 60)
    print("BACKTEST: 11 AM vs 12 PM Late Session Start")
    print("=" * 60)

    tickers = ['SPY', 'QQQ', 'IWM']

    # Use full year of 2025 data for larger sample
    end_date = '2025-12-19'
    start_date = '2025-01-02'

    results = {
        '11am': {'correct': 0, 'total': 0, 'by_ticker': {}},
        '12pm': {'correct': 0, 'total': 0, 'by_ticker': {}},
    }

    for ticker in tickers:
        print(f"\n{'='*40}")
        print(f"Processing {ticker}")
        print(f"{'='*40}")

        # Load V6 model
        model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_v6.pkl')
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            continue

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        feature_cols = model_data['feature_cols']

        # Fetch data
        hourly_df = fetch_hourly_data(ticker, start_date, end_date)
        daily_df = fetch_daily_data(ticker, '2025-11-01', end_date)

        if hourly_df.empty or daily_df.empty:
            print(f"  No data for {ticker}")
            continue

        # Get unique trading days
        trading_days = sorted(set(hourly_df.index.date))
        print(f"  Trading days: {len(trading_days)}")

        results['11am']['by_ticker'][ticker] = {'correct': 0, 'total': 0}
        results['12pm']['by_ticker'][ticker] = {'correct': 0, 'total': 0}

        print(f"\n  {'Date':<12} {'11AM Prob':>10} {'11AM Pred':>10} {'12PM Prob':>10} {'12PM Pred':>10} {'Actual':>8} {'11AM':>6} {'12PM':>6}")
        print(f"  {'-'*76}")

        for date in trading_days:
            # Find day index in daily data
            try:
                day_idx = list(daily_df.index).index(date)
            except ValueError:
                continue

            if day_idx < 5:
                continue

            # Get actual close
            today_bars = get_bars_up_to_hour(hourly_df, date, 16)
            if len(today_bars) < 2:
                continue

            today_open = today_bars.iloc[0]['Open']
            today_close = today_bars.iloc[-1]['Close']
            actual_bullish = today_close > today_open

            # Get 11 AM price
            price_11am = get_price_at_hour(hourly_df, date, 11)

            # Actual Target B (Close > 11 AM)
            actual_target_b = today_close > price_11am if price_11am else None

            # ===== 11 AM Prediction =====
            bars_11am = get_bars_up_to_hour(hourly_df, date, 11)
            if len(bars_11am) >= 2 and price_11am:
                features_11am = build_features(bars_11am, price_11am, daily_df, day_idx)
                if features_11am:
                    X_11am = np.array([[features_11am.get(col, 0) for col in feature_cols]])
                    X_11am = np.nan_to_num(X_11am, nan=0, posinf=0, neginf=0)

                    # Use late model for Target B at 11 AM
                    X_scaled = model_data['scaler_late'].transform(X_11am)
                    prob_b_11am = sum(
                        m.predict_proba(X_scaled)[0][1] * model_data['weights_late_b'][name]
                        for name, m in model_data['models_late_b'].items()
                    )
                    pred_11am = prob_b_11am > 0.5
                    correct_11am = pred_11am == actual_target_b

                    results['11am']['total'] += 1
                    results['11am']['by_ticker'][ticker]['total'] += 1
                    if correct_11am:
                        results['11am']['correct'] += 1
                        results['11am']['by_ticker'][ticker]['correct'] += 1
                else:
                    prob_b_11am = None
                    pred_11am = None
                    correct_11am = None
            else:
                prob_b_11am = None
                pred_11am = None
                correct_11am = None

            # ===== 12 PM Prediction =====
            bars_12pm = get_bars_up_to_hour(hourly_df, date, 12)
            if len(bars_12pm) >= 2 and price_11am:
                features_12pm = build_features(bars_12pm, price_11am, daily_df, day_idx)
                if features_12pm:
                    X_12pm = np.array([[features_12pm.get(col, 0) for col in feature_cols]])
                    X_12pm = np.nan_to_num(X_12pm, nan=0, posinf=0, neginf=0)

                    X_scaled = model_data['scaler_late'].transform(X_12pm)
                    prob_b_12pm = sum(
                        m.predict_proba(X_scaled)[0][1] * model_data['weights_late_b'][name]
                        for name, m in model_data['models_late_b'].items()
                    )
                    pred_12pm = prob_b_12pm > 0.5
                    correct_12pm = pred_12pm == actual_target_b

                    results['12pm']['total'] += 1
                    results['12pm']['by_ticker'][ticker]['total'] += 1
                    if correct_12pm:
                        results['12pm']['correct'] += 1
                        results['12pm']['by_ticker'][ticker]['correct'] += 1
                else:
                    prob_b_12pm = None
                    pred_12pm = None
                    correct_12pm = None
            else:
                prob_b_12pm = None
                pred_12pm = None
                correct_12pm = None

            # Print row
            actual_str = "BULL" if actual_target_b else "BEAR" if actual_target_b is not None else "N/A"

            prob_11_str = f"{prob_b_11am:.1%}" if prob_b_11am is not None else "N/A"
            pred_11_str = "BULL" if pred_11am else "BEAR" if pred_11am is not None else "N/A"
            c11_str = "✓" if correct_11am else "✗" if correct_11am is not None else "-"

            prob_12_str = f"{prob_b_12pm:.1%}" if prob_b_12pm is not None else "N/A"
            pred_12_str = "BULL" if pred_12pm else "BEAR" if pred_12pm is not None else "N/A"
            c12_str = "✓" if correct_12pm else "✗" if correct_12pm is not None else "-"

            print(f"  {date}  {prob_11_str:>10} {pred_11_str:>10} {prob_12_str:>10} {pred_12_str:>10} {actual_str:>8} {c11_str:>6} {c12_str:>6}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Target B Accuracy Comparison")
    print("=" * 60)

    print(f"\n{'Session Start':<15} {'Correct':>10} {'Total':>10} {'Accuracy':>12}")
    print("-" * 50)

    for session, data in results.items():
        if data['total'] > 0:
            acc = data['correct'] / data['total']
            print(f"{session:<15} {data['correct']:>10} {data['total']:>10} {acc:>11.1%}")

    print("\n\nBy Ticker:")
    print(f"{'Ticker':<10} {'11 AM Acc':>12} {'12 PM Acc':>12} {'Difference':>12}")
    print("-" * 50)

    for ticker in tickers:
        if ticker in results['11am']['by_ticker'] and ticker in results['12pm']['by_ticker']:
            data_11 = results['11am']['by_ticker'][ticker]
            data_12 = results['12pm']['by_ticker'][ticker]

            acc_11 = data_11['correct'] / data_11['total'] if data_11['total'] > 0 else 0
            acc_12 = data_12['correct'] / data_12['total'] if data_12['total'] > 0 else 0
            diff = acc_11 - acc_12

            diff_str = f"+{diff:.1%}" if diff > 0 else f"{diff:.1%}"
            print(f"{ticker:<10} {acc_11:>11.1%} {acc_12:>11.1%} {diff_str:>12}")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    acc_11 = results['11am']['correct'] / results['11am']['total'] if results['11am']['total'] > 0 else 0
    acc_12 = results['12pm']['correct'] / results['12pm']['total'] if results['12pm']['total'] > 0 else 0

    if acc_11 >= acc_12 - 0.05:  # Within 5%
        print(f"\n✓ 11 AM predictions are viable! ({acc_11:.1%} vs {acc_12:.1%})")
        print("  You can use Target B starting at 11 AM.")
    else:
        print(f"\n✗ 12 PM is significantly better ({acc_12:.1%} vs {acc_11:.1%})")
        print(f"  Recommend waiting until 12 PM for Target B.")


if __name__ == '__main__':
    run_backtest()
