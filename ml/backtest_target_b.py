"""
Backtest Target B (Close > 11 AM) predictions at 12 PM session start
Uses V6 time-split models for out-of-sample testing
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
        chunk_end = min(current_start + timedelta(days=60), end)
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


def calculate_multi_day_features(daily_df, day_idx):
    features = {}
    if day_idx < 10:
        return {'return_3d': 0, 'return_5d': 0, 'volatility_5d': 0, 'mean_reversion_signal': 0, 'consecutive_up': 0, 'consecutive_down': 0}

    close_today = daily_df.iloc[day_idx - 1]['Close']

    for days, name in [(3, '3d'), (5, '5d')]:
        if day_idx > days:
            close_past = daily_df.iloc[day_idx - days - 1]['Close']
            features[f'return_{name}'] = (close_today - close_past) / close_past
        else:
            features[f'return_{name}'] = 0

    returns = []
    for i in range(1, min(6, day_idx)):
        if day_idx - i - 1 >= 0:
            c1 = daily_df.iloc[day_idx - i]['Close']
            c2 = daily_df.iloc[day_idx - i - 1]['Close']
            returns.append((c1 - c2) / c2)
    features['volatility_5d'] = np.std(returns) if len(returns) >= 3 else 0

    prev_return = (daily_df.iloc[day_idx - 1]['Close'] - daily_df.iloc[day_idx - 1]['Open']) / daily_df.iloc[day_idx - 1]['Open']
    features['mean_reversion_signal'] = -prev_return

    consecutive_up = 0
    consecutive_down = 0
    for i in range(1, min(6, day_idx)):
        if daily_df.iloc[day_idx - i]['Close'] > daily_df.iloc[day_idx - i]['Open']:
            if consecutive_down == 0:
                consecutive_up += 1
            else:
                break
        elif daily_df.iloc[day_idx - i]['Close'] < daily_df.iloc[day_idx - i]['Open']:
            if consecutive_up == 0:
                consecutive_down += 1
            else:
                break
    features['consecutive_up'] = consecutive_up
    features['consecutive_down'] = consecutive_down

    return features


def create_features(bars_so_far, today_open, prev_day, prev_prev_day, daily_df, day_idx, price_11am=None):
    current_bar = bars_so_far.iloc[-1]
    current_price = current_bar['Close']

    high_so_far = bars_so_far['High'].max()
    low_so_far = bars_so_far['Low'].min()

    hours_since_open = (current_bar.name.hour - 9) + (current_bar.name.minute / 60)
    time_pct = min(max(hours_since_open / 6.5, 0), 1)

    prev_close = prev_day['Close']
    prev_open = prev_day['Open']
    prev_high = prev_day['High']
    prev_low = prev_day['Low']

    gap = (today_open - prev_close) / prev_close
    prev_return = (prev_close - prev_prev_day['Close']) / prev_prev_day['Close']
    prev_range = (prev_high - prev_low) / prev_close
    prev_body = (prev_close - prev_open) / prev_open
    range_so_far = max(high_so_far - low_so_far, 0.0001)

    features = {
        'time_pct': time_pct,
        'gap': gap,
        'gap_size': abs(gap),
        'gap_direction': 1 if gap > 0 else (-1 if gap < 0 else 0),
        'prev_return': prev_return,
        'prev_range': prev_range,
        'prev_body': prev_body,
        'prev_bullish': 1 if prev_close > prev_open else 0,
        'current_vs_open': (current_price - today_open) / today_open,
        'current_vs_open_direction': 1 if current_price > today_open else (-1 if current_price < today_open else 0),
        'position_in_range': (current_price - low_so_far) / range_so_far,
        'range_so_far_pct': range_so_far / today_open,
        'above_open': 1 if current_price > today_open else 0,
        'near_high': 1 if (high_so_far - current_price) < (current_price - low_so_far) else 0,
        'gap_filled': 1 if (gap > 0 and low_so_far <= prev_close) or (gap <= 0 and high_so_far >= prev_close) else 0,
        'morning_reversal': 1 if (gap > 0 and current_price < today_open) or (gap < 0 and current_price > today_open) else 0,
    }

    if len(bars_so_far) >= 2:
        features['last_hour_return'] = (current_price - bars_so_far['Close'].iloc[-2]) / bars_so_far['Close'].iloc[-2]
    else:
        features['last_hour_return'] = 0

    bullish_bars = sum(1 for i in range(len(bars_so_far)) if bars_so_far['Close'].iloc[i] > bars_so_far['Open'].iloc[i])
    features['bullish_bar_ratio'] = bullish_bars / len(bars_so_far)

    if len(bars_so_far) >= 1:
        features['first_hour_return'] = (bars_so_far['Close'].iloc[0] - today_open) / today_open
    else:
        features['first_hour_return'] = 0

    multi_day = calculate_multi_day_features(daily_df, day_idx)
    features.update(multi_day)

    if price_11am is not None:
        features['current_vs_11am'] = (current_price - price_11am) / price_11am
        features['above_11am'] = 1 if current_price > price_11am else 0
    else:
        features['current_vs_11am'] = 0
        features['above_11am'] = 0

    features['is_monday'] = 1 if current_bar.name.dayofweek == 0 else 0
    features['is_friday'] = 1 if current_bar.name.dayofweek == 4 else 0

    return features


def backtest_target_b(ticker):
    """Backtest Target B predictions at 12 PM"""
    print(f"\n{'='*60}")
    print(f"  TARGET B BACKTEST: {ticker}")
    print(f"{'='*60}")

    # Load model
    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_v6.pkl')
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}")
        return None

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    print(f"  Model version: {model_data.get('version', 'unknown')}")
    print(f"  Trained at: {model_data.get('trained_at', 'unknown')}")

    # Fetch test data (2025 only - out of sample)
    TEST_START = '2025-01-02'
    TEST_END = '2025-12-19'

    daily_df = fetch_daily_data(ticker, '2024-01-01', TEST_END)  # Need prev days for features
    hourly_df = fetch_hourly_data(ticker, TEST_START, TEST_END)

    feature_cols = model_data['feature_cols']
    scaler = model_data['scaler_late']
    models_b = model_data['models_late_b']
    weights_b = model_data['weights_late_b']

    # Backtest at 12 PM
    results = {
        'total': 0,
        'correct': 0,
        'high_conf_total': 0,
        'high_conf_correct': 0,
        'by_prediction': {'bullish': {'total': 0, 'correct': 0}, 'bearish': {'total': 0, 'correct': 0}},
        'by_confidence': {}
    }

    trading_days = sorted(set(hourly_df.index.date))
    start_date = pd.to_datetime(TEST_START).date()

    for day in trading_days:
        if day < start_date:
            continue
        if day not in daily_df.index:
            continue

        daily_dates = list(daily_df.index)
        day_idx = daily_dates.index(day)
        if day_idx < 10:
            continue

        today = daily_df.loc[day]
        prev_day = daily_df.iloc[day_idx - 1]
        prev_prev_day = daily_df.iloc[day_idx - 2]

        today_open = today['Open']
        today_close = today['Close']

        # Get 11 AM and 12 PM prices
        price_11am = get_price_at_hour(hourly_df, day, 11)
        if price_11am is None:
            continue

        # Get bars up to 12 PM
        day_start = pd.Timestamp(day).tz_localize('America/New_York').replace(hour=9, minute=0)
        noon = pd.Timestamp(day).tz_localize('America/New_York').replace(hour=12, minute=0)
        bars_to_noon = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index <= noon)]

        if len(bars_to_noon) < 3:
            continue

        # Actual target
        actual = 1 if today_close > price_11am else 0

        # Create features at 12 PM
        features = create_features(bars_to_noon, today_open, prev_day, prev_prev_day, daily_df, day_idx, price_11am)

        # Prepare for model
        X = pd.DataFrame([features])[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
        X_scaled = scaler.transform(X)

        # Ensemble prediction
        pred_prob = 0
        for name, model in models_b.items():
            pred_prob += model.predict_proba(X_scaled)[:, 1][0] * weights_b[name]

        predicted = 1 if pred_prob >= 0.5 else 0
        correct = 1 if predicted == actual else 0

        results['total'] += 1
        results['correct'] += correct

        # By prediction direction
        direction = 'bullish' if predicted == 1 else 'bearish'
        results['by_prediction'][direction]['total'] += 1
        results['by_prediction'][direction]['correct'] += correct

        # High confidence (>= 70%)
        confidence = max(pred_prob, 1 - pred_prob)
        if confidence >= 0.70:
            results['high_conf_total'] += 1
            results['high_conf_correct'] += correct

        # By confidence bucket
        bucket = f"{int(confidence * 10) * 10}%"
        if bucket not in results['by_confidence']:
            results['by_confidence'][bucket] = {'total': 0, 'correct': 0}
        results['by_confidence'][bucket]['total'] += 1
        results['by_confidence'][bucket]['correct'] += correct

    # Print results
    acc = results['correct'] / results['total'] if results['total'] > 0 else 0
    high_conf_acc = results['high_conf_correct'] / results['high_conf_total'] if results['high_conf_total'] > 0 else 0

    print(f"\n  RESULTS ({ticker}):")
    print(f"  {'='*50}")
    print(f"  Total trading days tested: {results['total']}")
    print(f"  Overall accuracy: {acc:.1%} ({results['correct']}/{results['total']})")
    print(f"  High confidence (>=70%) accuracy: {high_conf_acc:.1%} ({results['high_conf_correct']}/{results['high_conf_total']})")

    print(f"\n  BY PREDICTION:")
    for direction, stats in results['by_prediction'].items():
        if stats['total'] > 0:
            dir_acc = stats['correct'] / stats['total']
            print(f"    {direction.upper()}: {dir_acc:.1%} ({stats['correct']}/{stats['total']})")

    print(f"\n  BY CONFIDENCE:")
    for bucket in sorted(results['by_confidence'].keys(), reverse=True):
        stats = results['by_confidence'][bucket]
        if stats['total'] > 0:
            bucket_acc = stats['correct'] / stats['total']
            print(f"    {bucket}: {bucket_acc:.1%} ({stats['correct']}/{stats['total']})")

    return {
        'ticker': ticker,
        'accuracy': acc,
        'high_conf_accuracy': high_conf_acc,
        'total_days': results['total'],
        'high_conf_days': results['high_conf_total']
    }


def main():
    print("="*60)
    print("  TARGET B BACKTEST (Close > 11 AM)")
    print("  Testing at 12 PM Session Start")
    print("="*60)

    all_results = []
    for ticker in ['SPY', 'QQQ', 'IWM']:
        result = backtest_target_b(ticker)
        if result:
            all_results.append(result)

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"\n  {'Ticker':<8} {'Overall':>12} {'High Conf':>12} {'Days':>8}")
    print(f"  {'-'*44}")

    total_correct = 0
    total_days = 0
    hc_correct = 0
    hc_days = 0

    for r in all_results:
        print(f"  {r['ticker']:<8} {r['accuracy']:>11.1%} {r['high_conf_accuracy']:>11.1%} {r['total_days']:>8}")
        total_correct += r['accuracy'] * r['total_days']
        total_days += r['total_days']
        hc_correct += r['high_conf_accuracy'] * r['high_conf_days']
        hc_days += r['high_conf_days']

    avg_acc = total_correct / total_days if total_days > 0 else 0
    avg_hc = hc_correct / hc_days if hc_days > 0 else 0
    print(f"  {'-'*44}")
    print(f"  {'AVERAGE':<8} {avg_acc:>11.1%} {avg_hc:>11.1%} {total_days:>8}")


if __name__ == '__main__':
    main()
