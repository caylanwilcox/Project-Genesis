"""
Backtest V6 Time-Split Model - Last 3 Weeks

Tests the time-split model:
- Early model (9-11 AM)
- Late model (12-4 PM)
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
    current_hour = current_bar.name.hour

    high_so_far = bars_so_far['High'].max()
    low_so_far = bars_so_far['Low'].min()
    volume_so_far = bars_so_far['Volume'].sum()

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


def predict_v6(model_data, features, current_hour):
    """Make prediction using V6 time-split model"""
    feature_cols = model_data['feature_cols']

    X = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}])[feature_cols]
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    # Use early or late model based on time
    if current_hour <= 11:
        scaler = model_data['scaler_early']
        models = model_data['models_early']
        weights = model_data['weights_early']
        X_scaled = scaler.transform(X)

        prob_a = sum(m.predict_proba(X_scaled)[0][1] * weights[n] for n, m in models.items())
        prob_b = None  # No Target B for early session
    else:
        scaler = model_data['scaler_late']
        models_a = model_data['models_late_a']
        models_b = model_data['models_late_b']
        weights_a = model_data['weights_late_a']
        weights_b = model_data['weights_late_b']
        X_scaled = scaler.transform(X)

        prob_a = sum(m.predict_proba(X_scaled)[0][1] * weights_a[n] for n, m in models_a.items())
        prob_b = sum(m.predict_proba(X_scaled)[0][1] * weights_b[n] for n, m in models_b.items())

    return prob_a, prob_b


def backtest_ticker(ticker: str, start_date: str, end_date: str):
    print(f"\n{'='*70}")
    print(f"  BACKTEST V6: {ticker}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"{'='*70}")

    # Load model
    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_v6.pkl')
    if not os.path.exists(model_path):
        print(f"  No V6 model found for {ticker}")
        return None

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    print(f"  Model version: {model_data.get('version', 'unknown')}")

    # Fetch data
    fetch_start = (pd.to_datetime(start_date) - timedelta(days=15)).strftime('%Y-%m-%d')
    hourly_df = fetch_hourly_data(ticker, fetch_start, end_date)
    daily_df = fetch_daily_data(ticker, fetch_start, end_date)

    test_start = pd.to_datetime(start_date).date()
    test_end = pd.to_datetime(end_date).date()
    trading_days = sorted(set(hourly_df.index.date))
    trading_days = [d for d in trading_days if test_start <= d <= test_end]

    print(f"  Testing {len(trading_days)} trading days")

    results_early = []
    results_late = []

    for day in trading_days:
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
        actual_bullish = 1 if today_close > today_open else 0

        price_11am = get_price_at_hour(hourly_df, day, 11)
        actual_above_11am = 1 if (price_11am and today_close > price_11am) else 0

        day_start = pd.Timestamp(day).tz_localize('America/New_York').replace(hour=9, minute=0)
        day_end = pd.Timestamp(day).tz_localize('America/New_York').replace(hour=16, minute=30)
        day_bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index <= day_end)]

        if len(day_bars) < 2:
            continue

        for j in range(1, len(day_bars) + 1):
            bars_so_far = day_bars.iloc[:j]
            current_hour = bars_so_far.iloc[-1].name.hour
            current_price = bars_so_far.iloc[-1]['Close']

            p11 = price_11am if current_hour >= 11 else None
            features = create_features(bars_so_far, today_open, prev_day, prev_prev_day, daily_df, day_idx, p11)

            prob_a, prob_b = predict_v6(model_data, features, current_hour)

            pred_a = 1 if prob_a > 0.5 else 0
            correct_a = pred_a == actual_bullish
            high_conf_a = prob_a > 0.6 or prob_a < 0.4

            result = {
                'date': day,
                'hour': f"{current_hour:02d}:00",
                'current_price': current_price,
                'today_open': today_open,
                'today_close': today_close,
                'prob_a': prob_a,
                'pred_a': pred_a,
                'actual_a': actual_bullish,
                'correct_a': correct_a,
                'high_conf_a': high_conf_a,
            }

            if prob_b is not None and price_11am is not None:
                pred_b = 1 if prob_b > 0.5 else 0
                correct_b = pred_b == actual_above_11am
                high_conf_b = prob_b > 0.6 or prob_b < 0.4
                result['price_11am'] = price_11am
                result['prob_b'] = prob_b
                result['pred_b'] = pred_b
                result['actual_b'] = actual_above_11am
                result['correct_b'] = correct_b
                result['high_conf_b'] = high_conf_b

            if current_hour <= 11:
                results_early.append(result)
            else:
                results_late.append(result)

    # ========== RESULTS ==========
    print(f"\n  {'='*60}")
    print(f"  EARLY SESSION (9-11 AM) - Target A")
    print(f"  {'='*60}")

    df_early = pd.DataFrame(results_early)
    if len(df_early) > 0:
        for hour in sorted(df_early['hour'].unique()):
            h_df = df_early[df_early['hour'] == hour]
            total = len(h_df)
            correct = h_df['correct_a'].sum()
            acc = correct / total
            hc = h_df[h_df['high_conf_a']]
            hc_correct = hc['correct_a'].sum() if len(hc) > 0 else 0
            print(f"    {hour}: {correct}/{total} = {acc:.1%}  |  HiConf: {hc_correct}/{len(hc)} = {hc_correct/len(hc)*100 if len(hc) > 0 else 0:.0f}%")

        total_early = len(df_early)
        correct_early = df_early['correct_a'].sum()
        print(f"    {'─'*50}")
        print(f"    TOTAL: {correct_early}/{total_early} = {correct_early/total_early:.1%}")

    print(f"\n  {'='*60}")
    print(f"  LATE SESSION (12-4 PM) - Target A")
    print(f"  {'='*60}")

    df_late = pd.DataFrame(results_late)
    if len(df_late) > 0:
        for hour in sorted(df_late['hour'].unique()):
            h_df = df_late[df_late['hour'] == hour]
            total = len(h_df)
            correct = h_df['correct_a'].sum()
            acc = correct / total
            hc = h_df[h_df['high_conf_a']]
            hc_correct = hc['correct_a'].sum() if len(hc) > 0 else 0
            print(f"    {hour}: {correct}/{total} = {acc:.1%}  |  HiConf: {hc_correct}/{len(hc)} = {hc_correct/len(hc)*100 if len(hc) > 0 else 0:.0f}%")

        total_late = len(df_late)
        correct_late = df_late['correct_a'].sum()
        print(f"    {'─'*50}")
        print(f"    TOTAL: {correct_late}/{total_late} = {correct_late/total_late:.1%}")

    print(f"\n  {'='*60}")
    print(f"  LATE SESSION (12-4 PM) - Target B (close > 11 AM)")
    print(f"  {'='*60}")

    df_late_b = df_late[df_late['correct_b'].notna()] if 'correct_b' in df_late.columns else pd.DataFrame()
    if len(df_late_b) > 0:
        for hour in sorted(df_late_b['hour'].unique()):
            h_df = df_late_b[df_late_b['hour'] == hour]
            total = len(h_df)
            correct = h_df['correct_b'].sum()
            acc = correct / total
            hc = h_df[h_df['high_conf_b']]
            hc_correct = hc['correct_b'].sum() if len(hc) > 0 else 0
            print(f"    {hour}: {correct}/{total} = {acc:.1%}  |  HiConf: {hc_correct}/{len(hc)} = {hc_correct/len(hc)*100 if len(hc) > 0 else 0:.0f}%")

        total_b = len(df_late_b)
        correct_b = df_late_b['correct_b'].sum()
        print(f"    {'─'*50}")
        print(f"    TOTAL: {correct_b}/{total_b} = {correct_b/total_b:.1%}")

    # ========== RAW PREDICTIONS AT 2 PM ==========
    print(f"\n  {'='*60}")
    print(f"  RAW PREDICTIONS AT 2:00 PM")
    print(f"  {'='*60}")

    pm2 = df_late[df_late['hour'] == '14:00']
    if len(pm2) > 0:
        print(f"\n  {'Date':<12} {'Open':>8} {'Close':>8} {'Prob_A':>7} {'Pred':>6} {'Act':>6} {'OK':>4}")
        print(f"  {'-'*60}")
        for _, row in pm2.iterrows():
            pred = "BULL" if row['pred_a'] else "BEAR"
            act = "BULL" if row['actual_a'] else "BEAR"
            ok = "Y" if row['correct_a'] else "N"
            conf = "*" if row['high_conf_a'] else ""
            print(f"  {str(row['date']):<12} {row['today_open']:>8.2f} {row['today_close']:>8.2f} {row['prob_a']:>6.1%}{conf} {pred:>6} {act:>6} {ok:>4}")

    return df_early, df_late


def main():
    print("="*70)
    print("  V6 TIME-SPLIT MODEL BACKTEST - LAST 3 WEEKS")
    print("="*70)
    print()
    print("  Model: V6 Time-Split")
    print("  - Early (9-11 AM): Separate model for opening dynamics")
    print("  - Late (12-4 PM): Optimized for afternoon predictions")
    print()

    for ticker in ['SPY', 'QQQ', 'IWM']:
        backtest_ticker(ticker, '2025-12-02', '2025-12-19')


if __name__ == '__main__':
    main()
