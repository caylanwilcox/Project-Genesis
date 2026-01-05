"""
Time-Split Models - Separate models for different parts of the day

Insight: Early session (9-11 AM) has different dynamics than late session (1-4 PM)
- Early: More uncertainty, gap dynamics dominate
- Late: Trend continuation, momentum matters more

This creates:
1. Early model (9-11 AM predictions)
2. Late model (1-4 PM predictions)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from datetime import datetime, timedelta
import pickle
import os
import requests
import warnings
warnings.filterwarnings('ignore')

# Import centralized training periods
from config import TRAIN_START, TRAIN_END, TEST_START, TEST_END

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

    # Volatility
    returns = []
    for i in range(1, min(6, day_idx)):
        if day_idx - i - 1 >= 0:
            c1 = daily_df.iloc[day_idx - i]['Close']
            c2 = daily_df.iloc[day_idx - i - 1]['Close']
            returns.append((c1 - c2) / c2)
    features['volatility_5d'] = np.std(returns) if len(returns) >= 3 else 0

    # Mean reversion
    prev_return = (daily_df.iloc[day_idx - 1]['Close'] - daily_df.iloc[day_idx - 1]['Open']) / daily_df.iloc[day_idx - 1]['Open']
    features['mean_reversion_signal'] = -prev_return

    # Consecutive
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
    """Create features optimized for time-split models"""
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

    # Momentum
    if len(bars_so_far) >= 2:
        features['last_hour_return'] = (current_price - bars_so_far['Close'].iloc[-2]) / bars_so_far['Close'].iloc[-2]
    else:
        features['last_hour_return'] = 0

    bullish_bars = sum(1 for i in range(len(bars_so_far)) if bars_so_far['Close'].iloc[i] > bars_so_far['Open'].iloc[i])
    features['bullish_bar_ratio'] = bullish_bars / len(bars_so_far)

    # First hour (important for early model)
    if len(bars_so_far) >= 1:
        features['first_hour_return'] = (bars_so_far['Close'].iloc[0] - today_open) / today_open
    else:
        features['first_hour_return'] = 0

    # Multi-day
    multi_day = calculate_multi_day_features(daily_df, day_idx)
    features.update(multi_day)

    # 11 AM features
    if price_11am is not None:
        features['current_vs_11am'] = (current_price - price_11am) / price_11am
        features['above_11am'] = 1 if current_price > price_11am else 0
    else:
        features['current_vs_11am'] = 0
        features['above_11am'] = 0

    # Day of week
    features['is_monday'] = 1 if current_bar.name.dayofweek == 0 else 0
    features['is_friday'] = 1 if current_bar.name.dayofweek == 4 else 0

    return features


def create_time_split_samples(ticker, daily_df, hourly_df, start_date, end_date):
    """Create samples split by time of day"""
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()

    early_samples = []  # 9-11 AM
    late_samples = []   # 1-4 PM

    trading_days = sorted(set(hourly_df.index.date))

    for day in trading_days:
        if day < start or day > end:
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

        target_bullish = 1 if today_close > today_open else 0
        price_11am = get_price_at_hour(hourly_df, day, 11)
        target_above_11am = 1 if (price_11am and today_close > price_11am) else 0

        day_start = pd.Timestamp(day).tz_localize('America/New_York').replace(hour=9, minute=0)
        day_end = pd.Timestamp(day).tz_localize('America/New_York').replace(hour=16, minute=30)
        day_bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index <= day_end)]

        if len(day_bars) < 2:
            continue

        for j in range(1, len(day_bars) + 1):
            bars_so_far = day_bars.iloc[:j]
            current_hour = bars_so_far.iloc[-1].name.hour

            p11 = price_11am if current_hour >= 11 else None
            features = create_features(bars_so_far, today_open, prev_day, prev_prev_day, daily_df, day_idx, p11)
            features['target_bullish'] = target_bullish
            features['target_above_11am'] = target_above_11am

            # Split by time
            if current_hour <= 11:
                early_samples.append(features)
            elif current_hour >= 12:
                late_samples.append(features)

    return pd.DataFrame(early_samples), pd.DataFrame(late_samples)


def train_ensemble(X_train, y_train, X_test, y_test, name):
    """Train optimized ensemble"""
    models = {
        'xgb': XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.02, subsample=0.8, random_state=42, verbosity=0),
        'rf': RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1),
        'gb': GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.03, random_state=42),
        'et': ExtraTreesClassifier(n_estimators=200, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1)
    }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[model_name] = {'model': model, 'accuracy': acc}

    total_acc = sum(r['accuracy'] for r in results.values())
    weights = {n: r['accuracy'] / total_acc for n, r in results.items()}

    y_pred = np.zeros(len(y_test))
    for n, r in results.items():
        y_pred += r['model'].predict_proba(X_test)[:, 1] * weights[n]

    ensemble_acc = accuracy_score(y_test, (y_pred > 0.5).astype(int))
    print(f"    {name}: {ensemble_acc:.1%}")

    return {n: r['model'] for n, r in results.items()}, weights, ensemble_acc


def optimize_ticker(ticker):
    """Create time-split optimized models

    Uses centralized training periods from config.py:
    - Train: 2000-2024
    - Test: 2025
    """
    print(f"\n{'='*70}")
    print(f"  TIME-SPLIT OPTIMIZATION: {ticker}")
    print(f"{'='*70}")

    daily_df = fetch_daily_data(ticker, TRAIN_START, TEST_END)
    hourly_df = fetch_hourly_data(ticker, TRAIN_START, TEST_END)

    # Create time-split samples
    print(f"  Creating time-split samples...")
    train_early, train_late = create_time_split_samples(ticker, daily_df, hourly_df, TRAIN_START, TRAIN_END)
    test_early, test_late = create_time_split_samples(ticker, daily_df, hourly_df, TEST_START, TEST_END)

    print(f"    Early Train: {len(train_early)}, Test: {len(test_early)}")
    print(f"    Late Train: {len(train_late)}, Test: {len(test_late)}")

    feature_cols = [c for c in train_early.columns if c not in ['target_bullish', 'target_above_11am']]

    results = {}

    # ========== EARLY MODEL (9-11 AM) ==========
    print(f"\n  --- Early Session (9-11 AM) ---")

    X_train_early = train_early[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y_train_early = train_early['target_bullish']
    X_test_early = test_early[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y_test_early = test_early['target_bullish']

    scaler_early = RobustScaler()
    X_train_early_scaled = scaler_early.fit_transform(X_train_early)
    X_test_early_scaled = scaler_early.transform(X_test_early)

    models_early, weights_early, acc_early = train_ensemble(
        X_train_early_scaled, y_train_early, X_test_early_scaled, y_test_early, "Early Target A"
    )

    # ========== LATE MODEL (1-4 PM) ==========
    print(f"\n  --- Late Session (12-4 PM) ---")

    X_train_late = train_late[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y_train_late_a = train_late['target_bullish']
    y_train_late_b = train_late['target_above_11am']
    X_test_late = test_late[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y_test_late_a = test_late['target_bullish']
    y_test_late_b = test_late['target_above_11am']

    scaler_late = RobustScaler()
    X_train_late_scaled = scaler_late.fit_transform(X_train_late)
    X_test_late_scaled = scaler_late.transform(X_test_late)

    models_late_a, weights_late_a, acc_late_a = train_ensemble(
        X_train_late_scaled, y_train_late_a, X_test_late_scaled, y_test_late_a, "Late Target A"
    )

    models_late_b, weights_late_b, acc_late_b = train_ensemble(
        X_train_late_scaled, y_train_late_b, X_test_late_scaled, y_test_late_b, "Late Target B"
    )

    # Save
    model_data = {
        'ticker': ticker,
        'version': 'v6_time_split',
        'trained_at': datetime.now().isoformat(),
        'feature_cols': feature_cols,

        # Early model
        'scaler_early': scaler_early,
        'models_early': models_early,
        'weights_early': weights_early,
        'acc_early': acc_early,

        # Late model A
        'scaler_late': scaler_late,
        'models_late_a': models_late_a,
        'weights_late_a': weights_late_a,
        'acc_late_a': acc_late_a,

        # Late model B
        'models_late_b': models_late_b,
        'weights_late_b': weights_late_b,
        'acc_late_b': acc_late_b,
    }

    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_v6.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n  RESULTS for {ticker}:")
    print(f"    Early (9-11 AM) Target A:  {acc_early:.1%}")
    print(f"    Late (12-4 PM) Target A:  {acc_late_a:.1%}")
    print(f"    Late (12-4 PM) Target B:  {acc_late_b:.1%}")

    return {
        'early_a': acc_early,
        'late_a': acc_late_a,
        'late_b': acc_late_b
    }


def main():
    print("="*70)
    print("  TIME-SPLIT INTRADAY MODELS")
    print("="*70)
    print()
    print("  Strategy: Separate models for different times of day")
    print("  - Early model (9-11 AM): Gap dynamics, opening momentum")
    print("  - Late model (1-4 PM): Trend continuation, closing bias")
    print()

    results = {}
    for ticker in ['SPY', 'QQQ', 'IWM']:
        results[ticker] = optimize_ticker(ticker)

    print("\n" + "="*70)
    print("  FINAL RESULTS")
    print("="*70)
    print(f"\n  {'Ticker':<8} {'Early A':>10} {'Late A':>10} {'Late B':>10}")
    print(f"  {'-'*42}")
    for ticker, r in results.items():
        print(f"  {ticker:<8} {r['early_a']:>9.1%} {r['late_a']:>9.1%} {r['late_b']:>9.1%}")


if __name__ == '__main__':
    main()
