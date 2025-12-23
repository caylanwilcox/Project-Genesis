"""
Intraday Model V2 - Using REAL Hourly Data

This version trains on actual intraday (hourly) data from Polygon,
not simulated prices from daily OHLC.

Target A: Will close > open? (same as before, works well)
Target B: Will close > 11:00 AM price? (anchored to fixed time)

The key insight: Target B needs a FIXED reference point that we
actually know at prediction time. Using "current price" was flawed
because training simulated it from daily data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from datetime import datetime, timedelta
import pickle
import os
import requests
import warnings
warnings.filterwarnings('ignore')

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def fetch_hourly_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly intraday data from Polygon"""
    print(f"  Fetching hourly data for {ticker} ({start_date} to {end_date})...")

    all_data = []
    current_start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Polygon limits results, so fetch in chunks
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

    # Convert to Eastern Time
    df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')

    print(f"    Got {len(df)} hourly bars")
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def fetch_daily_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV data"""
    print(f"  Fetching daily data for {ticker}...")

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

    print(f"    Got {len(df)} daily bars")
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def get_price_at_hour(hourly_df, date, hour):
    """Get the closing price at a specific hour on a given date"""
    try:
        # Find bars for this date around this hour
        day_start = pd.Timestamp(date).tz_localize('America/New_York').replace(hour=hour, minute=0)
        day_end = day_start + timedelta(hours=1)

        bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index < day_end)]
        if len(bars) > 0:
            return bars['Close'].iloc[0]
    except:
        pass
    return None


def create_training_samples_from_real_data(ticker: str, daily_df: pd.DataFrame, hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create training samples using REAL hourly prices.

    For each trading day, we create samples at different hours:
    - Features: gap, prev_day stats, price relative to open, high/low so far
    - Target A: Will close > open?
    - Target B: Will close > 11:00 AM price? (only available after 11 AM)
    """

    samples = []

    # Get unique trading days
    trading_days = sorted(set(hourly_df.index.date))

    print(f"  Processing {len(trading_days)} trading days...")

    for i, day in enumerate(trading_days):
        if i < 2:  # Need prev days
            continue

        # Get this day's data
        if day not in daily_df.index:
            continue

        today = daily_df.loc[day]

        # Get previous days
        daily_dates = list(daily_df.index)
        day_idx = daily_dates.index(day)
        if day_idx < 2:
            continue

        prev_day = daily_df.iloc[day_idx - 1]
        prev_prev_day = daily_df.iloc[day_idx - 2]

        today_open = today['Open']
        today_close = today['Close']
        today_high = today['High']
        today_low = today['Low']

        # Gap
        gap = (today_open - prev_day['Close']) / prev_day['Close']
        prev_return = (prev_day['Close'] - prev_prev_day['Close']) / prev_prev_day['Close']
        prev_range = (prev_day['High'] - prev_day['Low']) / prev_day['Close']

        # Target A: close > open (same for all samples on this day)
        target_bullish = 1 if today_close > today_open else 0

        # Get 11:00 AM price for Target B
        price_11am = get_price_at_hour(hourly_df, day, 11)
        target_above_11am = None
        if price_11am is not None:
            target_above_11am = 1 if today_close > price_11am else 0

        # Get hourly bars for this day
        day_start = pd.Timestamp(day).tz_localize('America/New_York').replace(hour=9, minute=0)
        day_end = pd.Timestamp(day).tz_localize('America/New_York').replace(hour=16, minute=30)
        day_bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index <= day_end)]

        if len(day_bars) < 2:
            continue

        # Create sample at each hour
        for j in range(1, len(day_bars) + 1):
            bars_so_far = day_bars.iloc[:j]
            current_bar = bars_so_far.iloc[-1]
            current_hour = current_bar.name.hour

            current_price = current_bar['Close']
            high_so_far = bars_so_far['High'].max()
            low_so_far = bars_so_far['Low'].min()

            # Time through day (9:30 AM to 4:00 PM = 6.5 hours)
            hours_since_open = (current_bar.name.hour - 9) + (current_bar.name.minute / 60)
            time_pct = min(max(hours_since_open / 6.5, 0), 1)

            range_so_far = max(high_so_far - low_so_far, 0.0001)

            sample = {
                'date': day,
                'hour': current_hour,
                'time_pct': time_pct,
                'time_remaining': 1 - time_pct,

                # Gap features
                'gap': gap,
                'gap_direction': 1 if gap > 0 else (-1 if gap < 0 else 0),
                'gap_size': abs(gap),

                # Previous day
                'prev_return': prev_return,
                'prev_range': prev_range,

                # Current position (REAL prices)
                'current_vs_open': (current_price - today_open) / today_open,
                'current_vs_open_direction': 1 if current_price > today_open else (-1 if current_price < today_open else 0),
                'position_in_range': (current_price - low_so_far) / range_so_far if range_so_far > 0 else 0.5,
                'range_so_far_pct': range_so_far / today_open,
                'high_so_far_pct': (high_so_far - today_open) / today_open,
                'low_so_far_pct': (today_open - low_so_far) / today_open,

                # Momentum
                'above_open': 1 if current_price > today_open else 0,
                'near_high': 1 if (high_so_far - current_price) < (current_price - low_so_far) else 0,

                # Gap fill
                'gap_filled': 1 if (gap > 0 and low_so_far <= prev_day['Close']) or (gap <= 0 and high_so_far >= prev_day['Close']) else 0,

                # Targets
                'target_bullish': target_bullish,  # Target A
            }

            # Target B: Only add if we're past 11 AM and have the 11 AM price
            if current_hour >= 11 and target_above_11am is not None:
                sample['target_above_11am'] = target_above_11am
                sample['current_vs_11am'] = (current_price - price_11am) / price_11am

            samples.append(sample)

    df = pd.DataFrame(samples)
    print(f"  Created {len(df)} training samples")
    return df


def train_model(ticker: str, train_start: str = '2020-01-01', train_end: str = '2024-06-30',
                test_start: str = '2024-07-01', test_end: str = '2025-11-30'):
    """
    Train intraday model using real hourly data.

    Note: Using 2020+ because hourly data availability is better.
    Test on 2024-07 to 2025-11 (out of sample).
    """

    print(f"\n{'='*70}")
    print(f"  TRAINING: {ticker}")
    print(f"  Train: {train_start} to {train_end}")
    print(f"  Test: {test_start} to {test_end}")
    print(f"{'='*70}")

    # Fetch data
    daily_df = fetch_daily_data(ticker, train_start, test_end)
    hourly_df = fetch_hourly_data(ticker, train_start, test_end)

    if len(daily_df) < 100 or len(hourly_df) < 500:
        print("  Not enough data!")
        return None

    # Create training samples
    samples_df = create_training_samples_from_real_data(ticker, daily_df, hourly_df)

    if len(samples_df) < 1000:
        print("  Not enough training samples!")
        return None

    # Split train/test
    train_df = samples_df[samples_df['date'] < pd.to_datetime(test_start).date()]
    test_df = samples_df[samples_df['date'] >= pd.to_datetime(test_start).date()]

    print(f"\n  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")

    # Feature columns (same as original model for compatibility)
    feature_cols = [
        'time_pct', 'time_remaining',
        'gap', 'gap_direction', 'gap_size',
        'prev_return', 'prev_range',
        'current_vs_open', 'current_vs_open_direction',
        'position_in_range', 'range_so_far_pct',
        'high_so_far_pct', 'low_so_far_pct',
        'above_open', 'near_high', 'gap_filled'
    ]

    # ============ TARGET A: close > open ============
    print(f"\n  --- Training Target A: close > open ---")

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y_train = train_df['target_bullish']
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y_test = test_df['target_bullish']

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models_a, weights_a, metrics_a = _train_ensemble(X_train_scaled, y_train, X_test_scaled, y_test, "A")

    # ============ TARGET B: close > 11 AM price ============
    print(f"\n  --- Training Target B: close > 11 AM price ---")

    # Filter to samples that have Target B (after 11 AM)
    train_b = train_df[train_df['target_above_11am'].notna()].copy()
    test_b = test_df[test_df['target_above_11am'].notna()].copy()

    print(f"  Train B samples: {len(train_b)} (after 11 AM)")
    print(f"  Test B samples: {len(test_b)}")

    models_b = None
    weights_b = None

    if len(train_b) > 500 and len(test_b) > 100:
        # Add current_vs_11am as a feature
        feature_cols_b = feature_cols + ['current_vs_11am']

        X_train_b = train_b[feature_cols_b].replace([np.inf, -np.inf], 0).fillna(0)
        y_train_b = train_b['target_above_11am']
        X_test_b = test_b[feature_cols_b].replace([np.inf, -np.inf], 0).fillna(0)
        y_test_b = test_b['target_above_11am']

        scaler_b = RobustScaler()
        X_train_b_scaled = scaler_b.fit_transform(X_train_b)
        X_test_b_scaled = scaler_b.transform(X_test_b)

        models_b, weights_b, metrics_b = _train_ensemble(X_train_b_scaled, y_train_b, X_test_b_scaled, y_test_b, "B")
    else:
        print("  Not enough samples for Target B")
        scaler_b = None
        feature_cols_b = None

    # Save model
    model_data = {
        'ticker': ticker,
        'feature_cols': feature_cols,
        'scaler': scaler,
        'models': models_a,
        'weights': weights_a,
        'trained_at': datetime.now().isoformat(),
        'train_period': f"{train_start} to {train_end}",
        'test_period': f"{test_start} to {test_end}",
        'version': 'v2_real_hourly',
    }

    if models_b is not None:
        model_data['models_11am'] = models_b
        model_data['weights_11am'] = weights_b
        model_data['scaler_11am'] = scaler_b
        model_data['feature_cols_11am'] = feature_cols_b

    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_v2.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n  Saved model to {model_path}")

    return model_data


def _train_ensemble(X_train, y_train, X_test, y_test, label=""):
    """Train 4-model ensemble"""

    models = {
        'xgb': XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbosity=0),
        'gb': GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42),
        'rf': RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=20, random_state=42, n_jobs=-1),
        'et': ExtraTreesClassifier(n_estimators=150, max_depth=5, min_samples_leaf=20, random_state=42, n_jobs=-1)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {'model': model, 'accuracy': acc}
        print(f"    {name}: {acc:.1%}")

    # Weight by accuracy
    total_acc = sum(r['accuracy'] for r in results.values())
    weights = {name: r['accuracy'] / total_acc for name, r in results.items()}

    # Ensemble prediction
    y_pred_ensemble = np.zeros(len(y_test))
    for name, r in results.items():
        y_pred_ensemble += r['model'].predict_proba(X_test)[:, 1] * weights[name]
    y_pred_final = (y_pred_ensemble > 0.5).astype(int)
    ensemble_acc = accuracy_score(y_test, y_pred_final)

    print(f"    Ensemble{label}: {ensemble_acc:.1%}")

    fitted_models = {name: r['model'] for name, r in results.items()}

    return fitted_models, weights, {'accuracy': ensemble_acc}


def main():
    print("="*70)
    print("  INTRADAY MODEL V2 - REAL HOURLY DATA")
    print("="*70)
    print()
    print("  Training with REAL intraday prices (not simulated)")
    print("  Target A: Will close > open?")
    print("  Target B: Will close > 11:00 AM price? (after 11 AM)")
    print()

    for ticker in ['SPY', 'QQQ', 'IWM']:
        train_model(ticker)

    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
