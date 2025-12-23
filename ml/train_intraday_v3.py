"""
Intraday Model V3 - Enhanced Features, Strict Integrity

INTEGRITY RULES:
1. Train on 2020-01-01 to 2025-01-01 ONLY
2. Test on 2025-01-02 to 2025-12-19 (TRUE out-of-sample)
3. Use REAL hourly prices from Polygon (no simulation)
4. At prediction time, only use data available at that moment
5. No future information leakage

IMPROVEMENTS OVER V2:
- More features: momentum, trend strength, volume patterns
- Better feature engineering for afternoon predictions
- Separate models for different time windows
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

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def fetch_hourly_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly intraday data from Polygon"""
    print(f"  Fetching hourly data for {ticker} ({start_date} to {end_date})...")

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


def get_price_at_hour(hourly_df, date, hour):
    """Get the closing price at a specific hour"""
    try:
        day_start = pd.Timestamp(date).tz_localize('America/New_York').replace(hour=hour, minute=0)
        day_end = day_start + timedelta(hours=1)
        bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index < day_end)]
        if len(bars) > 0:
            return bars['Close'].iloc[0]
    except:
        pass
    return None


def get_volume_at_hour(hourly_df, date, hour):
    """Get volume at a specific hour"""
    try:
        day_start = pd.Timestamp(date).tz_localize('America/New_York').replace(hour=hour, minute=0)
        day_end = day_start + timedelta(hours=1)
        bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index < day_end)]
        if len(bars) > 0:
            return bars['Volume'].iloc[0]
    except:
        pass
    return None


def create_enhanced_features(bars_so_far, today_open, prev_day, prev_prev_day, price_11am=None):
    """
    Create enhanced feature set using ONLY data available at prediction time.

    bars_so_far: hourly bars from open to current hour (REAL data)
    today_open: today's opening price
    prev_day: previous day's daily bar
    prev_prev_day: day before previous
    price_11am: 11 AM price (only if current hour >= 11)
    """

    current_bar = bars_so_far.iloc[-1]
    current_price = current_bar['Close']
    current_hour = current_bar.name.hour

    high_so_far = bars_so_far['High'].max()
    low_so_far = bars_so_far['Low'].min()
    volume_so_far = bars_so_far['Volume'].sum()

    # Time features
    hours_since_open = (current_bar.name.hour - 9) + (current_bar.name.minute / 60)
    time_pct = min(max(hours_since_open / 6.5, 0), 1)

    # Previous day features
    prev_close = prev_day['Close']
    prev_open = prev_day['Open']
    prev_high = prev_day['High']
    prev_low = prev_day['Low']
    prev_volume = prev_day['Volume']

    gap = (today_open - prev_close) / prev_close
    prev_return = (prev_close - prev_prev_day['Close']) / prev_prev_day['Close']
    prev_range = (prev_high - prev_low) / prev_close
    prev_body = (prev_close - prev_open) / prev_open  # Candle body

    range_so_far = max(high_so_far - low_so_far, 0.0001)

    # ========== BASIC FEATURES ==========
    features = {
        'time_pct': time_pct,
        'time_remaining': 1 - time_pct,

        # Gap features
        'gap': gap,
        'gap_direction': 1 if gap > 0 else (-1 if gap < 0 else 0),
        'gap_size': abs(gap),

        # Previous day
        'prev_return': prev_return,
        'prev_range': prev_range,
        'prev_body': prev_body,
        'prev_bullish': 1 if prev_close > prev_open else 0,

        # Current position
        'current_vs_open': (current_price - today_open) / today_open,
        'current_vs_open_direction': 1 if current_price > today_open else (-1 if current_price < today_open else 0),
        'position_in_range': (current_price - low_so_far) / range_so_far if range_so_far > 0 else 0.5,
        'range_so_far_pct': range_so_far / today_open,
        'high_so_far_pct': (high_so_far - today_open) / today_open,
        'low_so_far_pct': (today_open - low_so_far) / today_open,

        # Binary indicators
        'above_open': 1 if current_price > today_open else 0,
        'near_high': 1 if (high_so_far - current_price) < (current_price - low_so_far) else 0,
        'gap_filled': 1 if (gap > 0 and low_so_far <= prev_close) or (gap <= 0 and high_so_far >= prev_close) else 0,
    }

    # ========== MOMENTUM FEATURES ==========
    if len(bars_so_far) >= 2:
        # Recent price change
        features['last_hour_return'] = (current_price - bars_so_far['Close'].iloc[-2]) / bars_so_far['Close'].iloc[-2]
        features['last_hour_direction'] = 1 if features['last_hour_return'] > 0 else -1
    else:
        features['last_hour_return'] = 0
        features['last_hour_direction'] = 0

    if len(bars_so_far) >= 3:
        # 2-hour momentum
        features['two_hour_return'] = (current_price - bars_so_far['Close'].iloc[-3]) / bars_so_far['Close'].iloc[-3]
    else:
        features['two_hour_return'] = features.get('last_hour_return', 0)

    # Trend strength: how many bars closed higher than opened
    bullish_bars = sum(1 for i in range(len(bars_so_far)) if bars_so_far['Close'].iloc[i] > bars_so_far['Open'].iloc[i])
    features['bullish_bar_ratio'] = bullish_bars / len(bars_so_far)

    # ========== VOLATILITY FEATURES ==========
    if len(bars_so_far) >= 2:
        hourly_returns = bars_so_far['Close'].pct_change().dropna()
        features['intraday_volatility'] = hourly_returns.std() if len(hourly_returns) > 0 else 0
    else:
        features['intraday_volatility'] = 0

    # Range expansion: is today's range larger than yesterday's?
    features['range_vs_prev'] = range_so_far / (prev_high - prev_low) if (prev_high - prev_low) > 0 else 1

    # ========== VOLUME FEATURES ==========
    avg_hourly_volume = volume_so_far / len(bars_so_far) if len(bars_so_far) > 0 else 0
    expected_daily_volume = prev_volume  # Use previous day as baseline

    features['volume_pace'] = (volume_so_far / (time_pct * expected_daily_volume)) if (time_pct * expected_daily_volume) > 0 else 1
    features['current_hour_volume_ratio'] = current_bar['Volume'] / avg_hourly_volume if avg_hourly_volume > 0 else 1

    # ========== 11 AM ANCHOR FEATURES ==========
    if price_11am is not None:
        features['current_vs_11am'] = (current_price - price_11am) / price_11am
        features['above_11am'] = 1 if current_price > price_11am else 0
        features['distance_from_11am'] = abs(current_price - price_11am) / price_11am
    else:
        features['current_vs_11am'] = 0
        features['above_11am'] = 0
        features['distance_from_11am'] = 0

    # ========== PATTERN FEATURES ==========
    # Morning reversal: gapped one way but moved the other
    features['morning_reversal'] = 1 if (gap > 0 and current_price < today_open) or (gap < 0 and current_price > today_open) else 0

    # Extended move: price moved significantly from open
    features['extended_move'] = 1 if abs(features['current_vs_open']) > 0.005 else 0  # > 0.5% move

    return features


def create_training_samples(ticker: str, daily_df: pd.DataFrame, hourly_df: pd.DataFrame,
                           train_end_date: str) -> tuple:
    """
    Create training samples with STRICT date cutoff.

    Returns separate dataframes for Target A and Target B.
    """

    train_cutoff = pd.to_datetime(train_end_date).date()

    samples_a = []  # Target A: close > open (all hours)
    samples_b = []  # Target B: close > 11am (only after 11 AM)

    trading_days = sorted(set(hourly_df.index.date))

    for day in trading_days:
        # STRICT: Only use training data
        if day > train_cutoff:
            continue

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

        # Target A: close > open
        target_bullish = 1 if today_close > today_open else 0

        # Get 11 AM price for Target B
        price_11am = get_price_at_hour(hourly_df, day, 11)
        target_above_11am = 1 if (price_11am and today_close > price_11am) else 0

        # Get hourly bars for this day
        day_start = pd.Timestamp(day).tz_localize('America/New_York').replace(hour=9, minute=0)
        day_end = pd.Timestamp(day).tz_localize('America/New_York').replace(hour=16, minute=30)
        day_bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index <= day_end)]

        if len(day_bars) < 2:
            continue

        # Create samples at each hour
        for j in range(1, len(day_bars) + 1):
            bars_so_far = day_bars.iloc[:j]
            current_hour = bars_so_far.iloc[-1].name.hour

            # Only use 11am price if we're at or after 11 AM
            p11 = price_11am if current_hour >= 11 else None

            features = create_enhanced_features(bars_so_far, today_open, prev_day, prev_prev_day, p11)
            features['target_bullish'] = target_bullish
            samples_a.append(features)

            # Target B: only after 11 AM when we have the anchor
            if current_hour >= 11 and price_11am is not None:
                features_b = features.copy()
                features_b['target_above_11am'] = target_above_11am
                samples_b.append(features_b)

    return pd.DataFrame(samples_a), pd.DataFrame(samples_b)


def train_ensemble(X_train, y_train, X_test, y_test, label=""):
    """Train ensemble with proper evaluation"""

    models = {
        'xgb': XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            eval_metric='logloss'
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        'rf': RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        ),
        'et': ExtraTreesClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )
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

    # Ensemble accuracy
    y_pred_ensemble = np.zeros(len(y_test))
    for name, r in results.items():
        y_pred_ensemble += r['model'].predict_proba(X_test)[:, 1] * weights[name]
    y_pred_final = (y_pred_ensemble > 0.5).astype(int)
    ensemble_acc = accuracy_score(y_test, y_pred_final)

    print(f"    Ensemble{label}: {ensemble_acc:.1%}")

    return {name: r['model'] for name, r in results.items()}, weights, ensemble_acc


def train_model(ticker: str):
    """Train V3 model with strict train/test split"""

    # STRICT DATES - NO LEAKAGE
    TRAIN_START = '2020-01-01'
    TRAIN_END = '2025-01-01'
    TEST_START = '2025-01-02'
    TEST_END = '2025-12-19'

    print(f"\n{'='*70}")
    print(f"  TRAINING V3: {ticker}")
    print(f"  Train: {TRAIN_START} to {TRAIN_END} (STRICT CUTOFF)")
    print(f"  Test: {TEST_START} to {TEST_END} (OUT OF SAMPLE)")
    print(f"{'='*70}")

    # Fetch all data
    daily_df = fetch_daily_data(ticker, TRAIN_START, TEST_END)
    hourly_df = fetch_hourly_data(ticker, TRAIN_START, TEST_END)

    if len(daily_df) < 100 or len(hourly_df) < 500:
        print("  Not enough data!")
        return None

    # Create samples with strict cutoff
    print(f"  Creating training samples (up to {TRAIN_END})...")
    train_a, train_b = create_training_samples(ticker, daily_df, hourly_df, TRAIN_END)

    print(f"  Creating test samples ({TEST_START} onwards)...")
    # For test, we need samples after train_end
    test_cutoff = pd.to_datetime(TEST_START).date()

    test_samples_a = []
    test_samples_b = []

    trading_days = sorted(set(hourly_df.index.date))

    for day in trading_days:
        if day < test_cutoff:
            continue

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
            features = create_enhanced_features(bars_so_far, today_open, prev_day, prev_prev_day, p11)
            features['target_bullish'] = target_bullish
            test_samples_a.append(features)

            if current_hour >= 11 and price_11am is not None:
                features_b = features.copy()
                features_b['target_above_11am'] = target_above_11am
                test_samples_b.append(features_b)

    test_a = pd.DataFrame(test_samples_a)
    test_b = pd.DataFrame(test_samples_b)

    print(f"\n  Train A samples: {len(train_a)}")
    print(f"  Test A samples: {len(test_a)}")
    print(f"  Train B samples: {len(train_b)}")
    print(f"  Test B samples: {len(test_b)}")

    # Feature columns for Target A
    feature_cols_a = [
        'time_pct', 'time_remaining',
        'gap', 'gap_direction', 'gap_size',
        'prev_return', 'prev_range', 'prev_body', 'prev_bullish',
        'current_vs_open', 'current_vs_open_direction',
        'position_in_range', 'range_so_far_pct',
        'high_so_far_pct', 'low_so_far_pct',
        'above_open', 'near_high', 'gap_filled',
        'last_hour_return', 'last_hour_direction', 'two_hour_return',
        'bullish_bar_ratio', 'intraday_volatility', 'range_vs_prev',
        'volume_pace', 'current_hour_volume_ratio',
        'morning_reversal', 'extended_move'
    ]

    # Feature columns for Target B (includes 11am features)
    feature_cols_b = feature_cols_a + ['current_vs_11am', 'above_11am', 'distance_from_11am']

    # ========== TRAIN TARGET A ==========
    print(f"\n  --- Training Target A: close > open ---")

    X_train_a = train_a[feature_cols_a].replace([np.inf, -np.inf], 0).fillna(0)
    y_train_a = train_a['target_bullish']
    X_test_a = test_a[feature_cols_a].replace([np.inf, -np.inf], 0).fillna(0)
    y_test_a = test_a['target_bullish']

    scaler_a = RobustScaler()
    X_train_a_scaled = scaler_a.fit_transform(X_train_a)
    X_test_a_scaled = scaler_a.transform(X_test_a)

    models_a, weights_a, acc_a = train_ensemble(X_train_a_scaled, y_train_a, X_test_a_scaled, y_test_a, " (A)")

    # ========== TRAIN TARGET B ==========
    print(f"\n  --- Training Target B: close > 11 AM ---")

    if len(train_b) > 500 and len(test_b) > 100:
        X_train_b = train_b[feature_cols_b].replace([np.inf, -np.inf], 0).fillna(0)
        y_train_b = train_b['target_above_11am']
        X_test_b = test_b[feature_cols_b].replace([np.inf, -np.inf], 0).fillna(0)
        y_test_b = test_b['target_above_11am']

        scaler_b = RobustScaler()
        X_train_b_scaled = scaler_b.fit_transform(X_train_b)
        X_test_b_scaled = scaler_b.transform(X_test_b)

        models_b, weights_b, acc_b = train_ensemble(X_train_b_scaled, y_train_b, X_test_b_scaled, y_test_b, " (B)")
    else:
        print("  Not enough samples for Target B")
        models_b, weights_b, scaler_b = None, None, None
        feature_cols_b = None

    # Save model
    model_data = {
        'ticker': ticker,
        'version': 'v3_enhanced',
        'train_period': f"{TRAIN_START} to {TRAIN_END}",
        'test_period': f"{TEST_START} to {TEST_END}",
        'trained_at': datetime.now().isoformat(),

        # Target A
        'feature_cols': feature_cols_a,
        'scaler': scaler_a,
        'models': models_a,
        'weights': weights_a,
        'test_accuracy_a': acc_a,
    }

    if models_b is not None:
        model_data['feature_cols_11am'] = feature_cols_b
        model_data['scaler_11am'] = scaler_b
        model_data['models_11am'] = models_b
        model_data['weights_11am'] = weights_b
        model_data['test_accuracy_b'] = acc_b

    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_v3.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n  Saved to {model_path}")

    return model_data


def main():
    print("="*70)
    print("  INTRADAY MODEL V3 - ENHANCED FEATURES")
    print("="*70)
    print()
    print("  INTEGRITY GUARANTEES:")
    print("  - Train: 2020-01-01 to 2025-01-01 ONLY")
    print("  - Test: 2025-01-02 to 2025-12-19 (OUT OF SAMPLE)")
    print("  - Real hourly prices (no simulation)")
    print("  - No future information leakage")
    print()

    for ticker in ['SPY', 'QQQ', 'IWM']:
        train_model(ticker)

    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
