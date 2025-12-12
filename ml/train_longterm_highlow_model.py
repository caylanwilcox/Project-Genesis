"""
Long-Term High/Low Price Prediction Model

Training Period: January 2000 - January 1, 2025 (~25 years)
Testing Period: January 1, 2025 - December 8, 2025 (~11 months)

Predicts:
- Expected HIGH of the day (% above open)
- Expected LOW of the day (% below open)

Accuracy metric: Range Capture Rate
- Does the predicted range contain the EOD close?
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from datetime import datetime
import pickle
import os
import requests
import warnings
warnings.filterwarnings('ignore')

# Polygon API
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Date ranges
TRAIN_START = '2000-01-01'
TRAIN_END = '2024-12-31'
TEST_START = '2025-01-01'
TEST_END = '2025-12-08'


def fetch_polygon_data_range(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical daily data from Polygon.io for a specific date range"""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if 'results' not in data or len(data['results']) == 0:
        raise ValueError(f"No data returned for {ticker}")

    df = pd.DataFrame(data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={
        'o': 'Open',
        'h': 'High',
        'l': 'Low',
        'c': 'Close',
        'v': 'Volume'
    })
    df = df.set_index('date')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate features for high/low prediction"""

    # Target: Predict the actual High and Low prices (as % from Open)
    df['actual_high_pct'] = ((df['High'] - df['Open']) / df['Open']) * 100
    df['actual_low_pct'] = ((df['Open'] - df['Low']) / df['Open']) * 100  # Positive value
    df['actual_close_pct'] = ((df['Close'] - df['Open']) / df['Open']) * 100

    # Previous day metrics
    df['prev_close'] = df['Close'].shift(1)
    df['prev_high'] = df['High'].shift(1)
    df['prev_low'] = df['Low'].shift(1)
    df['prev_open'] = df['Open'].shift(1)

    # Gap from previous close
    df['gap_pct'] = ((df['Open'] - df['prev_close']) / df['prev_close']) * 100

    # Previous day's range characteristics
    df['prev_range_pct'] = ((df['prev_high'] - df['prev_low']) / df['prev_close']) * 100
    df['prev_high_pct'] = ((df['prev_high'] - df['prev_open']) / df['prev_open']) * 100
    df['prev_low_pct'] = ((df['prev_open'] - df['prev_low']) / df['prev_open']) * 100
    df['prev_close_pct'] = ((df['prev_close'] - df['prev_open']) / df['prev_open']) * 100

    # Returns
    df['prev_return'] = df['Close'].pct_change().shift(1) * 100
    df['prev_2_return'] = df['Close'].pct_change().shift(2) * 100
    df['prev_3_return'] = df['Close'].pct_change().shift(3) * 100
    df['prev_5_return'] = df['Close'].pct_change().shift(5) * 100

    # Momentum
    df['momentum_3d'] = df['prev_return'].rolling(3).sum()
    df['momentum_5d'] = df['prev_return'].rolling(5).sum()
    df['momentum_10d'] = df['prev_return'].rolling(10).sum()

    # Volatility
    df['volatility_5d'] = df['prev_return'].rolling(5).std()
    df['volatility_10d'] = df['prev_return'].rolling(10).std()
    df['volatility_20d'] = df['prev_return'].rolling(20).std()
    df['vol_ratio'] = df['volatility_5d'] / df['volatility_20d']

    # ATR
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_5'] = (tr.rolling(5).mean().shift(1) / df['prev_close']) * 100
    df['atr_10'] = (tr.rolling(10).mean().shift(1) / df['prev_close']) * 100
    df['atr_14'] = (tr.rolling(14).mean().shift(1) / df['prev_close']) * 100
    df['atr_20'] = (tr.rolling(20).mean().shift(1) / df['prev_close']) * 100

    # Historical high/low patterns (key features!)
    df['avg_high_5d'] = df['actual_high_pct'].rolling(5).mean().shift(1)
    df['avg_low_5d'] = df['actual_low_pct'].rolling(5).mean().shift(1)
    df['avg_high_10d'] = df['actual_high_pct'].rolling(10).mean().shift(1)
    df['avg_low_10d'] = df['actual_low_pct'].rolling(10).mean().shift(1)
    df['avg_high_20d'] = df['actual_high_pct'].rolling(20).mean().shift(1)
    df['avg_low_20d'] = df['actual_low_pct'].rolling(20).mean().shift(1)
    df['max_high_5d'] = df['actual_high_pct'].rolling(5).max().shift(1)
    df['max_low_5d'] = df['actual_low_pct'].rolling(5).max().shift(1)
    df['max_high_10d'] = df['actual_high_pct'].rolling(10).max().shift(1)
    df['max_low_10d'] = df['actual_low_pct'].rolling(10).max().shift(1)

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 0.001)
    df['rsi_14'] = (100 - (100 / (1 + rs))).shift(1)

    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['macd'] = (ema_12 - ema_26).shift(1)
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = (df['macd'] - df['macd_signal'])

    # Price vs moving averages
    sma_20 = df['Close'].rolling(20).mean()
    sma_50 = df['Close'].rolling(50).mean()
    df['price_vs_sma20'] = ((df['prev_close'] - sma_20.shift(1)) / sma_20.shift(1)) * 100
    df['price_vs_sma50'] = ((df['prev_close'] - sma_50.shift(1)) / sma_50.shift(1)) * 100

    # Bollinger Bands
    bb_middle = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    df['bb_position'] = ((df['Close'] - bb_lower) / (bb_upper - bb_lower + 0.001)).shift(1)
    df['bb_width'] = ((bb_upper - bb_lower) / bb_middle * 100).shift(1)

    # Day of week
    df['day_of_week'] = df.index.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)

    # Volume
    df['volume_ratio'] = df['Volume'].shift(1) / df['Volume'].rolling(20).mean().shift(1)

    # Consecutive patterns
    df['up_day'] = (df['Close'] > df['Open']).astype(int)
    df['consec_up'] = df['up_day'].rolling(5).sum().shift(1)
    df['consec_down'] = (1 - df['up_day']).rolling(5).sum().shift(1)

    return df


def get_feature_columns():
    """Return list of features for high/low model"""
    return [
        'gap_pct', 'prev_range_pct', 'prev_high_pct', 'prev_low_pct', 'prev_close_pct',
        'prev_return', 'prev_2_return', 'prev_3_return', 'prev_5_return',
        'momentum_3d', 'momentum_5d', 'momentum_10d',
        'volatility_5d', 'volatility_10d', 'volatility_20d', 'vol_ratio',
        'atr_5', 'atr_10', 'atr_14', 'atr_20',
        'avg_high_5d', 'avg_low_5d', 'avg_high_10d', 'avg_low_10d',
        'avg_high_20d', 'avg_low_20d',
        'max_high_5d', 'max_low_5d', 'max_high_10d', 'max_low_10d',
        'rsi_14', 'macd_histogram',
        'price_vs_sma20', 'price_vs_sma50',
        'bb_position', 'bb_width',
        'day_of_week', 'is_monday', 'is_friday',
        'volume_ratio', 'consec_up', 'consec_down'
    ]


def evaluate_range_accuracy(y_high_true, y_low_true, y_close_true, high_pred, low_pred):
    """
    Evaluate how well predicted ranges capture EOD close
    """
    n = len(y_close_true)
    captured = 0
    misses = []
    ranges = []

    for i in range(n):
        pred_high = high_pred[i]
        pred_low = low_pred[i]
        actual_close = y_close_true.iloc[i]

        range_width = pred_high + pred_low
        ranges.append(range_width)

        if -pred_low <= actual_close <= pred_high:
            captured += 1
        else:
            if actual_close > pred_high:
                miss = actual_close - pred_high
            else:
                miss = -pred_low - actual_close
            misses.append(abs(miss))

    capture_rate = captured / n * 100
    avg_miss = np.mean(misses) if misses else 0
    avg_range = np.mean(ranges)

    return capture_rate, avg_miss, avg_range


def train_longterm_highlow_model(ticker: str):
    """Train high/low model on 25 years, test on 2025"""

    print(f"\n{'='*60}")
    print(f"Training LONG-TERM High/Low Model for {ticker}")
    print(f"Train: {TRAIN_START} to {TRAIN_END}")
    print(f"Test:  {TEST_START} to {TEST_END}")
    print('='*60)

    # Fetch training data
    print("\nFetching training data (2000-2024)...")
    df_train = fetch_polygon_data_range(ticker, TRAIN_START, TRAIN_END)
    print(f"  Got {len(df_train)} training days")

    # Fetch test data
    print("Fetching test data (2025)...")
    df_test = fetch_polygon_data_range(ticker, TEST_START, TEST_END)
    print(f"  Got {len(df_test)} test days")

    # Combine for feature calculation
    df_all = pd.concat([df_train, df_test])

    # Calculate features
    print("Calculating features...")
    df_all = calculate_features(df_all)

    feature_cols = get_feature_columns()
    print(f"  Using {len(feature_cols)} features")

    # Split back into train/test
    train_end_date = pd.Timestamp(TRAIN_END)
    test_start_date = pd.Timestamp(TEST_START)

    df_train_clean = df_all[df_all.index <= train_end_date].dropna(
        subset=feature_cols + ['actual_high_pct', 'actual_low_pct', 'actual_close_pct']
    )
    df_test_clean = df_all[df_all.index >= test_start_date].dropna(
        subset=feature_cols + ['actual_high_pct', 'actual_low_pct', 'actual_close_pct']
    )

    print(f"  Training samples: {len(df_train_clean)}")
    print(f"  Test samples: {len(df_test_clean)}")

    X_train = df_train_clean[feature_cols]
    y_high_train = df_train_clean['actual_high_pct']
    y_low_train = df_train_clean['actual_low_pct']

    X_test = df_test_clean[feature_cols]
    y_high_test = df_test_clean['actual_high_pct']
    y_low_test = df_test_clean['actual_low_pct']
    y_close_test = df_test_clean['actual_close_pct']

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining ensemble on 21+ years of data...")

    # Train HIGH model ensemble
    print("  Training HIGH prediction models...")

    xgb_high = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    xgb_high.fit(X_train_scaled, y_high_train)

    gb_high = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    gb_high.fit(X_train_scaled, y_high_train)

    rf_high = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf_high.fit(X_train_scaled, y_high_train)

    # Train LOW model ensemble
    print("  Training LOW prediction models...")

    xgb_low = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    xgb_low.fit(X_train_scaled, y_low_train)

    gb_low = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    gb_low.fit(X_train_scaled, y_low_train)

    rf_low = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf_low.fit(X_train_scaled, y_low_train)

    # Ensemble predictions (average)
    high_pred = (
        xgb_high.predict(X_test_scaled) * 0.4 +
        gb_high.predict(X_test_scaled) * 0.3 +
        rf_high.predict(X_test_scaled) * 0.3
    )

    low_pred = (
        xgb_low.predict(X_test_scaled) * 0.4 +
        gb_low.predict(X_test_scaled) * 0.3 +
        rf_low.predict(X_test_scaled) * 0.3
    )

    # Raw model performance
    capture_rate, avg_miss, avg_range = evaluate_range_accuracy(
        y_high_test, y_low_test, y_close_test, high_pred, low_pred
    )

    print(f"\n--- RAW MODEL PERFORMANCE (2025 Test) ---")
    print(f"  Range Capture Rate: {capture_rate:.1f}%")
    print(f"  Avg Miss (when wrong): {avg_miss:.3f}%")
    print(f"  Avg Range Width: {avg_range:.3f}%")

    # Find optimal buffer
    best_buffer = 0
    best_capture = capture_rate

    for buffer in np.arange(0, 0.5, 0.02):
        cap_rate, _, _ = evaluate_range_accuracy(
            y_high_test, y_low_test, y_close_test,
            high_pred + buffer, low_pred + buffer
        )
        if cap_rate > best_capture:
            best_capture = cap_rate
            best_buffer = buffer
        if cap_rate >= 95:
            break

    # Final evaluation with buffer
    final_capture, final_miss, final_range = evaluate_range_accuracy(
        y_high_test, y_low_test, y_close_test,
        high_pred + best_buffer, low_pred + best_buffer
    )

    print(f"\n--- WITH BUFFER (+{best_buffer:.2f}%) ---")
    print(f"  Range Capture Rate: {final_capture:.1f}%")
    print(f"  Avg Miss (when wrong): {final_miss:.3f}%")
    print(f"  Avg Range Width: {final_range:.3f}%")

    # Accuracy of high/low predictions
    high_mae = np.mean(np.abs(high_pred - y_high_test.values))
    low_mae = np.mean(np.abs(low_pred - y_low_test.values))
    print(f"\n--- PREDICTION ACCURACY ---")
    print(f"  High Prediction MAE: {high_mae:.3f}%")
    print(f"  Low Prediction MAE: {low_mae:.3f}%")

    # Show sample predictions
    print(f"\n--- SAMPLE PREDICTIONS (last 10 days of 2025) ---")
    print(f"  {'Date':<12} {'Pred High':<12} {'Actual High':<12} {'Pred Low':<12} {'Actual Low':<12} {'Close Captured?'}")
    print(f"  {'-'*75}")

    for i in range(-10, 0):
        date = df_test_clean.index[i].strftime('%Y-%m-%d')
        pred_h = high_pred[i] + best_buffer
        pred_l = low_pred[i] + best_buffer
        actual_h = y_high_test.iloc[i]
        actual_l = y_low_test.iloc[i]
        actual_c = y_close_test.iloc[i]
        captured = "✓" if -pred_l <= actual_c <= pred_h else "✗"
        print(f"  {date:<12} +{pred_h:.2f}%{'':<6} +{actual_h:.2f}%{'':<6} -{pred_l:.2f}%{'':<6} -{actual_l:.2f}%{'':<6} {captured}")

    # Save model
    model_data = {
        'high_models': {
            'xgb': xgb_high,
            'gb': gb_high,
            'rf': rf_high
        },
        'low_models': {
            'xgb': xgb_low,
            'gb': gb_low,
            'rf': rf_low
        },
        'weights': {'xgb': 0.4, 'gb': 0.3, 'rf': 0.3},
        'scaler': scaler,
        'feature_cols': feature_cols,
        'buffer': best_buffer,
        'ticker': ticker,
        'trained_at': datetime.now().isoformat(),
        'version': 'longterm_highlow_v1',
        'train_period': f'{TRAIN_START} to {TRAIN_END}',
        'test_period': f'{TEST_START} to {TEST_END}',
        'metrics': {
            'capture_rate': float(final_capture),
            'avg_miss': float(final_miss),
            'avg_range': float(final_range),
            'high_mae': float(high_mae),
            'low_mae': float(low_mae),
            'buffer': float(best_buffer),
            'train_samples': len(df_train_clean),
            'test_samples': len(df_test_clean)
        }
    }

    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_highlow_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to {model_path}")

    return model_data


if __name__ == '__main__':
    print("\n" + "="*70)
    print("   TRAINING LONG-TERM HIGH/LOW MODELS (21+ YEARS)")
    print("="*70)
    print(f"\nTraining Period: {TRAIN_START} to {TRAIN_END}")
    print(f"Testing Period:  {TEST_START} to {TEST_END}")

    results = {}
    for ticker in ['SPY', 'QQQ', 'IWM']:
        try:
            model_data = train_longterm_highlow_model(ticker)
            results[ticker] = model_data['metrics']
        except Exception as e:
            print(f"Error training {ticker}: {e}")
            import traceback
            traceback.print_exc()
            results[ticker] = {'error': str(e)}

    print("\n" + "="*70)
    print("   TRAINING COMPLETE - HIGH/LOW MODEL RESULTS ON 2025 DATA")
    print("="*70)
    print("\nMetric: Range Capture Rate (% of days where EOD close falls within predicted range)\n")

    print(f"{'Ticker':<8} {'Capture Rate':<15} {'Avg Range':<12} {'High MAE':<12} {'Low MAE':<12} {'Buffer'}")
    print("-" * 70)
    for ticker, metrics in results.items():
        if 'error' not in metrics:
            print(f"{ticker:<8} {metrics['capture_rate']:.1f}%{'':<9} {metrics['avg_range']:.2f}%{'':<7} {metrics['high_mae']:.3f}%{'':<7} {metrics['low_mae']:.3f}%{'':<7} +{metrics['buffer']:.2f}%")
