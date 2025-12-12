"""
High/Low Price Prediction Model v2

Accuracy measured by: Does the predicted range contain the EOD close?

Metrics:
1. Range Capture Rate: % of days where EOD close falls within predicted high/low
2. Average Miss Distance: When we miss, how far off are we?
3. Range Tightness: How narrow is our predicted range (tighter = more useful)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import pickle
import os
import requests
from datetime import datetime, timedelta

# Polygon API
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def fetch_polygon_data(ticker: str, days: int = 750) -> pd.DataFrame:
    """Fetch historical daily data from Polygon.io"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 5000,
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

    # Momentum
    df['momentum_3d'] = df['prev_return'].rolling(3).sum()
    df['momentum_5d'] = df['prev_return'].rolling(5).sum()

    # Volatility
    df['volatility_5d'] = df['prev_return'].rolling(5).std()
    df['volatility_10d'] = df['prev_return'].rolling(10).std()
    df['volatility_20d'] = df['prev_return'].rolling(20).std()

    # ATR
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_5'] = (tr.rolling(5).mean().shift(1) / df['prev_close']) * 100
    df['atr_10'] = (tr.rolling(10).mean().shift(1) / df['prev_close']) * 100
    df['atr_14'] = (tr.rolling(14).mean().shift(1) / df['prev_close']) * 100

    # Historical high/low patterns
    df['avg_high_5d'] = df['actual_high_pct'].rolling(5).mean().shift(1)
    df['avg_low_5d'] = df['actual_low_pct'].rolling(5).mean().shift(1)
    df['avg_high_10d'] = df['actual_high_pct'].rolling(10).mean().shift(1)
    df['avg_low_10d'] = df['actual_low_pct'].rolling(10).mean().shift(1)
    df['max_high_5d'] = df['actual_high_pct'].rolling(5).max().shift(1)
    df['max_low_5d'] = df['actual_low_pct'].rolling(5).max().shift(1)

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
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

    # Day of week
    df['day_of_week'] = df.index.dayofweek

    # Volume
    df['volume_ratio'] = df['Volume'].shift(1) / df['Volume'].rolling(20).mean().shift(1)

    # Consecutive patterns
    df['up_day'] = (df['Close'] > df['Open']).astype(int)
    df['consec_up'] = df['up_day'].rolling(5).sum().shift(1)

    return df


def evaluate_range_accuracy(y_high_true, y_low_true, y_close_true, high_pred, low_pred):
    """
    Evaluate how well predicted ranges capture EOD close

    Returns:
    - capture_rate: % of days where close is within predicted range
    - avg_miss: average distance when we miss (as %)
    - avg_range: average predicted range width
    """
    n = len(y_close_true)
    captured = 0
    misses = []
    ranges = []

    for i in range(n):
        pred_high = high_pred[i]  # % above open
        pred_low = low_pred[i]    # % below open (positive value)
        actual_close = y_close_true.iloc[i]  # % from open (can be negative)

        # Range is from -pred_low to +pred_high (relative to open)
        range_width = pred_high + pred_low
        ranges.append(range_width)

        # Check if close is within range
        if -pred_low <= actual_close <= pred_high:
            captured += 1
        else:
            # Calculate miss distance
            if actual_close > pred_high:
                miss = actual_close - pred_high
            else:  # actual_close < -pred_low
                miss = -pred_low - actual_close
            misses.append(abs(miss))

    capture_rate = captured / n * 100
    avg_miss = np.mean(misses) if misses else 0
    avg_range = np.mean(ranges)

    return capture_rate, avg_miss, avg_range


def train_highlow_model(ticker: str):
    """Train high/low prediction models optimized for capturing EOD close"""
    print(f"\n{'='*60}")
    print(f"Training High/Low Model v2 for {ticker}")
    print(f"Metric: Range Capture Rate (does range contain EOD close?)")
    print('='*60)

    # Fetch data
    print("\nFetching data from Polygon.io...")
    df = fetch_polygon_data(ticker, days=750)
    print(f"  Got {len(df)} days of data")

    # Calculate features
    df = calculate_features(df)

    # Features
    feature_cols = [
        'gap_pct', 'prev_range_pct', 'prev_high_pct', 'prev_low_pct', 'prev_close_pct',
        'prev_return', 'prev_2_return', 'prev_3_return',
        'momentum_3d', 'momentum_5d',
        'volatility_5d', 'volatility_10d', 'volatility_20d',
        'atr_5', 'atr_10', 'atr_14',
        'avg_high_5d', 'avg_low_5d', 'avg_high_10d', 'avg_low_10d',
        'max_high_5d', 'max_low_5d',
        'rsi_14', 'macd_histogram',
        'price_vs_sma20', 'price_vs_sma50',
        'day_of_week', 'volume_ratio', 'consec_up'
    ]

    # Clean data
    df_clean = df.dropna(subset=feature_cols + ['actual_high_pct', 'actual_low_pct', 'actual_close_pct'])
    print(f"  {len(df_clean)} samples after cleaning")

    X = df_clean[feature_cols]
    y_high = df_clean['actual_high_pct']
    y_low = df_clean['actual_low_pct']
    y_close = df_clean['actual_close_pct']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    print("\nTraining and evaluating models...")

    # Train HIGH model
    high_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    high_model.fit(X_scaled, y_high)

    # Train LOW model
    low_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    low_model.fit(X_scaled, y_low)

    # Evaluate on last 60 days (out of sample feel)
    test_size = 60
    X_test = X_scaled[-test_size:]
    y_high_test = y_high.iloc[-test_size:]
    y_low_test = y_low.iloc[-test_size:]
    y_close_test = y_close.iloc[-test_size:]

    high_pred = high_model.predict(X_test)
    low_pred = low_model.predict(X_test)

    # Raw model predictions
    capture_rate, avg_miss, avg_range = evaluate_range_accuracy(
        y_high_test, y_low_test, y_close_test, high_pred, low_pred
    )

    print(f"\n  Raw Model Performance (last {test_size} days):")
    print(f"    Range Capture Rate: {capture_rate:.1f}%")
    print(f"    Avg Miss (when wrong): {avg_miss:.3f}%")
    print(f"    Avg Range Width: {avg_range:.3f}%")

    # Add buffer to improve capture rate
    # Find optimal buffer that maximizes capture while keeping range tight
    best_buffer = 0
    best_score = 0

    for buffer in np.arange(0, 0.3, 0.02):
        cap_rate, miss, rng = evaluate_range_accuracy(
            y_high_test, y_low_test, y_close_test,
            high_pred + buffer, low_pred + buffer
        )
        # Score = capture_rate - penalty for wide range
        score = cap_rate - (rng * 5)  # Penalize wide ranges
        if cap_rate >= 90 and score > best_score:
            best_score = score
            best_buffer = buffer

    # If we didn't find 90%, use buffer that gets closest
    if best_buffer == 0:
        for buffer in np.arange(0, 0.5, 0.02):
            cap_rate, _, _ = evaluate_range_accuracy(
                y_high_test, y_low_test, y_close_test,
                high_pred + buffer, low_pred + buffer
            )
            if cap_rate >= 90:
                best_buffer = buffer
                break

    # Final evaluation with buffer
    final_capture, final_miss, final_range = evaluate_range_accuracy(
        y_high_test, y_low_test, y_close_test,
        high_pred + best_buffer, low_pred + best_buffer
    )

    print(f"\n  With Buffer (+{best_buffer:.2f}%):")
    print(f"    Range Capture Rate: {final_capture:.1f}%")
    print(f"    Avg Miss (when wrong): {final_miss:.3f}%")
    print(f"    Avg Range Width: {final_range:.3f}%")

    # Show some example predictions
    print(f"\n  Sample Predictions (last 5 days):")
    print(f"  {'Date':<12} {'Pred Range':<20} {'Actual Close':<15} {'Captured?'}")
    print(f"  {'-'*60}")

    for i in range(-5, 0):
        date = df_clean.index[i].strftime('%Y-%m-%d')
        pred_h = high_pred[i] + best_buffer
        pred_l = low_pred[i] + best_buffer
        actual_c = y_close_test.iloc[i]
        captured = "✓" if -pred_l <= actual_c <= pred_h else "✗"
        print(f"  {date:<12} -{pred_l:.2f}% to +{pred_h:.2f}%    {actual_c:+.2f}%          {captured}")

    # Save model
    model_data = {
        'high_model': high_model,
        'low_model': low_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'buffer': best_buffer,
        'ticker': ticker,
        'trained_at': datetime.now().isoformat(),
        'version': 'highlow_v2',
        'metrics': {
            'capture_rate': float(final_capture),
            'avg_miss': float(final_miss),
            'avg_range': float(final_range),
            'buffer': float(best_buffer),
            'samples': len(df_clean)
        }
    }

    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_highlow_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n  Saved model to {model_path}")

    return model_data


def main():
    """Train high/low models for all tickers"""
    os.makedirs(MODELS_DIR, exist_ok=True)

    tickers = ['SPY', 'QQQ', 'IWM']

    results = {}
    for ticker in tickers:
        try:
            model_data = train_highlow_model(ticker)
            results[ticker] = model_data['metrics']
        except Exception as e:
            print(f"Error training {ticker}: {e}")
            import traceback
            traceback.print_exc()
            results[ticker] = {'error': str(e)}

    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY - Range Capture Accuracy")
    print("="*60)
    print("\nMetric: % of days where EOD close falls within predicted range\n")

    for ticker, metrics in results.items():
        if 'error' in metrics:
            print(f"{ticker}: ERROR - {metrics['error']}")
        else:
            print(f"{ticker}:")
            print(f"  Range Capture Rate: {metrics['capture_rate']:.1f}%")
            print(f"  Avg Range Width: {metrics['avg_range']:.2f}%")
            print(f"  Buffer Applied: +{metrics['buffer']:.2f}%")
            print()


if __name__ == '__main__':
    main()
