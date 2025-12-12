"""
Shrinking Range ML Model

Trains a model to predict the REMAINING high/low from current price
at different times of day.

Key insight: At 2pm, if price is at $682, we want to predict:
- What's the likely HIGH for the rest of the day?
- What's the likely LOW for the rest of the day?

This requires training on intraday data at different time slices.

Accuracy metric: Does the predicted remaining range capture the EOD close?
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import pickle
import os
import requests
from datetime import datetime, timedelta

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def fetch_intraday_data(ticker: str, date: str):
    """Fetch 5-minute intraday bars for a specific date"""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/{date}/{date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 5000,
        'apiKey': POLYGON_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if 'results' not in data or len(data['results']) == 0:
        return None

    df = pd.DataFrame(data['results'])
    df['datetime'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df.set_index('datetime')

    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_daily_data(ticker: str, days: int = 500):
    """Fetch daily OHLCV data"""
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

    if 'results' not in data:
        return None

    df = pd.DataFrame(data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
    df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})

    return df


def build_time_slice_dataset(ticker: str, num_days: int = 200):
    """
    Build training data by simulating different time points during each day.

    For each day, we create multiple samples at different times:
    - 10:00 AM (30 min into session)
    - 11:00 AM
    - 12:00 PM
    - 1:00 PM
    - 2:00 PM
    - 3:00 PM

    Features at each time slice:
    - Time remaining (as fraction of day)
    - Current price vs open
    - High so far vs open
    - Low so far vs open
    - Range so far
    - Previous day's range, close, etc.

    Targets:
    - Remaining upside: (day_high - current_price) / current_price
    - Remaining downside: (current_price - day_low) / current_price
    """
    print(f"Building time-slice dataset for {ticker}...")

    # Get daily data for context
    daily_df = fetch_daily_data(ticker, days=num_days + 50)
    if daily_df is None:
        raise ValueError(f"No daily data for {ticker}")

    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.set_index('date')

    # Calculate daily features
    daily_df['prev_close'] = daily_df['Close'].shift(1)
    daily_df['prev_range_pct'] = ((daily_df['High'].shift(1) - daily_df['Low'].shift(1)) / daily_df['Close'].shift(1)) * 100
    daily_df['prev_return'] = daily_df['Close'].pct_change().shift(1) * 100
    daily_df['volatility_5d'] = daily_df['Close'].pct_change().rolling(5).std().shift(1) * 100
    daily_df['gap_pct'] = ((daily_df['Open'] - daily_df['prev_close']) / daily_df['prev_close']) * 100

    # Day high/low as % from open
    daily_df['day_high_pct'] = ((daily_df['High'] - daily_df['Open']) / daily_df['Open']) * 100
    daily_df['day_low_pct'] = ((daily_df['Open'] - daily_df['Low']) / daily_df['Open']) * 100
    daily_df['day_close_pct'] = ((daily_df['Close'] - daily_df['Open']) / daily_df['Open']) * 100

    # Time slices (hours after market open at 9:30)
    time_slices = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]  # hours after open

    samples = []
    dates = daily_df.index.tolist()[-num_days:]

    for i, date in enumerate(dates):
        if i < 5:  # Need previous days for features
            continue

        try:
            row = daily_df.loc[date]

            if pd.isna(row['prev_close']) or pd.isna(row['volatility_5d']):
                continue

            day_open = row['Open']
            day_high = row['High']
            day_low = row['Low']
            day_close = row['Close']

            # For each time slice, simulate what we'd know at that point
            for hours_elapsed in time_slices:
                time_remaining = 1 - (hours_elapsed / 6.5)  # 6.5 hour trading day

                # Simulate high/low achieved by this time
                # Use a simple model: high/low develop somewhat linearly with sqrt(time)
                # This is an approximation - ideally we'd use actual intraday data
                time_factor = np.sqrt(hours_elapsed / 6.5)

                # Simulate current high/low at this time point
                high_so_far_pct = row['day_high_pct'] * time_factor * np.random.uniform(0.7, 1.0)
                low_so_far_pct = row['day_low_pct'] * time_factor * np.random.uniform(0.7, 1.0)

                # Simulate current price (somewhere between high and low so far)
                price_position = np.random.uniform(0.3, 0.7)
                current_pct = -low_so_far_pct + (high_so_far_pct + low_so_far_pct) * price_position
                current_price = day_open * (1 + current_pct / 100)

                # Actual remaining move from this simulated point
                remaining_upside = ((day_high - current_price) / current_price) * 100
                remaining_downside = ((current_price - day_low) / current_price) * 100

                # Target: remaining range that captures close
                close_from_current = ((day_close - current_price) / current_price) * 100

                sample = {
                    # Time features
                    'time_remaining': time_remaining,
                    'hours_elapsed': hours_elapsed,

                    # Current state features
                    'current_vs_open_pct': current_pct,
                    'high_so_far_pct': high_so_far_pct,
                    'low_so_far_pct': low_so_far_pct,
                    'range_so_far_pct': high_so_far_pct + low_so_far_pct,

                    # Daily context
                    'gap_pct': row['gap_pct'],
                    'prev_range_pct': row['prev_range_pct'],
                    'prev_return': row['prev_return'],
                    'volatility_5d': row['volatility_5d'],

                    # Targets
                    'remaining_upside': remaining_upside,
                    'remaining_downside': remaining_downside,
                    'close_from_current': close_from_current,
                }
                samples.append(sample)

        except Exception as e:
            continue

    df = pd.DataFrame(samples)
    print(f"  Built {len(df)} time-slice samples")
    return df


def train_shrinking_model(ticker: str):
    """Train model to predict remaining range at different times of day"""
    print(f"\n{'='*60}")
    print(f"Training Shrinking Range Model for {ticker}")
    print('='*60)

    # Build dataset
    df = build_time_slice_dataset(ticker, num_days=400)

    feature_cols = [
        'time_remaining', 'hours_elapsed',
        'current_vs_open_pct', 'high_so_far_pct', 'low_so_far_pct', 'range_so_far_pct',
        'gap_pct', 'prev_range_pct', 'prev_return', 'volatility_5d'
    ]

    df_clean = df.dropna()
    print(f"  {len(df_clean)} samples after cleaning")

    X = df_clean[feature_cols]
    y_up = df_clean['remaining_upside']
    y_down = df_clean['remaining_downside']

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train upside model
    print("\nTraining REMAINING UPSIDE model...")
    up_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    up_model.fit(X_scaled, y_up)

    # Train downside model
    print("Training REMAINING DOWNSIDE model...")
    down_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    down_model.fit(X_scaled, y_down)

    # Evaluate: Does predicted range capture the close?
    print("\nEvaluating range capture accuracy...")

    up_pred = up_model.predict(X_scaled)
    down_pred = down_model.predict(X_scaled)
    close_actual = df_clean['close_from_current'].values

    # Add buffer for safety margin
    buffers = [0, 0.05, 0.1, 0.15, 0.2]

    for buffer in buffers:
        captured = 0
        for i in range(len(close_actual)):
            pred_high = up_pred[i] + buffer
            pred_low = -down_pred[i] - buffer
            actual_close = close_actual[i]

            if pred_low <= actual_close <= pred_high:
                captured += 1

        capture_rate = captured / len(close_actual) * 100
        print(f"  Buffer +{buffer:.2f}%: {capture_rate:.1f}% capture rate")

    # Find optimal buffer for 90%+ capture
    best_buffer = 0
    for buffer in np.arange(0, 0.5, 0.02):
        captured = 0
        for i in range(len(close_actual)):
            pred_high = up_pred[i] + buffer
            pred_low = -down_pred[i] - buffer
            actual_close = close_actual[i]
            if pred_low <= actual_close <= pred_high:
                captured += 1
        capture_rate = captured / len(close_actual) * 100
        if capture_rate >= 90:
            best_buffer = buffer
            break

    # Final evaluation
    final_captured = 0
    for i in range(len(close_actual)):
        pred_high = up_pred[i] + best_buffer
        pred_low = -down_pred[i] - best_buffer
        if pred_low <= close_actual[i] <= pred_high:
            final_captured += 1
    final_capture_rate = final_captured / len(close_actual) * 100

    print(f"\n  Optimal buffer: +{best_buffer:.2f}%")
    print(f"  Final capture rate: {final_capture_rate:.1f}%")

    # MAE
    up_mae = mean_absolute_error(y_up, up_pred)
    down_mae = mean_absolute_error(y_down, down_pred)
    print(f"  Upside MAE: {up_mae:.3f}%")
    print(f"  Downside MAE: {down_mae:.3f}%")

    # Save model
    model_data = {
        'up_model': up_model,
        'down_model': down_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'buffer': best_buffer,
        'ticker': ticker,
        'trained_at': datetime.now().isoformat(),
        'version': 'shrinking_v1',
        'metrics': {
            'capture_rate': float(final_capture_rate),
            'up_mae': float(up_mae),
            'down_mae': float(down_mae),
            'buffer': float(best_buffer),
            'samples': len(df_clean)
        }
    }

    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_shrinking_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n  Saved to {model_path}")

    return model_data


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    tickers = ['SPY', 'QQQ', 'IWM']

    results = {}
    for ticker in tickers:
        try:
            model_data = train_shrinking_model(ticker)
            results[ticker] = model_data['metrics']
        except Exception as e:
            print(f"Error training {ticker}: {e}")
            import traceback
            traceback.print_exc()
            results[ticker] = {'error': str(e)}

    print("\n" + "="*60)
    print("SHRINKING RANGE MODEL SUMMARY")
    print("="*60)
    for ticker, metrics in results.items():
        if 'error' in metrics:
            print(f"{ticker}: ERROR - {metrics['error']}")
        else:
            print(f"{ticker}:")
            print(f"  Capture Rate: {metrics['capture_rate']:.1f}%")
            print(f"  Upside MAE: {metrics['up_mae']:.3f}%")
            print(f"  Downside MAE: {metrics['down_mae']:.3f}%")
            print(f"  Buffer: +{metrics['buffer']:.2f}%")
            print()


if __name__ == '__main__':
    main()
