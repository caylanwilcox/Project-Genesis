"""
Multi-Timeframe Target ML Model

Trains a model to predict accurate price targets using:
- Aftermarket High/Low (4 PM - 8 PM ET previous day)
- 24-hour High/Low (rolling)
- Pre-market High/Low (4 AM - 9:30 AM ET)
- First 30 minute High/Low (9:30 AM - 10:00 AM ET)

The model learns which ranges are most predictive of the day's actual high/low
and how to combine them for better target accuracy.
"""

import pandas as pd
import numpy as np
import requests
import pickle
import os
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '').strip()
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

def fetch_intraday_data(ticker: str, date: str, timespan: str = 'minute'):
    """Fetch intraday minute bars for a specific date"""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timespan}/{date}/{date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get('status') != 'OK' or 'results' not in data:
        return None

    df = pd.DataFrame(data['results'])
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})

    return df

def fetch_extended_hours_data(ticker: str, date: str):
    """Fetch extended hours data (pre-market and after-market)"""
    # Polygon doesn't separate extended hours, so we use the full day
    # and filter by time ranges
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get('status') != 'OK' or 'results' not in data:
        return None

    df = pd.DataFrame(data['results'])
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
    # Convert to Eastern Time
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})

    return df

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

    if data.get('status') != 'OK' or 'results' not in data:
        return None

    df = pd.DataFrame(data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df.set_index('date')

    return df

def calculate_timeframe_ranges(intraday_df):
    """
    Calculate high/low for different timeframes from intraday data.

    Returns dict with:
    - premarket: High/Low from 4 AM - 9:30 AM ET
    - first_30min: High/Low from 9:30 AM - 10:00 AM ET
    - regular_session: High/Low from 9:30 AM - 4:00 PM ET
    - aftermarket: High/Low from 4:00 PM - 8:00 PM ET (previous day)
    """
    if intraday_df is None or len(intraday_df) == 0:
        return None

    ranges = {}

    # Pre-market: 4 AM - 9:30 AM ET
    premarket = intraday_df[
        ((intraday_df['hour'] >= 4) & (intraday_df['hour'] < 9)) |
        ((intraday_df['hour'] == 9) & (intraday_df['minute'] < 30))
    ]
    if len(premarket) > 0:
        ranges['premarket_high'] = premarket['high'].max()
        ranges['premarket_low'] = premarket['low'].min()
        ranges['premarket_open'] = premarket.iloc[0]['open']
        ranges['premarket_close'] = premarket.iloc[-1]['close']
        ranges['premarket_volume'] = premarket['volume'].sum()
    else:
        ranges['premarket_high'] = None
        ranges['premarket_low'] = None
        ranges['premarket_open'] = None
        ranges['premarket_close'] = None
        ranges['premarket_volume'] = 0

    # First 30 minutes: 9:30 AM - 10:00 AM ET
    first_30 = intraday_df[
        ((intraday_df['hour'] == 9) & (intraday_df['minute'] >= 30)) |
        ((intraday_df['hour'] == 10) & (intraday_df['minute'] == 0))
    ]
    if len(first_30) > 0:
        ranges['first_30_high'] = first_30['high'].max()
        ranges['first_30_low'] = first_30['low'].min()
        ranges['first_30_range'] = ranges['first_30_high'] - ranges['first_30_low']
        ranges['first_30_volume'] = first_30['volume'].sum()
    else:
        ranges['first_30_high'] = None
        ranges['first_30_low'] = None
        ranges['first_30_range'] = None
        ranges['first_30_volume'] = 0

    # Regular session: 9:30 AM - 4:00 PM ET
    regular = intraday_df[
        ((intraday_df['hour'] == 9) & (intraday_df['minute'] >= 30)) |
        ((intraday_df['hour'] >= 10) & (intraday_df['hour'] < 16))
    ]
    if len(regular) > 0:
        ranges['session_high'] = regular['high'].max()
        ranges['session_low'] = regular['low'].min()
        ranges['session_open'] = regular.iloc[0]['open']
        ranges['session_close'] = regular.iloc[-1]['close']
        ranges['session_volume'] = regular['volume'].sum()
    else:
        ranges['session_high'] = None
        ranges['session_low'] = None
        ranges['session_open'] = None
        ranges['session_close'] = None
        ranges['session_volume'] = 0

    # After-market: 4:00 PM - 8:00 PM ET
    aftermarket = intraday_df[
        (intraday_df['hour'] >= 16) & (intraday_df['hour'] < 20)
    ]
    if len(aftermarket) > 0:
        ranges['aftermarket_high'] = aftermarket['high'].max()
        ranges['aftermarket_low'] = aftermarket['low'].min()
        ranges['aftermarket_volume'] = aftermarket['volume'].sum()
    else:
        ranges['aftermarket_high'] = None
        ranges['aftermarket_low'] = None
        ranges['aftermarket_volume'] = 0

    return ranges

def build_training_data(ticker: str, days: int = 365):
    """
    Build training dataset with multi-timeframe ranges and actual outcomes.
    """
    print(f"\nBuilding training data for {ticker}...")

    # Get daily data for context
    daily_df = fetch_daily_data(ticker, days + 30)
    if daily_df is None or len(daily_df) < 50:
        print(f"Insufficient daily data for {ticker}")
        return None

    # Calculate technical indicators
    daily_df['atr_14'] = calculate_atr(daily_df, 14)
    daily_df['sma_20'] = daily_df['close'].rolling(20).mean()
    daily_df['returns_1d'] = daily_df['close'].pct_change()
    daily_df['returns_5d'] = daily_df['close'].pct_change(5)
    daily_df['volatility_20d'] = daily_df['returns_1d'].rolling(20).std() * np.sqrt(252)
    daily_df['range_pct'] = (daily_df['high'] - daily_df['low']) / daily_df['close']

    training_rows = []

    # Get dates to process
    dates = daily_df.index[-days:].tolist()

    for i, date in enumerate(dates):
        if i < 5:  # Need previous days for context
            continue

        date_str = date.strftime('%Y-%m-%d')

        # Fetch intraday data for this date
        intraday_df = fetch_extended_hours_data(ticker, date_str)

        if intraday_df is None or len(intraday_df) < 100:
            continue

        # Calculate timeframe ranges
        ranges = calculate_timeframe_ranges(intraday_df)
        if ranges is None:
            continue

        # Skip if missing key data
        if ranges['session_high'] is None or ranges['session_low'] is None:
            continue

        # Get previous day's aftermarket data
        prev_date = dates[i-1].strftime('%Y-%m-%d')
        prev_intraday = fetch_extended_hours_data(ticker, prev_date)
        prev_ranges = calculate_timeframe_ranges(prev_intraday) if prev_intraday is not None else {}

        # Get daily context
        daily_row = daily_df.loc[date]
        prev_daily = daily_df.iloc[daily_df.index.get_loc(date) - 1]

        # Build feature row
        row = {
            'date': date,
            'ticker': ticker,

            # Actual outcomes (targets to predict)
            'actual_high': float(daily_row['high']),
            'actual_low': float(daily_row['low']),
            'actual_range': float(daily_row['high'] - daily_row['low']),

            # Current price reference
            'prev_close': float(prev_daily['close']),
            'open_price': float(daily_row['open']),

            # Pre-market features
            'premarket_high': ranges.get('premarket_high'),
            'premarket_low': ranges.get('premarket_low'),
            'premarket_range': (ranges.get('premarket_high', 0) or 0) - (ranges.get('premarket_low', 0) or 0),

            # First 30 min features
            'first_30_high': ranges.get('first_30_high'),
            'first_30_low': ranges.get('first_30_low'),
            'first_30_range': ranges.get('first_30_range'),

            # Previous aftermarket features
            'prev_aftermarket_high': prev_ranges.get('aftermarket_high') if prev_ranges else None,
            'prev_aftermarket_low': prev_ranges.get('aftermarket_low') if prev_ranges else None,

            # Technical features
            'atr_14': float(daily_row['atr_14']) if not pd.isna(daily_row.get('atr_14')) else None,
            'volatility_20d': float(daily_row['volatility_20d']) if not pd.isna(daily_row.get('volatility_20d')) else None,
            'range_pct_avg': float(daily_df['range_pct'].iloc[i-5:i].mean()),

            # Gap analysis
            'gap_pct': (float(daily_row['open']) - float(prev_daily['close'])) / float(prev_daily['close']),

            # Day of week (for patterns)
            'day_of_week': date.dayofweek,
        }

        training_rows.append(row)

        if len(training_rows) % 50 == 0:
            print(f"  Processed {len(training_rows)} days...")

    print(f"  Total training samples: {len(training_rows)}")

    return pd.DataFrame(training_rows)

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def train_target_model(ticker: str, df: pd.DataFrame):
    """
    Train ML model to predict accurate targets using multi-timeframe ranges.

    Predicts:
    1. high_offset: How much above first_30_high the day's high will be
    2. low_offset: How much below first_30_low the day's low will be
    """
    print(f"\nTraining target model for {ticker}...")

    # Filter valid rows
    df = df.dropna(subset=['first_30_high', 'first_30_low', 'atr_14'])

    if len(df) < 100:
        print(f"Insufficient data for {ticker}: {len(df)} samples")
        return None

    # Calculate targets: offset from first 30 min range
    df['high_offset'] = df['actual_high'] - df['first_30_high']
    df['low_offset'] = df['first_30_low'] - df['actual_low']
    df['high_extension_pct'] = df['high_offset'] / df['prev_close']
    df['low_extension_pct'] = df['low_offset'] / df['prev_close']

    # Feature engineering
    df['premarket_range_pct'] = df['premarket_range'] / df['prev_close']
    df['first_30_range_pct'] = df['first_30_range'] / df['prev_close']
    df['gap_direction'] = np.sign(df['gap_pct'])
    df['gap_size'] = np.abs(df['gap_pct'])
    df['atr_pct'] = df['atr_14'] / df['prev_close']

    # Premarket extension relative to ATR
    df['premarket_high_vs_close'] = (df['premarket_high'] - df['prev_close']) / df['atr_14']
    df['premarket_low_vs_close'] = (df['prev_close'] - df['premarket_low']) / df['atr_14']

    # First 30 min position
    df['first_30_high_vs_open'] = (df['first_30_high'] - df['open_price']) / df['atr_14']
    df['first_30_low_vs_open'] = (df['open_price'] - df['first_30_low']) / df['atr_14']

    # Use previous aftermarket if available
    df['prev_ah_range'] = df['prev_aftermarket_high'].fillna(df['prev_close']) - df['prev_aftermarket_low'].fillna(df['prev_close'])
    df['prev_ah_range_pct'] = df['prev_ah_range'] / df['prev_close']

    # Features for prediction
    feature_cols = [
        'premarket_range_pct',
        'first_30_range_pct',
        'gap_direction',
        'gap_size',
        'atr_pct',
        'volatility_20d',
        'range_pct_avg',
        'premarket_high_vs_close',
        'premarket_low_vs_close',
        'first_30_high_vs_open',
        'first_30_low_vs_open',
        'prev_ah_range_pct',
        'day_of_week',
    ]

    # Drop rows with NaN in features
    df_clean = df.dropna(subset=feature_cols + ['high_extension_pct', 'low_extension_pct'])

    if len(df_clean) < 50:
        print(f"Insufficient clean data for {ticker}: {len(df_clean)} samples")
        return None

    X = df_clean[feature_cols]
    y_high = df_clean['high_extension_pct']
    y_low = df_clean['low_extension_pct']

    # Split data
    X_train, X_test, y_high_train, y_high_test = train_test_split(X, y_high, test_size=0.2, random_state=42)
    _, _, y_low_train, y_low_test = train_test_split(X, y_low, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models for high extension
    high_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=5,
        random_state=42
    )
    high_model.fit(X_train_scaled, y_high_train)

    # Train models for low extension
    low_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=5,
        random_state=42
    )
    low_model.fit(X_train_scaled, y_low_train)

    # Evaluate
    high_pred = high_model.predict(X_test_scaled)
    low_pred = low_model.predict(X_test_scaled)

    high_mae = mean_absolute_error(y_high_test, high_pred) * 100  # Convert to %
    low_mae = mean_absolute_error(y_low_test, low_pred) * 100

    high_r2 = r2_score(y_high_test, high_pred)
    low_r2 = r2_score(y_low_test, low_pred)

    print(f"\n{ticker} Target Model Results:")
    print(f"  High Extension - MAE: {high_mae:.3f}%, R2: {high_r2:.3f}")
    print(f"  Low Extension  - MAE: {low_mae:.3f}%, R2: {low_r2:.3f}")

    # Calculate actual target accuracy
    # Simulate predictions and see if actual high/low fell within predicted range
    df_clean['pred_high_ext'] = high_model.predict(scaler.transform(df_clean[feature_cols]))
    df_clean['pred_low_ext'] = low_model.predict(scaler.transform(df_clean[feature_cols]))

    df_clean['pred_target_high'] = df_clean['first_30_high'] + df_clean['pred_high_ext'] * df_clean['prev_close']
    df_clean['pred_target_low'] = df_clean['first_30_low'] - df_clean['pred_low_ext'] * df_clean['prev_close']

    # How often does actual high/low stay within predicted targets?
    high_captured = (df_clean['actual_high'] <= df_clean['pred_target_high'] * 1.005).mean()  # 0.5% buffer
    low_captured = (df_clean['actual_low'] >= df_clean['pred_target_low'] * 0.995).mean()
    both_captured = ((df_clean['actual_high'] <= df_clean['pred_target_high'] * 1.005) &
                     (df_clean['actual_low'] >= df_clean['pred_target_low'] * 0.995)).mean()

    print(f"\n  Target Capture Rates:")
    print(f"    High target captures actual high: {high_captured:.1%}")
    print(f"    Low target captures actual low: {low_captured:.1%}")
    print(f"    Both targets capture range: {both_captured:.1%}")

    # Feature importance
    print(f"\n  Top Features (High Model):")
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': high_model.feature_importances_
    }).sort_values('importance', ascending=False)
    for _, row in importance_df.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']:.3f}")

    # Save model
    model_data = {
        'ticker': ticker,
        'high_model': high_model,
        'low_model': low_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'metrics': {
            'high_mae_pct': high_mae,
            'low_mae_pct': low_mae,
            'high_r2': high_r2,
            'low_r2': low_r2,
            'high_capture_rate': high_captured,
            'low_capture_rate': low_captured,
            'both_capture_rate': both_captured,
            'training_samples': len(df_clean),
        },
        'version': 'target_v1',
        'trained_at': datetime.now().isoformat(),
    }

    return model_data

def main():
    """Train target models for all tickers"""
    if not POLYGON_API_KEY:
        print("ERROR: POLYGON_API_KEY not set")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)

    tickers = ['SPY', 'QQQ', 'IWM']

    for ticker in tickers:
        try:
            # Build training data
            df = build_training_data(ticker, days=180)  # 6 months of data

            if df is None or len(df) < 50:
                print(f"Skipping {ticker} - insufficient data")
                continue

            # Train model
            model_data = train_target_model(ticker, df)

            if model_data:
                # Save model
                model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_target_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                print(f"\n  Saved {ticker} target model to {model_path}")

        except Exception as e:
            print(f"Error training {ticker}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("Target model training complete!")

if __name__ == '__main__':
    main()
