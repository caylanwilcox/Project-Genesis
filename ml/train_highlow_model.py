"""
High/Low Price Prediction Model

Trains ML models to predict the daily high and low prices for SPY, QQQ, IWM.
Uses regression models to predict the percentage move from open to high/low.

Features:
- Recent price action (returns, momentum)
- Volatility metrics (ATR, range, std dev)
- Technical indicators (RSI, MACD)
- Day of week patterns
- Gap analysis (open vs previous close)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import requests
from datetime import datetime, timedelta

# Polygon API
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def fetch_polygon_data(ticker: str, days: int = 750) -> pd.DataFrame:
    """Fetch historical daily data from Polygon.io (3 years for training)"""
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

    # Target variables: percentage from open to high/low
    df['high_pct'] = ((df['High'] - df['Open']) / df['Open']) * 100
    df['low_pct'] = ((df['Open'] - df['Low']) / df['Open']) * 100  # Positive = how far down

    # Previous day metrics (shifted to avoid lookahead)
    df['prev_close'] = df['Close'].shift(1)
    df['prev_high'] = df['High'].shift(1)
    df['prev_low'] = df['Low'].shift(1)
    df['prev_open'] = df['Open'].shift(1)
    df['prev_volume'] = df['Volume'].shift(1)

    # Gap from previous close to today's open
    df['gap_pct'] = ((df['Open'] - df['prev_close']) / df['prev_close']) * 100

    # Previous day's range
    df['prev_range_pct'] = ((df['prev_high'] - df['prev_low']) / df['prev_close']) * 100
    df['prev_high_pct'] = ((df['prev_high'] - df['prev_open']) / df['prev_open']) * 100
    df['prev_low_pct'] = ((df['prev_open'] - df['prev_low']) / df['prev_open']) * 100

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

    # Average True Range (shifted)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_5'] = (tr.rolling(5).mean().shift(1) / df['prev_close']) * 100
    df['atr_10'] = (tr.rolling(10).mean().shift(1) / df['prev_close']) * 100
    df['atr_14'] = (tr.rolling(14).mean().shift(1) / df['prev_close']) * 100

    # Average high/low from open (what we're trying to predict)
    df['avg_high_5d'] = df['high_pct'].rolling(5).mean().shift(1)
    df['avg_low_5d'] = df['low_pct'].rolling(5).mean().shift(1)
    df['avg_high_10d'] = df['high_pct'].rolling(10).mean().shift(1)
    df['avg_low_10d'] = df['low_pct'].rolling(10).mean().shift(1)

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
    df['macd_histogram'] = (df['macd'] - df['macd_signal']).shift(1)

    # Price vs moving averages
    sma_20 = df['Close'].rolling(20).mean()
    sma_50 = df['Close'].rolling(50).mean()
    df['price_vs_sma20'] = ((df['prev_close'] - sma_20.shift(1)) / sma_20.shift(1)) * 100
    df['price_vs_sma50'] = ((df['prev_close'] - sma_50.shift(1)) / sma_50.shift(1)) * 100

    # Day of week (Monday=0, Friday=4)
    df['day_of_week'] = df.index.dayofweek

    # Volume metrics
    df['volume_ratio'] = df['prev_volume'] / df['Volume'].rolling(20).mean().shift(1)

    # Consecutive up/down days
    df['up_day'] = (df['Close'] > df['Open']).astype(int)
    df['consec_up'] = df['up_day'].rolling(5).sum().shift(1)
    df['consec_down'] = 5 - df['consec_up']

    return df


def train_highlow_model(ticker: str):
    """Train high/low prediction models for a ticker"""
    print(f"\n{'='*50}")
    print(f"Training High/Low Model for {ticker}")
    print('='*50)

    # Fetch data
    print("Fetching data from Polygon.io...")
    df = fetch_polygon_data(ticker, days=750)  # ~3 years
    print(f"  Got {len(df)} days of data")

    # Calculate features
    df = calculate_features(df)

    # Define features
    feature_cols = [
        'gap_pct', 'prev_range_pct', 'prev_high_pct', 'prev_low_pct',
        'prev_return', 'prev_2_return', 'prev_3_return',
        'momentum_3d', 'momentum_5d',
        'volatility_5d', 'volatility_10d', 'volatility_20d',
        'atr_5', 'atr_10', 'atr_14',
        'avg_high_5d', 'avg_low_5d', 'avg_high_10d', 'avg_low_10d',
        'rsi_14', 'macd_histogram',
        'price_vs_sma20', 'price_vs_sma50',
        'day_of_week', 'volume_ratio',
        'consec_up', 'consec_down'
    ]

    # Drop NaN rows
    df_clean = df.dropna(subset=feature_cols + ['high_pct', 'low_pct'])
    print(f"  {len(df_clean)} samples after cleaning")

    X = df_clean[feature_cols]
    y_high = df_clean['high_pct']
    y_low = df_clean['low_pct']

    # Time series split for validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train High prediction model
    print("\nTraining HIGH prediction model...")
    high_scores = []
    for train_idx, val_idx in tscv.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_high.iloc[train_idx], y_high.iloc[val_idx]

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        high_scores.append(mae)

    print(f"  Cross-val MAE: {np.mean(high_scores):.4f}% (±{np.std(high_scores):.4f})")

    # Train final high model on all data
    high_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    high_model.fit(X_scaled, y_high)

    # Train Low prediction model
    print("\nTraining LOW prediction model...")
    low_scores = []
    for train_idx, val_idx in tscv.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_low.iloc[train_idx], y_low.iloc[val_idx]

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        low_scores.append(mae)

    print(f"  Cross-val MAE: {np.mean(low_scores):.4f}% (±{np.std(low_scores):.4f})")

    # Train final low model on all data
    low_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    low_model.fit(X_scaled, y_low)

    # Evaluate on recent data (last 60 days)
    print("\nBacktest on last 60 days:")
    recent_X = X_scaled[-60:]
    recent_y_high = y_high.iloc[-60:]
    recent_y_low = y_low.iloc[-60:]

    high_pred = high_model.predict(recent_X)
    low_pred = low_model.predict(recent_X)

    high_mae = mean_absolute_error(recent_y_high, high_pred)
    low_mae = mean_absolute_error(recent_y_low, low_pred)

    # Calculate hit rates (prediction within X% of actual)
    high_within_025 = np.mean(np.abs(recent_y_high.values - high_pred) < 0.25) * 100
    high_within_05 = np.mean(np.abs(recent_y_high.values - high_pred) < 0.5) * 100
    low_within_025 = np.mean(np.abs(recent_y_low.values - low_pred) < 0.25) * 100
    low_within_05 = np.mean(np.abs(recent_y_low.values - low_pred) < 0.5) * 100

    print(f"  HIGH - MAE: {high_mae:.4f}%")
    print(f"    Within 0.25%: {high_within_025:.1f}%")
    print(f"    Within 0.50%: {high_within_05:.1f}%")
    print(f"  LOW - MAE: {low_mae:.4f}%")
    print(f"    Within 0.25%: {low_within_025:.1f}%")
    print(f"    Within 0.50%: {low_within_05:.1f}%")

    # Average actual high/low for context
    avg_high = recent_y_high.mean()
    avg_low = recent_y_low.mean()
    print(f"\n  Avg actual high from open: +{avg_high:.3f}%")
    print(f"  Avg actual low from open: -{avg_low:.3f}%")

    # Save model
    model_data = {
        'high_model': high_model,
        'low_model': low_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'ticker': ticker,
        'trained_at': datetime.now().isoformat(),
        'metrics': {
            'high_mae': float(high_mae),
            'low_mae': float(low_mae),
            'high_within_05': float(high_within_05),
            'low_within_05': float(low_within_05),
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
            results[ticker] = {'error': str(e)}

    # Summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    for ticker, metrics in results.items():
        if 'error' in metrics:
            print(f"{ticker}: ERROR - {metrics['error']}")
        else:
            print(f"{ticker}:")
            print(f"  High MAE: {metrics['high_mae']:.4f}% | Within 0.5%: {metrics['high_within_05']:.1f}%")
            print(f"  Low MAE: {metrics['low_mae']:.4f}% | Within 0.5%: {metrics['low_within_05']:.1f}%")


if __name__ == '__main__':
    main()
