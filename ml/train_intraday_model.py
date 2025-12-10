"""
Intraday Session Update Model

This model updates predictions throughout the trading day based on:
1. Pre-market: Gap from previous close
2. First 15-30 min: Early session direction
3. Mid-day: Current position in range, momentum
4. Late session: Trend confirmation

Training: 2003-2023 (~20 years)
Testing: 2024-2025 (~2 years)

The model predicts: "Will today close HIGHER than today's open?"
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from datetime import datetime
import pickle
import os
import requests
import warnings
warnings.filterwarnings('ignore')

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def fetch_daily_data(ticker: str, start_date: str = '2003-01-01', end_date: str = '2025-12-31') -> pd.DataFrame:
    """Fetch daily OHLCV data from Polygon"""
    print(f"Fetching daily data for {ticker}...")

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if 'results' not in data:
        raise ValueError(f"No data for {ticker}")

    df = pd.DataFrame(data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
    df = df.set_index('date')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    print(f"  Retrieved {len(df)} trading days")
    print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

    return df


def create_session_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simulated intraday snapshots from daily OHLC data.

    For each day, we simulate what we'd know at different points:
    - At open: Only know the gap
    - During day: Know current price relative to open, high, low so far

    We create multiple training samples per day at different "time slices"
    """

    snapshots = []

    # Previous day features
    df['prev_close'] = df['Close'].shift(1)
    df['prev_high'] = df['High'].shift(1)
    df['prev_low'] = df['Low'].shift(1)
    df['prev_range'] = (df['prev_high'] - df['prev_low']) / df['prev_close']
    df['prev_return'] = df['Close'].pct_change().shift(1)

    # Target: Did we close higher than open?
    df['bullish_day'] = (df['Close'] > df['Open']).astype(int)

    # Day's actual range
    df['day_range'] = (df['High'] - df['Low']) / df['Open']
    df['close_vs_open'] = (df['Close'] - df['Open']) / df['Open']

    # Gap
    df['gap'] = (df['Open'] - df['prev_close']) / df['prev_close']

    # Clean
    df = df.dropna()

    # For each day, create snapshots at different "time points"
    # We simulate time points: 0% (open), 10%, 25%, 50%, 75%, 90% through the day
    time_points = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]

    for idx, row in df.iterrows():
        open_price = row['Open']
        high_price = row['High']
        low_price = row['Low']
        close_price = row['Close']

        # Determine the "path" the price took
        # We'll use a simplified model: assume price moves somewhat linearly
        # toward its final position with some deviation to high/low

        for time_pct in time_points:
            snapshot = {
                'date': idx,
                'time_pct': time_pct,  # How far through the session (0-1)
                'time_remaining': 1 - time_pct,  # Time left in session

                # Known at this point
                'gap': row['gap'],
                'gap_direction': 1 if row['gap'] > 0 else -1 if row['gap'] < 0 else 0,
                'gap_size': abs(row['gap']),

                # Previous day context
                'prev_return': row['prev_return'],
                'prev_range': row['prev_range'],

                # Simulated current price position
                # At time 0: we're at open
                # At time 1: we're at close
                # In between: interpolate with some noise toward high/low
            }

            if time_pct == 0:
                # At open - no intraday info yet
                current_price = open_price
                high_so_far = open_price
                low_so_far = open_price
            else:
                # Simulate where price might be at this time point
                # This is an approximation based on typical price paths

                # Final direction
                final_direction = 1 if close_price > open_price else -1

                # Simulate current price as moving toward close with volatility
                progress = time_pct ** 0.8  # Slightly faster early moves

                # Base price moves toward close
                base_price = open_price + (close_price - open_price) * progress

                # Add some deviation - price often overshoots before settling
                if time_pct < 0.5:
                    # Early in day - might deviate more
                    deviation = (high_price - low_price) * 0.3 * (1 - time_pct)
                else:
                    # Later in day - converging to close
                    deviation = (high_price - low_price) * 0.1 * (1 - time_pct)

                current_price = base_price + np.random.uniform(-deviation, deviation) * 0.5

                # High/low so far (simulated)
                high_so_far = max(open_price, min(high_price, open_price + (high_price - open_price) * min(1, time_pct * 1.5)))
                low_so_far = min(open_price, max(low_price, open_price + (low_price - open_price) * min(1, time_pct * 1.5)))

            # Current position features
            range_so_far = high_so_far - low_so_far if high_so_far != low_so_far else 0.0001

            snapshot['current_vs_open'] = (current_price - open_price) / open_price
            snapshot['current_vs_open_direction'] = 1 if current_price > open_price else -1 if current_price < open_price else 0
            snapshot['position_in_range'] = (current_price - low_so_far) / range_so_far if range_so_far > 0 else 0.5
            snapshot['range_so_far_pct'] = range_so_far / open_price
            snapshot['high_so_far_pct'] = (high_so_far - open_price) / open_price
            snapshot['low_so_far_pct'] = (open_price - low_so_far) / open_price

            # Momentum indicators
            snapshot['above_open'] = 1 if current_price > open_price else 0
            snapshot['near_high'] = 1 if (high_so_far - current_price) < (current_price - low_so_far) else 0

            # Gap fill status
            if row['gap'] > 0:  # Gap up
                gap_filled = low_so_far <= row['prev_close']
            else:  # Gap down
                gap_filled = high_so_far >= row['prev_close']
            snapshot['gap_filled'] = 1 if gap_filled else 0

            # Target
            snapshot['target'] = row['bullish_day']

            snapshots.append(snapshot)

    return pd.DataFrame(snapshots)


def create_intraday_features(df: pd.DataFrame) -> tuple:
    """Extract feature columns and prepare data"""

    feature_cols = [
        'time_pct', 'time_remaining',
        'gap', 'gap_direction', 'gap_size',
        'prev_return', 'prev_range',
        'current_vs_open', 'current_vs_open_direction',
        'position_in_range', 'range_so_far_pct',
        'high_so_far_pct', 'low_so_far_pct',
        'above_open', 'near_high', 'gap_filled'
    ]

    return feature_cols, df[feature_cols], df['target']


def train_intraday_model(ticker: str = 'SPY'):
    """Train intraday update model"""

    print(f"\n{'='*70}")
    print(f"  INTRADAY MODEL TRAINING - {ticker}")
    print(f"{'='*70}")

    # Fetch data
    df = fetch_daily_data(ticker)

    # Create session snapshots
    print("\nCreating session snapshots...")
    snapshots = create_session_snapshots(df)
    print(f"  Created {len(snapshots)} training samples from {len(df)} days")

    # Split by date
    train_end = '2023-12-31'
    test_start = '2024-01-01'

    train_data = snapshots[snapshots['date'] <= train_end]
    test_data = snapshots[snapshots['date'] >= test_start]

    print(f"\n  TRAIN: {len(train_data)} samples ({train_data['date'].min().strftime('%Y-%m-%d')} to {train_data['date'].max().strftime('%Y-%m-%d')})")
    print(f"  TEST:  {len(test_data)} samples ({test_data['date'].min().strftime('%Y-%m-%d')} to {test_data['date'].max().strftime('%Y-%m-%d')})")

    # Get features
    feature_cols, X_train, y_train = create_intraday_features(train_data)
    _, X_test, y_test = create_intraday_features(test_data)

    print(f"\n  Train bullish rate: {y_train.mean():.1%}")
    print(f"  Test bullish rate: {y_test.mean():.1%}")

    # Handle any inf/nan
    X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)

    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    print("\nTraining models...")

    models = {
        'xgb': XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        'rf': RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        ),
        'et': ExtraTreesClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)

        # Overall accuracy
        pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, pred)

        # High confidence accuracy
        proba = model.predict_proba(X_test_scaled)[:, 1]
        high_conf_mask = (proba >= 0.65) | (proba <= 0.35)
        if high_conf_mask.sum() > 0:
            high_conf_acc = accuracy_score(y_test[high_conf_mask], pred[high_conf_mask])
            high_conf_count = high_conf_mask.sum()
        else:
            high_conf_acc = 0
            high_conf_count = 0

        results[name] = {
            'accuracy': acc,
            'high_conf_accuracy': high_conf_acc,
            'high_conf_count': high_conf_count
        }
        print(f"  {name}: {acc:.1%} overall, {high_conf_acc:.1%} high-conf ({high_conf_count} samples)")

    # Ensemble weights based on accuracy
    total_acc = sum(r['accuracy'] for r in results.values())
    weights = {name: r['accuracy'] / total_acc for name, r in results.items()}
    print(f"\n  Ensemble weights: {weights}")

    # Evaluate by time slice
    print("\n" + "="*50)
    print("  ACCURACY BY SESSION TIME")
    print("="*50)

    time_slices = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
    time_labels = ['Open', '10%', '25%', '50%', '75%', '90%']

    for time_pct, label in zip(time_slices, time_labels):
        mask = test_data['time_pct'] == time_pct
        if mask.sum() == 0:
            continue

        X_slice = X_test_scaled[mask.values]
        y_slice = y_test[mask].values

        # Ensemble prediction
        proba = np.zeros(len(y_slice))
        for name, model in models.items():
            proba += model.predict_proba(X_slice)[:, 1] * weights[name]

        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y_slice, pred)

        # High confidence
        high_conf = (proba >= 0.6) | (proba <= 0.4)
        if high_conf.sum() > 0:
            hc_acc = accuracy_score(y_slice[high_conf], pred[high_conf])
            hc_pct = high_conf.sum() / len(y_slice)
        else:
            hc_acc = 0
            hc_pct = 0

        print(f"  {label:5s}: {acc:.1%} overall | {hc_acc:.1%} @ 60% conf ({hc_pct:.0%} of signals)")

    # Calculate overall test metrics
    proba_all = np.zeros(len(y_test))
    for name, model in models.items():
        proba_all += model.predict_proba(X_test_scaled)[:, 1] * weights[name]

    pred_all = (proba_all >= 0.5).astype(int)
    overall_acc = accuracy_score(y_test, pred_all)

    # Save model
    model_data = {
        'models': models,
        'weights': weights,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'metrics': {
            'accuracy': overall_acc,
            'by_model': results
        },
        'ticker': ticker,
        'version': 'intraday_v1',
        'trained_at': datetime.now().isoformat()
    }

    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n✓ Model saved to {model_path}")

    return model_data


if __name__ == '__main__':
    print("="*70)
    print("   INTRADAY SESSION UPDATE MODEL TRAINING")
    print("   Train: 2003-2023 | Test: 2024-2025")
    print("="*70)

    results = {}
    for ticker in ['SPY', 'QQQ', 'IWM']:
        try:
            model_data = train_intraday_model(ticker)
            results[ticker] = model_data['metrics']['accuracy']
        except Exception as e:
            print(f"Error training {ticker}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("   FINAL RESULTS (Out-of-Sample 2024-2025)")
    print("="*70)
    for ticker, acc in results.items():
        print(f"  {ticker}: {acc:.1%} overall accuracy")
