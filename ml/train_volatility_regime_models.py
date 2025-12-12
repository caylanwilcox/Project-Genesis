"""
Volatility Regime Models

Trains SEPARATE models for different market volatility conditions:
- LOW volatility (VIX < 15 or ATR percentile < 30%)
- NORMAL volatility (VIX 15-25 or ATR percentile 30-70%)
- HIGH volatility (VIX > 25 or ATR percentile > 70%)

The prediction server automatically detects the current regime
and uses the appropriate model.

This improves accuracy because:
- Low-vol days have tighter ranges and more predictable patterns
- High-vol days have wider ranges and momentum-driven moves
- Different features matter in each regime
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
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

# Volatility regime thresholds (using ATR percentile)
LOW_VOL_THRESHOLD = 0.30   # Bottom 30% of historical volatility
HIGH_VOL_THRESHOLD = 0.70  # Top 30% of historical volatility


def fetch_polygon_data_range(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical daily data from Polygon.io"""
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
        return pd.DataFrame()

    df = pd.DataFrame(data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={
        'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
    })
    df = df.set_index('date')
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df


def calculate_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volatility regime for each day"""

    # ATR calculation
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_pct'] = (df['atr_14'] / df['close']) * 100

    # Rolling ATR percentile (where does current ATR rank vs last 252 days?)
    df['atr_percentile'] = df['atr_14'].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 252 else 0.5
    )

    # Volatility of returns
    df['daily_return'] = df['close'].pct_change() * 100
    df['volatility_20d'] = df['daily_return'].rolling(20).std()
    df['vol_percentile'] = df['volatility_20d'].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 252 else 0.5
    )

    # Combined volatility score (average of ATR and return volatility percentiles)
    df['vol_score'] = (df['atr_percentile'] + df['vol_percentile']) / 2

    # Assign regime
    def get_regime(score):
        if pd.isna(score):
            return 'NORMAL'
        elif score < LOW_VOL_THRESHOLD:
            return 'LOW'
        elif score > HIGH_VOL_THRESHOLD:
            return 'HIGH'
        else:
            return 'NORMAL'

    df['volatility_regime'] = df['vol_score'].shift(1).apply(get_regime)

    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate features for prediction"""

    # Targets
    df['target'] = (df['daily_return'] > 0).astype(int)
    df['actual_high_pct'] = ((df['high'] - df['open']) / df['open']) * 100
    df['actual_low_pct'] = ((df['open'] - df['low']) / df['open']) * 100

    # Previous returns
    df['prev_return'] = df['daily_return'].shift(1)
    df['prev_2_return'] = df['daily_return'].shift(2)
    df['prev_3_return'] = df['daily_return'].shift(3)
    df['prev_5_return'] = df['daily_return'].shift(5)

    # Momentum
    df['momentum_3d'] = df['daily_return'].rolling(3).sum().shift(1)
    df['momentum_5d'] = df['daily_return'].rolling(5).sum().shift(1)
    df['momentum_10d'] = df['daily_return'].rolling(10).sum().shift(1)

    # Volatility features
    df['volatility_5d'] = df['daily_return'].rolling(5).std().shift(1)
    df['volatility_10d'] = df['daily_return'].rolling(10).std().shift(1)
    df['vol_ratio'] = (df['volatility_5d'] / df['volatility_20d']).shift(1)

    # ATR features
    df['prev_atr_pct'] = df['atr_pct'].shift(1)
    df['atr_5'] = df['atr_14'].rolling(5).mean().shift(1) / df['close'].shift(1) * 100

    # Range features
    df['daily_range'] = ((df['high'] - df['low']) / df['close']) * 100
    df['prev_range'] = df['daily_range'].shift(1)
    df['avg_range_5d'] = df['daily_range'].rolling(5).mean().shift(1)
    df['avg_range_10d'] = df['daily_range'].rolling(10).mean().shift(1)

    # Historical high/low patterns
    df['avg_high_5d'] = df['actual_high_pct'].rolling(5).mean().shift(1)
    df['avg_low_5d'] = df['actual_low_pct'].rolling(5).mean().shift(1)
    df['avg_high_10d'] = df['actual_high_pct'].rolling(10).mean().shift(1)
    df['avg_low_10d'] = df['actual_low_pct'].rolling(10).mean().shift(1)

    # Gap
    df['gap'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) * 100
    df['abs_gap'] = df['gap'].abs()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 0.001)
    df['rsi_14'] = (100 - (100 / (1 + rs))).shift(1)

    # Moving averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['price_vs_sma20'] = ((df['close'].shift(1) - df['sma_20'].shift(1)) / df['sma_20'].shift(1)) * 100
    df['price_vs_sma50'] = ((df['close'].shift(1) - df['sma_50'].shift(1)) / df['sma_50'].shift(1)) * 100

    # Bollinger
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_position'] = ((df['close'] - (bb_middle - 2*bb_std)) / (4*bb_std + 0.001)).shift(1)
    df['bb_width'] = ((4 * bb_std) / bb_middle * 100).shift(1)

    # Calendar
    df['day_of_week'] = df.index.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)

    # Consecutive days
    df['up_day'] = (df['close'] > df['open']).astype(int)
    df['consec_up'] = df['up_day'].rolling(5).sum().shift(1)

    # Volume
    df['volume_ratio'] = (df['volume'] / df['volume'].rolling(20).mean()).shift(1)

    return df


def get_feature_columns():
    """Return feature columns for models"""
    return [
        'prev_return', 'prev_2_return', 'prev_3_return', 'prev_5_return',
        'momentum_3d', 'momentum_5d', 'momentum_10d',
        'volatility_5d', 'volatility_10d', 'vol_ratio',
        'prev_atr_pct', 'atr_5', 'vol_score',
        'prev_range', 'avg_range_5d', 'avg_range_10d',
        'avg_high_5d', 'avg_low_5d', 'avg_high_10d', 'avg_low_10d',
        'gap', 'abs_gap',
        'rsi_14', 'price_vs_sma20', 'price_vs_sma50',
        'bb_position', 'bb_width',
        'day_of_week', 'is_monday', 'is_friday',
        'consec_up', 'volume_ratio'
    ]


def train_direction_model(X_train, y_train, X_test, y_test, regime_name):
    """Train direction prediction model for a specific regime"""

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb.fit(X_train_scaled, y_train)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    gb.fit(X_train_scaled, y_train)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    # Accuracies
    xgb_acc = xgb.score(X_test_scaled, y_test)
    rf_acc = rf.score(X_test_scaled, y_test)
    gb_acc = gb.score(X_test_scaled, y_test)
    lr_acc = lr.score(X_test_scaled, y_test)

    # Weights
    total_acc = xgb_acc + rf_acc + gb_acc + lr_acc
    weights = {
        'xgb': xgb_acc / total_acc,
        'rf': rf_acc / total_acc,
        'gb': gb_acc / total_acc,
        'lr': lr_acc / total_acc
    }

    # Ensemble prediction
    y_pred_proba = (
        xgb.predict_proba(X_test_scaled)[:, 1] * weights['xgb'] +
        rf.predict_proba(X_test_scaled)[:, 1] * weights['rf'] +
        gb.predict_proba(X_test_scaled)[:, 1] * weights['gb'] +
        lr.predict_proba(X_test_scaled)[:, 1] * weights['lr']
    )
    y_pred = (y_pred_proba >= 0.5).astype(int)
    ensemble_acc = (y_pred == y_test.values).mean()

    # High confidence accuracy
    high_conf_mask = (y_pred_proba >= 0.65) | (y_pred_proba <= 0.35)
    high_conf_acc = (y_pred[high_conf_mask] == y_test.values[high_conf_mask]).mean() if high_conf_mask.sum() > 0 else 0

    print(f"    {regime_name} Direction: {ensemble_acc:.1%} overall, {high_conf_acc:.1%} high-conf ({high_conf_mask.sum()} signals)")

    return {
        'models': {'xgb': xgb, 'rf': rf, 'gb': gb, 'lr': lr},
        'weights': weights,
        'scaler': scaler,
        'accuracy': ensemble_acc,
        'high_conf_accuracy': high_conf_acc
    }


def train_highlow_model(X_train, y_high_train, y_low_train, X_test, y_high_test, y_low_test, regime_name):
    """Train high/low prediction model for a specific regime"""

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # High model
    xgb_high = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    xgb_high.fit(X_train_scaled, y_high_train)

    gb_high = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
    gb_high.fit(X_train_scaled, y_high_train)

    rf_high = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf_high.fit(X_train_scaled, y_high_train)

    # Low model
    xgb_low = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    xgb_low.fit(X_train_scaled, y_low_train)

    gb_low = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
    gb_low.fit(X_train_scaled, y_low_train)

    rf_low = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf_low.fit(X_train_scaled, y_low_train)

    # Predictions
    high_pred = (xgb_high.predict(X_test_scaled) * 0.4 +
                 gb_high.predict(X_test_scaled) * 0.3 +
                 rf_high.predict(X_test_scaled) * 0.3)

    low_pred = (xgb_low.predict(X_test_scaled) * 0.4 +
                gb_low.predict(X_test_scaled) * 0.3 +
                rf_low.predict(X_test_scaled) * 0.3)

    # MAE
    high_mae = np.mean(np.abs(high_pred - y_high_test.values))
    low_mae = np.mean(np.abs(low_pred - y_low_test.values))

    print(f"    {regime_name} High/Low MAE: {high_mae:.3f}% / {low_mae:.3f}%")

    return {
        'high_models': {'xgb': xgb_high, 'gb': gb_high, 'rf': rf_high},
        'low_models': {'xgb': xgb_low, 'gb': gb_low, 'rf': rf_low},
        'weights': {'xgb': 0.4, 'gb': 0.3, 'rf': 0.3},
        'scaler': scaler,
        'high_mae': high_mae,
        'low_mae': low_mae
    }


def train_regime_models(ticker: str):
    """Train separate models for each volatility regime"""

    print(f"\n{'='*60}")
    print(f"Training VOLATILITY REGIME Models for {ticker}")
    print('='*60)

    # Fetch data
    print("\nFetching data...")
    df_train = fetch_polygon_data_range(ticker, TRAIN_START, TRAIN_END)
    df_test = fetch_polygon_data_range(ticker, TEST_START, TEST_END)
    print(f"  {len(df_train)} train + {len(df_test)} test days")

    # Combine and calculate features
    df_all = pd.concat([df_train, df_test])
    df_all = calculate_volatility_regime(df_all)
    df_all = calculate_features(df_all)

    feature_cols = get_feature_columns()
    required_cols = feature_cols + ['target', 'actual_high_pct', 'actual_low_pct', 'volatility_regime']

    # Split back
    train_end_date = pd.Timestamp(TRAIN_END)
    test_start_date = pd.Timestamp(TEST_START)

    df_train_clean = df_all[df_all.index <= train_end_date].dropna(subset=required_cols)
    df_test_clean = df_all[df_all.index >= test_start_date].dropna(subset=required_cols)

    print(f"  Clean: {len(df_train_clean)} train + {len(df_test_clean)} test")

    # Regime distribution
    print("\n  Regime Distribution (Train):")
    for regime in ['LOW', 'NORMAL', 'HIGH']:
        count = (df_train_clean['volatility_regime'] == regime).sum()
        pct = count / len(df_train_clean) * 100
        print(f"    {regime}: {count} days ({pct:.1f}%)")

    print("\n  Regime Distribution (Test 2025):")
    for regime in ['LOW', 'NORMAL', 'HIGH']:
        count = (df_test_clean['volatility_regime'] == regime).sum()
        pct = count / len(df_test_clean) * 100
        print(f"    {regime}: {count} days ({pct:.1f}%)")

    # Train models for each regime
    regime_models = {}

    print("\nTraining regime-specific models...")

    for regime in ['LOW', 'NORMAL', 'HIGH']:
        train_regime = df_train_clean[df_train_clean['volatility_regime'] == regime]
        test_regime = df_test_clean[df_test_clean['volatility_regime'] == regime]

        if len(train_regime) < 100 or len(test_regime) < 10:
            print(f"  Skipping {regime} regime (not enough data)")
            continue

        print(f"\n  {regime} VOLATILITY ({len(train_regime)} train, {len(test_regime)} test):")

        X_train = train_regime[feature_cols]
        y_train = train_regime['target']
        y_high_train = train_regime['actual_high_pct']
        y_low_train = train_regime['actual_low_pct']

        X_test = test_regime[feature_cols]
        y_test = test_regime['target']
        y_high_test = test_regime['actual_high_pct']
        y_low_test = test_regime['actual_low_pct']

        # Train direction model
        direction_model = train_direction_model(X_train, y_train, X_test, y_test, regime)

        # Train high/low model
        highlow_model = train_highlow_model(
            X_train, y_high_train, y_low_train,
            X_test, y_high_test, y_low_test, regime
        )

        regime_models[regime] = {
            'direction': direction_model,
            'highlow': highlow_model,
            'train_samples': len(train_regime),
            'test_samples': len(test_regime)
        }

    # Also train ALL-regime model as fallback
    print(f"\n  ALL REGIMES (fallback model):")
    X_train_all = df_train_clean[feature_cols]
    y_train_all = df_train_clean['target']
    y_high_train_all = df_train_clean['actual_high_pct']
    y_low_train_all = df_train_clean['actual_low_pct']

    X_test_all = df_test_clean[feature_cols]
    y_test_all = df_test_clean['target']
    y_high_test_all = df_test_clean['actual_high_pct']
    y_low_test_all = df_test_clean['actual_low_pct']

    direction_model_all = train_direction_model(X_train_all, y_train_all, X_test_all, y_test_all, "ALL")
    highlow_model_all = train_highlow_model(
        X_train_all, y_high_train_all, y_low_train_all,
        X_test_all, y_high_test_all, y_low_test_all, "ALL"
    )

    regime_models['ALL'] = {
        'direction': direction_model_all,
        'highlow': highlow_model_all,
        'train_samples': len(df_train_clean),
        'test_samples': len(df_test_clean)
    }

    # Save model
    model_data = {
        'regime_models': regime_models,
        'feature_cols': feature_cols,
        'thresholds': {
            'low_vol': LOW_VOL_THRESHOLD,
            'high_vol': HIGH_VOL_THRESHOLD
        },
        'ticker': ticker,
        'version': 'volatility_regime_v1',
        'trained_at': datetime.now().isoformat()
    }

    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_regime_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\nSaved to {model_path}")

    return model_data


def compare_regime_vs_single():
    """Compare regime-based models vs single model performance"""

    print("\n" + "="*70)
    print("   VOLATILITY REGIME MODEL TRAINING")
    print("="*70)

    results = {}
    for ticker in ['SPY', 'QQQ', 'IWM']:
        try:
            model_data = train_regime_models(ticker)
            results[ticker] = model_data
        except Exception as e:
            print(f"Error training {ticker}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("   RESULTS SUMMARY")
    print("="*70)

    for ticker, data in results.items():
        print(f"\n{ticker}:")
        print(f"  {'Regime':<10} {'Direction':<15} {'High-Conf':<15} {'High MAE':<12} {'Low MAE':<12}")
        print(f"  {'-'*60}")

        for regime, model in data['regime_models'].items():
            dir_acc = model['direction']['accuracy']
            high_conf = model['direction']['high_conf_accuracy']
            high_mae = model['highlow']['high_mae']
            low_mae = model['highlow']['low_mae']
            print(f"  {regime:<10} {dir_acc:.1%}{'':<10} {high_conf:.1%}{'':<10} {high_mae:.3f}%{'':<7} {low_mae:.3f}%")


if __name__ == '__main__':
    compare_regime_vs_single()
