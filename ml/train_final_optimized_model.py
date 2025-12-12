"""
Final Optimized Daily Direction Prediction Model

Uses ONLY the 18 consensus features identified by feature_optimizer.py:
- Features selected by 2+ tickers during forward selection
- Removes redundant/noisy features that hurt generalization
- Targets better out-of-sample performance

Consensus Features (18):
- 3/3 tickers: gap, day_of_week, prev_2_return
- 2/3 tickers: atr_percentile, avg_range_5d, bb_position, momentum_3d,
               momentum_5d, momentum_10d, prev_3_return, prev_5_return,
               prev_atr_pct, prev_return, prev_volume_ratio, range_expansion,
               sma10_vs_sma50, vol_ratio_5_20, volume_price_trend
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from datetime import datetime, timedelta
import pickle
import os
import requests
import warnings
warnings.filterwarnings('ignore')

# Polygon API
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Consensus features from optimizer (sorted by selection frequency)
CONSENSUS_FEATURES = [
    # Selected by all 3 tickers
    'gap',
    'day_of_week',
    'prev_2_return',
    # Selected by 2/3 tickers
    'atr_percentile',
    'avg_range_5d',
    'bb_position',
    'momentum_3d',
    'momentum_5d',
    'momentum_10d',
    'prev_3_return',
    'prev_5_return',
    'prev_atr_pct',
    'prev_return',
    'prev_volume_ratio',
    'range_expansion',
    'sma10_vs_sma50',
    'vol_ratio_5_20',
    'volume_price_trend',
]


def fetch_polygon_data(ticker: str, days: int = 800) -> pd.DataFrame:
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
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume'
    })
    df = df.set_index('date')
    df = df[['open', 'high', 'low', 'close', 'volume']]

    return df


def calculate_consensus_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ONLY the 18 consensus features for final model"""

    # ========== PRICE ACTION ==========
    df['daily_return'] = df['close'].pct_change() * 100
    df['prev_return'] = df['daily_return'].shift(1)
    df['prev_2_return'] = df['daily_return'].shift(2)
    df['prev_3_return'] = df['daily_return'].shift(3)
    df['prev_5_return'] = df['daily_return'].shift(5)

    # ========== MOMENTUM ==========
    df['momentum_3d'] = df['daily_return'].rolling(3).sum().shift(1)
    df['momentum_5d'] = df['daily_return'].rolling(5).sum().shift(1)
    df['momentum_10d'] = df['daily_return'].rolling(10).sum().shift(1)

    # ========== VOLATILITY ==========
    df['volatility_5d'] = df['daily_return'].rolling(5).std().shift(1)
    df['volatility_20d'] = df['daily_return'].rolling(20).std().shift(1)
    df['vol_ratio_5_20'] = (df['volatility_5d'] / df['volatility_20d']).shift(1)

    # ATR-based volatility
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['prev_atr_pct'] = (df['atr_14'].shift(1) / df['close'].shift(1)) * 100

    # ATR percentile (VIX-like)
    df['atr_percentile'] = df['atr_14'].rolling(50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    ).shift(1)

    # ========== RANGE ANALYSIS ==========
    df['daily_range'] = ((df['high'] - df['low']) / df['close']) * 100
    df['prev_range'] = df['daily_range'].shift(1)
    df['avg_range_5d'] = df['daily_range'].rolling(5).mean().shift(1)
    df['avg_range_20d'] = df['daily_range'].rolling(20).mean().shift(1)
    df['range_expansion'] = (df['prev_range'] / df['avg_range_20d'])

    # Gap analysis
    df['gap'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) * 100

    # ========== MOVING AVERAGES ==========
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma10_vs_sma50'] = ((df['sma_10'].shift(1) - df['sma_50'].shift(1)) / df['sma_50'].shift(1)) * 100

    # ========== BOLLINGER BANDS ==========
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.001)).shift(1)

    # ========== VOLUME ==========
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['prev_volume_ratio'] = (df['volume'] / df['volume_sma_20']).shift(1)
    df['volume_price_trend'] = (df['daily_return'] * df['prev_volume_ratio']).shift(1)

    # ========== CALENDAR ==========
    df['day_of_week'] = df.index.dayofweek

    # ========== TARGET ==========
    df['target'] = (df['daily_return'] > 0).astype(int)

    return df


def train_final_model(ticker: str = 'SPY'):
    """Train final model with consensus features only"""

    print(f"\n{'='*60}")
    print(f"Training FINAL Model for {ticker}")
    print(f"Using {len(CONSENSUS_FEATURES)} consensus features")
    print('='*60)

    # Fetch data
    print("\nFetching historical data...")
    df = fetch_polygon_data(ticker, days=800)
    print(f"  Got {len(df)} days of data")

    # Calculate features
    print("Calculating consensus features...")
    df = calculate_consensus_features(df)

    # Drop NaN rows
    df_clean = df.dropna(subset=CONSENSUS_FEATURES + ['target'])
    print(f"  {len(df_clean)} samples after cleaning")

    # Time-based split: use last 60 days for testing
    test_size = 60
    train_df = df_clean.iloc[:-test_size]
    test_df = df_clean.iloc[-test_size:]

    X_train = train_df[CONSENSUS_FEATURES]
    y_train = train_df['target']
    X_test = test_df[CONSENSUS_FEATURES]
    y_test = test_df['target']

    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Bullish days (train): {y_train.mean():.1%}")
    print(f"  Bullish days (test): {y_test.mean():.1%}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train ensemble
    print("\nTraining final ensemble...")

    # XGBoost - tuned for smaller feature set
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb.fit(X_train_scaled, y_train)
    xgb_acc = xgb.score(X_test_scaled, y_test)
    print(f"  XGBoost: {xgb_acc:.1%}")

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=8,
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)
    rf_acc = rf.score(X_test_scaled, y_test)
    print(f"  Random Forest: {rf_acc:.1%}")

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=10,
        random_state=42
    )
    gb.fit(X_train_scaled, y_train)
    gb_acc = gb.score(X_test_scaled, y_test)
    print(f"  Gradient Boosting: {gb_acc:.1%}")

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_acc = lr.score(X_test_scaled, y_test)
    print(f"  Logistic Regression: {lr_acc:.1%}")

    # Ensemble weights based on performance
    total_acc = xgb_acc + rf_acc + gb_acc + lr_acc
    weights = {
        'xgb': xgb_acc / total_acc,
        'rf': rf_acc / total_acc,
        'gb': gb_acc / total_acc,
        'lr': lr_acc / total_acc
    }
    print(f"\n  Weights: XGB={weights['xgb']:.2f}, RF={weights['rf']:.2f}, GB={weights['gb']:.2f}, LR={weights['lr']:.2f}")

    # Ensemble prediction
    y_pred_proba = (
        xgb.predict_proba(X_test_scaled)[:, 1] * weights['xgb'] +
        rf.predict_proba(X_test_scaled)[:, 1] * weights['rf'] +
        gb.predict_proba(X_test_scaled)[:, 1] * weights['gb'] +
        lr.predict_proba(X_test_scaled)[:, 1] * weights['lr']
    )
    y_pred = (y_pred_proba >= 0.5).astype(int)
    ensemble_acc = (y_pred == y_test).mean()
    print(f"\n  ENSEMBLE: {ensemble_acc:.1%}")

    # High confidence analysis
    print("\n--- HIGH CONFIDENCE SIGNALS ---")
    for threshold in [0.55, 0.60, 0.65, 0.70]:
        high_conf_mask = (y_pred_proba >= threshold) | (y_pred_proba <= (1 - threshold))
        if high_conf_mask.sum() > 0:
            high_conf_acc = (y_pred[high_conf_mask] == y_test.values[high_conf_mask]).mean()
            print(f"  >= {threshold:.0%} conf: {high_conf_mask.sum()} signals, {high_conf_acc:.1%} accuracy")

    # Feature importance
    print("\nFeature importance (XGBoost):")
    importance = pd.DataFrame({
        'feature': CONSENSUS_FEATURES,
        'importance': xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Save model
    model_data = {
        'models': {
            'xgb': xgb,
            'rf': rf,
            'gb': gb,
            'lr': lr
        },
        'weights': weights,
        'scaler': scaler,
        'feature_cols': CONSENSUS_FEATURES,
        'metrics': {
            'accuracy': float(ensemble_acc),
            'xgb_accuracy': float(xgb_acc),
            'rf_accuracy': float(rf_acc),
            'gb_accuracy': float(gb_acc),
            'lr_accuracy': float(lr_acc),
            'bullish_rate_train': float(y_train.mean()),
            'bullish_rate_test': float(y_test.mean()),
        },
        'feature_importance': importance.to_dict('records'),
        'ticker': ticker,
        'version': 'final_consensus_v1',
        'trained_at': datetime.now().isoformat(),
    }

    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_daily_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved to {model_path}")

    return model_data


if __name__ == '__main__':
    print("\n" + "="*70)
    print("   TRAINING FINAL MODELS WITH CONSENSUS FEATURES")
    print("="*70)
    print(f"\nUsing {len(CONSENSUS_FEATURES)} optimized features:")
    for i, f in enumerate(CONSENSUS_FEATURES):
        marker = "(3/3)" if i < 3 else "(2/3)"
        print(f"  {i+1}. {f} {marker}")

    results = {}
    for ticker in ['SPY', 'QQQ', 'IWM']:
        try:
            model_data = train_final_model(ticker)
            results[ticker] = {
                'accuracy': model_data['metrics']['accuracy'],
                'xgb': model_data['metrics']['xgb_accuracy'],
                'rf': model_data['metrics']['rf_accuracy'],
                'gb': model_data['metrics']['gb_accuracy'],
                'lr': model_data['metrics']['lr_accuracy'],
            }
        except Exception as e:
            print(f"Error training {ticker}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("   TRAINING COMPLETE - FINAL RESULTS")
    print("="*70)
    print(f"\n{'Ticker':<8} {'Ensemble':<12} {'XGB':<10} {'RF':<10} {'GB':<10} {'LR':<10}")
    print("-" * 60)
    for ticker, metrics in results.items():
        print(f"{ticker:<8} {metrics['accuracy']:.1%}{'':<7} {metrics['xgb']:.1%}{'':<5} {metrics['rf']:.1%}{'':<5} {metrics['gb']:.1%}{'':<5} {metrics['lr']:.1%}")
