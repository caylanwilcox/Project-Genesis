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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
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
    # IMPORTANT: Keep units consistent with `ml/predict_server.py::calculate_daily_features`,
    # where `gap` is a DECIMAL return (e.g. 0.0025 for +0.25%), not percent.
    df['gap'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1))

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

    # Time-based holdout: use last 60 days for final testing
    # (DO NOT use this holdout to tune weights/thresholds; tune on earlier data via time-series CV)
    test_size = 60
    if len(df_clean) <= test_size + 120:
        raise ValueError(f"Insufficient samples for robust training+holdout: {len(df_clean)} rows")

    train_df = df_clean.iloc[:-test_size].copy()
    test_df = df_clean.iloc[-test_size:].copy()

    X_train = train_df[CONSENSUS_FEATURES].copy()
    y_train = train_df['target'].copy()
    X_test = test_df[CONSENSUS_FEATURES].copy()
    y_test = test_df['target'].copy()

    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Bullish days (train): {y_train.mean():.1%}")
    print(f"  Bullish days (test): {y_test.mean():.1%}")

    print("\nTraining final ensemble (leak-safe tuning)...")

    # Define base estimators (hyperparameters kept modest to reduce overfitting on ~800 samples)
    xgb_est = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=1.2,
        min_child_weight=3,
        random_state=42,
        eval_metric='logloss',
    )

    rf_est = RandomForestClassifier(
        n_estimators=400,
        max_depth=7,
        min_samples_leaf=10,
        min_samples_split=30,
        random_state=42,
        n_jobs=-1,
    )

    gb_est = GradientBoostingClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.03,
        min_samples_leaf=12,
        random_state=42,
    )

    lr_est = LogisticRegression(max_iter=2000, C=0.35, random_state=42)

    # Use a single, consistent scaler for all models via pipeline.
    # (Scaling doesn't materially change tree splits but does help LR and keeps the interface consistent.)
    pipelines = {
        'xgb': Pipeline([('scaler', StandardScaler()), ('model', xgb_est)]),
        'rf': Pipeline([('scaler', StandardScaler()), ('model', rf_est)]),
        'gb': Pipeline([('scaler', StandardScaler()), ('model', gb_est)]),
        'lr': Pipeline([('scaler', StandardScaler()), ('model', lr_est)]),
    }

    # --- 1) Tune weights + decision threshold on TRAIN ONLY using out-of-fold (OOF) predictions ---
    # Note: TimeSeriesSplit preserves temporal order and prevents lookahead.
    tscv = TimeSeriesSplit(n_splits=6)
    oof_proba = {}
    oof_acc = {}

    print("\nCross-validating (time-series) to tune weights/threshold...")
    # NOTE: sklearn's `cross_val_predict` does not support TimeSeriesSplit because it does not
    # produce a full partition of the dataset (early samples are never in any test fold).
    # We implement a safe manual OOF prediction loop instead.
    for name, pipe in pipelines.items():
        p = np.full(shape=(len(X_train),), fill_value=np.nan, dtype=float)
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            pipe.fit(X_tr, y_tr)
            p[val_idx] = pipe.predict_proba(X_val)[:, 1]

        valid = ~np.isnan(p)
        if valid.sum() < 50:
            raise ValueError(f"OOF coverage too small for {name}: {valid.sum()} samples")

        oof_proba[name] = p
        oof_pred = (p[valid] >= 0.5).astype(int)
        oof_acc[name] = float((oof_pred == y_train.values[valid]).mean())
        print(f"  {name.upper():>3} OOF acc: {oof_acc[name]:.1%} (n={int(valid.sum())})")

    # Weight models by OOF accuracy (normalize to sum to 1)
    total = sum(max(v, 1e-6) for v in oof_acc.values())
    weights = {k: float(max(v, 1e-6) / total) for k, v in oof_acc.items()}
    print(f"\n  Tuned weights (OOF): XGB={weights['xgb']:.2f}, RF={weights['rf']:.2f}, GB={weights['gb']:.2f}, LR={weights['lr']:.2f}")

    # Tune threshold on OOF ensemble predictions to maximize accuracy (on TRAIN only)
    # Use only rows that have OOF predictions for all models.
    valid_all = (
        ~np.isnan(oof_proba['xgb']) &
        ~np.isnan(oof_proba['rf']) &
        ~np.isnan(oof_proba['gb']) &
        ~np.isnan(oof_proba['lr'])
    )
    oof_ensemble = (
        oof_proba['xgb'][valid_all] * weights['xgb'] +
        oof_proba['rf'][valid_all] * weights['rf'] +
        oof_proba['gb'][valid_all] * weights['gb'] +
        oof_proba['lr'][valid_all] * weights['lr']
    )

    best_threshold = 0.5
    best_oof_acc = -1.0
    for thr in np.round(np.arange(0.40, 0.61, 0.01), 2):
        acc = float(((oof_ensemble >= thr).astype(int) == y_train.values[valid_all]).mean())
        if acc > best_oof_acc:
            best_oof_acc = acc
            best_threshold = float(thr)
    print(f"  Tuned threshold (OOF): {best_threshold:.2f} (OOF acc: {best_oof_acc:.1%})")

    # --- 2) Fit final models on ALL train data, evaluate ONCE on the holdout ---
    # IMPORTANT: Keep the saved artifact compatible with the serving layer:
    # - `model_data['scaler']` transforms features
    # - `model_data['models'][name]` expects SCALED features as input
    print("\nFitting final models on full training set...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit base estimators on scaled features (fresh instances to avoid any CV-fitted state bleed)
    xgb = XGBClassifier(**xgb_est.get_params())
    rf = RandomForestClassifier(**rf_est.get_params())
    gb = GradientBoostingClassifier(**gb_est.get_params())
    lr = LogisticRegression(**lr_est.get_params())

    xgb.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)
    gb.fit(X_train_scaled, y_train)
    lr.fit(X_train_scaled, y_train)

    # Holdout probabilities (unseen)
    test_proba = {
        'xgb': xgb.predict_proba(X_test_scaled)[:, 1],
        'rf': rf.predict_proba(X_test_scaled)[:, 1],
        'gb': gb.predict_proba(X_test_scaled)[:, 1],
        'lr': lr.predict_proba(X_test_scaled)[:, 1],
    }

    # Individual holdout accuracies (informational only)
    xgb_acc = float(((test_proba['xgb'] >= 0.5).astype(int) == y_test.values).mean())
    rf_acc = float(((test_proba['rf'] >= 0.5).astype(int) == y_test.values).mean())
    gb_acc = float(((test_proba['gb'] >= 0.5).astype(int) == y_test.values).mean())
    lr_acc = float(((test_proba['lr'] >= 0.5).astype(int) == y_test.values).mean())
    print(f"  XGBoost (holdout @0.50): {xgb_acc:.1%}")
    print(f"  Random Forest (holdout @0.50): {rf_acc:.1%}")
    print(f"  Gradient Boosting (holdout @0.50): {gb_acc:.1%}")
    print(f"  Logistic Regression (holdout @0.50): {lr_acc:.1%}")

    y_pred_proba = (
        test_proba['xgb'] * weights['xgb'] +
        test_proba['rf'] * weights['rf'] +
        test_proba['gb'] * weights['gb'] +
        test_proba['lr'] * weights['lr']
    )
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    ensemble_acc = float((y_pred == y_test.values).mean())
    print(f"\n  ENSEMBLE (holdout @thr={best_threshold:.2f}): {ensemble_acc:.1%}")

    # High confidence analysis
    print("\n--- HIGH CONFIDENCE SIGNALS (holdout) ---")
    for threshold in [0.55, 0.60, 0.65, 0.70]:
        high_conf_mask = (y_pred_proba >= threshold) | (y_pred_proba <= (1 - threshold))
        if high_conf_mask.sum() > 0:
            high_conf_acc = float((y_pred[high_conf_mask] == y_test.values[high_conf_mask]).mean())
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
            'lr': lr,
        },
        'weights': weights,
        'scaler': scaler,
        'feature_cols': CONSENSUS_FEATURES,
        'decision_threshold': best_threshold,
        'metrics': {
            'accuracy': float(ensemble_acc),
            'xgb_accuracy': float(xgb_acc),
            'rf_accuracy': float(rf_acc),
            'gb_accuracy': float(gb_acc),
            'lr_accuracy': float(lr_acc),
            'bullish_rate_train': float(y_train.mean()),
            'bullish_rate_test': float(y_test.mean()),
            'oof_accuracy': float(best_oof_acc),
        },
        'feature_importance': importance.to_dict('records'),
        'ticker': ticker,
        'version': 'final_consensus_v2_cv',
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
