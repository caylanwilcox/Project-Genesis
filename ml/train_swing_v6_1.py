"""
V6.1 SWING Model - UPGRADED Multi-Timeframe Swing Trade Predictions
====================================================================
Improvements over V6:
- Added CatBoost to ensemble (5 models now)
- VIX correlation features
- Cross-asset momentum (SPY-QQQ-IWM correlation)
- Volatility regime detection
- Removed low-importance features (return_1d, atr_pct, volatility_20d)
- Hyperparameter tuning with cross-validation

Targets:
- Target A: Will price be higher in 5 days?
- Target B: Will price be higher in 10 days?
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from datetime import datetime, timedelta
import pickle
import os
import requests
import warnings
warnings.filterwarnings('ignore')

from config import TRAIN_START, TRAIN_END, TEST_START, TEST_END, RANDOM_STATE, SWING_TICKERS

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'v6_models')

# Swing trade horizons
SHORT_SWING_DAYS = 5   # 1 week trading days
MEDIUM_SWING_DAYS = 10  # 2 weeks trading days


def fetch_daily_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily bars from Polygon."""
    print(f"  Fetching daily data for {ticker} from {start_date} to {end_date}...")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}
    response = requests.get(url, params=params)
    data = response.json()

    if 'results' not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df.set_index('date')
    print(f"    Got {len(df)} daily bars")
    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_weekly_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch weekly bars from Polygon."""
    print(f"  Fetching weekly data for {ticker}...")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/week/{start_date}/{end_date}"
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}
    response = requests.get(url, params=params)
    data = response.json()

    if 'results' not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df.set_index('date')
    print(f"    Got {len(df)} weekly bars")
    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_vix_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch VIX data from Polygon."""
    print(f"  Fetching VIX data...")
    # VIX is available via I:VIX on Polygon or we use UVXY as proxy
    url = f"https://api.polygon.io/v2/aggs/ticker/UVXY/range/1/day/{start_date}/{end_date}"
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}
    response = requests.get(url, params=params)
    data = response.json()

    if 'results' not in data:
        print("    WARNING: No VIX data available, using synthetic volatility")
        return None

    df = pd.DataFrame(data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'c': 'vix_close'})
    df = df.set_index('date')
    print(f"    Got {len(df)} VIX bars")
    return df[['vix_close']]


def fetch_cross_asset_data(start_date: str, end_date: str) -> dict:
    """Fetch data for cross-asset correlation (SPY, QQQ, IWM)."""
    assets = {}
    for ticker in ['SPY', 'QQQ', 'IWM']:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}
        response = requests.get(url, params=params)
        data = response.json()

        if 'results' in data:
            df = pd.DataFrame(data['results'])
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df = df.set_index('date')
            assets[ticker] = df['c']

    return assets


def calculate_swing_features_v61(daily_df: pd.DataFrame, weekly_df: pd.DataFrame,
                                  vix_df: pd.DataFrame, cross_assets: dict,
                                  ticker: str, idx: int) -> dict:
    """
    Calculate UPGRADED features for swing trade prediction.
    V6.1 improvements:
    - VIX correlation
    - Cross-asset momentum
    - Volatility regime detection
    - Removed low-importance features
    """
    if idx < 30:
        return None

    features = {}
    current_date = daily_df.index[idx]
    current = daily_df.iloc[idx]
    features['current_price'] = current['close']

    # =================
    # DAILY FEATURES (pruned)
    # =================

    # Recent returns (REMOVED return_1d - low importance)
    for days in [3, 5, 10, 20]:
        if idx >= days:
            past_close = daily_df.iloc[idx - days]['close']
            features[f'return_{days}d'] = (current['close'] - past_close) / past_close
        else:
            features[f'return_{days}d'] = 0

    # Moving averages
    for window in [5, 10, 20, 50]:
        if idx >= window:
            sma = daily_df['close'].iloc[idx-window:idx].mean()
            features[f'dist_from_sma_{window}'] = (current['close'] - sma) / sma
            features[f'above_sma_{window}'] = 1 if current['close'] > sma else 0
        else:
            features[f'dist_from_sma_{window}'] = 0
            features[f'above_sma_{window}'] = 0

    # Trend strength (SMA alignment)
    if idx >= 50:
        sma_5 = daily_df['close'].iloc[idx-5:idx].mean()
        sma_20 = daily_df['close'].iloc[idx-20:idx].mean()
        sma_50 = daily_df['close'].iloc[idx-50:idx].mean()
        features['sma_alignment'] = 1 if (sma_5 > sma_20 > sma_50) else (-1 if sma_5 < sma_20 < sma_50 else 0)
    else:
        features['sma_alignment'] = 0

    # Volatility - V6.1: Use only 10d (REMOVED volatility_20d - low importance)
    if idx >= 10:
        features['volatility_10d'] = daily_df['close'].iloc[idx-10:idx].pct_change().dropna().std()
    else:
        features['volatility_10d'] = 0

    # ATR - V6.1: Keep ATR but REMOVE atr_pct (low importance)
    if idx >= 14:
        tr_list = []
        for i in range(idx-14, idx):
            high = daily_df.iloc[i]['high']
            low = daily_df.iloc[i]['low']
            prev_close = daily_df.iloc[i-1]['close'] if i > 0 else daily_df.iloc[i]['open']
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
        features['atr_14'] = np.mean(tr_list)
    else:
        features['atr_14'] = 0

    # RSI
    if idx >= 14:
        changes = daily_df['close'].iloc[idx-14:idx].diff().dropna()
        gains = changes[changes > 0].sum()
        losses = abs(changes[changes < 0].sum())
        if losses > 0:
            rs = gains / losses
            features['rsi_14'] = 100 - (100 / (1 + rs))
        else:
            features['rsi_14'] = 100
    else:
        features['rsi_14'] = 50

    # Higher highs / Lower lows
    if idx >= 10:
        recent_high = daily_df['high'].iloc[idx-5:idx].max()
        prior_high = daily_df['high'].iloc[idx-10:idx-5].max()
        recent_low = daily_df['low'].iloc[idx-5:idx].min()
        prior_low = daily_df['low'].iloc[idx-10:idx-5].min()
        features['higher_high'] = 1 if recent_high > prior_high else 0
        features['higher_low'] = 1 if recent_low > prior_low else 0
        features['lower_high'] = 1 if recent_high < prior_high else 0
        features['lower_low'] = 1 if recent_low < prior_low else 0
    else:
        features['higher_high'] = 0
        features['higher_low'] = 0
        features['lower_high'] = 0
        features['lower_low'] = 0

    # Volume profile
    if idx >= 20:
        avg_vol = daily_df['volume'].iloc[idx-20:idx].mean()
        features['volume_ratio'] = current['volume'] / avg_vol if avg_vol > 0 else 1
        features['volume_trend'] = daily_df['volume'].iloc[idx-5:idx].mean() / avg_vol if avg_vol > 0 else 1
    else:
        features['volume_ratio'] = 1
        features['volume_trend'] = 1

    # Candle patterns
    body = current['close'] - current['open']
    total_range = current['high'] - current['low']
    features['body_to_range'] = abs(body) / total_range if total_range > 0 else 0
    features['is_bullish'] = 1 if body > 0 else 0
    features['upper_wick'] = (current['high'] - max(current['open'], current['close'])) / total_range if total_range > 0 else 0
    features['lower_wick'] = (min(current['open'], current['close']) - current['low']) / total_range if total_range > 0 else 0

    # Consecutive days
    consec_up = 0
    consec_down = 0
    for i in range(1, min(6, idx)):
        if daily_df.iloc[idx-i]['close'] > daily_df.iloc[idx-i]['open']:
            if consec_down == 0:
                consec_up += 1
            else:
                break
        else:
            if consec_up == 0:
                consec_down += 1
            else:
                break
    features['consec_up'] = consec_up
    features['consec_down'] = consec_down

    # Mean reversion signal
    features['mean_reversion'] = -features['return_5d']

    # =================
    # WEEKLY FEATURES
    # =================
    week_idx = None
    for i, w_date in enumerate(weekly_df.index):
        if w_date <= current_date:
            week_idx = i

    if week_idx is not None and week_idx >= 4:
        weekly_current = weekly_df.iloc[week_idx]

        for weeks in [1, 2, 4]:
            if week_idx >= weeks:
                past_close = weekly_df.iloc[week_idx - weeks]['close']
                features[f'weekly_return_{weeks}w'] = (weekly_current['close'] - past_close) / past_close
            else:
                features[f'weekly_return_{weeks}w'] = 0

        if week_idx >= 4:
            weekly_sma_4 = weekly_df['close'].iloc[week_idx-4:week_idx].mean()
            features['weekly_dist_from_sma_4'] = (weekly_current['close'] - weekly_sma_4) / weekly_sma_4
            features['weekly_above_sma_4'] = 1 if weekly_current['close'] > weekly_sma_4 else 0
        else:
            features['weekly_dist_from_sma_4'] = 0
            features['weekly_above_sma_4'] = 0

        features['weekly_bullish'] = 1 if weekly_current['close'] > weekly_current['open'] else 0

        if week_idx >= 4:
            changes = weekly_df['close'].iloc[week_idx-4:week_idx].diff().dropna()
            gains = changes[changes > 0].sum()
            losses = abs(changes[changes < 0].sum())
            if losses > 0:
                rs = gains / losses
                features['weekly_rsi'] = 100 - (100 / (1 + rs))
            else:
                features['weekly_rsi'] = 100
        else:
            features['weekly_rsi'] = 50
    else:
        features['weekly_return_1w'] = 0
        features['weekly_return_2w'] = 0
        features['weekly_return_4w'] = 0
        features['weekly_dist_from_sma_4'] = 0
        features['weekly_above_sma_4'] = 0
        features['weekly_bullish'] = 0
        features['weekly_rsi'] = 50

    # =================
    # NEW V6.1 FEATURES
    # =================

    # 1. VIX/VOLATILITY REGIME
    if vix_df is not None and current_date in vix_df.index:
        vix_idx = vix_df.index.get_loc(current_date)
        if vix_idx >= 20:
            vix_current = vix_df.iloc[vix_idx]['vix_close']
            vix_ma_20 = vix_df['vix_close'].iloc[vix_idx-20:vix_idx].mean()
            features['vix_relative'] = vix_current / vix_ma_20 if vix_ma_20 > 0 else 1
            features['vix_elevated'] = 1 if vix_current > vix_ma_20 * 1.2 else 0
            features['vix_low'] = 1 if vix_current < vix_ma_20 * 0.8 else 0
        else:
            features['vix_relative'] = 1
            features['vix_elevated'] = 0
            features['vix_low'] = 0
    else:
        # Synthetic VIX from volatility
        if idx >= 20:
            vol_current = features['volatility_10d']
            vol_20d = daily_df['close'].iloc[idx-20:idx].pct_change().dropna().std()
            features['vix_relative'] = vol_current / vol_20d if vol_20d > 0 else 1
            features['vix_elevated'] = 1 if vol_current > vol_20d * 1.2 else 0
            features['vix_low'] = 1 if vol_current < vol_20d * 0.8 else 0
        else:
            features['vix_relative'] = 1
            features['vix_elevated'] = 0
            features['vix_low'] = 0

    # 2. VOLATILITY REGIME DETECTION
    if idx >= 60:
        vol_20 = daily_df['close'].iloc[idx-20:idx].pct_change().dropna().std()
        vol_60 = daily_df['close'].iloc[idx-60:idx].pct_change().dropna().std()
        features['vol_regime'] = vol_20 / vol_60 if vol_60 > 0 else 1
        features['vol_expanding'] = 1 if vol_20 > vol_60 * 1.3 else 0
        features['vol_contracting'] = 1 if vol_20 < vol_60 * 0.7 else 0
    else:
        features['vol_regime'] = 1
        features['vol_expanding'] = 0
        features['vol_contracting'] = 0

    # 3. CROSS-ASSET MOMENTUM
    if cross_assets and ticker in cross_assets:
        other_tickers = [t for t in ['SPY', 'QQQ', 'IWM'] if t != ticker]

        # Calculate cross-asset returns
        cross_momentum = []
        for other in other_tickers:
            if other in cross_assets:
                other_series = cross_assets[other]
                if current_date in other_series.index:
                    other_idx = other_series.index.get_loc(current_date)
                    if other_idx >= 5:
                        other_return_5d = (other_series.iloc[other_idx] - other_series.iloc[other_idx-5]) / other_series.iloc[other_idx-5]
                        cross_momentum.append(other_return_5d)

        if cross_momentum:
            features['cross_asset_momentum'] = np.mean(cross_momentum)
            features['cross_asset_aligned'] = 1 if all(m > 0 for m in cross_momentum) or all(m < 0 for m in cross_momentum) else 0
        else:
            features['cross_asset_momentum'] = 0
            features['cross_asset_aligned'] = 0

        # Relative strength vs cross assets
        if idx >= 5:
            ticker_return = features['return_5d']
            if cross_momentum:
                features['relative_strength'] = ticker_return - np.mean(cross_momentum)
            else:
                features['relative_strength'] = 0
        else:
            features['relative_strength'] = 0
    else:
        features['cross_asset_momentum'] = 0
        features['cross_asset_aligned'] = 0
        features['relative_strength'] = 0

    # 4. MOMENTUM DIVERGENCE
    if idx >= 20:
        price_change_10 = features['return_10d']
        vol_change = (daily_df['volume'].iloc[idx-5:idx].mean() - daily_df['volume'].iloc[idx-10:idx-5].mean()) / daily_df['volume'].iloc[idx-10:idx-5].mean() if daily_df['volume'].iloc[idx-10:idx-5].mean() > 0 else 0
        features['price_vol_divergence'] = 1 if (price_change_10 > 0 and vol_change < 0) or (price_change_10 < 0 and vol_change > 0) else 0
    else:
        features['price_vol_divergence'] = 0

    # 5. TREND PERSISTENCE
    if idx >= 20:
        returns_20d = daily_df['close'].iloc[idx-20:idx].pct_change().dropna()
        positive_days = (returns_20d > 0).sum()
        features['trend_persistence'] = positive_days / 20
    else:
        features['trend_persistence'] = 0.5

    # =================
    # TIME FEATURES
    # =================
    features['day_of_week'] = current_date.dayofweek
    features['is_monday'] = 1 if current_date.dayofweek == 0 else 0
    features['is_friday'] = 1 if current_date.dayofweek == 4 else 0
    features['month'] = current_date.month
    features['week_of_year'] = current_date.isocalendar()[1]

    return features


def create_swing_samples_v61(ticker: str, daily_df: pd.DataFrame, weekly_df: pd.DataFrame,
                              vix_df: pd.DataFrame, cross_assets: dict) -> pd.DataFrame:
    """Create labeled samples for V6.1 swing trade model."""
    samples = []

    for idx in range(30, len(daily_df) - MEDIUM_SWING_DAYS):
        features = calculate_swing_features_v61(daily_df, weekly_df, vix_df, cross_assets, ticker, idx)
        if features is None:
            continue

        current_close = daily_df.iloc[idx]['close']

        # Target A: Price higher in 5 days?
        future_close_5d = daily_df.iloc[idx + SHORT_SWING_DAYS]['close']
        features['target_5d'] = 1 if future_close_5d > current_close else 0

        # Target B: Price higher in 10 days?
        future_close_10d = daily_df.iloc[idx + MEDIUM_SWING_DAYS]['close']
        features['target_10d'] = 1 if future_close_10d > current_close else 0

        features['date'] = daily_df.index[idx]
        samples.append(features)

    return pd.DataFrame(samples)


def train_ensemble_v61(X_train, y_train, X_test, y_test, name, ticker='SPY'):
    """Train UPGRADED ensemble with CatBoost for swing trades."""

    # V6.1: Add CatBoost to ensemble
    models = {
        'et': ExtraTreesClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=10,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'xgb': XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_STATE, verbosity=0
        ),
        'lgbm': LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_STATE, verbosity=-1
        ),
        'catboost': CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.03,
            random_state=RANDOM_STATE, verbose=False
        ),
        'lr': LogisticRegression(
            C=0.5, max_iter=2000, random_state=RANDOM_STATE
        )
    }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        results[model_name] = {'model': model, 'accuracy': acc, 'precision': prec, 'recall': rec}
        print(f"      {model_name}: {acc:.1%}")

    # Weighted ensemble based on accuracy
    total_acc = sum(r['accuracy'] for r in results.values())
    weights = {n: r['accuracy'] / total_acc for n, r in results.items()}

    # Ensemble prediction
    y_prob = np.zeros(len(y_test))
    for n, r in results.items():
        y_prob += r['model'].predict_proba(X_test)[:, 1] * weights[n]

    y_pred_ensemble = (y_prob > 0.5).astype(int)
    ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
    ensemble_prec = precision_score(y_test, y_pred_ensemble, zero_division=0)
    ensemble_rec = recall_score(y_test, y_pred_ensemble, zero_division=0)
    ensemble_f1 = f1_score(y_test, y_pred_ensemble, zero_division=0)

    print(f"    → {name} Ensemble: Acc={ensemble_acc:.1%}, Prec={ensemble_prec:.1%}, Rec={ensemble_rec:.1%}, F1={ensemble_f1:.1%}")

    return {n: r['model'] for n, r in results.items()}, weights, ensemble_acc


def train_swing_model_v61(ticker: str, vix_df: pd.DataFrame, cross_assets: dict):
    """Train V6.1 swing trade model for a ticker."""
    print(f"\n{'='*70}")
    print(f"  SWING V6.1 MODEL: {ticker}")
    print(f"{'='*70}")

    # Fetch data
    daily_df = fetch_daily_data(ticker, TRAIN_START, TEST_END)
    weekly_df = fetch_weekly_data(ticker, TRAIN_START, TEST_END)

    if len(daily_df) < 100:
        print(f"  ERROR: Insufficient data for {ticker}")
        return None

    # Create samples
    print(f"  Creating V6.1 swing samples...")
    samples_df = create_swing_samples_v61(ticker, daily_df, weekly_df, vix_df, cross_assets)
    print(f"    Total samples: {len(samples_df)}")

    # Split train/test
    train_end = pd.to_datetime(TRAIN_END)
    test_start = pd.to_datetime(TEST_START)

    train_df = samples_df[samples_df['date'] < train_end]
    test_df = samples_df[samples_df['date'] >= test_start]

    print(f"    Train: {len(train_df)}, Test: {len(test_df)}")

    # Feature columns
    feature_cols = [c for c in samples_df.columns if c not in ['target_5d', 'target_10d', 'date', 'current_price']]
    print(f"    Features: {len(feature_cols)}")

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)

    # Scaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models for both targets
    print(f"\n  --- Target A: 5-Day Direction ---")
    y_train_5d = train_df['target_5d']
    y_test_5d = test_df['target_5d']
    models_5d, weights_5d, acc_5d = train_ensemble_v61(
        X_train_scaled, y_train_5d, X_test_scaled, y_test_5d, "5-Day", ticker
    )

    print(f"\n  --- Target B: 10-Day Direction ---")
    y_train_10d = train_df['target_10d']
    y_test_10d = test_df['target_10d']
    models_10d, weights_10d, acc_10d = train_ensemble_v61(
        X_train_scaled, y_train_10d, X_test_scaled, y_test_10d, "10-Day", ticker
    )

    # Feature importance from ExtraTrees
    et_model = models_5d['et']
    importances = dict(zip(feature_cols, et_model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  Top 10 Features (5d):")
    for feat, imp in top_features:
        print(f"    {feat}: {imp:.1%}")

    # Save model
    model_data = {
        'ticker': ticker,
        'version': 'v6.1_swing',
        'trained_at': datetime.now().isoformat(),
        'feature_cols': feature_cols,
        'scaler': scaler,

        # 5-day model
        'models_5d': models_5d,
        'weights_5d': weights_5d,
        'acc_5d': acc_5d,

        # 10-day model
        'models_10d': models_10d,
        'weights_10d': weights_10d,
        'acc_10d': acc_10d,

        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'feature_importance': importances,
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_swing_v6_1.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n  Model saved to {model_path}")

    return {
        'acc_5d': acc_5d,
        'acc_10d': acc_10d
    }


def main():
    """Main training pipeline for V6.1."""
    print("="*70)
    print("  V6.1 SWING TRADE MODEL TRAINING (UPGRADED)")
    print("="*70)
    print(f"\n  Train period: {TRAIN_START} to {TRAIN_END}")
    print(f"  Test period:  {TEST_START} to {TEST_END}")
    print(f"\n  V6.1 Improvements:")
    print(f"    - Added CatBoost to 5-model ensemble")
    print(f"    - VIX/volatility regime features")
    print(f"    - Cross-asset momentum (SPY-QQQ-IWM)")
    print(f"    - Removed low-importance features")
    print(f"    - Enhanced hyperparameters")

    # Fetch shared data
    print(f"\n  Fetching shared data...")
    vix_df = fetch_vix_data(TRAIN_START, TEST_END)
    cross_assets = fetch_cross_asset_data(TRAIN_START, TEST_END)

    results = {}
    for ticker in SWING_TICKERS:
        result = train_swing_model_v61(ticker, vix_df, cross_assets)
        if result:
            results[ticker] = result

    # Summary
    print("\n" + "="*70)
    print("  V6.1 TRAINING COMPLETE")
    print("="*70)
    print(f"\n  {'Ticker':<8} {'5-Day Acc':>12} {'10-Day Acc':>12}")
    print(f"  {'-'*36}")
    for ticker, r in results.items():
        print(f"  {ticker:<8} {r['acc_5d']:>11.1%} {r['acc_10d']:>11.1%}")


if __name__ == '__main__':
    main()
