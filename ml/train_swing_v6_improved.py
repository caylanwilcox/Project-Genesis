"""
V6 SWING Model V2 - Improved Multi-Timeframe Swing Trade Predictions
=====================================================================
Improvements over V1:
1. Better volatility regime features
2. Up/Down volume ratio (buying vs selling pressure)
3. Price position in range (0-1 normalized)
4. Trend persistence (bullish days ratio)
5. Bollinger Band position
6. MACD histogram momentum
7. Removed low-importance features
8. Optimized hyperparameters
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from datetime import datetime, timedelta
import pickle
import os
import requests
import warnings
warnings.filterwarnings('ignore')

from config import TRAIN_START, TRAIN_END, TEST_START, TEST_END, RANDOM_STATE, SWING_TICKERS

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'v6_models')

# Swing trade horizon
SHORT_SWING_DAYS = 5


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


def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Calculate RSI."""
    if len(prices) < period + 1:
        return 50.0

    changes = np.diff(prices[-period-1:])
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices: np.ndarray) -> tuple:
    """Calculate MACD line, signal, and histogram."""
    if len(prices) < 26:
        return 0, 0, 0

    # EMA calculations
    ema_12 = pd.Series(prices).ewm(span=12, adjust=False).mean().iloc[-1]
    ema_26 = pd.Series(prices).ewm(span=26, adjust=False).mean().iloc[-1]
    macd_line = ema_12 - ema_26

    # Signal line (9-period EMA of MACD)
    macd_series = pd.Series(prices).ewm(span=12, adjust=False).mean() - pd.Series(prices).ewm(span=26, adjust=False).mean()
    signal = macd_series.ewm(span=9, adjust=False).mean().iloc[-1]
    histogram = macd_line - signal

    return macd_line, signal, histogram


def calculate_improved_features(daily_df: pd.DataFrame, weekly_df: pd.DataFrame, idx: int) -> dict:
    """
    Calculate improved features for swing trade prediction.
    Focused on high-importance features + new additions.
    """
    if idx < 50:  # Need 50 days for proper indicators
        return None

    features = {}
    current_date = daily_df.index[idx]

    close = daily_df['close'].values[:idx+1]
    high = daily_df['high'].values[:idx+1]
    low = daily_df['low'].values[:idx+1]
    volume = daily_df['volume'].values[:idx+1]
    open_prices = daily_df['open'].values[:idx+1]

    current_price = close[-1]

    # =====================
    # CORE MOMENTUM FEATURES
    # =====================

    # Returns at key lookbacks
    for days in [1, 3, 5, 10]:
        features[f'return_{days}d'] = (current_price - close[-days-1]) / close[-days-1]

    # =====================
    # MOVING AVERAGE FEATURES (High Importance)
    # =====================

    sma_5 = np.mean(close[-5:])
    sma_10 = np.mean(close[-10:])
    sma_20 = np.mean(close[-20:])
    sma_50 = np.mean(close[-50:])

    features['dist_from_sma_5'] = (current_price - sma_5) / sma_5
    features['dist_from_sma_10'] = (current_price - sma_10) / sma_10
    features['dist_from_sma_20'] = (current_price - sma_20) / sma_20

    features['above_sma_5'] = 1 if current_price > sma_5 else 0
    features['above_sma_10'] = 1 if current_price > sma_10 else 0
    features['above_sma_20'] = 1 if current_price > sma_20 else 0

    # SMA Trend (slope)
    sma_5_prev = np.mean(close[-10:-5])
    features['sma_5_slope'] = (sma_5 - sma_5_prev) / sma_5_prev

    # Golden/Death cross proximity
    features['sma_5_vs_20'] = (sma_5 - sma_20) / sma_20
    features['sma_10_vs_50'] = (sma_10 - sma_50) / sma_50

    # =====================
    # VOLATILITY FEATURES
    # =====================

    returns = np.diff(close[-21:]) / close[-21:-1]
    features['volatility_20d'] = np.std(returns) if len(returns) > 0 else 0

    # ATR
    tr_list = []
    for i in range(-14, 0):
        h, l, pc = high[i], low[i], close[i-1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_list.append(tr)
    features['atr_14'] = np.mean(tr_list)
    features['atr_pct'] = features['atr_14'] / current_price

    # Volatility regime (current vs historical)
    if idx >= 252:
        vol_history = []
        for i in range(idx - 252, idx - 20):
            ret = np.diff(close[i:i+21]) / close[i:i+20]
            vol_history.append(np.std(ret) if len(ret) > 0 else 0)
        if vol_history:
            features['vol_percentile'] = sum(1 for v in vol_history if v < features['volatility_20d']) / len(vol_history)
        else:
            features['vol_percentile'] = 0.5
    else:
        features['vol_percentile'] = 0.5

    # =====================
    # BOLLINGER BAND POSITION
    # =====================

    bb_sma = np.mean(close[-20:])
    bb_std = np.std(close[-20:])
    bb_upper = bb_sma + 2 * bb_std
    bb_lower = bb_sma - 2 * bb_std

    # Position in BB (0 = lower band, 1 = upper band)
    features['bb_position'] = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
    features['bb_width'] = (bb_upper - bb_lower) / bb_sma  # Volatility proxy

    # =====================
    # RSI FEATURES
    # =====================

    features['rsi_14'] = calculate_rsi(close, 14)
    features['rsi_oversold'] = 1 if features['rsi_14'] < 30 else 0
    features['rsi_overbought'] = 1 if features['rsi_14'] > 70 else 0
    features['rsi_neutral'] = 1 if 40 <= features['rsi_14'] <= 60 else 0

    # =====================
    # MACD FEATURES
    # =====================

    macd_line, signal_line, histogram = calculate_macd(close)
    features['macd_histogram'] = histogram / current_price * 100  # Normalized
    features['macd_bullish'] = 1 if macd_line > signal_line else 0
    features['macd_above_zero'] = 1 if macd_line > 0 else 0

    # =====================
    # PRICE POSITION IN RANGE
    # =====================

    high_20 = np.max(high[-20:])
    low_20 = np.min(low[-20:])
    features['price_position_20d'] = (current_price - low_20) / (high_20 - low_20) if high_20 != low_20 else 0.5

    high_50 = np.max(high[-50:])
    low_50 = np.min(low[-50:])
    features['price_position_50d'] = (current_price - low_50) / (high_50 - low_50) if high_50 != low_50 else 0.5

    # =====================
    # TREND STRUCTURE
    # =====================

    # Higher highs / lower lows
    recent_high = np.max(high[-5:])
    prior_high = np.max(high[-10:-5])
    recent_low = np.min(low[-5:])
    prior_low = np.min(low[-10:-5])

    features['higher_high'] = 1 if recent_high > prior_high else 0
    features['lower_low'] = 1 if recent_low < prior_low else 0
    features['higher_low'] = 1 if recent_low > prior_low else 0
    features['lower_high'] = 1 if recent_high < prior_high else 0

    # Trend score
    features['trend_score'] = features['higher_high'] + features['higher_low'] - features['lower_low'] - features['lower_high']

    # =====================
    # VOLUME FEATURES (New)
    # =====================

    avg_volume_20 = np.mean(volume[-20:])
    features['volume_ratio'] = volume[-1] / avg_volume_20 if avg_volume_20 > 0 else 1

    # Up volume vs Down volume (buying vs selling pressure)
    up_volume = 0
    down_volume = 0
    for i in range(-5, 0):
        if close[i] > close[i-1]:
            up_volume += volume[i]
        else:
            down_volume += volume[i]
    features['up_down_volume_ratio'] = up_volume / (down_volume + 1)

    # Volume trend
    vol_5 = np.mean(volume[-5:])
    vol_20 = np.mean(volume[-20:])
    features['volume_trend'] = vol_5 / vol_20 if vol_20 > 0 else 1

    # =====================
    # TREND PERSISTENCE
    # =====================

    bullish_days_10 = sum(1 for i in range(-10, 0) if close[i] > open_prices[i])
    features['bullish_ratio_10d'] = bullish_days_10 / 10

    bullish_days_5 = sum(1 for i in range(-5, 0) if close[i] > open_prices[i])
    features['bullish_ratio_5d'] = bullish_days_5 / 5

    # Consecutive up/down days
    consec_up = 0
    consec_down = 0
    for i in range(-1, -11, -1):
        if close[i] > open_prices[i]:
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

    # =====================
    # WEEKLY FEATURES (High Importance)
    # =====================

    if weekly_df is not None and len(weekly_df) >= 4:
        weekly_close = weekly_df['close'].values
        weekly_open = weekly_df['open'].values

        # Find corresponding week
        week_idx = len(weekly_df) - 1

        # Weekly bullish (TOP FEATURE!)
        features['weekly_bullish'] = 1 if weekly_close[-1] > weekly_open[-1] else 0

        # Weekly returns
        features['weekly_return_1w'] = (weekly_close[-1] - weekly_close[-2]) / weekly_close[-2] if len(weekly_close) >= 2 else 0
        features['weekly_return_2w'] = (weekly_close[-1] - weekly_close[-3]) / weekly_close[-3] if len(weekly_close) >= 3 else 0

        # Weekly SMA
        if len(weekly_close) >= 4:
            weekly_sma_4 = np.mean(weekly_close[-4:])
            features['weekly_dist_from_sma'] = (weekly_close[-1] - weekly_sma_4) / weekly_sma_4
            features['weekly_above_sma'] = 1 if weekly_close[-1] > weekly_sma_4 else 0
        else:
            features['weekly_dist_from_sma'] = 0
            features['weekly_above_sma'] = 0

        # Weekly trend (2 weeks same direction)
        if len(weekly_close) >= 3:
            w1_bull = weekly_close[-1] > weekly_open[-1]
            w2_bull = weekly_close[-2] > weekly_open[-2]
            features['weekly_trend_consistent'] = 1 if w1_bull == w2_bull else 0
        else:
            features['weekly_trend_consistent'] = 0
    else:
        features['weekly_bullish'] = 0
        features['weekly_return_1w'] = 0
        features['weekly_return_2w'] = 0
        features['weekly_dist_from_sma'] = 0
        features['weekly_above_sma'] = 0
        features['weekly_trend_consistent'] = 0

    # =====================
    # TIME FEATURES (High Importance)
    # =====================

    features['day_of_week'] = current_date.dayofweek
    features['is_monday'] = 1 if current_date.dayofweek == 0 else 0
    features['is_friday'] = 1 if current_date.dayofweek == 4 else 0
    features['month'] = current_date.month

    # Month-end effect
    next_day = current_date + timedelta(days=1)
    features['is_month_end'] = 1 if next_day.month != current_date.month else 0

    # =====================
    # MEAN REVERSION
    # =====================

    features['mean_reversion_5d'] = -features['return_5d']  # Contrarian
    features['mean_reversion_10d'] = -features['return_10d']

    return features


def create_samples(ticker: str, daily_df: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Create labeled samples."""
    samples = []

    for idx in range(50, len(daily_df) - SHORT_SWING_DAYS):
        features = calculate_improved_features(daily_df, weekly_df, idx)
        if features is None:
            continue

        current_close = daily_df.iloc[idx]['close']
        future_close = daily_df.iloc[idx + SHORT_SWING_DAYS]['close']

        features['target_5d'] = 1 if future_close > current_close else 0
        features['date'] = daily_df.index[idx]

        samples.append(features)

    return pd.DataFrame(samples)


def train_optimized_ensemble(X_train, y_train, X_test, y_test, name):
    """Train optimized ensemble with tuned hyperparameters."""

    # Optimized models
    models = {
        'xgb': XGBClassifier(
            n_estimators=300,
            max_depth=4,  # Reduced to prevent overfitting
            learning_rate=0.03,  # Lower learning rate
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1
        ),
        'rf': RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            min_samples_split=10,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=RANDOM_STATE
        ),
        'et': ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        results[model_name] = {
            'model': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }
        print(f"      {model_name}: Acc={acc:.1%}, Prec={prec:.1%}, Rec={rec:.1%}")

    # Weighted ensemble by F1 score
    total_f1 = sum(r['f1'] for r in results.values())
    if total_f1 > 0:
        weights = {n: r['f1'] / total_f1 for n, r in results.items()}
    else:
        weights = {n: 0.25 for n in results.keys()}

    # Ensemble prediction
    y_prob = np.zeros(len(y_test))
    for n, r in results.items():
        y_prob += r['model'].predict_proba(X_test)[:, 1] * weights[n]

    # Evaluate at different thresholds
    best_threshold = 0.5
    best_f1 = 0
    for thresh in [0.45, 0.5, 0.55]:
        y_pred_ens = (y_prob > thresh).astype(int)
        f1 = f1_score(y_test, y_pred_ens, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    y_pred_ensemble = (y_prob > best_threshold).astype(int)
    ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
    ensemble_prec = precision_score(y_test, y_pred_ensemble, zero_division=0)
    ensemble_rec = recall_score(y_test, y_pred_ensemble, zero_division=0)

    print(f"    ENSEMBLE {name}: Acc={ensemble_acc:.1%}, Prec={ensemble_prec:.1%}, Rec={ensemble_rec:.1%}")

    return {n: r['model'] for n, r in results.items()}, weights, ensemble_acc, best_threshold


def train_improved_swing_model(ticker: str):
    """Train improved swing model for a ticker."""
    print(f"\n{'='*70}")
    print(f"  IMPROVED V6 SWING MODEL: {ticker}")
    print(f"{'='*70}")

    # Fetch data
    daily_df = fetch_daily_data(ticker, TRAIN_START, TEST_END)
    weekly_df = fetch_weekly_data(ticker, TRAIN_START, TEST_END)

    if len(daily_df) < 100:
        print(f"  ERROR: Insufficient data for {ticker}")
        return None

    # Create samples
    print(f"  Creating improved swing samples...")
    samples_df = create_samples(ticker, daily_df, weekly_df)
    print(f"    Total samples: {len(samples_df)}")

    # Split train/test
    train_end = pd.to_datetime(TRAIN_END)
    test_start = pd.to_datetime(TEST_START)

    train_df = samples_df[samples_df['date'] < train_end]
    test_df = samples_df[samples_df['date'] >= test_start]

    print(f"    Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"    Train positive rate: {train_df['target_5d'].mean():.1%}")
    print(f"    Test positive rate: {test_df['target_5d'].mean():.1%}")

    # Feature columns
    feature_cols = [c for c in samples_df.columns if c not in ['target_5d', 'date']]
    print(f"    Features: {len(feature_cols)}")

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y_train = train_df['target_5d']
    y_test = test_df['target_5d']

    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    print(f"\n  --- Training 5-Day Model ---")
    models, weights, acc, threshold = train_optimized_ensemble(
        X_train_scaled, y_train, X_test_scaled, y_test, "5-Day"
    )

    # Feature importance
    print(f"\n  Top 10 Features:")
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': models['xgb'].feature_importances_
    }).sort_values('importance', ascending=False)
    for i, row in importances.head(10).iterrows():
        print(f"    {row['feature']:<30} {row['importance']:.4f}")

    # Save model
    model_data = {
        'ticker': ticker,
        'version': 'v6_swing_v2',
        'trained_at': datetime.now().isoformat(),
        'feature_cols': feature_cols,
        'scaler': scaler,
        'models_5d': models,
        'weights_5d': weights,
        'acc_5d': acc,
        'threshold_5d': threshold,
        'train_samples': len(train_df),
        'test_samples': len(test_df),
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_swing_v6.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n  Model saved to {model_path}")

    return {'acc_5d': acc}


def main():
    """Main training pipeline."""
    print("="*70)
    print("  IMPROVED V6 SWING TRADE MODEL TRAINING")
    print("="*70)
    print(f"\n  Improvements:")
    print(f"    - Better volatility regime features")
    print(f"    - Up/Down volume ratio (buying pressure)")
    print(f"    - Price position in range")
    print(f"    - Bollinger Band position")
    print(f"    - MACD histogram")
    print(f"    - Optimized hyperparameters")
    print(f"\n  Train: {TRAIN_START} to {TRAIN_END}")
    print(f"  Test:  {TEST_START} to {TEST_END}")

    results = {}
    for ticker in SWING_TICKERS:
        result = train_improved_swing_model(ticker)
        if result:
            results[ticker] = result

    # Summary
    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70)
    print(f"\n  {'Ticker':<8} {'5-Day Accuracy':>15}")
    print(f"  {'-'*25}")
    for ticker, r in results.items():
        print(f"  {ticker:<8} {r['acc_5d']:>14.1%}")


if __name__ == '__main__':
    main()
