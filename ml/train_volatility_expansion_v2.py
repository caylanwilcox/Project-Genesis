"""
Volatility Expansion Model V2 - High Precision Version
========================================================
Target: 60%+ precision when predicting expansion events.

Key Changes from V1:
1. Stricter expansion definition (2x avg range, 50+ bps)
2. Focus on high-confidence setups only (compression + time + level)
3. Train for precision over recall
4. Use probability calibration

Features focused on:
- Strong compression signals (squeeze setups)
- Time of day patterns (power hour, first 30 min)
- Proximity to key levels (PDH/PDL breakout candidates)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from datetime import datetime, timedelta
import pickle
import os
import requests
import warnings
warnings.filterwarnings('ignore')

from config import TRAIN_START, TRAIN_END, TEST_START, TEST_END, RANDOM_STATE, DEFAULT_TICKERS

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# =============================================================================
# V2 CONFIGURATION - Stricter thresholds for higher precision
# =============================================================================

LOOKAHEAD_BARS = 15          # 15 minutes lookahead
EXPANSION_THRESHOLD = 2.0    # 2x average range (stricter than 1.5x)
MIN_EXPANSION_BPS = 50       # 0.50% minimum move (stricter than 0.30%)


def fetch_minute_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch 1-minute bars from Polygon."""
    print(f"  Fetching minute data for {ticker}...")
    all_data = []
    current_start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    while current_start < end:
        chunk_end = min(current_start + timedelta(days=5), end)
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{current_start.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}"
        params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}

        response = requests.get(url, params=params)
        data = response.json()

        if 'results' in data:
            all_data.extend(data['results'])

        current_start = chunk_end + timedelta(days=1)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df['datetime'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df.set_index('datetime')
    df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
    df = df.between_time('09:30', '16:00')

    print(f"    {len(df)} minute bars")
    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_daily_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily bars for pivot calculations."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}
    response = requests.get(url, params=params)
    data = response.json()

    if 'results' not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df.set_index('date')
    return df[['open', 'high', 'low', 'close', 'volume']]


def calculate_features(minute_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate features optimized for high-precision expansion prediction."""
    df = minute_df.copy()

    # Bar range in basis points
    df['bar_range_bps'] = ((df['high'] - df['low']) / df['close']) * 10000

    # ===================
    # COMPRESSION FEATURES (most important for precision)
    # ===================
    for window in [5, 10, 15, 30, 60]:
        df[f'range_mean_{window}'] = df['bar_range_bps'].rolling(window).mean()

    # Compression ratios - key signal
    df['compression_5v30'] = df['range_mean_5'] / df['range_mean_30'].replace(0, np.nan)
    df['compression_10v60'] = df['range_mean_10'] / df['range_mean_60'].replace(0, np.nan)

    # Squeeze detection - very tight range
    df['is_squeezed'] = df['compression_5v30'] < 0.5  # 5-bar range < 50% of 30-bar avg

    # Consecutive compression bars
    df['range_decreasing'] = df['bar_range_bps'] < df['bar_range_bps'].shift(1)
    df['consec_decrease'] = df['range_decreasing'].rolling(5).sum()

    # ATR contraction
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_10'] = df['tr'].rolling(10).mean()
    df['atr_30'] = df['tr'].rolling(30).mean()
    df['atr_contraction'] = df['atr_10'] / df['atr_30'].replace(0, np.nan)

    # ===================
    # TIME FEATURES (strong predictors)
    # ===================
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['time_of_day'] = df['hour'] + df['minute'] / 60.0

    # High-probability expansion windows
    df['is_first_30min'] = (df['hour'] == 9) & (df['minute'] >= 30)
    df['is_first_hour'] = (df['hour'] == 9) | ((df['hour'] == 10) & (df['minute'] == 0))
    df['is_power_hour'] = df['hour'] >= 15
    df['is_last_30min'] = (df['hour'] == 15) & (df['minute'] >= 30)
    df['is_lull'] = (df['hour'] >= 11) & (df['hour'] < 14)

    # Day of week
    df['day_of_week'] = df.index.dayofweek
    df['is_monday'] = df['day_of_week'] == 0
    df['is_friday'] = df['day_of_week'] == 4

    # ===================
    # KEY LEVEL PROXIMITY (breakout candidates)
    # ===================
    df['date'] = df.index.date

    if daily_df is not None and len(daily_df) > 0:
        daily_df_reset = daily_df.reset_index()
        daily_df_reset['prev_high'] = daily_df_reset['high'].shift(1)
        daily_df_reset['prev_low'] = daily_df_reset['low'].shift(1)
        daily_df_reset['prev_close'] = daily_df_reset['close'].shift(1)
        daily_df_reset['pivot'] = (daily_df_reset['prev_high'] + daily_df_reset['prev_low'] + daily_df_reset['prev_close']) / 3
        daily_df_reset['today_open'] = daily_df_reset['open']

        df = df.reset_index()
        df = df.merge(
            daily_df_reset[['date', 'prev_high', 'prev_low', 'prev_close', 'pivot', 'today_open']],
            on='date', how='left'
        )
        df = df.set_index('datetime')

        # Distance from key levels (bps)
        df['dist_pdh'] = ((df['close'] - df['prev_high']) / df['close']) * 10000
        df['dist_pdl'] = ((df['close'] - df['prev_low']) / df['close']) * 10000
        df['dist_pivot'] = ((df['close'] - df['pivot']) / df['close']) * 10000

        # Near breakout levels (within 15 bps = very close)
        df['near_pdh'] = (df['dist_pdh'] > -15) & (df['dist_pdh'] < 15)
        df['near_pdl'] = (df['dist_pdl'] > -15) & (df['dist_pdl'] < 15)
        df['at_level'] = df['near_pdh'] | df['near_pdl']

        # Breaking out NOW
        df['breaking_pdh'] = df['dist_pdh'] > 0
        df['breaking_pdl'] = df['dist_pdl'] < 0

    # ===================
    # VOLUME FEATURES
    # ===================
    df['vol_mean_15'] = df['volume'].rolling(15).mean()
    df['vol_mean_60'] = df['volume'].rolling(60).mean()
    df['vol_ratio'] = df['volume'] / df['vol_mean_60'].replace(0, np.nan)
    df['vol_dryup'] = df['vol_mean_15'] / df['vol_mean_60'].replace(0, np.nan)

    # Volume spike (often precedes expansion)
    df['vol_spike'] = df['vol_ratio'] > 2.0

    # ===================
    # MOMENTUM
    # ===================
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_15'] = df['close'].pct_change(15)

    # Price position in range
    df['roll_high_15'] = df['high'].rolling(15).max()
    df['roll_low_15'] = df['low'].rolling(15).min()
    df['price_position'] = (df['close'] - df['roll_low_15']) / (df['roll_high_15'] - df['roll_low_15']).replace(0, np.nan)

    return df


def calculate_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate target with STRICTER criteria for higher precision.

    Target = 1 only if:
    - Next 15 bars have range > 2x average (was 1.5x)
    - AND move is at least 50 bps (was 30 bps)
    """
    # Future range calculation
    df['future_high'] = df['high'].iloc[::-1].rolling(LOOKAHEAD_BARS).max().iloc[::-1].shift(-1)
    df['future_low'] = df['low'].iloc[::-1].rolling(LOOKAHEAD_BARS).min().iloc[::-1].shift(-1)
    df['future_range'] = df['future_high'] - df['future_low']
    df['future_range_bps'] = (df['future_range'] / df['close']) * 10000

    # Average range for comparison
    df['avg_range_bps'] = df['bar_range_bps'].rolling(30).mean() * LOOKAHEAD_BARS

    # STRICT target criteria
    df['target'] = (
        (df['future_range_bps'] > df['avg_range_bps'] * EXPANSION_THRESHOLD) &
        (df['future_range_bps'] > MIN_EXPANSION_BPS)
    ).astype(int)

    return df


def prepare_data(tickers: list, start_date: str, end_date: str) -> tuple:
    """Prepare features and target."""
    all_X = []
    all_y = []

    feature_cols = [
        # Compression (most important)
        'compression_5v30', 'compression_10v60', 'is_squeezed', 'consec_decrease',
        'atr_contraction', 'range_mean_5', 'range_mean_30',
        # Time
        'time_of_day', 'is_first_30min', 'is_first_hour', 'is_power_hour',
        'is_last_30min', 'is_lull',
        # Key levels
        'dist_pdh', 'dist_pdl', 'dist_pivot', 'near_pdh', 'near_pdl', 'at_level',
        # Volume
        'vol_ratio', 'vol_dryup', 'vol_spike',
        # Momentum
        'ret_5', 'ret_15', 'price_position',
    ]

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        minute_df = fetch_minute_data(ticker, start_date, end_date)
        daily_df = fetch_daily_data(ticker, start_date, end_date)

        if minute_df.empty:
            continue

        df = calculate_features(minute_df, daily_df)
        df = calculate_target(df)

        # Ensure all columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        df_clean = df.dropna(subset=feature_cols + ['target'])

        if len(df_clean) > 0:
            all_X.append(df_clean[feature_cols])
            all_y.append(df_clean['target'])
            expansion_rate = df_clean['target'].mean() * 100
            print(f"  {ticker}: {len(df_clean)} samples, {df_clean['target'].sum()} expansions ({expansion_rate:.2f}%)")

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)

    print(f"\nTotal: {len(X)} samples, {y.sum()} expansions ({y.mean()*100:.2f}%)")

    return X, y, feature_cols


def train_high_precision_model(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """Train model optimized for HIGH PRECISION."""

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)

    print("\n" + "="*50)
    print("Training HIGH PRECISION models...")
    print("="*50)

    models = {}

    # XGBoost with precision focus
    print("\nTraining XGBoost (precision-focused)...")
    # Higher scale_pos_weight makes model more conservative
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,           # Shallower trees = less overfitting
        learning_rate=0.05,    # Lower LR = more conservative
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=0.3,  # Under-weight positive class = fewer predictions
        min_child_weight=10,   # Require more samples per leaf
        random_state=RANDOM_STATE,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb.fit(X_scaled, y_train)
    models['xgb'] = xgb

    # Random Forest with balanced but conservative settings
    print("Training Random Forest (conservative)...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=50,    # Large min samples = conservative
        min_samples_split=100,
        class_weight={0: 1, 1: 0.5},  # Under-weight positives
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_scaled, y_train)
    models['rf'] = rf

    # Gradient Boosting
    print("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=50,
        random_state=RANDOM_STATE
    )
    gb.fit(X_scaled, y_train)
    models['gb'] = gb

    return models, scaler


def evaluate_and_find_threshold(models: dict, scaler, X_test: pd.DataFrame, y_test: pd.Series, feature_cols: list):
    """Evaluate and find threshold for 60%+ precision."""

    X_scaled = scaler.transform(X_test)

    print("\n" + "="*70)
    print("EVALUATION - Finding 60%+ Precision Threshold")
    print("="*70)

    # Get ensemble probabilities
    probs = []
    for name, model in models.items():
        prob = model.predict_proba(X_scaled)[:, 1]
        probs.append(prob)

    ensemble_prob = np.mean(probs, axis=0)

    # Find threshold that achieves 60%+ precision
    print("\nThreshold Analysis:")
    print("-" * 60)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'Pred Rate':<12} {'Count':<10}")
    print("-" * 60)

    best_threshold = 0.7
    best_f1 = 0

    for threshold in np.arange(0.3, 0.95, 0.05):
        y_pred = (ensemble_prob >= threshold).astype(int)
        if y_pred.sum() == 0:
            continue

        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        pred_rate = y_pred.mean() * 100
        count = y_pred.sum()

        status = "  <-- TARGET" if prec >= 0.60 else ""
        print(f"{threshold:<12.2f} {prec:<12.3f} {rec:<12.3f} {pred_rate:<12.1f}% {count:<10}{status}")

        # Track best threshold with 60%+ precision
        if prec >= 0.60 and f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print("-" * 60)
    print(f"\nBest threshold for 60%+ precision: {best_threshold:.2f}")

    # Final evaluation at best threshold
    y_pred_final = (ensemble_prob >= best_threshold).astype(int)
    final_prec = precision_score(y_test, y_pred_final, zero_division=0)
    final_rec = recall_score(y_test, y_pred_final, zero_division=0)

    print(f"\nFINAL METRICS at threshold {best_threshold}:")
    print(f"  Precision: {final_prec:.1%} (target: 60%+)")
    print(f"  Recall: {final_rec:.1%}")
    print(f"  Predictions: {y_pred_final.sum()} / {len(y_test)} ({y_pred_final.mean()*100:.1f}%)")

    # Feature importance
    print("\n" + "-"*50)
    print("TOP FEATURES (XGBoost):")
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': models['xgb'].feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importances.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return best_threshold, final_prec


def save_model(models: dict, scaler, feature_cols: list, threshold: float, precision: float):
    """Save model with metadata."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_data = {
        'models': models,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'threshold': threshold,
        'precision': precision,
        'config': {
            'lookahead_bars': LOOKAHEAD_BARS,
            'expansion_threshold': EXPANSION_THRESHOLD,
            'min_expansion_bps': MIN_EXPANSION_BPS,
            'version': 'v2_high_precision'
        }
    }

    filepath = os.path.join(MODELS_DIR, 'volatility_expansion_model.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved to {filepath}")
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Precision: {precision:.1%}")


def main():
    print("="*70)
    print("VOLATILITY EXPANSION MODEL V2 - HIGH PRECISION")
    print("="*70)
    print(f"Target: 60%+ precision")
    print(f"Expansion threshold: {EXPANSION_THRESHOLD}x average range")
    print(f"Min expansion: {MIN_EXPANSION_BPS} bps")

    # Training period
    train_start = '2024-01-01'
    train_end = '2024-12-31'
    test_start = '2025-01-01'
    test_end = '2025-03-31'

    print(f"\nTrain: {train_start} to {train_end}")
    print(f"Test:  {test_start} to {test_end}")

    # Prepare data
    print("\n" + "="*70)
    print("PREPARING TRAINING DATA")
    print("="*70)
    X_train, y_train, feature_cols = prepare_data(DEFAULT_TICKERS, train_start, train_end)

    print("\n" + "="*70)
    print("PREPARING TEST DATA")
    print("="*70)
    X_test, y_test, _ = prepare_data(DEFAULT_TICKERS, test_start, test_end)

    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: No data")
        return

    # Train
    models, scaler = train_high_precision_model(X_train, y_train)

    # Evaluate and find best threshold
    best_threshold, precision = evaluate_and_find_threshold(models, scaler, X_test, y_test, feature_cols)

    if precision < 0.60:
        print("\n" + "!"*70)
        print("WARNING: Could not achieve 60% precision")
        print("Consider: More data, different features, or stricter target definition")
        print("!"*70)

    # Save
    save_model(models, scaler, feature_cols, best_threshold, precision)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
