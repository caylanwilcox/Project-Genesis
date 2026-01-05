"""
Volatility Expansion Model
===========================
Predicts when the market is likely to experience a significant range expansion.

Key Insight: Volatility is mean-reverting. Periods of compression (tight ranges)
tend to precede periods of expansion. This model identifies compression setups
and predicts imminent breakouts.

Features:
1. Range Compression: Current range vs historical average
2. Time-Based: Time of day, day of week effects
3. Volume Profile: Volume relative to average
4. Price Position: Where price is within recent range
5. Structural: Proximity to key levels (PDH/PDL, Pivot)

Target: Whether the NEXT N bars will have range > X * average range
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
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
# CONFIGURATION
# =============================================================================

# How many bars ahead to look for expansion
LOOKAHEAD_BARS = 15  # 15 minutes

# What multiple of average range counts as "expansion"
EXPANSION_THRESHOLD = 1.5  # 1.5x average range = expansion

# Minimum range expansion in basis points to be considered significant
MIN_EXPANSION_BPS = 30  # 0.30% minimum move


def fetch_minute_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch 1-minute bars from Polygon."""
    print(f"  Fetching minute data for {ticker} from {start_date} to {end_date}...")
    all_data = []
    current_start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    while current_start < end:
        # Polygon allows ~5000 results per request, chunk by week for 1-min data
        chunk_end = min(current_start + timedelta(days=5), end)
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{current_start.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}"
        params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}

        response = requests.get(url, params=params)
        data = response.json()

        if 'results' in data:
            all_data.extend(data['results'])
            print(f"    Chunk {current_start.date()} to {chunk_end.date()}: {len(data['results'])} bars")

        current_start = chunk_end + timedelta(days=1)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df['datetime'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df.set_index('datetime')
    df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')

    # Filter to market hours only (9:30 AM - 4:00 PM ET)
    df = df.between_time('09:30', '16:00')

    print(f"  Total: {len(df)} minute bars for {ticker}")
    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_daily_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily bars for pivot/level calculations."""
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
    """
    Calculate features for volatility expansion prediction.

    Features are designed to detect compression before expansion.
    """
    df = minute_df.copy()

    # =================
    # 1. RANGE FEATURES
    # =================
    # Bar range in basis points
    df['bar_range_bps'] = ((df['high'] - df['low']) / df['close']) * 10000

    # Rolling range statistics
    for window in [5, 10, 15, 30]:
        df[f'range_mean_{window}'] = df['bar_range_bps'].rolling(window).mean()
        df[f'range_std_{window}'] = df['bar_range_bps'].rolling(window).std()

    # Range compression ratio (current vs recent average)
    df['range_compression_5v30'] = df['range_mean_5'] / df['range_mean_30'].replace(0, np.nan)
    df['range_compression_10v30'] = df['range_mean_10'] / df['range_mean_30'].replace(0, np.nan)

    # ATR (Average True Range)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_15'] = df['tr'].rolling(15).mean()
    df['atr_30'] = df['tr'].rolling(30).mean()
    df['atr_ratio'] = df['atr_15'] / df['atr_30'].replace(0, np.nan)

    # ==================
    # 2. VOLUME FEATURES
    # ==================
    df['volume_mean_15'] = df['volume'].rolling(15).mean()
    df['volume_mean_30'] = df['volume'].rolling(30).mean()
    df['volume_ratio'] = df['volume'] / df['volume_mean_30'].replace(0, np.nan)
    df['volume_dryup'] = df['volume_mean_15'] / df['volume_mean_30'].replace(0, np.nan)

    # =====================
    # 3. PRICE POSITION
    # =====================
    # Rolling high/low
    df['rolling_high_15'] = df['high'].rolling(15).max()
    df['rolling_low_15'] = df['low'].rolling(15).min()
    df['rolling_range_15'] = df['rolling_high_15'] - df['rolling_low_15']

    # Position within range (0 = at low, 1 = at high)
    df['price_position'] = (df['close'] - df['rolling_low_15']) / df['rolling_range_15'].replace(0, np.nan)

    # Squeeze: tight range relative to price
    df['squeeze_ratio'] = df['rolling_range_15'] / df['close'] * 100

    # =====================
    # 4. TIME FEATURES
    # =====================
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['time_of_day'] = df['hour'] + df['minute'] / 60.0

    # Time windows (known volatility patterns)
    df['is_open_30'] = ((df['hour'] == 9) & (df['minute'] >= 30)) | ((df['hour'] == 10) & (df['minute'] == 0))
    df['is_first_hour'] = (df['hour'] == 9) | ((df['hour'] == 10) & (df['minute'] == 0))
    df['is_midday_lull'] = (df['hour'] >= 11) & (df['hour'] < 14)
    df['is_power_hour'] = df['hour'] >= 15
    df['is_last_30'] = (df['hour'] == 15) & (df['minute'] >= 30)

    # Day of week
    df['day_of_week'] = df.index.dayofweek

    # ======================
    # 5. MOMENTUM FEATURES
    # ======================
    df['returns_1'] = df['close'].pct_change(1)
    df['returns_5'] = df['close'].pct_change(5)
    df['returns_15'] = df['close'].pct_change(15)

    # Momentum consistency (are returns in same direction?)
    df['momentum_consistency'] = np.sign(df['returns_5']) == np.sign(df['returns_15'])

    # ========================
    # 6. STRUCTURAL FEATURES
    # ========================
    # Distance from daily levels
    df['date'] = df.index.date

    # Merge daily data for pivot calculations
    daily_df_reset = daily_df.reset_index()
    daily_df_reset['date'] = daily_df_reset['date']

    # Previous day high/low/close for pivot calculation
    daily_df_reset['prev_high'] = daily_df_reset['high'].shift(1)
    daily_df_reset['prev_low'] = daily_df_reset['low'].shift(1)
    daily_df_reset['prev_close'] = daily_df_reset['close'].shift(1)
    daily_df_reset['pivot'] = (daily_df_reset['prev_high'] + daily_df_reset['prev_low'] + daily_df_reset['prev_close']) / 3
    daily_df_reset['today_open'] = daily_df_reset['open']

    # Merge with minute data
    df = df.reset_index()
    df = df.merge(
        daily_df_reset[['date', 'prev_high', 'prev_low', 'prev_close', 'pivot', 'today_open']],
        on='date',
        how='left'
    )
    df = df.set_index('datetime')

    # Distance from key levels (in basis points)
    df['dist_from_pdh'] = ((df['close'] - df['prev_high']) / df['close']) * 10000
    df['dist_from_pdl'] = ((df['close'] - df['prev_low']) / df['close']) * 10000
    df['dist_from_pivot'] = ((df['close'] - df['pivot']) / df['close']) * 10000
    df['dist_from_open'] = ((df['close'] - df['today_open']) / df['close']) * 10000

    # Near key levels (within 20 bps)
    df['near_pdh'] = abs(df['dist_from_pdh']) < 20
    df['near_pdl'] = abs(df['dist_from_pdl']) < 20
    df['near_pivot'] = abs(df['dist_from_pivot']) < 20

    # =======================
    # 7. CONTRACTION SIGNALS
    # =======================
    # Consecutive bars with decreasing range
    df['range_decreasing'] = df['bar_range_bps'] < df['bar_range_bps'].shift(1)
    df['consec_range_decrease'] = df['range_decreasing'].rolling(5).sum()

    # Bollinger Band width (volatility proxy)
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_width'] = (df['bb_std'] * 2) / df['bb_middle'] * 100
    df['bb_width_percentile'] = df['bb_width'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    )

    return df


def calculate_target(df: pd.DataFrame, lookahead: int = LOOKAHEAD_BARS,
                     threshold: float = EXPANSION_THRESHOLD,
                     min_bps: float = MIN_EXPANSION_BPS) -> pd.DataFrame:
    """
    Calculate target variable: Will there be significant volatility expansion?

    Target = 1 if:
    - Next N bars have total range > threshold * average range
    - AND the move is at least min_bps basis points
    """
    # Calculate future range (high to low over next N bars)
    df['future_high'] = df['high'].shift(-1).rolling(lookahead).max()
    df['future_low'] = df['low'].shift(-lookahead).rolling(lookahead).min()

    # Shift to align: we want the range from the NEXT bar onwards
    df['future_high'] = df['high'].iloc[::-1].rolling(lookahead).max().iloc[::-1].shift(-1)
    df['future_low'] = df['low'].iloc[::-1].rolling(lookahead).min().iloc[::-1].shift(-1)
    df['future_range'] = df['future_high'] - df['future_low']
    df['future_range_bps'] = (df['future_range'] / df['close']) * 10000

    # Compare to recent average range
    df['avg_range_bps'] = df['bar_range_bps'].rolling(30).mean() * lookahead

    # Target: expansion if future range exceeds threshold AND minimum bps
    df['target'] = (
        (df['future_range_bps'] > df['avg_range_bps'] * threshold) &
        (df['future_range_bps'] > min_bps)
    ).astype(int)

    return df


def prepare_training_data(tickers: list, start_date: str, end_date: str) -> tuple:
    """Prepare features and target for training."""
    all_features = []
    all_targets = []

    for ticker in tickers:
        print(f"\nPreparing data for {ticker}...")

        # Fetch data
        minute_df = fetch_minute_data(ticker, start_date, end_date)
        daily_df = fetch_daily_data(ticker, start_date, end_date)

        if minute_df.empty or daily_df.empty:
            print(f"  Skipping {ticker} - no data")
            continue

        # Calculate features
        df = calculate_features(minute_df, daily_df)

        # Calculate target
        df = calculate_target(df)

        # Select feature columns
        feature_cols = [
            # Range features
            'bar_range_bps', 'range_mean_5', 'range_mean_10', 'range_mean_15', 'range_mean_30',
            'range_std_5', 'range_std_10',
            'range_compression_5v30', 'range_compression_10v30',
            'atr_15', 'atr_30', 'atr_ratio',
            # Volume features
            'volume_ratio', 'volume_dryup',
            # Price position
            'price_position', 'squeeze_ratio',
            # Time features
            'time_of_day', 'is_first_hour', 'is_midday_lull', 'is_power_hour',
            # Momentum
            'returns_1', 'returns_5', 'returns_15',
            # Structural
            'dist_from_pdh', 'dist_from_pdl', 'dist_from_pivot', 'dist_from_open',
            'near_pdh', 'near_pdl', 'near_pivot',
            # Contraction signals
            'consec_range_decrease', 'bb_width_percentile',
        ]

        # Drop rows with NaN in features or target
        df_clean = df.dropna(subset=feature_cols + ['target'])

        if len(df_clean) > 0:
            all_features.append(df_clean[feature_cols])
            all_targets.append(df_clean['target'])
            print(f"  {ticker}: {len(df_clean)} samples, {df_clean['target'].sum()} expansion events ({df_clean['target'].mean()*100:.1f}%)")

    # Combine all tickers
    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_targets, ignore_index=True)

    print(f"\nTotal: {len(X)} samples, {y.sum()} expansion events ({y.mean()*100:.1f}%)")

    return X, y, feature_cols


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """Train ensemble of models for volatility expansion prediction."""

    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Train multiple models and ensemble
    models = {}

    # XGBoost - good at capturing non-linear patterns
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb.fit(X_scaled, y_train)
    models['xgb'] = xgb

    # Random Forest - robust to overfitting
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    rf.fit(X_scaled, y_train)
    models['rf'] = rf

    # Gradient Boosting
    print("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=RANDOM_STATE
    )
    gb.fit(X_scaled, y_train)
    models['gb'] = gb

    return models, scaler


def evaluate_model(models: dict, scaler, X_test: pd.DataFrame, y_test: pd.Series, feature_cols: list):
    """Evaluate model performance."""
    X_scaled = scaler.transform(X_test)

    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)

    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"\n{name.upper()}:")
        print(f"  Accuracy:  {acc:.3f}")
        print(f"  Precision: {prec:.3f} (when we predict expansion, how often correct)")
        print(f"  Recall:    {rec:.3f} (what % of actual expansions we catch)")
        print(f"  F1 Score:  {f1:.3f}")

        results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

    # Ensemble prediction (average probabilities)
    print("\n" + "-"*50)
    print("ENSEMBLE (Average Probabilities):")
    ensemble_prob = np.mean([
        models['xgb'].predict_proba(X_scaled)[:, 1],
        models['rf'].predict_proba(X_scaled)[:, 1],
        models['gb'].predict_proba(X_scaled)[:, 1]
    ], axis=0)

    # Evaluate at different thresholds
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_ens = (ensemble_prob >= threshold).astype(int)
        prec = precision_score(y_test, y_pred_ens, zero_division=0)
        rec = recall_score(y_test, y_pred_ens, zero_division=0)
        f1 = f1_score(y_test, y_pred_ens, zero_division=0)
        pred_rate = y_pred_ens.mean()
        print(f"  Threshold {threshold}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, Pred Rate={pred_rate*100:.1f}%")

    # Feature importance
    print("\n" + "-"*50)
    print("TOP 10 FEATURE IMPORTANCE (XGBoost):")
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': models['xgb'].feature_importances_
    }).sort_values('importance', ascending=False)

    for i, row in importances.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return results


def save_models(models: dict, scaler, feature_cols: list):
    """Save trained models to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_data = {
        'models': models,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'config': {
            'lookahead_bars': LOOKAHEAD_BARS,
            'expansion_threshold': EXPANSION_THRESHOLD,
            'min_expansion_bps': MIN_EXPANSION_BPS,
            'train_end': TRAIN_END,
            'test_start': TEST_START
        }
    }

    filepath = os.path.join(MODELS_DIR, 'volatility_expansion_model.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModels saved to {filepath}")


def main():
    """Main training pipeline."""
    print("="*70)
    print("VOLATILITY EXPANSION MODEL TRAINING")
    print("="*70)
    print(f"Train period: {TRAIN_START} to {TRAIN_END}")
    print(f"Test period:  {TEST_START} to {TEST_END}")
    print(f"Lookahead: {LOOKAHEAD_BARS} bars")
    print(f"Expansion threshold: {EXPANSION_THRESHOLD}x average range")
    print(f"Min expansion: {MIN_EXPANSION_BPS} bps")

    # For faster iteration during development, use shorter periods
    # Uncomment for full training:
    train_start = '2024-01-01'  # Use 2024 for training (faster iteration)
    train_end = TRAIN_END
    test_start = TEST_START
    test_end = '2025-03-31'  # Test on Q1 2025

    print(f"\n[Development Mode] Using:")
    print(f"  Train: {train_start} to {train_end}")
    print(f"  Test:  {test_start} to {test_end}")

    # Prepare training data
    print("\n" + "="*70)
    print("PREPARING TRAINING DATA")
    print("="*70)
    X_train, y_train, feature_cols = prepare_training_data(DEFAULT_TICKERS, train_start, train_end)

    # Prepare test data
    print("\n" + "="*70)
    print("PREPARING TEST DATA")
    print("="*70)
    X_test, y_test, _ = prepare_training_data(DEFAULT_TICKERS, test_start, test_end)

    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: No data available")
        return

    # Train models
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    models, scaler = train_model(X_train, y_train)

    # Evaluate
    results = evaluate_model(models, scaler, X_test, y_test, feature_cols)

    # Save
    save_models(models, scaler, feature_cols)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
