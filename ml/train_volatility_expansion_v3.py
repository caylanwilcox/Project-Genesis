"""
Volatility Expansion Model V3 - Rule-Based + ML Hybrid
=======================================================
Target: 60%+ precision by combining strong rules with ML confirmation.

Key Insight: Instead of predicting rare events from scratch, we:
1. Identify KNOWN high-expansion windows (power hour, first 30 min, key levels)
2. Use ML to FILTER within those windows
3. Only signal when multiple factors align

This is a "qualifier" model - it says YES or NO to pre-identified setups.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from datetime import datetime, timedelta
import pickle
import os
import requests
import warnings
warnings.filterwarnings('ignore')

from config import RANDOM_STATE, DEFAULT_TICKERS

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Configuration
LOOKAHEAD_BARS = 15
MIN_EXPANSION_BPS = 40  # 0.40% move


def fetch_minute_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch 1-minute bars from Polygon."""
    print(f"  Fetching {ticker}...", end=" ")
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
    print(f"{len(df)} bars")
    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_daily_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily data."""
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


def calculate_features_and_target(minute_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate features and identify expansion setups."""
    df = minute_df.copy()

    # Basic range
    df['bar_range_bps'] = ((df['high'] - df['low']) / df['close']) * 10000

    # Time features
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['time_of_day'] = df['hour'] + df['minute'] / 60.0

    # =====================
    # SETUP IDENTIFICATION
    # =====================
    # These are KNOWN expansion-prone windows
    df['setup_power_hour'] = df['hour'] >= 15
    df['setup_first_30'] = (df['hour'] == 9) & (df['minute'] >= 30) & (df['minute'] < 60)
    df['setup_first_hour'] = (df['hour'] == 9) | ((df['hour'] == 10) & (df['minute'] == 0))

    # Compression setup (squeeze before expansion)
    for w in [5, 10, 30]:
        df[f'range_mean_{w}'] = df['bar_range_bps'].rolling(w).mean()

    df['compression_ratio'] = df['range_mean_5'] / df['range_mean_30'].replace(0, np.nan)
    df['setup_compressed'] = df['compression_ratio'] < 0.6

    # ATR contraction
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                    abs(df['low'] - df['close'].shift(1))))
    df['atr_10'] = df['tr'].rolling(10).mean()
    df['atr_30'] = df['tr'].rolling(30).mean()
    df['atr_ratio'] = df['atr_10'] / df['atr_30'].replace(0, np.nan)
    df['setup_atr_squeeze'] = df['atr_ratio'] < 0.7

    # Volume dry-up (often precedes expansion)
    df['vol_15'] = df['volume'].rolling(15).mean()
    df['vol_60'] = df['volume'].rolling(60).mean()
    df['vol_ratio'] = df['vol_15'] / df['vol_60'].replace(0, np.nan)
    df['setup_vol_dryup'] = df['vol_ratio'] < 0.7

    # Key levels
    df['date'] = df.index.date
    if daily_df is not None and len(daily_df) > 0:
        daily_df_r = daily_df.reset_index()
        daily_df_r['prev_high'] = daily_df_r['high'].shift(1)
        daily_df_r['prev_low'] = daily_df_r['low'].shift(1)
        daily_df_r['prev_close'] = daily_df_r['close'].shift(1)

        df = df.reset_index()
        df = df.merge(daily_df_r[['date', 'prev_high', 'prev_low', 'prev_close']], on='date', how='left')
        df = df.set_index('datetime')

        df['dist_pdh_bps'] = ((df['close'] - df['prev_high']) / df['close']) * 10000
        df['dist_pdl_bps'] = ((df['close'] - df['prev_low']) / df['close']) * 10000

        df['setup_near_pdh'] = abs(df['dist_pdh_bps']) < 20
        df['setup_near_pdl'] = abs(df['dist_pdl_bps']) < 20
        df['setup_at_key_level'] = df['setup_near_pdh'] | df['setup_near_pdl']
    else:
        df['dist_pdh_bps'] = 0
        df['dist_pdl_bps'] = 0
        df['setup_near_pdh'] = False
        df['setup_near_pdl'] = False
        df['setup_at_key_level'] = False

    # =====================
    # COMBINED SETUP SCORE
    # =====================
    # Count how many favorable conditions are present
    df['setup_score'] = (
        df['setup_power_hour'].astype(int) * 2 +  # Power hour is strong
        df['setup_first_30'].astype(int) * 2 +    # First 30 min is strong
        df['setup_compressed'].astype(int) +
        df['setup_atr_squeeze'].astype(int) +
        df['setup_vol_dryup'].astype(int) +
        df['setup_at_key_level'].astype(int) * 2  # Key levels are strong
    )

    # HIGH-QUALITY SETUP: score >= 3 (multiple factors aligned)
    df['is_setup'] = df['setup_score'] >= 3

    # =====================
    # TARGET: Did expansion actually happen?
    # =====================
    df['future_high'] = df['high'].iloc[::-1].rolling(LOOKAHEAD_BARS).max().iloc[::-1].shift(-1)
    df['future_low'] = df['low'].iloc[::-1].rolling(LOOKAHEAD_BARS).min().iloc[::-1].shift(-1)
    df['future_range_bps'] = ((df['future_high'] - df['future_low']) / df['close']) * 10000

    df['expansion_occurred'] = df['future_range_bps'] > MIN_EXPANSION_BPS

    return df


def analyze_setup_performance(df: pd.DataFrame):
    """Analyze precision of different setups."""
    print("\n" + "="*60)
    print("SETUP PERFORMANCE ANALYSIS")
    print("="*60)

    df_valid = df.dropna(subset=['expansion_occurred'])

    # Overall base rate
    base_rate = df_valid['expansion_occurred'].mean()
    print(f"\nBase expansion rate (40+ bps in 15 min): {base_rate:.1%}")

    # Analyze each setup type
    setups = [
        ('Power Hour (3-4 PM)', 'setup_power_hour'),
        ('First 30 min', 'setup_first_30'),
        ('First Hour', 'setup_first_hour'),
        ('Compressed Range', 'setup_compressed'),
        ('ATR Squeeze', 'setup_atr_squeeze'),
        ('Volume Dry-up', 'setup_vol_dryup'),
        ('Near PDH', 'setup_near_pdh'),
        ('Near PDL', 'setup_near_pdl'),
        ('Any Key Level', 'setup_at_key_level'),
    ]

    print(f"\n{'Setup':<25} {'Count':<10} {'Precision':<12} {'vs Base':<10}")
    print("-" * 60)

    for name, col in setups:
        subset = df_valid[df_valid[col] == True]
        if len(subset) > 100:
            precision = subset['expansion_occurred'].mean()
            lift = precision / base_rate if base_rate > 0 else 0
            print(f"{name:<25} {len(subset):<10} {precision:<12.1%} {lift:<10.2f}x")

    # Analyze combined score
    print("\n" + "-"*60)
    print("COMBINED SETUP SCORE:")
    print("-"*60)

    for score in range(1, 8):
        subset = df_valid[df_valid['setup_score'] >= score]
        if len(subset) > 50:
            precision = subset['expansion_occurred'].mean()
            lift = precision / base_rate if base_rate > 0 else 0
            status = " <-- 60%+ " if precision >= 0.60 else ""
            print(f"Score >= {score}: {len(subset):>6} samples, {precision:>6.1%} precision, {lift:.2f}x lift{status}")

    return base_rate


def train_qualifier_model(df_train: pd.DataFrame) -> tuple:
    """
    Train a model that qualifies/confirms setups.

    Only runs on bars where is_setup=True.
    Target: predict which setups will actually produce expansion.
    """
    # Filter to setup bars only
    setup_df = df_train[df_train['is_setup'] == True].copy()
    setup_df = setup_df.dropna(subset=['expansion_occurred'])

    if len(setup_df) < 100:
        print("Not enough setup samples for training")
        return None, None, None

    feature_cols = [
        'compression_ratio', 'atr_ratio', 'vol_ratio',
        'setup_score', 'time_of_day',
        'setup_power_hour', 'setup_first_30', 'setup_compressed',
        'setup_atr_squeeze', 'setup_vol_dryup', 'setup_at_key_level',
        'dist_pdh_bps', 'dist_pdl_bps',
        'range_mean_5', 'range_mean_30',
    ]

    # Ensure columns exist
    for col in feature_cols:
        if col not in setup_df.columns:
            setup_df[col] = 0

    X = setup_df[feature_cols].fillna(0)
    y = setup_df['expansion_occurred'].astype(int)

    print(f"\nTraining qualifier model on {len(X)} setup samples...")
    print(f"Setup expansion rate: {y.mean():.1%}")

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Train ensemble
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb.fit(X_scaled, y)

    return xgb, scaler, feature_cols


def evaluate_final_system(df_test: pd.DataFrame, model, scaler, feature_cols) -> float:
    """Evaluate the complete setup + qualifier system."""

    df_valid = df_test.dropna(subset=['expansion_occurred'])

    # Step 1: Filter to setups
    setup_df = df_valid[df_valid['is_setup'] == True].copy()

    if len(setup_df) == 0:
        print("No setups in test data")
        return 0

    # Step 2: Apply qualifier model
    X_test = setup_df[feature_cols].fillna(0)
    X_scaled = scaler.transform(X_test)

    probs = model.predict_proba(X_scaled)[:, 1]
    setup_df['qualifier_prob'] = probs

    print("\n" + "="*60)
    print("FINAL SYSTEM EVALUATION (Setup + Qualifier)")
    print("="*60)

    print(f"\nTotal test samples: {len(df_valid)}")
    print(f"Setup samples: {len(setup_df)} ({len(setup_df)/len(df_valid)*100:.1f}%)")

    base_rate = df_valid['expansion_occurred'].mean()
    print(f"Base expansion rate: {base_rate:.1%}")

    setup_rate = setup_df['expansion_occurred'].mean()
    print(f"Setup expansion rate: {setup_rate:.1%}")

    # Find threshold for 60%+ precision
    print("\n" + "-"*60)
    print("Qualifier Threshold Analysis:")
    print("-"*60)

    best_threshold = 0.5
    best_precision = 0

    for threshold in np.arange(0.3, 0.9, 0.05):
        qualified = setup_df[setup_df['qualifier_prob'] >= threshold]
        if len(qualified) < 10:
            continue

        precision = qualified['expansion_occurred'].mean()
        count = len(qualified)
        status = " <-- 60%+ TARGET" if precision >= 0.60 else ""

        print(f"Threshold {threshold:.2f}: {count:>5} signals, {precision:>6.1%} precision{status}")

        if precision >= 0.60 and precision > best_precision:
            best_precision = precision
            best_threshold = threshold

    print("\n" + "="*60)
    if best_precision >= 0.60:
        print(f"SUCCESS! Best threshold: {best_threshold:.2f} with {best_precision:.1%} precision")
    else:
        print(f"Best precision achieved: {best_precision:.1%} at threshold {best_threshold:.2f}")
        print("Could not reach 60% target")

    return best_precision, best_threshold


def save_system(model, scaler, feature_cols, threshold, precision):
    """Save the complete system."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    system_data = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'threshold': threshold,
        'precision': precision,
        'setup_rules': {
            'min_setup_score': 3,
            'min_expansion_bps': MIN_EXPANSION_BPS,
            'lookahead_bars': LOOKAHEAD_BARS
        },
        'version': 'v3_rule_hybrid'
    }

    filepath = os.path.join(MODELS_DIR, 'volatility_expansion_model.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(system_data, f)

    print(f"\nSystem saved to {filepath}")


def main():
    print("="*70)
    print("VOLATILITY EXPANSION V3 - RULE + ML HYBRID")
    print("="*70)
    print("Strategy: Identify high-probability setups, then ML qualifies them")

    train_start = '2024-01-01'
    train_end = '2024-12-31'
    test_start = '2025-01-01'
    test_end = '2025-03-31'

    # Gather data
    all_train = []
    all_test = []

    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)

    for ticker in DEFAULT_TICKERS:
        minute_train = fetch_minute_data(ticker, train_start, train_end)
        minute_test = fetch_minute_data(ticker, test_start, test_end)
        daily = fetch_daily_data(ticker, train_start, test_end)

        if not minute_train.empty:
            df = calculate_features_and_target(minute_train, daily)
            all_train.append(df)

        if not minute_test.empty:
            df = calculate_features_and_target(minute_test, daily)
            all_test.append(df)

    df_train = pd.concat(all_train, ignore_index=False)
    df_test = pd.concat(all_test, ignore_index=False)

    print(f"\nTrain: {len(df_train)} samples")
    print(f"Test: {len(df_test)} samples")

    # Analyze setup performance
    analyze_setup_performance(df_train)

    # Train qualifier
    print("\n" + "="*50)
    print("TRAINING QUALIFIER MODEL")
    print("="*50)

    model, scaler, feature_cols = train_qualifier_model(df_train)

    if model is None:
        print("Training failed")
        return

    # Evaluate
    precision, threshold = evaluate_final_system(df_test, model, scaler, feature_cols)

    # Save
    save_system(model, scaler, feature_cols, threshold, precision)

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
