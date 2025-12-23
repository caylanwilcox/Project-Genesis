"""
Enhanced Daily Direction Prediction Model v3

This model uses the 80 new features added in the Feature Implementation Schedule:
- TTM Squeeze (6 features)
- KDJ (9,3,3) (8 features)
- Volatility Regime (9 features)
- Divergences (10 features)
- Bar Patterns (5 features)
- Trend Structure (5 features)
- Pivot Points (9 features)
- Range Patterns (8 features)
- Fibonacci Levels (7 features)
- Swing S/R (7 features)
- Enhanced Calendar (6 features)

This model saves separately from existing models to allow A/B comparison.
Model files: models/{ticker}_enhanced_v3_model.pkl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
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

# ============================================================
# NEW FEATURES FROM PHASE 1-3 IMPLEMENTATION
# ============================================================

# Features removed due to 0.0000 or very low (<0.005) importance across all tickers:
# - ttm_squeeze_fired, kdj_j_overbought, kdj_j_oversold
# - vol_expanding, vol_contracting
# - rsi_bearish_div, macd_bullish_div, macd_bearish_div
# - obv_bullish_div, obv_bearish_div, div_signal
# - at_resistance, at_support, above_pivot
# - range_contraction, near_fib_level
# - is_quarter_start, is_friday, macd_cross

NEW_FEATURES = [
    # Phase 1 - TTM Squeeze (keep momentum-related, remove binary fired)
    'ttm_squeeze_on', 'ttm_squeeze_off',  # Important: volatility compression state
    'ttm_squeeze_momentum', 'ttm_squeeze_momentum_rising', 'ttm_squeeze_bars',

    # Phase 1 - KDJ (keep core values and crossovers, remove extreme zones)
    'kdj_k', 'kdj_d', 'kdj_j', 'kdj_golden_cross', 'kdj_death_cross',
    'kdj_zone',  # Categorical zone is useful

    # Phase 1 - Volatility Regime (keep continuous values, remove binary expand/contract)
    'hv_10', 'hv_20', 'hv_ratio', 'vol_regime', 'vol_percentile',
    'atr_pct', 'atr_regime',

    # Phase 2 - Divergences (keep RSI bullish and counts, remove others)
    'rsi_bullish_div', 'rsi_div_strength',
    'bullish_div_count', 'bearish_div_count',

    # Phase 2 - Bar Patterns (keep all - they showed decent importance)
    'inside_bar', 'outside_bar', 'narrow_range_4', 'narrow_range_7', 'wide_range_bar',

    # Phase 2 - Trend Structure (keep all - these are top features!)
    'higher_high', 'lower_low', 'higher_low', 'lower_high', 'trend_structure_3',

    # Phase 2 - Pivot Points (keep distances, remove binary above_pivot)
    'dist_to_pivot', 'dist_to_r1', 'dist_to_s1',

    # Phase 3 - Range Patterns (keep continuous, remove binary contraction)
    'avg_range_10', 'avg_range_20', 'range_vs_avg', 'range_expansion',
    'range_rank_20', 'consec_narrow', 'breakout_potential',

    # Phase 3 - Fibonacci (keep distances, remove binary near_fib)
    'dist_to_fib_382', 'dist_to_fib_618',

    # Phase 3 - Swing S/R (keep distances and position, remove binary at_support/resistance)
    'dist_to_resistance', 'dist_to_support', 'swing_position',

    # Phase 3 - Calendar (remove low-importance is_quarter_start)
    'week_of_month', 'is_quarter_end', 'is_opex_week',
    'is_first_5_days', 'is_last_5_days'
]

# Full list for reference (80 original features)
ALL_NEW_FEATURES_ORIGINAL = [
    'ttm_squeeze_on', 'ttm_squeeze_off', 'ttm_squeeze_fired',
    'ttm_squeeze_momentum', 'ttm_squeeze_momentum_rising', 'ttm_squeeze_bars',
    'kdj_k', 'kdj_d', 'kdj_j', 'kdj_golden_cross', 'kdj_death_cross',
    'kdj_j_overbought', 'kdj_j_oversold', 'kdj_zone',
    'hv_10', 'hv_20', 'hv_ratio', 'vol_regime', 'vol_percentile',
    'vol_expanding', 'vol_contracting', 'atr_pct', 'atr_regime',
    'rsi_bullish_div', 'rsi_bearish_div', 'rsi_div_strength',
    'macd_bullish_div', 'macd_bearish_div', 'obv_bullish_div', 'obv_bearish_div',
    'bullish_div_count', 'bearish_div_count', 'div_signal',
    'inside_bar', 'outside_bar', 'narrow_range_4', 'narrow_range_7', 'wide_range_bar',
    'higher_high', 'lower_low', 'higher_low', 'lower_high', 'trend_structure_3',
    'dist_to_pivot', 'dist_to_r1', 'dist_to_s1', 'above_pivot',
    'avg_range_10', 'avg_range_20', 'range_vs_avg', 'range_expansion',
    'range_contraction', 'range_rank_20', 'consec_narrow', 'breakout_potential',
    'dist_to_fib_382', 'dist_to_fib_618', 'near_fib_level',
    'dist_to_resistance', 'dist_to_support', 'at_resistance', 'at_support', 'swing_position',
    'week_of_month', 'is_quarter_end', 'is_quarter_start', 'is_opex_week',
    'is_first_5_days', 'is_last_5_days'
]

# Base features from existing models
# Removed: macd_cross (0.0000 importance), is_friday (0.0000 importance)
BASE_FEATURES = [
    # Returns
    'daily_return', 'ret_lag_1', 'ret_lag_2', 'ret_lag_3', 'ret_lag_5',

    # Momentum
    'momentum_3d', 'momentum_5d', 'momentum_10d', 'roc_5', 'roc_10',

    # Volatility
    'volatility_5d', 'volatility_10d', 'volatility_20d',

    # RSI
    'rsi_14', 'rsi_9', 'rsi_change', 'rsi_oversold', 'rsi_overbought',

    # MACD (removed macd_cross - 0.0000 importance)
    'macd_hist', 'macd_hist_change',

    # Moving Averages
    'price_vs_sma20', 'price_vs_sma50', 'ema_cross', 'trend_alignment',

    # Bollinger Bands
    'bb_position', 'bb_width', 'bb_squeeze',

    # Stochastic
    'stoch_k', 'stoch_d',

    # Volume
    'volume_ratio', 'volume_trend', 'obv_trend',

    # Calendar (removed is_friday - 0.0000 importance)
    'day_of_week', 'is_monday',

    # Streak/Mean reversion
    'streak', 'zscore_20',

    # ADX
    'adx', 'di_diff'
]

ALL_FEATURES = BASE_FEATURES + NEW_FEATURES


def fetch_polygon_data(ticker: str, start_date: str = '2003-01-01', end_date: str = None) -> pd.DataFrame:
    """Fetch historical daily data from Polygon.io

    Args:
        ticker: Stock symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (defaults to today)
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    all_results = []
    current_start = start_date

    # Polygon limits to 50000 results, so we may need multiple requests
    while True:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{current_start}/{end_date}"
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000,
            'apiKey': POLYGON_API_KEY
        }

        response = requests.get(url, params=params)
        data = response.json()

        if 'results' not in data or len(data['results']) == 0:
            break

        all_results.extend(data['results'])

        # Check if there's more data
        if len(data['results']) < 50000:
            break

        # Move start date forward
        last_date = pd.to_datetime(data['results'][-1]['t'], unit='ms')
        current_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')

        if current_start >= end_date:
            break

    if len(all_results) == 0:
        raise ValueError(f"No data returned for {ticker}")

    df = pd.DataFrame(all_results)
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

    # Remove duplicates if any
    df = df[~df.index.duplicated(keep='first')]

    return df


def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ALL features including new Phase 1-3 features"""

    # Import feature calculation from predict_server
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from predict_server import calculate_daily_features

    # Calculate all features using the updated predict_server function
    df = calculate_daily_features(df)

    # Add momentum features not in predict_server
    for period in [3, 5, 10]:
        df[f'momentum_{period}d'] = df['daily_return'].rolling(period).sum().shift(1)

    # Add ROC features
    for period in [5, 10]:
        df[f'roc_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period) * 100).shift(1)

    # Add volatility features
    for period in [5, 10, 20]:
        df[f'volatility_{period}d'] = df['daily_return'].rolling(period).std().shift(1)

    # Add lagged returns
    for i in [1, 2, 3, 5]:
        df[f'ret_lag_{i}'] = df['daily_return'].shift(i)

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create prediction target: next day direction"""
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df


def train_enhanced_model(ticker: str):
    """Train enhanced model with all new features

    Training: 2003-01-01 to 2023-12-31
    Testing: 2024-01-01 to present (2025)
    """
    print(f"\n{'='*60}")
    print(f"Training Enhanced v3 Model for {ticker}")
    print(f"{'='*60}")

    # Fetch ALL data from 2003 to present
    print(f"Fetching data for {ticker} (2003-2025)...")
    df = fetch_polygon_data(ticker, start_date='2003-01-01', end_date=None)
    print(f"  Total data: {len(df)} days ({df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')})")

    # Calculate features
    print("Calculating features...")
    df = calculate_all_features(df)
    df = create_target(df)

    # Get available features
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    missing_features = [f for f in ALL_FEATURES if f not in df.columns]

    print(f"  Available features: {len(available_features)}/{len(ALL_FEATURES)}")
    if missing_features:
        print(f"  Missing features: {missing_features[:5]}...")

    # Prepare data
    df_clean = df.dropna(subset=available_features + ['target'])
    print(f"  Clean samples: {len(df_clean)}")

    if len(df_clean) < 500:
        print(f"  ERROR: Not enough data for training")
        return None

    # Split by date: Train on 2003-2023, Test on 2024-2025
    train_end_date = '2023-12-31'
    test_start_date = '2024-01-01'

    train_data = df_clean[df_clean.index <= train_end_date]
    test_data = df_clean[df_clean.index >= test_start_date]

    X_train = train_data[available_features]
    y_train = train_data['target']
    X_test = test_data[available_features]
    y_test = test_data['target']

    print(f"  Training period: {train_data.index.min().strftime('%Y-%m-%d')} to {train_data.index.max().strftime('%Y-%m-%d')}")
    print(f"  Testing period: {test_data.index.min().strftime('%Y-%m-%d')} to {test_data.index.max().strftime('%Y-%m-%d')}")
    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train ensemble models
    print("\nTraining ensemble...")

    # Tuned hyperparameters for better out-of-sample performance
    # Key: reduce overfitting (lower depth, more regularization, more samples per leaf)
    models = {
        'xgb': XGBClassifier(
            n_estimators=300,
            max_depth=3,  # Reduced from 4 to prevent overfitting
            learning_rate=0.03,  # Slower learning
            subsample=0.7,  # More regularization
            colsample_bytree=0.7,
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            min_child_weight=5,  # Require more samples per leaf
            random_state=42,
            verbosity=0
        ),
        'rf': RandomForestClassifier(
            n_estimators=300,
            max_depth=5,  # Reduced from 6
            min_samples_split=30,  # Increased from 20
            min_samples_leaf=10,  # New: require more samples per leaf
            max_features='sqrt',  # Limit features per tree
            random_state=42,
            n_jobs=-1
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,  # Reduced from 4
            learning_rate=0.03,  # Slower learning
            subsample=0.7,  # More regularization
            min_samples_split=30,
            min_samples_leaf=10,
            random_state=42
        ),
        'et': ExtraTreesClassifier(
            n_estimators=300,
            max_depth=5,  # Reduced from 6
            min_samples_split=30,  # Increased from 20
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    }

    model_results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        train_acc = model.score(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)
        model_results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc
        }
        print(f"  {name.upper()}: Train={train_acc:.1%}, Test={test_acc:.1%}")

    # Ensemble prediction (weighted average)
    ensemble_probs = np.zeros(len(X_test))
    weights = {'xgb': 0.35, 'rf': 0.25, 'gb': 0.25, 'et': 0.15}

    for name, weight in weights.items():
        probs = model_results[name]['model'].predict_proba(X_test_scaled)[:, 1]
        ensemble_probs += weight * probs

    ensemble_pred = (ensemble_probs > 0.5).astype(int)
    ensemble_acc = (ensemble_pred == y_test).mean()

    # High confidence accuracy (prob > 0.6 or < 0.4)
    high_conf_mask = (ensemble_probs > 0.6) | (ensemble_probs < 0.4)
    if high_conf_mask.sum() > 0:
        high_conf_acc = (ensemble_pred[high_conf_mask] == y_test.values[high_conf_mask]).mean()
        high_conf_pct = high_conf_mask.mean() * 100
    else:
        high_conf_acc = 0
        high_conf_pct = 0

    print(f"\n{'='*40}")
    print(f"ENSEMBLE RESULTS:")
    print(f"  Overall Accuracy: {ensemble_acc:.1%}")
    print(f"  High Confidence Accuracy: {high_conf_acc:.1%} ({high_conf_pct:.1f}% of signals)")
    print(f"{'='*40}")

    # Feature importance from XGBoost
    print("\nTop 15 Most Important Features:")
    importance = pd.Series(
        model_results['xgb']['model'].feature_importances_,
        index=available_features
    ).sort_values(ascending=False)

    for i, (feat, imp) in enumerate(importance.head(15).items()):
        new_flag = " [NEW]" if feat in NEW_FEATURES else ""
        print(f"  {i+1:2d}. {feat}: {imp:.4f}{new_flag}")

    # Count new features in top 20
    top_20_features = importance.head(20).index.tolist()
    new_in_top_20 = sum(1 for f in top_20_features if f in NEW_FEATURES)
    print(f"\nNew features in top 20: {new_in_top_20}/20")

    # Save model
    model_data = {
        'version': 'enhanced_v3',
        'ticker': ticker,
        'models': {name: data['model'] for name, data in model_results.items()},
        'scaler': scaler,
        'features': available_features,
        'weights': weights,
        'metrics': {
            'ensemble_accuracy': ensemble_acc,
            'high_conf_accuracy': high_conf_acc,
            'high_conf_pct': high_conf_pct,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_period': f"{train_data.index.min().strftime('%Y-%m-%d')} to {train_data.index.max().strftime('%Y-%m-%d')}",
            'test_period': f"{test_data.index.min().strftime('%Y-%m-%d')} to {test_data.index.max().strftime('%Y-%m-%d')}",
            'feature_importance': importance.to_dict()
        },
        'new_features_used': [f for f in available_features if f in NEW_FEATURES],
        'trained_at': datetime.now().isoformat()
    }

    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_enhanced_v3_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved to: {model_path}")

    return model_data


def compare_with_existing(ticker: str, new_model_data: dict):
    """Compare new model with existing improved_v2 model"""
    print(f"\n{'='*60}")
    print(f"COMPARISON: Enhanced v3 vs Improved v2 ({ticker})")
    print(f"{'='*60}")

    # Load existing model
    existing_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_improved_v2_model.pkl')
    if not os.path.exists(existing_path):
        print("  Existing model not found, skipping comparison")
        return

    with open(existing_path, 'rb') as f:
        existing_model = pickle.load(f)

    # Compare metrics
    existing_acc = existing_model.get('metrics', {}).get('test_accuracy', 0)
    existing_hc = existing_model.get('metrics', {}).get('high_conf_win_rate', 0)

    new_acc = new_model_data['metrics']['ensemble_accuracy']
    new_hc = new_model_data['metrics']['high_conf_accuracy']

    print(f"\n  {'Metric':<25} {'Improved v2':>12} {'Enhanced v3':>12} {'Change':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Overall Accuracy':<25} {existing_acc:>11.1%} {new_acc:>11.1%} {(new_acc-existing_acc)*100:>+9.1f}%")
    print(f"  {'High Conf Accuracy':<25} {existing_hc:>11.1%} {new_hc:>11.1%} {(new_hc-existing_hc)*100:>+9.1f}%")
    print(f"  {'Features Used':<25} {len(existing_model.get('features', [])):>12} {len(new_model_data['features']):>12}")
    print(f"  {'New Features':<25} {'0':>12} {len(new_model_data['new_features_used']):>12}")


def main():
    """Train enhanced v3 models for all tickers"""
    print("="*60)
    print("ENHANCED v3 MODEL TRAINING")
    print("Using 80 new features from Phase 1-3 implementation")
    print("="*60)

    results = {}

    for ticker in ['SPY', 'QQQ', 'IWM']:
        try:
            model_data = train_enhanced_model(ticker)
            if model_data:
                results[ticker] = model_data
                compare_with_existing(ticker, model_data)
        except Exception as e:
            print(f"ERROR training {ticker}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    for ticker, data in results.items():
        metrics = data['metrics']
        print(f"\n{ticker}:")
        print(f"  Accuracy: {metrics['ensemble_accuracy']:.1%}")
        print(f"  High Conf: {metrics['high_conf_accuracy']:.1%} ({metrics['high_conf_pct']:.1f}%)")
        print(f"  New features used: {len(data['new_features_used'])}")

    print("\nModels saved to: ml/models/*_enhanced_v3_model.pkl")
    print("To use: Set MODEL_VERSION='enhanced_v3' in predict_server.py")


if __name__ == '__main__':
    main()
