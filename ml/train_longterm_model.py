"""
Long-Term Daily Direction Prediction Model

Training Period: January 2000 - January 1, 2025 (~25 years)
Testing Period: January 1, 2025 - December 8, 2025 (~11 months)

This provides a proper out-of-sample test on 2025 data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
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


def fetch_polygon_data_range(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical daily data from Polygon.io for a specific date range"""

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


def calculate_optimized_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate enhanced features for daily prediction"""

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

    # Rate of Change (ROC)
    df['roc_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100).shift(1)
    df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100).shift(1)
    df['roc_20'] = ((df['close'] - df['close'].shift(20)) / df['close'].shift(20) * 100).shift(1)

    # ========== VOLATILITY ==========
    df['volatility_5d'] = df['daily_return'].rolling(5).std().shift(1)
    df['volatility_10d'] = df['daily_return'].rolling(10).std().shift(1)
    df['volatility_20d'] = df['daily_return'].rolling(20).std().shift(1)

    # Volatility ratio (current vs historical)
    df['vol_ratio_5_20'] = (df['volatility_5d'] / df['volatility_20d']).shift(1)

    # ATR-based volatility regime
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_5'] = tr.rolling(5).mean()
    df['prev_atr_pct'] = (df['atr_14'].shift(1) / df['close'].shift(1)) * 100

    # ATR expansion/contraction (VIX-like)
    df['atr_ratio'] = (df['atr_5'] / df['atr_14']).shift(1)
    df['atr_percentile'] = df['atr_14'].rolling(50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    ).shift(1)

    # ========== RANGE ANALYSIS ==========
    df['daily_range'] = ((df['high'] - df['low']) / df['close']) * 100
    df['prev_range'] = df['daily_range'].shift(1)
    df['avg_range_5d'] = df['daily_range'].rolling(5).mean().shift(1)
    df['avg_range_20d'] = df['daily_range'].rolling(20).mean().shift(1)
    df['range_expansion'] = (df['prev_range'] / df['avg_range_20d'])

    # Price position within day's range
    df['close_position'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 0.001)).shift(1)

    # Gap analysis
    df['gap'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) * 100
    df['gap_filled'] = ((df['close'].shift(1) >= df['low']) & (df['close'].shift(1) <= df['high'])).astype(int).shift(1)

    # ========== CONSECUTIVE DAYS (Mean Reversion) ==========
    df['up_day'] = (df['close'] > df['open']).astype(int)
    df['down_day'] = (df['close'] < df['open']).astype(int)

    # Count consecutive up/down days
    def count_consecutive(series):
        result = []
        count = 0
        prev_val = None
        for val in series:
            if val == prev_val and val == 1:
                count += 1
            else:
                count = 1 if val == 1 else 0
            result.append(count)
            prev_val = val
        return result

    df['consec_up'] = count_consecutive(df['up_day'].values)
    df['consec_down'] = count_consecutive(df['down_day'].values)
    df['prev_consec_up'] = pd.Series(df['consec_up'].values, index=df.index).shift(1)
    df['prev_consec_down'] = pd.Series(df['consec_down'].values, index=df.index).shift(1)

    # Net streak (positive = up streak, negative = down streak)
    df['streak'] = df['prev_consec_up'] - df['prev_consec_down']

    # ========== RSI ==========
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 0.001)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['prev_rsi'] = df['rsi_14'].shift(1)

    # RSI zones
    df['rsi_oversold'] = (df['prev_rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['prev_rsi'] > 70).astype(int)
    df['rsi_momentum'] = df['prev_rsi'] - df['rsi_14'].shift(2)

    # ========== STOCHASTIC ==========
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = ((df['close'] - low_14) / (high_14 - low_14 + 0.001) * 100).shift(1)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['stoch_crossover'] = (df['stoch_k'] > df['stoch_d']).astype(int) - (df['stoch_k'] < df['stoch_d']).astype(int)

    # ========== WILLIAMS %R ==========
    df['williams_r'] = ((high_14 - df['close']) / (high_14 - low_14 + 0.001) * -100).shift(1)

    # ========== MOVING AVERAGES ==========
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()

    # Price vs MAs
    df['price_vs_sma5'] = ((df['close'].shift(1) - df['sma_5'].shift(1)) / df['sma_5'].shift(1)) * 100
    df['price_vs_sma10'] = ((df['close'].shift(1) - df['sma_10'].shift(1)) / df['sma_10'].shift(1)) * 100
    df['price_vs_sma20'] = ((df['close'].shift(1) - df['sma_20'].shift(1)) / df['sma_20'].shift(1)) * 100
    df['price_vs_sma50'] = ((df['close'].shift(1) - df['sma_50'].shift(1)) / df['sma_50'].shift(1)) * 100
    df['price_vs_ema9'] = ((df['close'].shift(1) - df['ema_9'].shift(1)) / df['ema_9'].shift(1)) * 100
    df['price_vs_ema21'] = ((df['close'].shift(1) - df['ema_21'].shift(1)) / df['ema_21'].shift(1)) * 100

    # MA crossovers
    df['sma5_vs_sma20'] = ((df['sma_5'].shift(1) - df['sma_20'].shift(1)) / df['sma_20'].shift(1)) * 100
    df['sma10_vs_sma50'] = ((df['sma_10'].shift(1) - df['sma_50'].shift(1)) / df['sma_50'].shift(1)) * 100
    df['ema9_vs_ema21'] = ((df['ema_9'].shift(1) - df['ema_21'].shift(1)) / df['ema_21'].shift(1)) * 100

    # Trend alignment score
    df['trend_alignment'] = (
        (df['close'].shift(1) > df['sma_5'].shift(1)).astype(int) +
        (df['close'].shift(1) > df['sma_20'].shift(1)).astype(int) +
        (df['close'].shift(1) > df['sma_50'].shift(1)).astype(int) +
        (df['sma_5'].shift(1) > df['sma_20'].shift(1)).astype(int) +
        (df['sma_20'].shift(1) > df['sma_50'].shift(1)).astype(int)
    ) / 5

    # ========== BOLLINGER BANDS ==========
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100).shift(1)
    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.001)).shift(1)
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.2)).astype(int).shift(1)

    # ========== MACD ==========
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['prev_macd_hist'] = df['macd_histogram'].shift(1)
    df['macd_crossover'] = (
        (df['macd'].shift(1) > df['macd_signal'].shift(1)).astype(int) -
        (df['macd'].shift(2) > df['macd_signal'].shift(2)).astype(int)
    )
    df['macd_divergence'] = df['macd_histogram'].shift(1) - df['macd_histogram'].shift(2)

    # ========== ADX (Trend Strength) ==========
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr_14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
    df['adx'] = dx.rolling(14).mean().shift(1)
    df['plus_di'] = plus_di.shift(1)
    df['minus_di'] = minus_di.shift(1)
    df['di_diff'] = (df['plus_di'] - df['minus_di'])

    # ========== VOLUME ==========
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['prev_volume_ratio'] = (df['volume'] / df['volume_sma_20']).shift(1)
    df['volume_trend'] = (df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()).shift(1)

    # Volume-price relationship
    df['volume_price_trend'] = (df['daily_return'] * df['prev_volume_ratio']).shift(1)

    # ========== CALENDAR ==========
    df['day_of_week'] = df.index.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    df['is_month_end'] = (df.index.is_month_end).astype(int)
    df['is_month_start'] = (df.index.is_month_start).astype(int)

    # ========== TARGET ==========
    df['target'] = (df['daily_return'] > 0).astype(int)

    return df


def get_feature_columns():
    """Return the list of feature columns to use"""
    return [
        # Price action
        'prev_return', 'prev_2_return', 'prev_3_return', 'prev_5_return',
        # Momentum
        'momentum_3d', 'momentum_5d', 'momentum_10d',
        'roc_5', 'roc_10', 'roc_20',
        # Volatility
        'volatility_5d', 'volatility_10d', 'volatility_20d',
        'vol_ratio_5_20', 'prev_atr_pct', 'atr_ratio', 'atr_percentile',
        # Range
        'prev_range', 'avg_range_5d', 'range_expansion', 'close_position',
        'gap',
        # Mean reversion
        'prev_consec_up', 'prev_consec_down', 'streak',
        # RSI
        'prev_rsi', 'rsi_oversold', 'rsi_overbought', 'rsi_momentum',
        # Stochastic
        'stoch_k', 'stoch_d', 'stoch_crossover',
        # Williams %R
        'williams_r',
        # Moving averages
        'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20', 'price_vs_sma50',
        'price_vs_ema9', 'price_vs_ema21',
        'sma5_vs_sma20', 'sma10_vs_sma50', 'ema9_vs_ema21',
        'trend_alignment',
        # Bollinger
        'bb_width', 'bb_position', 'bb_squeeze',
        # MACD
        'prev_macd_hist', 'macd_crossover', 'macd_divergence',
        # ADX
        'adx', 'di_diff',
        # Volume
        'prev_volume_ratio', 'volume_trend', 'volume_price_trend',
        # Calendar
        'day_of_week', 'is_monday', 'is_friday',
    ]


def train_longterm_model(ticker: str = 'SPY'):
    """Train model on 25 years of data, test on 2025"""

    print(f"\n{'='*60}")
    print(f"Training LONG-TERM Model for {ticker}")
    print(f"Train: {TRAIN_START} to {TRAIN_END}")
    print(f"Test:  {TEST_START} to {TEST_END}")
    print('='*60)

    # Fetch training data (2000-2024)
    print("\nFetching training data (2000-2024)...")
    df_train = fetch_polygon_data_range(ticker, TRAIN_START, TRAIN_END)
    print(f"  Got {len(df_train)} training days")

    # Fetch test data (2025)
    print("Fetching test data (2025)...")
    df_test = fetch_polygon_data_range(ticker, TEST_START, TEST_END)
    print(f"  Got {len(df_test)} test days")

    # Combine for feature calculation (need history for indicators)
    df_all = pd.concat([df_train, df_test])

    # Calculate features
    print("Calculating features...")
    df_all = calculate_optimized_features(df_all)

    feature_cols = get_feature_columns()
    print(f"  Using {len(feature_cols)} features")

    # Split back into train/test
    train_end_date = pd.Timestamp(TRAIN_END)
    test_start_date = pd.Timestamp(TEST_START)

    df_train_clean = df_all[df_all.index <= train_end_date].dropna(subset=feature_cols + ['target'])
    df_test_clean = df_all[df_all.index >= test_start_date].dropna(subset=feature_cols + ['target'])

    print(f"  Training samples: {len(df_train_clean)}")
    print(f"  Test samples: {len(df_test_clean)}")

    X_train = df_train_clean[feature_cols]
    y_train = df_train_clean['target']
    X_test = df_test_clean[feature_cols]
    y_test = df_test_clean['target']

    print(f"\n  Bullish days (train): {y_train.mean():.1%}")
    print(f"  Bullish days (test): {y_test.mean():.1%}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train ensemble
    print("\nTraining ensemble on 25 years of data...")

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb.fit(X_train_scaled, y_train)
    xgb_acc = xgb.score(X_test_scaled, y_test)
    print(f"  XGBoost: {xgb_acc:.1%}")

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    rf_acc = rf.score(X_test_scaled, y_test)
    print(f"  Random Forest: {rf_acc:.1%}")

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.03,
        min_samples_leaf=10,
        random_state=42
    )
    gb.fit(X_train_scaled, y_train)
    gb_acc = gb.score(X_test_scaled, y_test)
    print(f"  Gradient Boosting: {gb_acc:.1%}")

    # Logistic Regression
    lr = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_acc = lr.score(X_test_scaled, y_test)
    print(f"  Logistic Regression: {lr_acc:.1%}")

    # Ensemble weights
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
    print("\n--- HIGH CONFIDENCE SIGNALS (2025 Test) ---")
    for threshold in [0.55, 0.60, 0.65, 0.70, 0.75]:
        high_conf_mask = (y_pred_proba >= threshold) | (y_pred_proba <= (1 - threshold))
        if high_conf_mask.sum() > 0:
            high_conf_acc = (y_pred[high_conf_mask] == y_test.values[high_conf_mask]).mean()
            print(f"  >= {threshold:.0%} conf: {high_conf_mask.sum()} signals, {high_conf_acc:.1%} accuracy")

    # BUY vs SELL breakdown
    print("\n--- BUY vs SELL (2025 Test) ---")
    buy_mask = y_pred_proba >= 0.55
    sell_mask = y_pred_proba <= 0.45

    if buy_mask.sum() > 0:
        buy_acc = (y_test.values[buy_mask] == 1).mean()
        print(f"  BUY signals (>=55%): {buy_mask.sum()} signals, {buy_acc:.1%} accuracy")

    if sell_mask.sum() > 0:
        sell_acc = (y_test.values[sell_mask] == 0).mean()
        print(f"  SELL signals (<=45%): {sell_mask.sum()} signals, {sell_acc:.1%} accuracy")

    # Feature importance
    print("\nTop 15 features (XGBoost):")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    for _, row in importance.head(15).iterrows():
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
        'feature_cols': feature_cols,
        'metrics': {
            'accuracy': float(ensemble_acc),
            'xgb_accuracy': float(xgb_acc),
            'rf_accuracy': float(rf_acc),
            'gb_accuracy': float(gb_acc),
            'lr_accuracy': float(lr_acc),
            'bullish_rate_train': float(y_train.mean()),
            'bullish_rate_test': float(y_test.mean()),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
        },
        'feature_importance': importance.to_dict('records'),
        'ticker': ticker,
        'version': 'longterm_v1',
        'train_period': f'{TRAIN_START} to {TRAIN_END}',
        'test_period': f'{TEST_START} to {TEST_END}',
        'trained_at': datetime.now().isoformat(),
    }

    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_daily_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved to {model_path}")

    return model_data, df_test_clean, y_pred_proba


if __name__ == '__main__':
    print("\n" + "="*70)
    print("   TRAINING LONG-TERM MODELS (25 YEARS)")
    print("="*70)
    print(f"\nTraining Period: {TRAIN_START} to {TRAIN_END}")
    print(f"Testing Period:  {TEST_START} to {TEST_END}")

    results = {}
    for ticker in ['SPY', 'QQQ', 'IWM']:
        try:
            model_data, test_df, probs = train_longterm_model(ticker)
            results[ticker] = {
                'accuracy': model_data['metrics']['accuracy'],
                'xgb': model_data['metrics']['xgb_accuracy'],
                'rf': model_data['metrics']['rf_accuracy'],
                'gb': model_data['metrics']['gb_accuracy'],
                'lr': model_data['metrics']['lr_accuracy'],
                'train_samples': model_data['metrics']['train_samples'],
                'test_samples': model_data['metrics']['test_samples'],
            }
        except Exception as e:
            print(f"Error training {ticker}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("   TRAINING COMPLETE - RESULTS ON 2025 DATA")
    print("="*70)
    print(f"\n{'Ticker':<8} {'Ensemble':<12} {'XGB':<10} {'RF':<10} {'GB':<10} {'LR':<10} {'Train':<8} {'Test':<8}")
    print("-" * 80)
    for ticker, metrics in results.items():
        print(f"{ticker:<8} {metrics['accuracy']:.1%}{'':<7} {metrics['xgb']:.1%}{'':<5} {metrics['rf']:.1%}{'':<5} {metrics['gb']:.1%}{'':<5} {metrics['lr']:.1%}{'':<5} {metrics['train_samples']:<8} {metrics['test_samples']:<8}")
