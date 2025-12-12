"""
Feature Optimizer for Daily Direction Prediction

Tests different feature combinations to find optimal set:
1. Individual feature importance (SHAP-like analysis)
2. Feature group ablation study
3. Forward feature selection
4. Backward feature elimination
5. Correlation-based redundancy removal

Goal: Find minimal feature set with maximum accuracy
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
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


def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ALL possible features for optimization testing"""

    # ========== PRICE ACTION ==========
    df['daily_return'] = df['close'].pct_change() * 100
    df['prev_return'] = df['daily_return'].shift(1)
    df['prev_2_return'] = df['daily_return'].shift(2)
    df['prev_3_return'] = df['daily_return'].shift(3)
    df['prev_5_return'] = df['daily_return'].shift(5)
    df['prev_10_return'] = df['daily_return'].shift(10)

    # ========== MOMENTUM ==========
    df['momentum_3d'] = df['daily_return'].rolling(3).sum().shift(1)
    df['momentum_5d'] = df['daily_return'].rolling(5).sum().shift(1)
    df['momentum_10d'] = df['daily_return'].rolling(10).sum().shift(1)
    df['momentum_20d'] = df['daily_return'].rolling(20).sum().shift(1)

    # Rate of Change (ROC)
    df['roc_3'] = ((df['close'] - df['close'].shift(3)) / df['close'].shift(3) * 100).shift(1)
    df['roc_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100).shift(1)
    df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100).shift(1)
    df['roc_20'] = ((df['close'] - df['close'].shift(20)) / df['close'].shift(20) * 100).shift(1)

    # ========== VOLATILITY ==========
    df['volatility_3d'] = df['daily_return'].rolling(3).std().shift(1)
    df['volatility_5d'] = df['daily_return'].rolling(5).std().shift(1)
    df['volatility_10d'] = df['daily_return'].rolling(10).std().shift(1)
    df['volatility_20d'] = df['daily_return'].rolling(20).std().shift(1)
    df['vol_ratio_5_20'] = (df['volatility_5d'] / df['volatility_20d']).shift(1)
    df['vol_ratio_3_10'] = (df['volatility_3d'] / df['volatility_10d']).shift(1)

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_5'] = tr.rolling(5).mean()
    df['atr_10'] = tr.rolling(10).mean()
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_20'] = tr.rolling(20).mean()
    df['prev_atr_pct'] = (df['atr_14'].shift(1) / df['close'].shift(1)) * 100
    df['atr_ratio_5_14'] = (df['atr_5'] / df['atr_14']).shift(1)
    df['atr_ratio_5_20'] = (df['atr_5'] / df['atr_20']).shift(1)
    df['atr_percentile'] = df['atr_14'].rolling(50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    ).shift(1)

    # ========== RANGE ANALYSIS ==========
    df['daily_range'] = ((df['high'] - df['low']) / df['close']) * 100
    df['prev_range'] = df['daily_range'].shift(1)
    df['avg_range_3d'] = df['daily_range'].rolling(3).mean().shift(1)
    df['avg_range_5d'] = df['daily_range'].rolling(5).mean().shift(1)
    df['avg_range_10d'] = df['daily_range'].rolling(10).mean().shift(1)
    df['avg_range_20d'] = df['daily_range'].rolling(20).mean().shift(1)
    df['range_expansion'] = (df['prev_range'] / df['avg_range_20d'])
    df['range_contraction'] = (df['prev_range'] < df['avg_range_10d']).astype(int)

    # Price position within range
    df['close_position'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 0.001)).shift(1)
    df['open_position'] = ((df['open'] - df['low']) / (df['high'] - df['low'] + 0.001)).shift(1)

    # Gap analysis
    # IMPORTANT: Use DECIMAL gap return for consistency with serving (`ml/predict_server.py`).
    # Example: +0.25% gap => gap = 0.0025
    df['gap'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1))
    df['gap_abs'] = abs(df['gap'])
    # 0.1% thresholds expressed as decimals
    df['gap_up'] = (df['gap'] > 0.001).astype(int)
    df['gap_down'] = (df['gap'] < -0.001).astype(int)

    # ========== CONSECUTIVE DAYS (Mean Reversion) ==========
    df['up_day'] = (df['close'] > df['open']).astype(int)
    df['down_day'] = (df['close'] < df['open']).astype(int)
    df['up_close'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['down_close'] = (df['close'] < df['close'].shift(1)).astype(int)

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
    df['consec_up_close'] = count_consecutive(df['up_close'].values)
    df['consec_down_close'] = count_consecutive(df['down_close'].values)

    df['prev_consec_up'] = pd.Series(df['consec_up'].values, index=df.index).shift(1)
    df['prev_consec_down'] = pd.Series(df['consec_down'].values, index=df.index).shift(1)
    df['streak'] = df['prev_consec_up'] - df['prev_consec_down']
    df['streak_abs'] = abs(df['streak'])

    # ========== RSI ==========
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 0.001)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    gain_7 = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss_7 = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    df['rsi_7'] = 100 - (100 / (1 + gain_7 / (loss_7 + 0.001)))

    df['prev_rsi'] = df['rsi_14'].shift(1)
    df['prev_rsi_7'] = df['rsi_7'].shift(1)
    df['rsi_oversold'] = (df['prev_rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['prev_rsi'] > 70).astype(int)
    df['rsi_extreme'] = ((df['prev_rsi'] < 25) | (df['prev_rsi'] > 75)).astype(int)
    df['rsi_momentum'] = df['prev_rsi'] - df['rsi_14'].shift(2)
    df['rsi_slope'] = df['rsi_14'].diff().shift(1)

    # ========== STOCHASTIC ==========
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = ((df['close'] - low_14) / (high_14 - low_14 + 0.001) * 100).shift(1)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['stoch_crossover'] = (df['stoch_k'] > df['stoch_d']).astype(int) - (df['stoch_k'] < df['stoch_d']).astype(int)
    df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
    df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)

    # ========== WILLIAMS %R ==========
    df['williams_r'] = ((high_14 - df['close']) / (high_14 - low_14 + 0.001) * -100).shift(1)
    df['williams_oversold'] = (df['williams_r'] < -80).astype(int)
    df['williams_overbought'] = (df['williams_r'] > -20).astype(int)

    # ========== CCI ==========
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(20).mean()
    mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = ((typical_price - sma_tp) / (0.015 * mad)).shift(1)
    df['cci_oversold'] = (df['cci'] < -100).astype(int)
    df['cci_overbought'] = (df['cci'] > 100).astype(int)

    # ========== MOVING AVERAGES ==========
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_5'] = df['close'].ewm(span=5).mean()
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    df['price_vs_sma5'] = ((df['close'].shift(1) - df['sma_5'].shift(1)) / df['sma_5'].shift(1)) * 100
    df['price_vs_sma10'] = ((df['close'].shift(1) - df['sma_10'].shift(1)) / df['sma_10'].shift(1)) * 100
    df['price_vs_sma20'] = ((df['close'].shift(1) - df['sma_20'].shift(1)) / df['sma_20'].shift(1)) * 100
    df['price_vs_sma50'] = ((df['close'].shift(1) - df['sma_50'].shift(1)) / df['sma_50'].shift(1)) * 100
    df['price_vs_ema5'] = ((df['close'].shift(1) - df['ema_5'].shift(1)) / df['ema_5'].shift(1)) * 100
    df['price_vs_ema9'] = ((df['close'].shift(1) - df['ema_9'].shift(1)) / df['ema_9'].shift(1)) * 100
    df['price_vs_ema21'] = ((df['close'].shift(1) - df['ema_21'].shift(1)) / df['ema_21'].shift(1)) * 100

    df['sma5_vs_sma10'] = ((df['sma_5'].shift(1) - df['sma_10'].shift(1)) / df['sma_10'].shift(1)) * 100
    df['sma5_vs_sma20'] = ((df['sma_5'].shift(1) - df['sma_20'].shift(1)) / df['sma_20'].shift(1)) * 100
    df['sma10_vs_sma20'] = ((df['sma_10'].shift(1) - df['sma_20'].shift(1)) / df['sma_20'].shift(1)) * 100
    df['sma10_vs_sma50'] = ((df['sma_10'].shift(1) - df['sma_50'].shift(1)) / df['sma_50'].shift(1)) * 100
    df['sma20_vs_sma50'] = ((df['sma_20'].shift(1) - df['sma_50'].shift(1)) / df['sma_50'].shift(1)) * 100
    df['ema9_vs_ema21'] = ((df['ema_9'].shift(1) - df['ema_21'].shift(1)) / df['ema_21'].shift(1)) * 100
    df['ema12_vs_ema26'] = ((df['ema_12'].shift(1) - df['ema_26'].shift(1)) / df['ema_26'].shift(1)) * 100

    # MA crossover signals
    df['golden_cross'] = ((df['sma_5'] > df['sma_20']) & (df['sma_5'].shift(1) <= df['sma_20'].shift(1))).astype(int).shift(1)
    df['death_cross'] = ((df['sma_5'] < df['sma_20']) & (df['sma_5'].shift(1) >= df['sma_20'].shift(1))).astype(int).shift(1)

    # Trend alignment
    df['trend_alignment'] = (
        (df['close'].shift(1) > df['sma_5'].shift(1)).astype(int) +
        (df['close'].shift(1) > df['sma_20'].shift(1)).astype(int) +
        (df['close'].shift(1) > df['sma_50'].shift(1)).astype(int) +
        (df['sma_5'].shift(1) > df['sma_20'].shift(1)).astype(int) +
        (df['sma_20'].shift(1) > df['sma_50'].shift(1)).astype(int)
    ) / 5

    df['bullish_alignment'] = (df['trend_alignment'] >= 0.8).astype(int)
    df['bearish_alignment'] = (df['trend_alignment'] <= 0.2).astype(int)

    # ========== BOLLINGER BANDS ==========
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100).shift(1)
    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.001)).shift(1)
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.2)).astype(int).shift(1)
    df['bb_breakout_up'] = (df['close'].shift(1) > df['bb_upper'].shift(1)).astype(int)
    df['bb_breakout_down'] = (df['close'].shift(1) < df['bb_lower'].shift(1)).astype(int)

    # ========== MACD ==========
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['prev_macd'] = df['macd'].shift(1)
    df['prev_macd_signal'] = df['macd_signal'].shift(1)
    df['prev_macd_hist'] = df['macd_histogram'].shift(1)
    df['macd_positive'] = (df['prev_macd'] > 0).astype(int)
    df['macd_above_signal'] = (df['prev_macd'] > df['prev_macd_signal']).astype(int)
    df['macd_crossover'] = (
        (df['macd'].shift(1) > df['macd_signal'].shift(1)).astype(int) -
        (df['macd'].shift(2) > df['macd_signal'].shift(2)).astype(int)
    )
    df['macd_divergence'] = df['macd_histogram'].shift(1) - df['macd_histogram'].shift(2)
    df['macd_acceleration'] = df['macd_divergence'] - (df['macd_histogram'].shift(2) - df['macd_histogram'].shift(3))

    # ========== ADX (Trend Strength) ==========
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr_14_adx = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14_adx)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14_adx)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
    df['adx'] = dx.rolling(14).mean().shift(1)
    df['plus_di'] = plus_di.shift(1)
    df['minus_di'] = minus_di.shift(1)
    df['di_diff'] = (plus_di - minus_di).shift(1)
    df['adx_strong'] = (df['adx'] > 25).astype(int)
    df['adx_weak'] = (df['adx'] < 20).astype(int)

    # ========== VOLUME ==========
    df['volume_sma_5'] = df['volume'].rolling(5).mean()
    df['volume_sma_10'] = df['volume'].rolling(10).mean()
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['prev_volume_ratio'] = (df['volume'] / df['volume_sma_20']).shift(1)
    df['volume_ratio_5'] = (df['volume'] / df['volume_sma_5']).shift(1)
    df['volume_trend'] = (df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()).shift(1)
    df['volume_spike'] = (df['prev_volume_ratio'] > 1.5).astype(int)
    df['volume_dry'] = (df['prev_volume_ratio'] < 0.5).astype(int)
    df['volume_price_trend'] = (df['daily_return'] * df['prev_volume_ratio']).shift(1)
    df['obv'] = (df['volume'] * np.sign(df['daily_return'])).cumsum()
    df['obv_slope'] = df['obv'].diff(5).shift(1)

    # ========== CALENDAR ==========
    df['day_of_week'] = df.index.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_tuesday'] = (df['day_of_week'] == 1).astype(int)
    df['is_wednesday'] = (df['day_of_week'] == 2).astype(int)
    df['is_thursday'] = (df['day_of_week'] == 3).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)

    # ========== TARGET ==========
    df['target'] = (df['daily_return'] > 0).astype(int)

    return df


# Define feature groups for ablation study
FEATURE_GROUPS = {
    'price_action': ['prev_return', 'prev_2_return', 'prev_3_return', 'prev_5_return'],
    'momentum': ['momentum_3d', 'momentum_5d', 'momentum_10d', 'roc_5', 'roc_10', 'roc_20'],
    'volatility': ['volatility_5d', 'volatility_10d', 'volatility_20d', 'vol_ratio_5_20', 'prev_atr_pct', 'atr_ratio_5_14', 'atr_percentile'],
    'range': ['prev_range', 'avg_range_5d', 'range_expansion', 'close_position', 'gap'],
    'mean_reversion': ['prev_consec_up', 'prev_consec_down', 'streak', 'streak_abs'],
    'rsi': ['prev_rsi', 'rsi_oversold', 'rsi_overbought', 'rsi_momentum'],
    'stochastic': ['stoch_k', 'stoch_d', 'stoch_crossover', 'stoch_oversold', 'stoch_overbought'],
    'williams': ['williams_r', 'williams_oversold', 'williams_overbought'],
    'cci': ['cci', 'cci_oversold', 'cci_overbought'],
    'ma_price': ['price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20', 'price_vs_sma50', 'price_vs_ema9', 'price_vs_ema21'],
    'ma_cross': ['sma5_vs_sma20', 'sma10_vs_sma50', 'ema9_vs_ema21', 'trend_alignment'],
    'bollinger': ['bb_width', 'bb_position', 'bb_squeeze'],
    'macd': ['prev_macd_hist', 'macd_crossover', 'macd_divergence', 'macd_above_signal'],
    'adx': ['adx', 'di_diff', 'adx_strong'],
    'volume': ['prev_volume_ratio', 'volume_trend', 'volume_spike', 'volume_price_trend'],
    'calendar': ['day_of_week', 'is_monday', 'is_friday'],
}

# All features flat list
ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]


def evaluate_features(X_train, y_train, X_test, y_test, feature_cols):
    """Evaluate a feature set using XGBoost"""
    if len(feature_cols) == 0:
        return 0.0

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[feature_cols])
    X_test_scaled = scaler.transform(X_test[feature_cols])

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test)

    # Also get predictions for more metrics
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # High confidence accuracy
    high_conf_mask = (y_pred_proba >= 0.65) | (y_pred_proba <= 0.35)
    if high_conf_mask.sum() > 5:
        y_pred = (y_pred_proba >= 0.5).astype(int)
        high_conf_acc = (y_pred[high_conf_mask] == y_test.values[high_conf_mask]).mean()
    else:
        high_conf_acc = accuracy

    return accuracy, high_conf_acc, model


def run_feature_optimization(ticker: str = 'SPY'):
    """Run comprehensive feature optimization"""

    print(f"\n{'='*70}")
    print(f"   FEATURE OPTIMIZATION FOR {ticker}")
    print('='*70)

    # Fetch and prepare data
    print("\nFetching data...")
    df = fetch_polygon_data(ticker, days=800)
    df = calculate_all_features(df)

    # Get available features (those that exist in df)
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    print(f"Total features available: {len(available_features)}")

    # Clean data
    df_clean = df.dropna(subset=available_features + ['target'])
    print(f"Samples after cleaning: {len(df_clean)}")

    # Split data
    test_size = 60
    train_df = df_clean.iloc[:-test_size]
    test_df = df_clean.iloc[-test_size:]

    X_train = train_df[available_features]
    y_train = train_df['target']
    X_test = test_df[available_features]
    y_test = test_df['target']

    print(f"Training: {len(X_train)} samples, Test: {len(X_test)} samples")

    results = {}

    # ========== 1. BASELINE: ALL FEATURES ==========
    print("\n" + "-"*50)
    print("1. BASELINE (All Features)")
    print("-"*50)

    acc, hc_acc, _ = evaluate_features(X_train, y_train, X_test, y_test, available_features)
    results['all_features'] = {'accuracy': acc, 'high_conf': hc_acc, 'n_features': len(available_features)}
    print(f"   Accuracy: {acc:.1%} | High-Conf: {hc_acc:.1%} | Features: {len(available_features)}")

    # ========== 2. FEATURE GROUP ANALYSIS ==========
    print("\n" + "-"*50)
    print("2. FEATURE GROUP ANALYSIS")
    print("-"*50)

    group_results = {}
    for group_name, features in FEATURE_GROUPS.items():
        valid_features = [f for f in features if f in available_features]
        if valid_features:
            acc, hc_acc, _ = evaluate_features(X_train, y_train, X_test, y_test, valid_features)
            group_results[group_name] = {'accuracy': acc, 'high_conf': hc_acc, 'n_features': len(valid_features)}
            print(f"   {group_name:15} -> {acc:.1%} ({len(valid_features)} features)")

    results['groups'] = group_results

    # ========== 3. ABLATION STUDY (Remove each group) ==========
    print("\n" + "-"*50)
    print("3. ABLATION STUDY (Impact of removing each group)")
    print("-"*50)

    ablation_results = {}
    baseline_acc = results['all_features']['accuracy']

    for group_name, features in FEATURE_GROUPS.items():
        remaining = [f for f in available_features if f not in features]
        if remaining:
            acc, hc_acc, _ = evaluate_features(X_train, y_train, X_test, y_test, remaining)
            impact = baseline_acc - acc
            ablation_results[group_name] = {'accuracy': acc, 'impact': impact}
            symbol = "↓" if impact > 0 else "↑"
            print(f"   Without {group_name:15} -> {acc:.1%} ({symbol}{abs(impact)*100:.1f}%)")

    results['ablation'] = ablation_results

    # ========== 4. FORWARD SELECTION ==========
    print("\n" + "-"*50)
    print("4. FORWARD FEATURE SELECTION (Top groups)")
    print("-"*50)

    # Sort groups by individual performance
    sorted_groups = sorted(group_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    selected_features = []
    best_acc = 0
    forward_results = []

    for group_name, _ in sorted_groups:
        test_features = selected_features + [f for f in FEATURE_GROUPS[group_name] if f in available_features]
        acc, hc_acc, _ = evaluate_features(X_train, y_train, X_test, y_test, test_features)

        if acc >= best_acc:
            selected_features = test_features
            best_acc = acc
            forward_results.append({'group': group_name, 'accuracy': acc, 'n_features': len(test_features)})
            print(f"   + {group_name:15} -> {acc:.1%} ({len(test_features)} features)")

    results['forward_selection'] = forward_results

    # ========== 5. INDIVIDUAL FEATURE IMPORTANCE ==========
    print("\n" + "-"*50)
    print("5. TOP INDIVIDUAL FEATURES")
    print("-"*50)

    # Train full model to get importances
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
    model.fit(X_train_scaled, y_train)

    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n   Top 20 features:")
    for i, row in importance_df.head(20).iterrows():
        print(f"   {row['feature']:25} {row['importance']:.4f}")

    results['feature_importance'] = importance_df.to_dict('records')

    # ========== 6. OPTIMAL FEATURE SET ==========
    print("\n" + "-"*50)
    print("6. OPTIMAL FEATURE SET SEARCH")
    print("-"*50)

    # Try top N features
    best_config = None
    best_score = 0

    for n in [10, 15, 20, 25, 30, 35, 40]:
        top_features = importance_df.head(n)['feature'].tolist()
        acc, hc_acc, _ = evaluate_features(X_train, y_train, X_test, y_test, top_features)

        # Score combines accuracy and high-conf accuracy
        score = acc * 0.4 + hc_acc * 0.6

        print(f"   Top {n:2} features -> Acc: {acc:.1%}, High-Conf: {hc_acc:.1%}, Score: {score:.3f}")

        if score > best_score:
            best_score = score
            best_config = {
                'n_features': n,
                'features': top_features,
                'accuracy': acc,
                'high_conf_accuracy': hc_acc,
                'score': score
            }

    results['optimal'] = best_config

    print(f"\n   OPTIMAL: Top {best_config['n_features']} features")
    print(f"   Accuracy: {best_config['accuracy']:.1%}")
    print(f"   High-Conf Accuracy: {best_config['high_conf_accuracy']:.1%}")

    # ========== 7. CORRELATION-BASED PRUNING ==========
    print("\n" + "-"*50)
    print("7. CORRELATION-BASED FEATURE PRUNING")
    print("-"*50)

    # Get correlation matrix for top features
    top_features = importance_df.head(30)['feature'].tolist()
    corr_matrix = X_train[top_features].corr().abs()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            if corr_matrix.iloc[i, j] > 0.8:
                high_corr_pairs.append((top_features[i], top_features[j], corr_matrix.iloc[i, j]))

    if high_corr_pairs:
        print("   Highly correlated pairs (>0.8):")
        for f1, f2, corr in high_corr_pairs[:10]:
            print(f"   {f1:25} <-> {f2:25} ({corr:.2f})")

    # Remove lower-importance feature from each correlated pair
    features_to_remove = set()
    feature_ranks = {f: i for i, f in enumerate(top_features)}

    for f1, f2, _ in high_corr_pairs:
        # Keep the one with higher importance (lower rank)
        if feature_ranks[f1] < feature_ranks[f2]:
            features_to_remove.add(f2)
        else:
            features_to_remove.add(f1)

    pruned_features = [f for f in top_features if f not in features_to_remove]
    print(f"\n   Pruned {len(features_to_remove)} correlated features")

    acc, hc_acc, _ = evaluate_features(X_train, y_train, X_test, y_test, pruned_features)
    print(f"   After pruning: {acc:.1%} accuracy ({len(pruned_features)} features)")

    results['pruned'] = {
        'features': pruned_features,
        'accuracy': acc,
        'high_conf_accuracy': hc_acc,
        'n_features': len(pruned_features)
    }

    # ========== FINAL RECOMMENDATION ==========
    print("\n" + "="*70)
    print("   FINAL RECOMMENDATION")
    print("="*70)

    # Compare all approaches
    approaches = [
        ('All Features', results['all_features']['accuracy'], results['all_features']['high_conf'], results['all_features']['n_features']),
        ('Optimal Top-N', best_config['accuracy'], best_config['high_conf_accuracy'], best_config['n_features']),
        ('Correlation Pruned', results['pruned']['accuracy'], results['pruned']['high_conf_accuracy'], results['pruned']['n_features']),
    ]

    print(f"\n   {'Approach':<25} {'Accuracy':<12} {'High-Conf':<12} {'Features':<10}")
    print(f"   {'-'*55}")
    for name, acc, hc, n in approaches:
        print(f"   {name:<25} {acc:.1%}{'':<6} {hc:.1%}{'':<6} {n}")

    # Best recommendation
    best_approach = max(approaches, key=lambda x: x[1] * 0.4 + x[2] * 0.6)
    print(f"\n   RECOMMENDED: {best_approach[0]}")

    # Save optimal features
    optimal_features = results['pruned']['features'] if results['pruned']['accuracy'] >= best_config['accuracy'] else best_config['features']

    results['recommended_features'] = optimal_features

    print(f"\n   Optimal Feature Set ({len(optimal_features)} features):")
    for f in optimal_features:
        print(f"      - {f}")

    return results


def run_all_tickers():
    """Run optimization for all tickers and find common optimal features"""

    all_results = {}
    feature_votes = {}

    for ticker in ['SPY', 'QQQ', 'IWM']:
        results = run_feature_optimization(ticker)
        all_results[ticker] = results

        # Vote for features
        for f in results['recommended_features']:
            feature_votes[f] = feature_votes.get(f, 0) + 1

    # Find features that work well for all tickers
    print("\n" + "="*70)
    print("   CROSS-TICKER FEATURE CONSENSUS")
    print("="*70)

    consensus_features = [f for f, votes in feature_votes.items() if votes >= 2]
    print(f"\nFeatures selected for 2+ tickers ({len(consensus_features)}):")
    for f in sorted(consensus_features):
        print(f"   - {f} ({feature_votes[f]}/3 tickers)")

    return all_results, consensus_features


if __name__ == '__main__':
    # Run for all tickers
    all_results, consensus = run_all_tickers()
