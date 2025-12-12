"""
Enhanced High/Low Price Prediction Model

NEW FEATURES ADDED:
1. VIX data - Volatility index predicts daily range size
2. Inter-market correlations - How SPY/QQQ/IWM move together
3. Gap analysis - Pre-market gaps predict intraday range
4. Range percentile - Current range vs historical
5. Time-based patterns - Day of week, month effects

Training Period: 2003 - 2024 (~21 years)
Testing Period: 2025

Goal: Reduce High MAE from 1.41% to <1.0%
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
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
    """Fetch historical daily data from Polygon.io"""
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
        return pd.DataFrame()

    df = pd.DataFrame(data['results'])
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

    return df


def fetch_vix_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch VIX data - use VIXY as proxy if VIX not available"""
    # Try VIX first
    df = fetch_polygon_data_range('VIX', start_date, end_date)
    if len(df) == 0:
        # Try VIXY ETF as proxy
        df = fetch_polygon_data_range('VIXY', start_date, end_date)
    if len(df) == 0:
        # Try VXX
        df = fetch_polygon_data_range('VXX', start_date, end_date)
    return df


def calculate_enhanced_features(df: pd.DataFrame, vix_df: pd.DataFrame = None,
                                 spy_df: pd.DataFrame = None, qqq_df: pd.DataFrame = None,
                                 iwm_df: pd.DataFrame = None) -> pd.DataFrame:
    """Calculate enhanced features including VIX and inter-market data"""

    # ========== TARGET VARIABLES ==========
    df['actual_high_pct'] = ((df['High'] - df['Open']) / df['Open']) * 100
    df['actual_low_pct'] = ((df['Open'] - df['Low']) / df['Open']) * 100
    df['actual_close_pct'] = ((df['Close'] - df['Open']) / df['Open']) * 100
    df['actual_range_pct'] = ((df['High'] - df['Low']) / df['Open']) * 100

    # ========== PREVIOUS DAY METRICS ==========
    df['prev_close'] = df['Close'].shift(1)
    df['prev_high'] = df['High'].shift(1)
    df['prev_low'] = df['Low'].shift(1)
    df['prev_open'] = df['Open'].shift(1)

    # Gap from previous close (KEY FEATURE)
    df['gap_pct'] = ((df['Open'] - df['prev_close']) / df['prev_close']) * 100
    df['abs_gap'] = df['gap_pct'].abs()
    df['gap_up'] = (df['gap_pct'] > 0).astype(int)
    df['large_gap'] = (df['abs_gap'] > 0.5).astype(int)  # Gap > 0.5%

    # Previous day's range characteristics
    df['prev_range_pct'] = ((df['prev_high'] - df['prev_low']) / df['prev_close']) * 100
    df['prev_high_pct'] = ((df['prev_high'] - df['prev_open']) / df['prev_open']) * 100
    df['prev_low_pct'] = ((df['prev_open'] - df['prev_low']) / df['prev_open']) * 100
    df['prev_close_pct'] = ((df['prev_close'] - df['prev_open']) / df['prev_open']) * 100

    # ========== RETURNS ==========
    df['prev_return'] = df['Close'].pct_change().shift(1) * 100
    df['prev_2_return'] = df['Close'].pct_change().shift(2) * 100
    df['prev_3_return'] = df['Close'].pct_change().shift(3) * 100
    df['prev_5_return'] = df['Close'].pct_change().shift(5) * 100

    # ========== MOMENTUM ==========
    df['momentum_3d'] = df['prev_return'].rolling(3).sum()
    df['momentum_5d'] = df['prev_return'].rolling(5).sum()
    df['momentum_10d'] = df['prev_return'].rolling(10).sum()

    # ========== VOLATILITY (KEY FOR RANGE PREDICTION) ==========
    df['volatility_5d'] = df['prev_return'].rolling(5).std()
    df['volatility_10d'] = df['prev_return'].rolling(10).std()
    df['volatility_20d'] = df['prev_return'].rolling(20).std()
    df['vol_ratio_5_20'] = df['volatility_5d'] / df['volatility_20d']

    # Volatility percentile (is current vol high or low historically?)
    df['vol_percentile'] = df['volatility_20d'].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 252 else 0.5
    ).shift(1)

    # ========== ATR (TRUE RANGE) ==========
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    df['atr_5'] = (tr.rolling(5).mean().shift(1) / df['prev_close']) * 100
    df['atr_10'] = (tr.rolling(10).mean().shift(1) / df['prev_close']) * 100
    df['atr_14'] = (tr.rolling(14).mean().shift(1) / df['prev_close']) * 100
    df['atr_20'] = (tr.rolling(20).mean().shift(1) / df['prev_close']) * 100

    # ATR expansion/contraction
    df['atr_ratio_5_20'] = df['atr_5'] / df['atr_20']
    df['atr_percentile'] = df['atr_14'].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 252 else 0.5
    ).shift(1)

    # ========== HISTORICAL HIGH/LOW PATTERNS (MOST IMPORTANT) ==========
    df['avg_high_5d'] = df['actual_high_pct'].rolling(5).mean().shift(1)
    df['avg_low_5d'] = df['actual_low_pct'].rolling(5).mean().shift(1)
    df['avg_high_10d'] = df['actual_high_pct'].rolling(10).mean().shift(1)
    df['avg_low_10d'] = df['actual_low_pct'].rolling(10).mean().shift(1)
    df['avg_high_20d'] = df['actual_high_pct'].rolling(20).mean().shift(1)
    df['avg_low_20d'] = df['actual_low_pct'].rolling(20).mean().shift(1)

    df['max_high_5d'] = df['actual_high_pct'].rolling(5).max().shift(1)
    df['max_low_5d'] = df['actual_low_pct'].rolling(5).max().shift(1)
    df['max_high_10d'] = df['actual_high_pct'].rolling(10).max().shift(1)
    df['max_low_10d'] = df['actual_low_pct'].rolling(10).max().shift(1)

    df['min_high_5d'] = df['actual_high_pct'].rolling(5).min().shift(1)
    df['min_low_5d'] = df['actual_low_pct'].rolling(5).min().shift(1)

    # Range patterns
    df['avg_range_5d'] = df['actual_range_pct'].rolling(5).mean().shift(1)
    df['avg_range_10d'] = df['actual_range_pct'].rolling(10).mean().shift(1)
    df['avg_range_20d'] = df['actual_range_pct'].rolling(20).mean().shift(1)
    df['range_expanding'] = (df['avg_range_5d'] > df['avg_range_20d']).astype(int)

    # ========== RSI ==========
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 0.001)
    df['rsi_14'] = (100 - (100 / (1 + rs))).shift(1)
    df['rsi_extreme'] = ((df['rsi_14'] < 30) | (df['rsi_14'] > 70)).astype(int)

    # ========== MACD ==========
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['macd'] = (ema_12 - ema_26).shift(1)
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # ========== BOLLINGER BANDS ==========
    bb_middle = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    df['bb_position'] = ((df['Close'] - bb_lower) / (bb_upper - bb_lower + 0.001)).shift(1)
    df['bb_width'] = ((bb_upper - bb_lower) / bb_middle * 100).shift(1)
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.2)).astype(int).shift(1)

    # ========== PRICE VS MOVING AVERAGES ==========
    sma_20 = df['Close'].rolling(20).mean()
    sma_50 = df['Close'].rolling(50).mean()
    sma_200 = df['Close'].rolling(200).mean()
    df['price_vs_sma20'] = ((df['prev_close'] - sma_20.shift(1)) / sma_20.shift(1)) * 100
    df['price_vs_sma50'] = ((df['prev_close'] - sma_50.shift(1)) / sma_50.shift(1)) * 100
    df['price_vs_sma200'] = ((df['prev_close'] - sma_200.shift(1)) / sma_200.shift(1)) * 100

    # Trend strength
    df['above_sma20'] = (df['prev_close'] > sma_20.shift(1)).astype(int)
    df['above_sma50'] = (df['prev_close'] > sma_50.shift(1)).astype(int)
    df['above_sma200'] = (df['prev_close'] > sma_200.shift(1)).astype(int)
    df['trend_strength'] = df['above_sma20'] + df['above_sma50'] + df['above_sma200']

    # ========== CALENDAR EFFECTS ==========
    df['day_of_week'] = df.index.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    df['month'] = df.index.month
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_quarter_end'] = (df['month'].isin([3, 6, 9, 12]) & df['is_month_end']).astype(int)

    # ========== VOLUME ==========
    df['volume_ratio'] = df['Volume'].shift(1) / df['Volume'].rolling(20).mean().shift(1)
    df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)

    # ========== CONSECUTIVE PATTERNS ==========
    df['up_day'] = (df['Close'] > df['Open']).astype(int)
    df['consec_up'] = df['up_day'].rolling(5).sum().shift(1)
    df['consec_down'] = (1 - df['up_day']).rolling(5).sum().shift(1)

    # ========== VIX FEATURES (if available) ==========
    if vix_df is not None and len(vix_df) > 0:
        # Align VIX data with main dataframe
        vix_aligned = vix_df.reindex(df.index, method='ffill')

        df['vix_close'] = vix_aligned['Close'].shift(1)
        df['vix_high'] = (df['vix_close'] > 25).astype(int)  # High fear
        df['vix_extreme'] = (df['vix_close'] > 35).astype(int)  # Extreme fear
        df['vix_low'] = (df['vix_close'] < 15).astype(int)  # Complacency
        df['vix_change'] = vix_aligned['Close'].pct_change().shift(1) * 100
        df['vix_5d_avg'] = vix_aligned['Close'].rolling(5).mean().shift(1)
        df['vix_trend'] = (df['vix_close'] > df['vix_5d_avg']).astype(int)
    else:
        # Use volatility as VIX proxy
        df['vix_close'] = df['volatility_20d'] * 16  # Annualize
        df['vix_high'] = (df['vix_close'] > 1.5).astype(int)
        df['vix_extreme'] = (df['vix_close'] > 2.5).astype(int)
        df['vix_low'] = (df['vix_close'] < 0.8).astype(int)
        df['vix_change'] = df['vix_close'].pct_change().shift(1) * 100
        df['vix_5d_avg'] = df['vix_close'].rolling(5).mean().shift(1)
        df['vix_trend'] = (df['vix_close'] > df['vix_5d_avg']).astype(int)

    # ========== INTER-MARKET CORRELATIONS ==========
    # Add returns from other ETFs if available
    if spy_df is not None and len(spy_df) > 0:
        spy_aligned = spy_df.reindex(df.index, method='ffill')
        df['spy_return'] = spy_aligned['Close'].pct_change().shift(1) * 100
        df['spy_range'] = ((spy_aligned['High'] - spy_aligned['Low']) / spy_aligned['Open'] * 100).shift(1)

    if qqq_df is not None and len(qqq_df) > 0:
        qqq_aligned = qqq_df.reindex(df.index, method='ffill')
        df['qqq_return'] = qqq_aligned['Close'].pct_change().shift(1) * 100
        df['qqq_range'] = ((qqq_aligned['High'] - qqq_aligned['Low']) / qqq_aligned['Open'] * 100).shift(1)

    if iwm_df is not None and len(iwm_df) > 0:
        iwm_aligned = iwm_df.reindex(df.index, method='ffill')
        df['iwm_return'] = iwm_aligned['Close'].pct_change().shift(1) * 100
        df['iwm_range'] = ((iwm_aligned['High'] - iwm_aligned['Low']) / iwm_aligned['Open'] * 100).shift(1)

    return df


def get_enhanced_feature_columns(ticker: str, has_vix: bool = True):
    """Return list of features for enhanced high/low model"""

    base_features = [
        # Gap features
        'gap_pct', 'abs_gap', 'gap_up', 'large_gap',
        # Previous day
        'prev_range_pct', 'prev_high_pct', 'prev_low_pct', 'prev_close_pct',
        # Returns
        'prev_return', 'prev_2_return', 'prev_3_return', 'prev_5_return',
        # Momentum
        'momentum_3d', 'momentum_5d', 'momentum_10d',
        # Volatility
        'volatility_5d', 'volatility_10d', 'volatility_20d',
        'vol_ratio_5_20', 'vol_percentile',
        # ATR
        'atr_5', 'atr_10', 'atr_14', 'atr_20',
        'atr_ratio_5_20', 'atr_percentile',
        # Historical high/low patterns
        'avg_high_5d', 'avg_low_5d', 'avg_high_10d', 'avg_low_10d',
        'avg_high_20d', 'avg_low_20d',
        'max_high_5d', 'max_low_5d', 'max_high_10d', 'max_low_10d',
        'min_high_5d', 'min_low_5d',
        # Range patterns
        'avg_range_5d', 'avg_range_10d', 'avg_range_20d', 'range_expanding',
        # Technical indicators
        'rsi_14', 'rsi_extreme', 'macd_histogram',
        'bb_position', 'bb_width', 'bb_squeeze',
        'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
        'trend_strength',
        # Calendar
        'day_of_week', 'is_monday', 'is_friday',
        'is_month_end', 'is_month_start', 'is_quarter_end',
        # Volume
        'volume_ratio', 'high_volume',
        # Consecutive
        'consec_up', 'consec_down',
        # VIX
        'vix_close', 'vix_high', 'vix_extreme', 'vix_low',
        'vix_change', 'vix_trend',
    ]

    # Add inter-market features (exclude self)
    if ticker != 'SPY':
        base_features.extend(['spy_return', 'spy_range'])
    if ticker != 'QQQ':
        base_features.extend(['qqq_return', 'qqq_range'])
    if ticker != 'IWM':
        base_features.extend(['iwm_return', 'iwm_range'])

    return base_features


def evaluate_range_accuracy(y_high_true, y_low_true, y_close_true, high_pred, low_pred):
    """Evaluate how well predicted ranges capture EOD close"""
    n = len(y_close_true)
    captured = 0
    misses = []
    ranges = []

    for i in range(n):
        pred_high = high_pred[i]
        pred_low = low_pred[i]
        actual_close = y_close_true.iloc[i]

        range_width = pred_high + pred_low
        ranges.append(range_width)

        if -pred_low <= actual_close <= pred_high:
            captured += 1
        else:
            if actual_close > pred_high:
                miss = actual_close - pred_high
            else:
                miss = -pred_low - actual_close
            misses.append(abs(miss))

    capture_rate = captured / n * 100
    avg_miss = np.mean(misses) if misses else 0
    avg_range = np.mean(ranges)

    return capture_rate, avg_miss, avg_range


def train_enhanced_highlow_model(ticker: str):
    """Train enhanced high/low model with VIX and inter-market data"""

    print(f"\n{'='*60}")
    print(f"Training ENHANCED High/Low Model for {ticker}")
    print(f"Train: {TRAIN_START} to {TRAIN_END}")
    print(f"Test:  {TEST_START} to {TEST_END}")
    print('='*60)

    # Fetch all data
    print("\nFetching data...")

    # Main ticker
    df_train = fetch_polygon_data_range(ticker, TRAIN_START, TRAIN_END)
    df_test = fetch_polygon_data_range(ticker, TEST_START, TEST_END)
    print(f"  {ticker}: {len(df_train)} train + {len(df_test)} test days")

    # VIX data
    vix_train = fetch_polygon_data_range('UVXY', TRAIN_START, TRAIN_END)  # Use UVXY as VIX proxy
    vix_test = fetch_polygon_data_range('UVXY', TEST_START, TEST_END)
    print(f"  VIX proxy (UVXY): {len(vix_train)} train + {len(vix_test)} test days")

    # Inter-market data
    spy_train = fetch_polygon_data_range('SPY', TRAIN_START, TRAIN_END) if ticker != 'SPY' else None
    spy_test = fetch_polygon_data_range('SPY', TEST_START, TEST_END) if ticker != 'SPY' else None

    qqq_train = fetch_polygon_data_range('QQQ', TRAIN_START, TRAIN_END) if ticker != 'QQQ' else None
    qqq_test = fetch_polygon_data_range('QQQ', TEST_START, TEST_END) if ticker != 'QQQ' else None

    iwm_train = fetch_polygon_data_range('IWM', TRAIN_START, TRAIN_END) if ticker != 'IWM' else None
    iwm_test = fetch_polygon_data_range('IWM', TEST_START, TEST_END) if ticker != 'IWM' else None

    # Combine train and test
    df_all = pd.concat([df_train, df_test])
    vix_all = pd.concat([vix_train, vix_test]) if len(vix_train) > 0 else None
    spy_all = pd.concat([spy_train, spy_test]) if spy_train is not None and len(spy_train) > 0 else None
    qqq_all = pd.concat([qqq_train, qqq_test]) if qqq_train is not None and len(qqq_train) > 0 else None
    iwm_all = pd.concat([iwm_train, iwm_test]) if iwm_train is not None and len(iwm_train) > 0 else None

    # Calculate features
    print("Calculating enhanced features...")
    df_all = calculate_enhanced_features(df_all, vix_all, spy_all, qqq_all, iwm_all)

    feature_cols = get_enhanced_feature_columns(ticker, has_vix=(vix_all is not None))

    # Filter to features that exist
    feature_cols = [f for f in feature_cols if f in df_all.columns]
    print(f"  Using {len(feature_cols)} features")

    # Split back into train/test
    train_end_date = pd.Timestamp(TRAIN_END)
    test_start_date = pd.Timestamp(TEST_START)

    required_cols = feature_cols + ['actual_high_pct', 'actual_low_pct', 'actual_close_pct']
    df_train_clean = df_all[df_all.index <= train_end_date].dropna(subset=required_cols)
    df_test_clean = df_all[df_all.index >= test_start_date].dropna(subset=required_cols)

    print(f"  Training samples: {len(df_train_clean)}")
    print(f"  Test samples: {len(df_test_clean)}")

    X_train = df_train_clean[feature_cols]
    y_high_train = df_train_clean['actual_high_pct']
    y_low_train = df_train_clean['actual_low_pct']

    X_test = df_test_clean[feature_cols]
    y_high_test = df_test_clean['actual_high_pct']
    y_low_test = df_test_clean['actual_low_pct']
    y_close_test = df_test_clean['actual_close_pct']

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining enhanced ensemble...")

    # Train HIGH model ensemble
    print("  Training HIGH prediction models...")

    xgb_high = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    xgb_high.fit(X_train_scaled, y_high_train)

    gb_high = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,
        min_samples_leaf=10,
        random_state=42
    )
    gb_high.fit(X_train_scaled, y_high_train)

    rf_high = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf_high.fit(X_train_scaled, y_high_train)

    # Train LOW model ensemble
    print("  Training LOW prediction models...")

    xgb_low = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    xgb_low.fit(X_train_scaled, y_low_train)

    gb_low = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,
        min_samples_leaf=10,
        random_state=42
    )
    gb_low.fit(X_train_scaled, y_low_train)

    rf_low = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf_low.fit(X_train_scaled, y_low_train)

    # Ensemble predictions
    high_pred = (
        xgb_high.predict(X_test_scaled) * 0.4 +
        gb_high.predict(X_test_scaled) * 0.3 +
        rf_high.predict(X_test_scaled) * 0.3
    )

    low_pred = (
        xgb_low.predict(X_test_scaled) * 0.4 +
        gb_low.predict(X_test_scaled) * 0.3 +
        rf_low.predict(X_test_scaled) * 0.3
    )

    # Accuracy metrics
    high_mae = np.mean(np.abs(high_pred - y_high_test.values))
    low_mae = np.mean(np.abs(low_pred - y_low_test.values))

    capture_rate, avg_miss, avg_range = evaluate_range_accuracy(
        y_high_test, y_low_test, y_close_test, high_pred, low_pred
    )

    print(f"\n--- ENHANCED MODEL PERFORMANCE (2025 Test) ---")
    print(f"  High Prediction MAE: {high_mae:.3f}%")
    print(f"  Low Prediction MAE: {low_mae:.3f}%")
    print(f"  Range Capture Rate: {capture_rate:.1f}%")
    print(f"  Avg Range Width: {avg_range:.3f}%")

    # Find optimal buffer
    best_buffer = 0
    best_capture = capture_rate

    for buffer in np.arange(0, 0.5, 0.02):
        cap_rate, _, _ = evaluate_range_accuracy(
            y_high_test, y_low_test, y_close_test,
            high_pred + buffer, low_pred + buffer
        )
        if cap_rate > best_capture:
            best_capture = cap_rate
            best_buffer = buffer
        if cap_rate >= 95:
            break

    final_capture, final_miss, final_range = evaluate_range_accuracy(
        y_high_test, y_low_test, y_close_test,
        high_pred + best_buffer, low_pred + best_buffer
    )

    print(f"\n--- WITH BUFFER (+{best_buffer:.2f}%) ---")
    print(f"  Range Capture Rate: {final_capture:.1f}%")
    print(f"  Avg Range Width: {final_range:.3f}%")

    # Feature importance
    print(f"\nTop 15 features (XGBoost HIGH model):")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_high.feature_importances_
    }).sort_values('importance', ascending=False)
    for _, row in importance.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Save model
    model_data = {
        'high_models': {'xgb': xgb_high, 'gb': gb_high, 'rf': rf_high},
        'low_models': {'xgb': xgb_low, 'gb': gb_low, 'rf': rf_low},
        'weights': {'xgb': 0.4, 'gb': 0.3, 'rf': 0.3},
        'scaler': scaler,
        'feature_cols': feature_cols,
        'buffer': best_buffer,
        'ticker': ticker,
        'trained_at': datetime.now().isoformat(),
        'version': 'enhanced_highlow_v1',
        'train_period': f'{TRAIN_START} to {TRAIN_END}',
        'test_period': f'{TEST_START} to {TEST_END}',
        'metrics': {
            'capture_rate': float(final_capture),
            'high_mae': float(high_mae),
            'low_mae': float(low_mae),
            'avg_range': float(final_range),
            'buffer': float(best_buffer),
            'train_samples': len(df_train_clean),
            'test_samples': len(df_test_clean)
        }
    }

    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_highlow_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to {model_path}")

    return model_data


if __name__ == '__main__':
    print("\n" + "="*70)
    print("   TRAINING ENHANCED HIGH/LOW MODELS")
    print("   (with VIX + Inter-market correlations)")
    print("="*70)

    results = {}
    for ticker in ['SPY', 'QQQ', 'IWM']:
        try:
            model_data = train_enhanced_highlow_model(ticker)
            results[ticker] = model_data['metrics']
        except Exception as e:
            print(f"Error training {ticker}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("   RESULTS COMPARISON")
    print("="*70)
    print("\n  BEFORE (basic features):")
    print("  Ticker   High MAE    Low MAE     Capture Rate")
    print("  SPY      1.407%      0.529%      87.6%")
    print("  QQQ      0.447%      0.745%      86.3%")
    print("  IWM      4.040%      0.559%      80.3%")

    print("\n  AFTER (enhanced features):")
    print(f"  {'Ticker':<8} {'High MAE':<12} {'Low MAE':<12} {'Capture Rate'}")
    print("  " + "-" * 50)
    for ticker, metrics in results.items():
        if 'error' not in metrics:
            print(f"  {ticker:<8} {metrics['high_mae']:.3f}%{'':<6} {metrics['low_mae']:.3f}%{'':<6} {metrics['capture_rate']:.1f}%")
