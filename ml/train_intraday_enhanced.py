"""
Enhanced Intraday Session Model with Phase 1-3 Features

Adds TTM Squeeze, KDJ, Volatility Regime, Divergences, Bar Patterns,
Trend Structure, Pivot Points, and more to the intraday prediction model.

Original intraday model: ~88% accuracy
Goal: Improve by incorporating daily technical context
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from datetime import datetime
import pickle
import os
import requests
import warnings
warnings.filterwarnings('ignore')

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def fetch_daily_data(ticker: str, start_date: str = '2003-01-01', end_date: str = '2025-12-31') -> pd.DataFrame:
    """Fetch daily OHLCV data from Polygon"""
    print(f"Fetching daily data for {ticker}...")

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if 'results' not in data:
        raise ValueError(f"No data for {ticker}")

    df = pd.DataFrame(data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
    df = df.set_index('date')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    print(f"  Retrieved {len(df)} trading days")
    return df


def calculate_daily_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all the Phase 1-3 features that provide DAILY CONTEXT
    for intraday predictions.

    These are features known at market open (based on previous close).
    """
    df = df.copy()

    # ========== BASIC FEATURES ==========
    df['daily_return'] = df['Close'].pct_change() * 100

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 0.001)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.001)

    # ATR
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # ========== TTM SQUEEZE ==========
    kc_middle = df['Close'].ewm(span=20).mean()
    kc_atr = tr.rolling(20).mean()
    kc_upper = kc_middle + 1.5 * kc_atr
    kc_lower = kc_middle - 1.5 * kc_atr

    squeeze_on = (df['bb_lower'] > kc_lower) & (df['bb_upper'] < kc_upper)
    df['ttm_squeeze_on'] = squeeze_on.astype(int).shift(1)
    df['ttm_squeeze_off'] = (~squeeze_on).astype(int).shift(1)
    df['ttm_squeeze_momentum'] = (df['Close'] - kc_middle).shift(1)

    # Count consecutive squeeze bars
    squeeze_bars = []
    count = 0
    for val in squeeze_on:
        if val:
            count += 1
        else:
            count = 0
        squeeze_bars.append(count)
    df['ttm_squeeze_bars'] = pd.Series(squeeze_bars, index=df.index).shift(1)

    # ========== KDJ (9,3,3) ==========
    lowest_9 = df['Low'].rolling(9).min()
    highest_9 = df['High'].rolling(9).max()
    rsv = (df['Close'] - lowest_9) / (highest_9 - lowest_9 + 0.001) * 100
    df['kdj_k'] = rsv.ewm(span=3, adjust=False).mean().shift(1)
    df['kdj_d'] = df['kdj_k'].ewm(span=3, adjust=False).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    # KDJ zone
    df['kdj_zone'] = 0
    df.loc[df['kdj_k'] > 80, 'kdj_zone'] = 1   # Overbought
    df.loc[df['kdj_k'] < 20, 'kdj_zone'] = -1  # Oversold

    # ========== VOLATILITY REGIME ==========
    df['hv_10'] = df['daily_return'].rolling(10).std() * np.sqrt(252)
    df['hv_20'] = df['daily_return'].rolling(20).std() * np.sqrt(252)
    df['hv_ratio'] = (df['hv_10'] / (df['hv_20'] + 0.001)).shift(1)

    # Vol percentile (50-day rolling with min 20)
    df['vol_percentile'] = df['hv_20'].rolling(50, min_periods=20).apply(
        lambda x: (x.iloc[-1] > x[:-1]).mean() * 100 if len(x) > 1 else 50
    ).shift(1)

    # Vol regime
    df['vol_regime'] = 1  # Normal
    df.loc[df['vol_percentile'] < 20, 'vol_regime'] = 0  # Low
    df.loc[df['vol_percentile'] > 80, 'vol_regime'] = 2  # High

    # ATR as % of price
    df['atr_pct'] = (df['atr'] / df['Close'] * 100).shift(1)

    # ========== TREND STRUCTURE ==========
    df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int).shift(1)
    df['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int).shift(1)
    df['higher_low'] = (df['Low'] > df['Low'].shift(1)).astype(int).shift(1)
    df['lower_high'] = (df['High'] < df['High'].shift(1)).astype(int).shift(1)

    # 3-bar trend structure
    df['trend_structure_3'] = (
        df['higher_high'].rolling(3).sum() +
        df['higher_low'].rolling(3).sum() -
        df['lower_high'].rolling(3).sum() -
        df['lower_low'].rolling(3).sum()
    ).shift(1)

    # ========== BAR PATTERNS ==========
    day_range = df['High'] - df['Low']
    prev_range = day_range.shift(1)

    df['inside_bar'] = ((df['High'] < df['High'].shift(1)) &
                        (df['Low'] > df['Low'].shift(1))).astype(int).shift(1)
    df['outside_bar'] = ((df['High'] > df['High'].shift(1)) &
                         (df['Low'] < df['Low'].shift(1))).astype(int).shift(1)

    # Narrow range
    min_range_4 = day_range.rolling(4).min()
    min_range_7 = day_range.rolling(7).min()
    df['narrow_range_4'] = (day_range == min_range_4).astype(int).shift(1)
    df['narrow_range_7'] = (day_range == min_range_7).astype(int).shift(1)

    # Wide range
    avg_range = day_range.rolling(20).mean()
    df['wide_range_bar'] = (day_range > avg_range * 2).astype(int).shift(1)

    # ========== RANGE PATTERNS ==========
    df['avg_range_10'] = (day_range / df['Close']).rolling(10).mean().shift(1) * 100
    df['avg_range_20'] = (day_range / df['Close']).rolling(20).mean().shift(1) * 100
    df['range_vs_avg'] = (day_range / (avg_range + 0.001)).shift(1)
    df['range_expansion'] = (day_range > avg_range * 1.5).astype(int).shift(1)

    # Consecutive narrow days
    narrow = day_range < avg_range * 0.5
    consec_narrow = []
    count = 0
    for val in narrow:
        if val:
            count += 1
        else:
            count = 0
        consec_narrow.append(count)
    df['consec_narrow'] = pd.Series(consec_narrow, index=df.index).shift(1)

    # ========== PIVOT POINTS ==========
    df['pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['pivot_r1'] = 2 * df['pivot'] - df['Low'].shift(1)
    df['pivot_s1'] = 2 * df['pivot'] - df['High'].shift(1)

    df['dist_to_pivot'] = ((df['Close'].shift(1) - df['pivot']) / df['pivot'] * 100)
    df['dist_to_r1'] = ((df['pivot_r1'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100)
    df['dist_to_s1'] = ((df['Close'].shift(1) - df['pivot_s1']) / df['Close'].shift(1) * 100)

    # ========== SWING S/R ==========
    df['swing_high_20'] = df['High'].rolling(20).max().shift(1)
    df['swing_low_20'] = df['Low'].rolling(20).min().shift(1)

    swing_range = df['swing_high_20'] - df['swing_low_20']
    df['dist_to_resistance'] = ((df['swing_high_20'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100)
    df['dist_to_support'] = ((df['Close'].shift(1) - df['swing_low_20']) / df['Close'].shift(1) * 100)
    df['swing_position'] = ((df['Close'].shift(1) - df['swing_low_20']) / (swing_range + 0.001))

    # ========== CALENDAR ==========
    df['day_of_week'] = df.index.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    df['week_of_month'] = (df.index.day - 1) // 7 + 1

    # OPEX week (3rd Friday of month)
    df['is_opex_week'] = ((df.index.day >= 15) & (df.index.day <= 21) &
                          (df['day_of_week'] <= 4)).astype(int)

    df['is_first_5_days'] = (df.index.day <= 5).astype(int)
    df['is_last_5_days'] = (df.index.day >= 25).astype(int)

    # ========== MOMENTUM ==========
    df['momentum_3d'] = df['daily_return'].rolling(3).sum().shift(1)
    df['momentum_5d'] = df['daily_return'].rolling(5).sum().shift(1)
    df['roc_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100).shift(1)

    # Moving average position
    df['price_vs_sma20'] = ((df['Close'] - sma20) / sma20 * 100).shift(1)
    sma50 = df['Close'].rolling(50).mean()
    df['price_vs_sma50'] = ((df['Close'] - sma50) / sma50 * 100).shift(1)

    # Z-score
    df['zscore_20'] = ((df['Close'] - sma20) / (std20 + 0.001)).shift(1)

    # Streak
    streak = []
    count = 0
    for ret in df['daily_return']:
        if pd.isna(ret):
            streak.append(0)
        elif ret > 0:
            count = count + 1 if count > 0 else 1
            streak.append(count)
        elif ret < 0:
            count = count - 1 if count < 0 else -1
            streak.append(count)
        else:
            streak.append(0)
    df['streak'] = pd.Series(streak, index=df.index).shift(1)

    # Volume
    df['volume_sma20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = (df['Volume'] / (df['volume_sma20'] + 1)).shift(1)

    return df


def create_session_snapshots_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    """Create session snapshots with enhanced daily context features"""

    # First calculate daily context features
    df = calculate_daily_context_features(df)

    snapshots = []

    # Previous day features
    df['prev_close'] = df['Close'].shift(1)
    df['prev_high'] = df['High'].shift(1)
    df['prev_low'] = df['Low'].shift(1)
    df['prev_range'] = (df['prev_high'] - df['prev_low']) / df['prev_close']
    df['prev_return'] = df['Close'].pct_change().shift(1)

    # Target: Did we close higher than open?
    df['bullish_day'] = (df['Close'] > df['Open']).astype(int)

    # Gap
    df['gap'] = (df['Open'] - df['prev_close']) / df['prev_close']

    # Clean
    df = df.dropna()

    # Time points through the session
    time_points = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]

    # Daily context feature columns (known at open)
    daily_context_cols = [
        'ttm_squeeze_on', 'ttm_squeeze_off', 'ttm_squeeze_momentum', 'ttm_squeeze_bars',
        'kdj_k', 'kdj_d', 'kdj_j', 'kdj_zone',
        'hv_ratio', 'vol_percentile', 'vol_regime', 'atr_pct',
        'higher_high', 'lower_low', 'higher_low', 'lower_high', 'trend_structure_3',
        'inside_bar', 'outside_bar', 'narrow_range_4', 'narrow_range_7', 'wide_range_bar',
        'avg_range_10', 'avg_range_20', 'range_vs_avg', 'range_expansion', 'consec_narrow',
        'dist_to_pivot', 'dist_to_r1', 'dist_to_s1',
        'dist_to_resistance', 'dist_to_support', 'swing_position',
        'day_of_week', 'is_monday', 'is_friday', 'week_of_month',
        'is_opex_week', 'is_first_5_days', 'is_last_5_days',
        'momentum_3d', 'momentum_5d', 'roc_5',
        'price_vs_sma20', 'price_vs_sma50', 'zscore_20', 'streak', 'volume_ratio',
        'rsi_14', 'macd_hist', 'bb_position', 'bb_width'
    ]

    for idx, row in df.iterrows():
        open_price = row['Open']
        high_price = row['High']
        low_price = row['Low']
        close_price = row['Close']

        for time_pct in time_points:
            snapshot = {
                'date': idx,
                'time_pct': time_pct,
                'time_remaining': 1 - time_pct,

                # Gap features
                'gap': row['gap'],
                'gap_direction': 1 if row['gap'] > 0 else -1 if row['gap'] < 0 else 0,
                'gap_size': abs(row['gap']),

                # Previous day
                'prev_return': row['prev_return'],
                'prev_range': row['prev_range'],
            }

            # Add daily context features
            for col in daily_context_cols:
                if col in row.index and pd.notna(row[col]):
                    snapshot[col] = row[col]
                else:
                    snapshot[col] = 0

            # Simulate current price position
            if time_pct == 0:
                current_price = open_price
                high_so_far = open_price
                low_so_far = open_price
            else:
                progress = time_pct ** 0.8
                base_price = open_price + (close_price - open_price) * progress

                if time_pct < 0.5:
                    deviation = (high_price - low_price) * 0.3 * (1 - time_pct)
                else:
                    deviation = (high_price - low_price) * 0.1 * (1 - time_pct)

                current_price = base_price + np.random.uniform(-deviation, deviation) * 0.5
                high_so_far = max(open_price, min(high_price, open_price + (high_price - open_price) * min(1, time_pct * 1.5)))
                low_so_far = min(open_price, max(low_price, open_price + (low_price - open_price) * min(1, time_pct * 1.5)))

            range_so_far = high_so_far - low_so_far if high_so_far != low_so_far else 0.0001

            snapshot['current_vs_open'] = (current_price - open_price) / open_price
            snapshot['current_vs_open_direction'] = 1 if current_price > open_price else -1 if current_price < open_price else 0
            snapshot['position_in_range'] = (current_price - low_so_far) / range_so_far if range_so_far > 0 else 0.5
            snapshot['range_so_far_pct'] = range_so_far / open_price
            snapshot['high_so_far_pct'] = (high_so_far - open_price) / open_price
            snapshot['low_so_far_pct'] = (open_price - low_so_far) / open_price

            snapshot['above_open'] = 1 if current_price > open_price else 0
            snapshot['near_high'] = 1 if (high_so_far - current_price) < (current_price - low_so_far) else 0

            # Gap fill status
            if row['gap'] > 0:
                gap_filled = low_so_far <= row['prev_close']
            else:
                gap_filled = high_so_far >= row['prev_close']
            snapshot['gap_filled'] = 1 if gap_filled else 0

            # Targets
            snapshot['target'] = row['bullish_day']
            snapshot['target_close_above_current'] = 1 if close_price > current_price else 0

            snapshots.append(snapshot)

    return pd.DataFrame(snapshots)


def train_enhanced_intraday_model(ticker: str = 'SPY'):
    """Train enhanced intraday model with Phase 1-3 features"""

    print(f"\n{'='*70}")
    print(f"  ENHANCED INTRADAY MODEL - {ticker}")
    print(f"  With TTM Squeeze, KDJ, Volatility, Trend Structure, etc.")
    print(f"{'='*70}")

    # Fetch data
    df = fetch_daily_data(ticker)

    # Create enhanced snapshots
    print("\nCreating enhanced session snapshots...")
    snapshots = create_session_snapshots_enhanced(df)
    print(f"  Created {len(snapshots)} training samples")

    # Split by date
    train_end = '2023-12-31'
    test_start = '2024-01-01'

    train_data = snapshots[snapshots['date'] <= train_end]
    test_data = snapshots[snapshots['date'] >= test_start]

    print(f"\n  TRAIN: {len(train_data)} samples")
    print(f"  TEST:  {len(test_data)} samples")

    # Feature columns - session features + daily context
    session_features = [
        'time_pct', 'time_remaining',
        'gap', 'gap_direction', 'gap_size',
        'prev_return', 'prev_range',
        'current_vs_open', 'current_vs_open_direction',
        'position_in_range', 'range_so_far_pct',
        'high_so_far_pct', 'low_so_far_pct',
        'above_open', 'near_high', 'gap_filled'
    ]

    daily_context_features = [
        'ttm_squeeze_on', 'ttm_squeeze_off', 'ttm_squeeze_momentum', 'ttm_squeeze_bars',
        'kdj_k', 'kdj_d', 'kdj_j', 'kdj_zone',
        'hv_ratio', 'vol_percentile', 'vol_regime', 'atr_pct',
        'higher_high', 'lower_low', 'higher_low', 'lower_high', 'trend_structure_3',
        'inside_bar', 'outside_bar', 'narrow_range_4', 'narrow_range_7', 'wide_range_bar',
        'avg_range_10', 'avg_range_20', 'range_vs_avg', 'range_expansion', 'consec_narrow',
        'dist_to_pivot', 'dist_to_r1', 'dist_to_s1',
        'dist_to_resistance', 'dist_to_support', 'swing_position',
        'day_of_week', 'is_monday', 'is_friday', 'week_of_month',
        'is_opex_week', 'is_first_5_days', 'is_last_5_days',
        'momentum_3d', 'momentum_5d', 'roc_5',
        'price_vs_sma20', 'price_vs_sma50', 'zscore_20', 'streak', 'volume_ratio',
        'rsi_14', 'macd_hist', 'bb_position', 'bb_width'
    ]

    all_features = session_features + daily_context_features
    available_features = [f for f in all_features if f in train_data.columns]

    print(f"\n  Using {len(available_features)} features ({len(session_features)} session + {len(available_features) - len(session_features)} daily context)")

    X_train = train_data[available_features].replace([np.inf, -np.inf], 0).fillna(0)
    X_test = test_data[available_features].replace([np.inf, -np.inf], 0).fillna(0)
    y_train = train_data['target'].astype(int)
    y_test = test_data['target'].astype(int)

    print(f"  Train bullish rate: {y_train.mean():.1%}")
    print(f"  Test bullish rate: {y_test.mean():.1%}")

    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train ensemble with tuned hyperparameters
    print("\nTraining ensemble...")

    models = {
        'xgb': XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        'rf': RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        ),
        'et': ExtraTreesClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)

        pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, pred)

        proba = model.predict_proba(X_test_scaled)[:, 1]
        high_conf_mask = (proba >= 0.65) | (proba <= 0.35)
        hc_acc = accuracy_score(y_test[high_conf_mask], pred[high_conf_mask]) if high_conf_mask.sum() > 0 else 0

        results[name] = {'accuracy': acc, 'high_conf_accuracy': hc_acc, 'high_conf_count': int(high_conf_mask.sum())}
        print(f"  {name}: {acc:.1%} overall, {hc_acc:.1%} high-conf ({high_conf_mask.sum()} samples)")

    # Calculate weights
    total_acc = sum(r['accuracy'] for r in results.values())
    weights = {name: (r['accuracy'] / total_acc) for name, r in results.items()}

    # Ensemble prediction
    proba_all = np.zeros(len(y_test))
    for name, model in models.items():
        proba_all += model.predict_proba(X_test_scaled)[:, 1] * weights[name]
    pred_all = (proba_all >= 0.5).astype(int)
    overall_acc = accuracy_score(y_test, pred_all)

    print(f"\n  ENSEMBLE ACCURACY: {overall_acc:.1%}")

    # By time slice
    print("\n" + "="*50)
    print("  ACCURACY BY SESSION TIME (Enhanced)")
    print("="*50)

    time_slices = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
    time_labels = ['Open', '10%', '25%', '50%', '75%', '90%']

    for time_pct, label in zip(time_slices, time_labels):
        mask = test_data['time_pct'] == time_pct
        if mask.sum() == 0:
            continue

        X_slice = X_test_scaled[mask.values]
        y_slice = y_test[mask].values

        proba = np.zeros(len(y_slice))
        for name, model in models.items():
            proba += model.predict_proba(X_slice)[:, 1] * weights[name]

        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y_slice, pred)

        high_conf = (proba >= 0.6) | (proba <= 0.4)
        hc_acc = accuracy_score(y_slice[high_conf], pred[high_conf]) if high_conf.sum() > 0 else 0
        hc_pct = high_conf.sum() / len(y_slice)

        print(f"  {label:5s}: {acc:.1%} overall | {hc_acc:.1%} @ 60% conf ({hc_pct:.0%} of signals)")

    # Feature importance
    print("\n" + "="*50)
    print("  TOP 20 FEATURE IMPORTANCE")
    print("="*50)

    importance = pd.Series(models['xgb'].feature_importances_, index=available_features).sort_values(ascending=False)
    for i, (feat, imp) in enumerate(importance.head(20).items()):
        is_new = " [DAILY]" if feat in daily_context_features else ""
        print(f"  {i+1:2d}. {feat}: {imp:.4f}{is_new}")

    # Count daily features in top 20
    top_20 = importance.head(20).index.tolist()
    daily_in_top_20 = sum(1 for f in top_20 if f in daily_context_features)
    print(f"\n  Daily context features in top 20: {daily_in_top_20}/20")

    # Save model
    model_data = {
        'models': models,
        'weights': weights,
        'scaler': scaler,
        'feature_cols': available_features,
        'metrics': {
            'accuracy': float(overall_acc),
            'by_model': results
        },
        'ticker': ticker,
        'version': 'intraday_enhanced_v1',
        'trained_at': datetime.now().isoformat(),
        'daily_context_features': daily_context_features
    }

    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_enhanced_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n✓ Model saved to {model_path}")

    return model_data


if __name__ == '__main__':
    print("="*70)
    print("   ENHANCED INTRADAY MODEL TRAINING")
    print("   With Phase 1-3 Daily Context Features")
    print("="*70)

    results = {}
    for ticker in ['SPY', 'QQQ', 'IWM']:
        try:
            model_data = train_enhanced_intraday_model(ticker)
            results[ticker] = model_data['metrics']['accuracy']
        except Exception as e:
            print(f"Error training {ticker}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("   FINAL RESULTS (Out-of-Sample 2024-2025)")
    print("="*70)
    for ticker, acc in results.items():
        print(f"  {ticker}: {acc:.1%}")
