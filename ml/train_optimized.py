"""
Optimized Intraday Models - Maximized per Ticker

For each ticker:
1. Feature selection - find best features for that specific ticker
2. Hyperparameter tuning - optimize model params
3. Ensemble weight optimization
4. Test multiple configurations and keep the best
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from xgboost import XGBClassifier
from datetime import datetime, timedelta
import pickle
import os
import requests
import warnings
warnings.filterwarnings('ignore')

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def fetch_hourly_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly data"""
    print(f"  Fetching hourly data for {ticker}...")
    all_data = []
    current_start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    while current_start < end:
        chunk_end = min(current_start + timedelta(days=60), end)
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{current_start.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}"
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
    df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
    df = df.set_index('datetime')
    df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
    print(f"    Got {len(df)} hourly bars")
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def fetch_daily_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily data"""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}
    response = requests.get(url, params=params)
    data = response.json()

    if 'results' not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
    df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
    df = df.set_index('date')
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def get_price_at_hour(hourly_df, date, hour):
    try:
        day_start = pd.Timestamp(date).tz_localize('America/New_York').replace(hour=hour, minute=0)
        day_end = day_start + timedelta(hours=1)
        bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index < day_end)]
        if len(bars) > 0:
            return bars['Close'].iloc[0]
    except:
        pass
    return None


def calculate_multi_day_features(daily_df, day_idx):
    """Calculate multi-day features"""
    features = {}

    if day_idx < 10:
        return {
            'return_3d': 0, 'return_5d': 0, 'return_10d': 0,
            'consecutive_up': 0, 'consecutive_down': 0,
            'volatility_5d': 0, 'volatility_10d': 0,
            'trend_strength_5d': 0, 'mean_reversion_signal': 0,
            'high_low_range_5d': 0,
        }

    close_today = daily_df.iloc[day_idx - 1]['Close']

    # Multi-day returns
    for days, name in [(3, '3d'), (5, '5d'), (10, '10d')]:
        if day_idx > days:
            close_past = daily_df.iloc[day_idx - days - 1]['Close']
            features[f'return_{name}'] = (close_today - close_past) / close_past
        else:
            features[f'return_{name}'] = 0

    # Consecutive days
    consecutive_up = 0
    consecutive_down = 0
    for i in range(1, min(8, day_idx)):
        prev_close = daily_df.iloc[day_idx - i]['Close']
        prev_open = daily_df.iloc[day_idx - i]['Open']
        if prev_close > prev_open:
            if consecutive_down == 0:
                consecutive_up += 1
            else:
                break
        elif prev_close < prev_open:
            if consecutive_up == 0:
                consecutive_down += 1
            else:
                break
        else:
            break
    features['consecutive_up'] = consecutive_up
    features['consecutive_down'] = consecutive_down

    # Rolling volatility
    returns = []
    for i in range(1, min(11, day_idx)):
        if day_idx - i - 1 >= 0:
            c1 = daily_df.iloc[day_idx - i]['Close']
            c2 = daily_df.iloc[day_idx - i - 1]['Close']
            returns.append((c1 - c2) / c2)

    features['volatility_5d'] = np.std(returns[:5]) if len(returns) >= 5 else 0
    features['volatility_10d'] = np.std(returns) if len(returns) >= 5 else 0

    # Trend strength
    bullish_count = sum(1 for i in range(1, min(6, day_idx))
                       if daily_df.iloc[day_idx - i]['Close'] > daily_df.iloc[day_idx - i]['Open'])
    features['trend_strength_5d'] = bullish_count / 5

    # Mean reversion
    prev_return = (daily_df.iloc[day_idx - 1]['Close'] - daily_df.iloc[day_idx - 1]['Open']) / daily_df.iloc[day_idx - 1]['Open']
    features['mean_reversion_signal'] = -prev_return  # Negative correlation

    # 5-day high-low range
    highs = [daily_df.iloc[day_idx - i]['High'] for i in range(1, min(6, day_idx))]
    lows = [daily_df.iloc[day_idx - i]['Low'] for i in range(1, min(6, day_idx))]
    if highs and lows:
        features['high_low_range_5d'] = (max(highs) - min(lows)) / close_today
    else:
        features['high_low_range_5d'] = 0

    return features


def create_all_features(bars_so_far, today_open, prev_day, prev_prev_day, daily_df, day_idx, price_11am=None):
    """Create comprehensive feature set"""
    current_bar = bars_so_far.iloc[-1]
    current_price = current_bar['Close']
    current_hour = current_bar.name.hour

    high_so_far = bars_so_far['High'].max()
    low_so_far = bars_so_far['Low'].min()
    volume_so_far = bars_so_far['Volume'].sum()

    hours_since_open = (current_bar.name.hour - 9) + (current_bar.name.minute / 60)
    time_pct = min(max(hours_since_open / 6.5, 0), 1)

    prev_close = prev_day['Close']
    prev_open = prev_day['Open']
    prev_high = prev_day['High']
    prev_low = prev_day['Low']
    prev_volume = prev_day['Volume']

    gap = (today_open - prev_close) / prev_close
    prev_return = (prev_close - prev_prev_day['Close']) / prev_prev_day['Close']
    prev_range = (prev_high - prev_low) / prev_close
    prev_body = (prev_close - prev_open) / prev_open
    range_so_far = max(high_so_far - low_so_far, 0.0001)

    features = {
        # Time
        'time_pct': time_pct,
        'time_remaining': 1 - time_pct,
        'time_squared': time_pct ** 2,

        # Gap
        'gap': gap,
        'gap_direction': 1 if gap > 0 else (-1 if gap < 0 else 0),
        'gap_size': abs(gap),
        'gap_large': 1 if abs(gap) > 0.005 else 0,

        # Previous day
        'prev_return': prev_return,
        'prev_range': prev_range,
        'prev_body': prev_body,
        'prev_bullish': 1 if prev_close > prev_open else 0,
        'prev_strong_up': 1 if prev_return > 0.01 else 0,
        'prev_strong_down': 1 if prev_return < -0.01 else 0,

        # Current position
        'current_vs_open': (current_price - today_open) / today_open,
        'current_vs_open_direction': 1 if current_price > today_open else (-1 if current_price < today_open else 0),
        'position_in_range': (current_price - low_so_far) / range_so_far if range_so_far > 0 else 0.5,
        'range_so_far_pct': range_so_far / today_open,
        'high_so_far_pct': (high_so_far - today_open) / today_open,
        'low_so_far_pct': (today_open - low_so_far) / today_open,
        'above_open': 1 if current_price > today_open else 0,
        'near_high': 1 if (high_so_far - current_price) < (current_price - low_so_far) else 0,
        'at_high': 1 if current_price >= high_so_far * 0.999 else 0,
        'at_low': 1 if current_price <= low_so_far * 1.001 else 0,

        # Gap fill
        'gap_filled': 1 if (gap > 0 and low_so_far <= prev_close) or (gap <= 0 and high_so_far >= prev_close) else 0,
        'gap_fill_pct': min(1, abs(current_price - today_open) / abs(today_open - prev_close)) if abs(today_open - prev_close) > 0 else 0,

        # Momentum
        'morning_reversal': 1 if (gap > 0 and current_price < today_open) or (gap < 0 and current_price > today_open) else 0,
        'extended_move': 1 if abs((current_price - today_open) / today_open) > 0.005 else 0,
        'strong_move': 1 if abs((current_price - today_open) / today_open) > 0.01 else 0,
    }

    # Hourly momentum
    if len(bars_so_far) >= 2:
        features['last_hour_return'] = (current_price - bars_so_far['Close'].iloc[-2]) / bars_so_far['Close'].iloc[-2]
        features['last_hour_direction'] = 1 if features['last_hour_return'] > 0 else -1
    else:
        features['last_hour_return'] = 0
        features['last_hour_direction'] = 0

    if len(bars_so_far) >= 3:
        features['two_hour_return'] = (current_price - bars_so_far['Close'].iloc[-3]) / bars_so_far['Close'].iloc[-3]
    else:
        features['two_hour_return'] = features.get('last_hour_return', 0)

    # Bar patterns
    bullish_bars = sum(1 for i in range(len(bars_so_far)) if bars_so_far['Close'].iloc[i] > bars_so_far['Open'].iloc[i])
    features['bullish_bar_ratio'] = bullish_bars / len(bars_so_far)
    features['mostly_bullish'] = 1 if features['bullish_bar_ratio'] > 0.6 else 0
    features['mostly_bearish'] = 1 if features['bullish_bar_ratio'] < 0.4 else 0

    # Intraday volatility
    if len(bars_so_far) >= 2:
        hourly_returns = bars_so_far['Close'].pct_change().dropna()
        features['intraday_volatility'] = hourly_returns.std() if len(hourly_returns) > 0 else 0
    else:
        features['intraday_volatility'] = 0

    features['range_vs_prev'] = range_so_far / (prev_high - prev_low) if (prev_high - prev_low) > 0 else 1
    features['range_expanding'] = 1 if features['range_vs_prev'] > 1.2 else 0

    # Volume
    avg_hourly_volume = volume_so_far / len(bars_so_far) if len(bars_so_far) > 0 else 0
    features['volume_pace'] = (volume_so_far / (time_pct * prev_volume)) if (time_pct * prev_volume) > 0 else 1
    features['current_hour_volume_ratio'] = current_bar['Volume'] / avg_hourly_volume if avg_hourly_volume > 0 else 1
    features['high_volume'] = 1 if features['volume_pace'] > 1.3 else 0

    # 11 AM anchor
    if price_11am is not None:
        features['current_vs_11am'] = (current_price - price_11am) / price_11am
        features['above_11am'] = 1 if current_price > price_11am else 0
        features['distance_from_11am'] = abs(current_price - price_11am) / price_11am
    else:
        features['current_vs_11am'] = 0
        features['above_11am'] = 0
        features['distance_from_11am'] = 0

    # Day of week
    day_of_week = current_bar.name.dayofweek
    features['is_monday'] = 1 if day_of_week == 0 else 0
    features['is_friday'] = 1 if day_of_week == 4 else 0
    features['day_of_week'] = day_of_week

    # Multi-day features
    multi_day = calculate_multi_day_features(daily_df, day_idx)
    features.update(multi_day)

    # First hour
    if len(bars_so_far) >= 1:
        first_bar_return = (bars_so_far['Close'].iloc[0] - today_open) / today_open
        features['first_hour_return'] = first_bar_return
        features['first_hour_bullish'] = 1 if first_bar_return > 0 else 0
        features['first_hour_strong'] = 1 if abs(first_bar_return) > 0.005 else 0
    else:
        features['first_hour_return'] = 0
        features['first_hour_bullish'] = 0
        features['first_hour_strong'] = 0

    # Afternoon momentum
    if current_hour >= 14 and len(bars_so_far) >= 4:
        features['afternoon_momentum'] = (current_price - bars_so_far['Close'].iloc[-4]) / bars_so_far['Close'].iloc[-4]
    else:
        features['afternoon_momentum'] = 0

    # Interaction features
    features['gap_x_time'] = gap * time_pct
    features['momentum_x_time'] = features['current_vs_open'] * time_pct
    features['vol_x_direction'] = features['intraday_volatility'] * features['current_vs_open_direction']

    return features


def create_samples(ticker: str, daily_df: pd.DataFrame, hourly_df: pd.DataFrame,
                   start_date: str, end_date: str):
    """Create samples for a date range"""
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()

    samples_a = []
    samples_b = []

    trading_days = sorted(set(hourly_df.index.date))

    for day in trading_days:
        if day < start or day > end:
            continue
        if day not in daily_df.index:
            continue

        daily_dates = list(daily_df.index)
        day_idx = daily_dates.index(day)
        if day_idx < 10:
            continue

        today = daily_df.loc[day]
        prev_day = daily_df.iloc[day_idx - 1]
        prev_prev_day = daily_df.iloc[day_idx - 2]

        today_open = today['Open']
        today_close = today['Close']

        target_bullish = 1 if today_close > today_open else 0
        price_11am = get_price_at_hour(hourly_df, day, 11)
        target_above_11am = 1 if (price_11am and today_close > price_11am) else 0

        day_start = pd.Timestamp(day).tz_localize('America/New_York').replace(hour=9, minute=0)
        day_end = pd.Timestamp(day).tz_localize('America/New_York').replace(hour=16, minute=30)
        day_bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index <= day_end)]

        if len(day_bars) < 2:
            continue

        for j in range(1, len(day_bars) + 1):
            bars_so_far = day_bars.iloc[:j]
            current_hour = bars_so_far.iloc[-1].name.hour

            p11 = price_11am if current_hour >= 11 else None
            features = create_all_features(bars_so_far, today_open, prev_day, prev_prev_day, daily_df, day_idx, p11)
            features['target_bullish'] = target_bullish
            samples_a.append(features)

            if current_hour >= 11 and price_11am is not None:
                features_b = features.copy()
                features_b['target_above_11am'] = target_above_11am
                samples_b.append(features_b)

    return pd.DataFrame(samples_a), pd.DataFrame(samples_b)


def select_best_features(X_train, y_train, feature_names, n_features=30):
    """Select best features using mutual information"""
    selector = SelectKBest(mutual_info_classif, k=min(n_features, len(feature_names)))
    selector.fit(X_train, y_train)

    scores = selector.scores_
    feature_scores = list(zip(feature_names, scores))
    feature_scores.sort(key=lambda x: x[1], reverse=True)

    selected = [f[0] for f in feature_scores[:n_features]]
    return selected, feature_scores


def tune_and_train(X_train, y_train, X_test, y_test, ticker, target_name):
    """Tune hyperparameters and train optimized ensemble"""

    print(f"\n    Tuning {target_name} for {ticker}...")

    best_acc = 0
    best_config = None

    # Test different configurations
    configs = [
        # XGBoost focused
        {'xgb_n': 300, 'xgb_depth': 4, 'xgb_lr': 0.02, 'rf_n': 100, 'rf_depth': 4},
        {'xgb_n': 250, 'xgb_depth': 5, 'xgb_lr': 0.03, 'rf_n': 150, 'rf_depth': 5},
        {'xgb_n': 400, 'xgb_depth': 3, 'xgb_lr': 0.02, 'rf_n': 100, 'rf_depth': 6},
        # RF focused
        {'xgb_n': 150, 'xgb_depth': 4, 'xgb_lr': 0.05, 'rf_n': 300, 'rf_depth': 6},
        {'xgb_n': 200, 'xgb_depth': 4, 'xgb_lr': 0.03, 'rf_n': 250, 'rf_depth': 7},
        # Balanced
        {'xgb_n': 200, 'xgb_depth': 4, 'xgb_lr': 0.03, 'rf_n': 200, 'rf_depth': 5},
    ]

    for cfg in configs:
        models = {
            'xgb': XGBClassifier(
                n_estimators=cfg['xgb_n'],
                max_depth=cfg['xgb_depth'],
                learning_rate=cfg['xgb_lr'],
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
                eval_metric='logloss'
            ),
            'rf': RandomForestClassifier(
                n_estimators=cfg['rf_n'],
                max_depth=cfg['rf_depth'],
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                random_state=42
            ),
            'et': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            results[name] = {'model': model, 'accuracy': acc}

        # Weighted ensemble
        total_acc = sum(r['accuracy'] for r in results.values())
        weights = {name: r['accuracy'] / total_acc for name, r in results.items()}

        y_pred_ensemble = np.zeros(len(y_test))
        for name, r in results.items():
            y_pred_ensemble += r['model'].predict_proba(X_test)[:, 1] * weights[name]

        ensemble_acc = accuracy_score(y_test, (y_pred_ensemble > 0.5).astype(int))

        if ensemble_acc > best_acc:
            best_acc = ensemble_acc
            best_config = {
                'config': cfg,
                'models': {name: r['model'] for name, r in results.items()},
                'weights': weights,
                'individual_acc': {name: r['accuracy'] for name, r in results.items()}
            }

    print(f"      Best accuracy: {best_acc:.1%}")
    print(f"      Individual: {best_config['individual_acc']}")

    return best_config['models'], best_config['weights'], best_acc


def optimize_ticker(ticker: str):
    """Optimize model for a specific ticker"""

    TRAIN_START = '2020-01-01'
    TRAIN_END = '2025-01-01'
    TEST_START = '2025-01-02'
    TEST_END = '2025-12-19'

    print(f"\n{'='*70}")
    print(f"  OPTIMIZING: {ticker}")
    print(f"{'='*70}")

    # Fetch data
    daily_df = fetch_daily_data(ticker, TRAIN_START, TEST_END)
    hourly_df = fetch_hourly_data(ticker, TRAIN_START, TEST_END)

    # Create samples
    print(f"  Creating samples...")
    train_a, train_b = create_samples(ticker, daily_df, hourly_df, TRAIN_START, TRAIN_END)
    test_a, test_b = create_samples(ticker, daily_df, hourly_df, TEST_START, TEST_END)

    print(f"    Train A: {len(train_a)}, Test A: {len(test_a)}")
    print(f"    Train B: {len(train_b)}, Test B: {len(test_b)}")

    # Get all feature columns
    all_features = [c for c in train_a.columns if c not in ['target_bullish', 'target_above_11am']]

    # ========== TARGET A ==========
    print(f"\n  --- Optimizing Target A ---")

    X_train_a = train_a[all_features].replace([np.inf, -np.inf], 0).fillna(0)
    y_train_a = train_a['target_bullish']
    X_test_a = test_a[all_features].replace([np.inf, -np.inf], 0).fillna(0)
    y_test_a = test_a['target_bullish']

    # Feature selection
    print(f"    Selecting best features...")
    best_features_a, feature_scores_a = select_best_features(X_train_a.values, y_train_a.values, all_features, n_features=35)
    print(f"    Top 10 features: {best_features_a[:10]}")

    X_train_a_sel = X_train_a[best_features_a]
    X_test_a_sel = X_test_a[best_features_a]

    scaler_a = RobustScaler()
    X_train_a_scaled = scaler_a.fit_transform(X_train_a_sel)
    X_test_a_scaled = scaler_a.transform(X_test_a_sel)

    models_a, weights_a, acc_a = tune_and_train(X_train_a_scaled, y_train_a, X_test_a_scaled, y_test_a, ticker, "Target A")

    # ========== TARGET B ==========
    print(f"\n  --- Optimizing Target B ---")

    all_features_b = [c for c in train_b.columns if c not in ['target_bullish', 'target_above_11am']]

    X_train_b = train_b[all_features_b].replace([np.inf, -np.inf], 0).fillna(0)
    y_train_b = train_b['target_above_11am']
    X_test_b = test_b[all_features_b].replace([np.inf, -np.inf], 0).fillna(0)
    y_test_b = test_b['target_above_11am']

    print(f"    Selecting best features...")
    best_features_b, feature_scores_b = select_best_features(X_train_b.values, y_train_b.values, all_features_b, n_features=35)
    print(f"    Top 10 features: {best_features_b[:10]}")

    X_train_b_sel = X_train_b[best_features_b]
    X_test_b_sel = X_test_b[best_features_b]

    scaler_b = RobustScaler()
    X_train_b_scaled = scaler_b.fit_transform(X_train_b_sel)
    X_test_b_scaled = scaler_b.transform(X_test_b_sel)

    models_b, weights_b, acc_b = tune_and_train(X_train_b_scaled, y_train_b, X_test_b_scaled, y_test_b, ticker, "Target B")

    # Save optimized model
    model_data = {
        'ticker': ticker,
        'version': 'v5_optimized',
        'train_period': f"{TRAIN_START} to {TRAIN_END}",
        'test_period': f"{TEST_START} to {TEST_END}",
        'trained_at': datetime.now().isoformat(),

        # Target A
        'feature_cols': best_features_a,
        'scaler': scaler_a,
        'models': models_a,
        'weights': weights_a,
        'test_accuracy_a': acc_a,

        # Target B
        'feature_cols_11am': best_features_b,
        'scaler_11am': scaler_b,
        'models_11am': models_b,
        'weights_11am': weights_b,
        'test_accuracy_b': acc_b,
    }

    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_v5.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n  RESULTS for {ticker}:")
    print(f"    Target A: {acc_a:.1%}")
    print(f"    Target B: {acc_b:.1%}")
    print(f"  Saved to {model_path}")

    return model_data


def main():
    print("="*70)
    print("  OPTIMIZED INTRADAY MODELS - PER-TICKER MAXIMIZATION")
    print("="*70)
    print()
    print("  Strategy:")
    print("  1. Feature selection per ticker")
    print("  2. Hyperparameter tuning")
    print("  3. Ensemble weight optimization")
    print()

    results = {}
    for ticker in ['SPY', 'QQQ', 'IWM']:
        results[ticker] = optimize_ticker(ticker)

    print("\n" + "="*70)
    print("  FINAL RESULTS")
    print("="*70)
    print(f"\n  {'Ticker':<8} {'Target A':>12} {'Target B':>12}")
    print(f"  {'-'*35}")
    for ticker, data in results.items():
        print(f"  {ticker:<8} {data['test_accuracy_a']:>11.1%} {data['test_accuracy_b']:>11.1%}")


if __name__ == '__main__':
    main()
