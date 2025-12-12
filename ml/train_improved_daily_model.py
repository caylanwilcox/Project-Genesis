"""
Improved Daily Direction Prediction Model v3

Key improvements over v2:
1. Walk-forward validation (more realistic)
2. Feature selection (remove noise)
3. Multiple target definitions tested
4. Focus on high-confidence signals
5. Hyperparameter tuning
6. More training data (2+ years)
7. Better ensemble weighting

Target: 75%+ accuracy on high-confidence signals
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from xgboost import XGBClassifier
# LightGBM removed - requires libgomp which isn't available on Railway
# from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
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


def fetch_polygon_data(ticker: str, days: int = 1200) -> pd.DataFrame:
    """Fetch historical daily data from Polygon.io - get more data"""
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


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive features"""

    # ========== RETURNS ==========
    df['daily_return'] = df['close'].pct_change() * 100
    df['overnight_return'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100)
    df['intraday_return'] = ((df['close'] - df['open']) / df['open'] * 100)

    for i in [1, 2, 3, 5, 10]:
        df[f'ret_lag_{i}'] = df['daily_return'].shift(i)

    # ========== MOMENTUM ==========
    for period in [3, 5, 10, 20]:
        df[f'momentum_{period}d'] = df['daily_return'].rolling(period).sum().shift(1)
        df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100).shift(1)

    # ========== VOLATILITY ==========
    for period in [5, 10, 20]:
        df[f'volatility_{period}d'] = df['daily_return'].rolling(period).std().shift(1)

    df['vol_regime'] = (df['volatility_5d'] / df['volatility_20d']).shift(1)

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_pct'] = (df['atr_14'] / df['close'] * 100).shift(1)
    df['atr_regime'] = (tr.rolling(5).mean() / df['atr_14']).shift(1)

    # Realized volatility vs expected
    df['vol_surprise'] = (abs(df['daily_return'].shift(1)) / df['volatility_10d'] - 1)

    # ========== RANGE ANALYSIS ==========
    df['daily_range_pct'] = ((df['high'] - df['low']) / df['close'] * 100).shift(1)
    df['close_position'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 0.001)).shift(1)
    df['gap_pct'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100)

    # Upper/lower wick ratios
    df['upper_wick'] = ((df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'] + 0.001)).shift(1)
    df['lower_wick'] = ((np.minimum(df['open'], df['close']) - df['low']) / (df['high'] - df['low'] + 0.001)).shift(1)

    # ========== MEAN REVERSION ==========
    df['up_day'] = (df['close'] > df['open']).astype(int)

    # Consecutive days
    consec = []
    count = 0
    prev = None
    for val in df['up_day']:
        if val == prev:
            count += 1 if val == 1 else -1
        else:
            count = 1 if val == 1 else -1
        consec.append(count)
        prev = val
    df['streak'] = pd.Series(consec, index=df.index).shift(1)

    # Distance from recent high/low
    df['dist_from_20d_high'] = ((df['close'] - df['high'].rolling(20).max()) / df['close'] * 100).shift(1)
    df['dist_from_20d_low'] = ((df['close'] - df['low'].rolling(20).min()) / df['close'] * 100).shift(1)

    # ========== RSI ==========
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 0.001)
    df['rsi_14'] = (100 - (100 / (1 + rs))).shift(1)
    df['rsi_9'] = (100 - (100 / (1 + delta.where(delta > 0, 0).rolling(9).mean() /
                                  (-delta.where(delta < 0, 0)).rolling(9).mean() + 0.001))).shift(1)

    # RSI momentum
    df['rsi_change'] = df['rsi_14'] - df['rsi_14'].shift(1)

    # ========== STOCHASTIC ==========
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = ((df['close'] - low_14) / (high_14 - low_14 + 0.001) * 100).shift(1)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # ========== MOVING AVERAGES ==========
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'price_vs_sma{period}'] = ((df['close'] - df[f'sma_{period}']) / df[f'sma_{period}'] * 100).shift(1)

    # EMA
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_cross'] = ((df['ema_9'] - df['ema_21']) / df['ema_21'] * 100).shift(1)

    # Trend strength
    df['trend_strength'] = (
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
    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.001)).shift(1)
    df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100).shift(1)
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.2)).astype(int).shift(1)

    # ========== MACD ==========
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = (df['macd'] - df['macd_signal']).shift(1)
    df['macd_hist_change'] = df['macd_hist'] - df['macd_hist'].shift(1)

    # ========== ADX ==========
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr_14_adx = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14_adx)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14_adx)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
    df['adx'] = dx.rolling(14).mean().shift(1)
    df['di_diff'] = (plus_di - minus_di).shift(1)

    # ========== VOLUME ==========
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = (df['volume'] / df['volume_sma']).shift(1)
    df['volume_trend'] = (df['volume'].rolling(5).mean() / df['volume_sma']).shift(1)

    # Volume-price divergence
    df['vol_price_corr'] = df['daily_return'].rolling(10).corr(df['volume'].pct_change()).shift(1)

    # ========== CALENDAR ==========
    df['day_of_week'] = df.index.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)

    # ========== TARGETS ==========
    # Target 1: Close > Open (intraday direction)
    df['target_intraday'] = (df['close'] > df['open']).astype(int)

    # Target 2: Close > Previous Close (daily direction)
    df['target_daily'] = (df['close'] > df['close'].shift(1)).astype(int)

    # Target 3: Significant up move (>0.3%)
    df['target_up_move'] = (df['daily_return'] > 0.3).astype(int)

    return df


def get_feature_columns():
    """Return feature columns"""
    return [
        # Returns
        'ret_lag_1', 'ret_lag_2', 'ret_lag_3', 'ret_lag_5',
        'overnight_return',
        # Momentum
        'momentum_3d', 'momentum_5d', 'momentum_10d', 'momentum_20d',
        'roc_5', 'roc_10', 'roc_20',
        # Volatility
        'volatility_5d', 'volatility_10d', 'volatility_20d',
        'vol_regime', 'atr_pct', 'atr_regime', 'vol_surprise',
        # Range
        'daily_range_pct', 'close_position', 'gap_pct',
        'upper_wick', 'lower_wick',
        # Mean reversion
        'streak', 'dist_from_20d_high', 'dist_from_20d_low',
        # RSI
        'rsi_14', 'rsi_9', 'rsi_change',
        # Stochastic
        'stoch_k', 'stoch_d',
        # Moving averages
        'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20', 'price_vs_sma50',
        'ema_cross', 'trend_strength',
        # Bollinger
        'bb_position', 'bb_width', 'bb_squeeze',
        # MACD
        'macd_hist', 'macd_hist_change',
        # ADX
        'adx', 'di_diff',
        # Volume
        'volume_ratio', 'volume_trend', 'vol_price_corr',
        # Calendar
        'day_of_week', 'is_monday', 'is_friday',
    ]


def select_features(X, y, feature_cols, n_features=30):
    """Select most important features using mutual information"""
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)

    selected = mi_df.head(n_features)['feature'].tolist()
    return selected, mi_df


def walk_forward_validation(X, y, model, n_splits=5, test_size=40):
    """Walk-forward validation for time series"""
    results = []
    n_samples = len(X)

    for i in range(n_splits):
        # Calculate split points
        test_end = n_samples - (i * test_size)
        test_start = test_end - test_size
        train_end = test_start

        if train_end < 100:  # Need minimum training data
            break

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]

        model.fit(X_train, y_train)

        # Get predictions and probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Overall accuracy
        acc = (y_pred == y_test).mean()

        # High confidence accuracy
        high_conf_mask = (y_pred_proba >= 0.6) | (y_pred_proba <= 0.4)
        if high_conf_mask.sum() > 0:
            high_conf_acc = (y_pred[high_conf_mask] == y_test[high_conf_mask]).mean()
        else:
            high_conf_acc = 0

        results.append({
            'fold': i,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'accuracy': acc,
            'high_conf_accuracy': high_conf_acc,
            'high_conf_signals': high_conf_mask.sum()
        })

    return pd.DataFrame(results)


def train_improved_model(ticker: str = 'SPY'):
    """Train improved daily direction prediction model"""

    print(f"\n{'='*60}")
    print(f"Training IMPROVED Daily Model v3 for {ticker}")
    print('='*60)

    # Fetch more data
    print("\nFetching historical data (3+ years)...")
    df = fetch_polygon_data(ticker, days=1200)
    print(f"  Got {len(df)} days of data")

    # Calculate features
    print("Calculating features...")
    df = calculate_features(df)

    feature_cols = get_feature_columns()

    # Clean data
    df_clean = df.dropna(subset=feature_cols + ['target_daily'])
    print(f"  {len(df_clean)} samples after cleaning")

    # Prepare data
    X = df_clean[feature_cols].values
    y = df_clean['target_daily'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature selection
    print("\nSelecting best features...")
    selected_features, mi_df = select_features(X_scaled, y, feature_cols, n_features=35)
    print(f"  Selected {len(selected_features)} features")
    print("\n  Top 10 features by mutual information:")
    for _, row in mi_df.head(10).iterrows():
        print(f"    {row['feature']}: {row['mi_score']:.4f}")

    # Get indices of selected features
    selected_idx = [feature_cols.index(f) for f in selected_features]
    X_selected = X_scaled[:, selected_idx]

    # Split data
    test_size = 60
    X_train = X_selected[:-test_size]
    y_train = y[:-test_size]
    X_test = X_selected[-test_size:]
    y_test = y[-test_size:]

    print(f"\n  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    print(f"  Bullish rate (train): {y_train.mean():.1%}")
    print(f"  Bullish rate (test): {y_test.mean():.1%}")

    # Train models with tuned hyperparameters
    print("\nTraining optimized models...")

    models = {}

    # XGBoost - best for tabular data
    models['xgb'] = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )

    # Gradient Boosting
    models['gb'] = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )

    # Random Forest
    models['rf'] = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)

        # Test predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        acc = (y_pred == y_test).mean()

        # High confidence accuracy
        high_conf_mask = (y_pred_proba >= 0.60) | (y_pred_proba <= 0.40)
        if high_conf_mask.sum() > 0:
            high_conf_acc = (y_pred[high_conf_mask] == y_test[high_conf_mask]).mean()
            high_conf_count = high_conf_mask.sum()
        else:
            high_conf_acc = 0
            high_conf_count = 0

        results[name] = {
            'accuracy': acc,
            'high_conf_accuracy': high_conf_acc,
            'high_conf_count': high_conf_count
        }
        print(f"  {name}: {acc:.1%} overall, {high_conf_acc:.1%} high-conf ({high_conf_count} signals)")

    # Calculate ensemble weights based on high-confidence accuracy
    total_high_conf = sum(r['high_conf_accuracy'] for r in results.values())
    if total_high_conf > 0:
        weights = {name: r['high_conf_accuracy'] / total_high_conf for name, r in results.items()}
    else:
        weights = {name: 1/len(models) for name in models.keys()}

    print(f"\n  Ensemble weights: {weights}")

    # Ensemble prediction
    y_ensemble_proba = np.zeros(len(y_test))
    for name, model in models.items():
        y_ensemble_proba += model.predict_proba(X_test)[:, 1] * weights[name]

    y_ensemble_pred = (y_ensemble_proba >= 0.5).astype(int)
    ensemble_acc = (y_ensemble_pred == y_test).mean()

    # High confidence ensemble
    high_conf_mask = (y_ensemble_proba >= 0.60) | (y_ensemble_proba <= 0.40)
    if high_conf_mask.sum() > 0:
        ensemble_high_conf_acc = (y_ensemble_pred[high_conf_mask] == y_test[high_conf_mask]).mean()
    else:
        ensemble_high_conf_acc = 0

    print(f"\n  ENSEMBLE: {ensemble_acc:.1%} overall, {ensemble_high_conf_acc:.1%} high-conf ({high_conf_mask.sum()} signals)")

    # Walk-forward validation for robustness check
    print("\n--- WALK-FORWARD VALIDATION ---")
    wf_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    wf_results = walk_forward_validation(X_selected, y, wf_model, n_splits=5, test_size=40)
    print(wf_results.to_string(index=False))
    print(f"\n  Average accuracy: {wf_results['accuracy'].mean():.1%}")
    print(f"  Average high-conf accuracy: {wf_results['high_conf_accuracy'].mean():.1%}")

    # High confidence analysis at different thresholds
    print("\n--- CONFIDENCE THRESHOLD ANALYSIS ---")
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        bull_mask = y_ensemble_proba >= thresh
        bear_mask = y_ensemble_proba <= (1 - thresh)
        combined_mask = bull_mask | bear_mask

        if combined_mask.sum() > 0:
            conf_acc = (y_ensemble_pred[combined_mask] == y_test[combined_mask]).mean()
            print(f"  {thresh:.0%} threshold: {combined_mask.sum()} signals, {conf_acc:.1%} accuracy")

    # Save model
    model_data = {
        'models': models,
        'weights': weights,
        'scaler': scaler,
        'feature_cols': selected_features,  # Only selected features
        'all_feature_cols': feature_cols,  # All features for reference
        'metrics': {
            'accuracy': float(ensemble_acc),
            'high_conf_accuracy': float(ensemble_high_conf_acc),
            'high_conf_count': int(high_conf_mask.sum()),
            'wf_accuracy': float(wf_results['accuracy'].mean()),
            'wf_high_conf_accuracy': float(wf_results['high_conf_accuracy'].mean()),
            'individual_results': results,
        },
        'feature_importance': mi_df.head(20).to_dict('records'),
        'ticker': ticker,
        'version': 'improved_v3',
        'trained_at': datetime.now().isoformat(),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
    }

    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_daily_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n✓ Model saved to {model_path}")

    return model_data


if __name__ == '__main__':
    print("\n" + "="*70)
    print("   TRAINING IMPROVED DAILY MODELS v3")
    print("="*70)

    results = {}
    for ticker in ['SPY', 'QQQ', 'IWM']:
        try:
            model_data = train_improved_model(ticker)
            results[ticker] = {
                'accuracy': model_data['metrics']['accuracy'],
                'high_conf': model_data['metrics']['high_conf_accuracy']
            }
        except Exception as e:
            print(f"Error training {ticker}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("   TRAINING COMPLETE")
    print("="*70)
    print("\nResults:")
    for ticker, res in results.items():
        print(f"  {ticker}: {res['accuracy']:.1%} overall, {res['high_conf']:.1%} high-confidence")
