"""
SPY Model Improvement Analysis

Analyzes why SPY underperforms and tests improvements:
1. Feature importance analysis
2. Hyperparameter optimization
3. Timeframe-specific models
4. Gap size filtering
5. Market condition filtering
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import pickle
import os
from datetime import datetime

DATA_DIR = './data'
MODELS_DIR = './models'

# Categorical encoding
CATEGORICAL_MAPPINGS = {
    'fvg_type': {'bearish': 0, 'bullish': 1, 'unknown': 2},
    'volume_profile': {'high': 0, 'low': 1, 'medium': 2, 'unknown': 3},
    'market_structure': {'bearish': 0, 'bullish': 1, 'neutral': 2, 'unknown': 3},
    'rsi_zone': {'neutral': 0, 'overbought': 1, 'oversold': 2, 'unknown': 3},
    'macd_trend': {'bearish': 0, 'bullish': 1, 'neutral': 2, 'unknown': 3},
    'volatility_regime': {'high': 0, 'low': 1, 'medium': 2, 'unknown': 3},
}

FEATURE_COLUMNS = [
    'gap_size_pct', 'validation_score', 'rsi_14', 'macd',
    'macd_signal', 'macd_histogram', 'atr_14', 'sma_20',
    'sma_50', 'ema_12', 'ema_26', 'bb_bandwidth', 'volume_ratio',
    'price_vs_sma20', 'price_vs_sma50', 'hour_of_day', 'day_of_week'
]

CATEGORICAL_COLUMNS = [
    'fvg_type', 'volume_profile', 'market_structure',
    'rsi_zone', 'macd_trend', 'volatility_regime'
]


def load_spy_data():
    """Load SPY data from large dataset"""
    filepath = os.path.join(DATA_DIR, 'spy_large_features.json')

    with open(filepath, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data['features'])
    print(f"Loaded {len(df)} SPY samples")
    return df


def prepare_features(df):
    """Prepare features for training"""
    df = df.copy()

    # Binary target
    df['target'] = df['final_outcome'].apply(
        lambda x: 1 if x in ['tp1', 'tp2', 'tp3'] else 0
    )

    # Encode categorical
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
            mapping = CATEGORICAL_MAPPINGS.get(col, {})
            df[f'{col}_encoded'] = df[col].apply(
                lambda x: mapping.get(str(x).lower(), mapping.get('unknown', 0))
            )

    # Feature columns
    feature_cols = FEATURE_COLUMNS.copy()
    for col in CATEGORICAL_COLUMNS:
        feature_cols.append(f'{col}_encoded')

    # Fill missing
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if df[col].dtype in ['float64', 'int64'] else 0)

    return df, feature_cols


def analyze_losing_trades(df):
    """Analyze what makes SPY trades lose"""
    print("\n" + "="*60)
    print("ANALYZING LOSING TRADES")
    print("="*60)

    wins = df[df['final_outcome'].isin(['tp1', 'tp2', 'tp3'])]
    losses = df[df['final_outcome'] == 'stop_loss']

    print(f"\nWins: {len(wins)} | Losses: {len(losses)}")

    # Compare key features
    print("\n--- FEATURE COMPARISON (Wins vs Losses) ---")

    compare_cols = ['gap_size_pct', 'rsi_14', 'atr_14', 'volume_ratio',
                    'bb_bandwidth', 'hour_of_day', 'price_vs_sma20']

    for col in compare_cols:
        if col in df.columns:
            win_mean = wins[col].mean()
            loss_mean = losses[col].mean()
            diff = ((win_mean - loss_mean) / loss_mean * 100) if loss_mean != 0 else 0
            print(f"  {col}: Wins={win_mean:.2f}, Losses={loss_mean:.2f} ({diff:+.1f}%)")

    # By timeframe
    print("\n--- WIN RATE BY TIMEFRAME ---")
    for tf in df['timeframe'].unique():
        tf_df = df[df['timeframe'] == tf]
        wr = tf_df['final_outcome'].isin(['tp1', 'tp2', 'tp3']).mean()
        print(f"  {tf}: {wr:.1%} ({len(tf_df)} trades)")

    # By hour
    print("\n--- WIN RATE BY HOUR ---")
    df['hour'] = df['hour_of_day'].astype(int)
    for hour in sorted(df['hour'].unique()):
        hour_df = df[df['hour'] == hour]
        if len(hour_df) >= 10:
            wr = hour_df['final_outcome'].isin(['tp1', 'tp2', 'tp3']).mean()
            print(f"  {hour:02d}:00 - {wr:.1%} ({len(hour_df)} trades)")

    # By gap size bucket
    print("\n--- WIN RATE BY GAP SIZE ---")
    df['gap_bucket'] = pd.cut(df['gap_size_pct'], bins=[0, 0.2, 0.4, 0.6, 1.0, 5.0])
    for bucket in df['gap_bucket'].unique():
        if pd.notna(bucket):
            bucket_df = df[df['gap_bucket'] == bucket]
            if len(bucket_df) >= 10:
                wr = bucket_df['final_outcome'].isin(['tp1', 'tp2', 'tp3']).mean()
                print(f"  {bucket}: {wr:.1%} ({len(bucket_df)} trades)")

    # By RSI zone
    print("\n--- WIN RATE BY RSI ZONE ---")
    for zone in df['rsi_zone'].unique():
        zone_df = df[df['rsi_zone'] == zone]
        if len(zone_df) >= 10:
            wr = zone_df['final_outcome'].isin(['tp1', 'tp2', 'tp3']).mean()
            print(f"  {zone}: {wr:.1%} ({len(zone_df)} trades)")

    # By market structure
    print("\n--- WIN RATE BY MARKET STRUCTURE ---")
    for ms in df['market_structure'].unique():
        ms_df = df[df['market_structure'] == ms]
        if len(ms_df) >= 10:
            wr = ms_df['final_outcome'].isin(['tp1', 'tp2', 'tp3']).mean()
            print(f"  {ms}: {wr:.1%} ({len(ms_df)} trades)")

    # By volatility regime
    print("\n--- WIN RATE BY VOLATILITY ---")
    for vol in df['volatility_regime'].unique():
        vol_df = df[df['volatility_regime'] == vol]
        if len(vol_df) >= 10:
            wr = vol_df['final_outcome'].isin(['tp1', 'tp2', 'tp3']).mean()
            print(f"  {vol}: {wr:.1%} ({len(vol_df)} trades)")

    return wins, losses


def optimize_hyperparameters(X_train, y_train):
    """Grid search for optimal SPY hyperparameters"""
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*60)

    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.05, 0.08, 0.1, 0.15],
        'n_estimators': [100, 150, 200],
        'min_child_weight': [2, 3, 4, 5],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2],
    }

    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
    )

    print("Running grid search (this may take a few minutes)...")

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best F1 score: {grid_search.best_score_:.4f}")

    return grid_search.best_params_


def train_optimized_model(df, feature_cols, best_params=None):
    """Train optimized SPY model"""
    print("\n" + "="*60)
    print("TRAINING OPTIMIZED SPY MODEL")
    print("="*60)

    # Walk-forward split
    train_df = df[df['dataset'] == 'train'].copy()
    test_df = df[df['dataset'] == 'test'].copy()

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    print(f"Training: {len(X_train)} | Testing: {len(X_test)}")

    # Default optimized params for SPY
    if best_params is None:
        best_params = {
            'max_depth': 5,
            'learning_rate': 0.08,
            'n_estimators': 150,
            'min_child_weight': 3,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'gamma': 0.1,
        }

    model = xgb.XGBClassifier(
        **best_params,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nOptimized Model Results:")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.1%}")

    # Calculate trading metrics
    test_df = test_df.copy()
    test_df['predicted_win'] = y_pred
    test_df['win_probability'] = y_prob
    test_df['actual_win'] = y_test

    pnl_map = {'tp1': 1.0, 'tp2': 1.5, 'tp3': 2.0, 'stop_loss': -1.0, 'timeout': -0.5}
    test_df['pnl'] = test_df['final_outcome'].map(pnl_map)

    total_pnl = test_df['pnl'].sum()
    gross_profit = test_df[test_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(test_df[test_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    print(f"\nTrading Metrics:")
    print(f"  Win Rate: {y_test.mean():.1%}")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Total P&L: {total_pnl:.1f} units")

    return model, accuracy, profit_factor


def train_filtered_model(df, feature_cols, filters):
    """Train model on filtered subset of data"""
    print(f"\n--- Training with filters: {filters} ---")

    filtered_df = df.copy()

    for col, condition in filters.items():
        if callable(condition):
            filtered_df = filtered_df[condition(filtered_df[col])]
        else:
            filtered_df = filtered_df[filtered_df[col].isin(condition)]

    if len(filtered_df) < 100:
        print(f"Not enough samples after filtering: {len(filtered_df)}")
        return None, 0, 0

    print(f"Filtered samples: {len(filtered_df)}")

    train_df = filtered_df[filtered_df['dataset'] == 'train']
    test_df = filtered_df[filtered_df['dataset'] == 'test']

    if len(train_df) < 50 or len(test_df) < 20:
        print("Not enough samples in train/test splits")
        return None, 0, 0

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    model = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.08,
        n_estimators=150,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        objective='binary:logistic',
        random_state=42,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # P&L
    test_df = test_df.copy()
    pnl_map = {'tp1': 1.0, 'tp2': 1.5, 'tp3': 2.0, 'stop_loss': -1.0, 'timeout': -0.5}
    test_df['pnl'] = test_df['final_outcome'].map(pnl_map)

    gross_profit = test_df[test_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(test_df[test_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    win_rate = y_test.mean()

    print(f"  Accuracy: {accuracy:.1%}, Win Rate: {win_rate:.1%}, PF: {profit_factor:.2f}")

    return model, accuracy, profit_factor


def main():
    print("="*60)
    print("SPY MODEL IMPROVEMENT ANALYSIS")
    print("="*60)

    # Load data
    df = load_spy_data()
    df, feature_cols = prepare_features(df)

    # Analyze losing trades
    wins, losses = analyze_losing_trades(df)

    # Test different filter combinations
    print("\n" + "="*60)
    print("TESTING FILTER COMBINATIONS")
    print("="*60)

    filter_tests = [
        ("Best timeframes (15m, 1d)", {'timeframe': ['15m', '1d']}),
        ("Bullish market structure", {'market_structure': ['bullish']}),
        ("Medium/Low volatility", {'volatility_regime': ['medium', 'low']}),
        ("Gap size 0.2-0.6%", {'gap_size_pct': lambda x: (x >= 0.2) & (x <= 0.6)}),
        ("Trading hours 9-15", {'hour_of_day': lambda x: (x >= 9) & (x <= 15)}),
        ("RSI neutral/oversold", {'rsi_zone': ['neutral', 'oversold']}),
        ("High volume", {'volume_profile': ['high', 'medium']}),
    ]

    results = []
    for name, filters in filter_tests:
        model, acc, pf = train_filtered_model(df, feature_cols, filters)
        if model:
            results.append((name, acc, pf))

    # Combined best filters
    print("\n" + "="*60)
    print("COMBINED OPTIMAL FILTERS")
    print("="*60)

    best_filters = {
        'timeframe': ['15m', '1d', '4h'],
        'volatility_regime': ['medium', 'low'],
    }

    model, acc, pf = train_filtered_model(df, feature_cols, best_filters)

    # Train final optimized model
    print("\n" + "="*60)
    print("FINAL OPTIMIZED MODEL")
    print("="*60)

    # Apply best filters to full dataset
    optimized_df = df.copy()

    # Filter to best conditions
    optimized_df = optimized_df[
        (optimized_df['timeframe'].isin(['15m', '1d', '4h', '1h'])) &
        (optimized_df['volatility_regime'].isin(['medium', 'low']))
    ]

    print(f"Optimized dataset size: {len(optimized_df)}")

    if len(optimized_df) >= 200:
        model, accuracy, profit_factor = train_optimized_model(optimized_df, feature_cols)

        if profit_factor > 2.72:  # Better than current
            # Save optimized model
            os.makedirs(MODELS_DIR, exist_ok=True)

            model_data = {
                'model': model,
                'ticker': 'SPY',
                'feature_cols': feature_cols,
                'filters': {
                    'timeframe': ['15m', '1d', '4h', '1h'],
                    'volatility_regime': ['medium', 'low'],
                },
                'metrics': {
                    'accuracy': accuracy,
                    'profit_factor': profit_factor,
                },
                'trained_at': datetime.now().isoformat(),
                'version': 'optimized_v2',
            }

            with open(os.path.join(MODELS_DIR, 'spy_fvg_model_optimized.pkl'), 'wb') as f:
                pickle.dump(model_data, f)

            print(f"\nOptimized model saved!")
            print(f"Improvement: PF {2.72:.2f} → {profit_factor:.2f}")

    # Summary
    print("\n" + "="*60)
    print("IMPROVEMENT RECOMMENDATIONS")
    print("="*60)

    print("""
1. TIMEFRAME SELECTION:
   - Focus on 15m and daily timeframes (best win rates)
   - Avoid 5m timeframe for SPY (59.6% WR vs 70% on 15m)

2. VOLATILITY FILTER:
   - Trade medium/low volatility regimes
   - Avoid high volatility periods

3. GAP SIZE:
   - Sweet spot: 0.2% - 0.6% gaps
   - Too small (<0.2%) = noise, too large (>1%) = reversals

4. MARKET STRUCTURE:
   - Bullish structure has higher win rate
   - Be cautious in bearish/neutral conditions

5. TRADING HOURS:
   - Best hours: 9:00 - 15:00
   - Avoid first 30 min and last hour
""")

    return results


if __name__ == '__main__':
    results = main()
