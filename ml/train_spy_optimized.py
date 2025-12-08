"""
Train Optimized SPY Model

Based on analysis, the best filters for SPY are:
1. Timeframes: 15m, 1d (best win rates)
2. RSI zone: neutral, oversold (avoid overbought)
3. Volume profile: high, medium (avoid low volume)
4. Gap size: 0.1-0.4% (sweet spot)
5. Trading hours: 14:00-20:00 (afternoon/evening session)
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import pickle
import os
from datetime import datetime

DATA_DIR = './data'
MODELS_DIR = './models'

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
    filepath = os.path.join(DATA_DIR, 'spy_large_features.json')
    with open(filepath, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data['features'])


def prepare_features(df):
    df = df.copy()

    df['target'] = df['final_outcome'].apply(
        lambda x: 1 if x in ['tp1', 'tp2', 'tp3'] else 0
    )

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
            mapping = CATEGORICAL_MAPPINGS.get(col, {})
            df[f'{col}_encoded'] = df[col].apply(
                lambda x: mapping.get(str(x).lower(), mapping.get('unknown', 0))
            )

    feature_cols = FEATURE_COLUMNS.copy()
    for col in CATEGORICAL_COLUMNS:
        feature_cols.append(f'{col}_encoded')

    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if df[col].dtype in ['float64', 'int64'] else 0)

    return df, feature_cols


def apply_optimal_filters(df):
    """Apply the optimal trading filters discovered from analysis"""

    # Start with full dataset
    filtered = df.copy()
    print(f"Starting samples: {len(filtered)}")

    # Filter 1: Best timeframes (15m, 1d have 73.7% and 69.2% WR)
    filtered = filtered[filtered['timeframe'].isin(['15m', '1d'])]
    print(f"After timeframe filter (15m, 1d): {len(filtered)}")

    # Filter 2: RSI neutral/oversold (71.0-71.5% WR vs 62% overbought)
    filtered = filtered[filtered['rsi_zone'].isin(['neutral', 'oversold'])]
    print(f"After RSI filter: {len(filtered)}")

    # Filter 3: High/Medium volume (better signals)
    filtered = filtered[filtered['volume_profile'].isin(['high', 'medium'])]
    print(f"After volume filter: {len(filtered)}")

    # Filter 4: Smaller gaps (0.1-0.4% have 69-74% WR)
    filtered = filtered[(filtered['gap_size_pct'] >= 0.1) & (filtered['gap_size_pct'] <= 0.4)]
    print(f"After gap size filter: {len(filtered)}")

    # Filter 5: Best hours (14:00-20:00 have 75-84% WR)
    filtered = filtered[(filtered['hour_of_day'] >= 14) & (filtered['hour_of_day'] <= 20)]
    print(f"After hour filter: {len(filtered)}")

    return filtered


def calculate_pnl(outcome):
    pnl_map = {
        'tp1': 1.0, 'tp2': 1.5, 'tp3': 2.0,
        'stop_loss': -1.0, 'timeout': -0.5
    }
    return pnl_map.get(outcome, 0)


def train_and_evaluate(df, feature_cols, name="Model"):
    """Train model and return metrics"""

    train_df = df[df['dataset'] == 'train'].copy()
    test_df = df[df['dataset'] == 'test'].copy()

    if len(train_df) < 30 or len(test_df) < 10:
        print(f"Not enough data: train={len(train_df)}, test={len(test_df)}")
        return None

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    # Optimized hyperparameters for SPY
    model = xgb.XGBClassifier(
        max_depth=4,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.05,
        objective='binary:logistic',
        random_state=42,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Trading metrics
    test_df['pnl'] = test_df['final_outcome'].apply(calculate_pnl)
    total_pnl = test_df['pnl'].sum()
    win_rate = y_test.mean()

    gross_profit = test_df[test_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(test_df[test_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    print(f"\n{name} Results:")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.1%}")
    print(f"  Win Rate:  {win_rate:.1%}")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Total P&L: {total_pnl:.1f} units")

    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_pnl': total_pnl,
        'train_size': len(X_train),
        'test_size': len(X_test),
    }


def main():
    print("="*60)
    print("TRAINING OPTIMIZED SPY MODEL")
    print("="*60)

    # Load data
    df = load_spy_data()
    df, feature_cols = prepare_features(df)
    print(f"\nTotal SPY samples: {len(df)}")

    # Baseline (current model on all data)
    print("\n" + "="*60)
    print("BASELINE MODEL (ALL DATA)")
    print("="*60)
    baseline = train_and_evaluate(df, feature_cols, "Baseline")

    # Optimized with filters
    print("\n" + "="*60)
    print("OPTIMIZED MODEL (WITH FILTERS)")
    print("="*60)
    filtered_df = apply_optimal_filters(df)

    if len(filtered_df) < 50:
        print("Not enough samples after filtering, relaxing filters...")
        # Relax filters
        filtered_df = df[
            (df['timeframe'].isin(['15m', '1d'])) &
            (df['rsi_zone'].isin(['neutral', 'oversold'])) &
            (df['volume_profile'].isin(['high', 'medium']))
        ]
        print(f"Relaxed filter samples: {len(filtered_df)}")

    optimized = train_and_evaluate(filtered_df, feature_cols, "Optimized")

    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    if baseline and optimized:
        print(f"\n{'Metric':<20} {'Baseline':>12} {'Optimized':>12} {'Change':>12}")
        print("-" * 60)
        print(f"{'Win Rate':<20} {baseline['win_rate']:>11.1%} {optimized['win_rate']:>11.1%} {(optimized['win_rate']-baseline['win_rate'])*100:>+11.1f}%")
        print(f"{'Profit Factor':<20} {baseline['profit_factor']:>12.2f} {optimized['profit_factor']:>12.2f} {optimized['profit_factor']-baseline['profit_factor']:>+12.2f}")
        print(f"{'Total P&L':<20} {baseline['total_pnl']:>12.1f} {optimized['total_pnl']:>12.1f} {optimized['total_pnl']-baseline['total_pnl']:>+12.1f}")
        print(f"{'Trade Count':<20} {baseline['test_size']:>12} {optimized['test_size']:>12} {optimized['test_size']-baseline['test_size']:>+12}")

        # Save optimized model if better
        if optimized['profit_factor'] > baseline['profit_factor']:
            print("\n✓ Optimized model is better! Saving...")

            os.makedirs(MODELS_DIR, exist_ok=True)

            model_data = {
                'model': optimized['model'],
                'ticker': 'SPY',
                'feature_cols': feature_cols,
                'filters': {
                    'timeframe': ['15m', '1d'],
                    'rsi_zone': ['neutral', 'oversold'],
                    'volume_profile': ['high', 'medium'],
                    'gap_size_pct': [0.1, 0.4],
                    'hour_of_day': [14, 20],
                },
                'metrics': {
                    'accuracy': optimized['accuracy'],
                    'precision': optimized['precision'],
                    'recall': optimized['recall'],
                    'f1': optimized['f1'],
                    'win_rate': optimized['win_rate'],
                    'profit_factor': optimized['profit_factor'],
                },
                'trained_at': datetime.now().isoformat(),
                'version': 'optimized_v2',
            }

            # Save as the main SPY model
            with open(os.path.join(MODELS_DIR, 'spy_fvg_model.pkl'), 'wb') as f:
                pickle.dump(model_data, f)

            print(f"Saved optimized model to {MODELS_DIR}/spy_fvg_model.pkl")

            # Also save model info
            info = {
                'ticker': 'SPY',
                'version': 'optimized_v2',
                'filters': model_data['filters'],
                'metrics': {k: float(v) for k, v in model_data['metrics'].items()},
                'trained_at': model_data['trained_at'],
                'improvement': {
                    'profit_factor_before': baseline['profit_factor'],
                    'profit_factor_after': optimized['profit_factor'],
                    'win_rate_before': baseline['win_rate'],
                    'win_rate_after': optimized['win_rate'],
                }
            }

            with open(os.path.join(MODELS_DIR, 'spy_model_info.json'), 'w') as f:
                json.dump(info, f, indent=2)

    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == '__main__':
    main()
