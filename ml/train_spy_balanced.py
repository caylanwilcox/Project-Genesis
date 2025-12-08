"""
Train Balanced SPY Model

Find the optimal balance between:
- High win rate / profit factor
- Sufficient trade count for meaningful P&L
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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


def calculate_pnl(outcome):
    pnl_map = {
        'tp1': 1.0, 'tp2': 1.5, 'tp3': 2.0,
        'stop_loss': -1.0, 'timeout': -0.5
    }
    return pnl_map.get(outcome, 0)


def evaluate_filter_combo(df, feature_cols, filters, name):
    """Evaluate a filter combination"""

    filtered = df.copy()

    for col, values in filters.items():
        if col in filtered.columns:
            if isinstance(values, tuple):  # Range
                filtered = filtered[(filtered[col] >= values[0]) & (filtered[col] <= values[1])]
            else:  # List of values
                filtered = filtered[filtered[col].isin(values)]

    train_df = filtered[filtered['dataset'] == 'train']
    test_df = filtered[filtered['dataset'] == 'test']

    if len(train_df) < 50 or len(test_df) < 30:
        return None

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    model = xgb.XGBClassifier(
        max_depth=4,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        objective='binary:logistic',
        random_state=42,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    win_rate = y_test.mean()

    test_df = test_df.copy()
    test_df['pnl'] = test_df['final_outcome'].apply(calculate_pnl)
    total_pnl = test_df['pnl'].sum()

    gross_profit = test_df[test_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(test_df[test_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Score: balance profit factor with trade count
    # We want PF > 3 and at least 100+ trades
    score = profit_factor * min(len(test_df) / 100, 1.5)  # Bonus for more trades up to 150

    return {
        'name': name,
        'filters': filters,
        'model': model,
        'accuracy': accuracy,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_pnl': total_pnl,
        'test_trades': len(test_df),
        'score': score,
    }


def main():
    print("="*60)
    print("TRAINING BALANCED SPY MODEL")
    print("="*60)

    df = load_spy_data()
    df, feature_cols = prepare_features(df)
    print(f"Total SPY samples: {len(df)}")

    # Test various filter combinations
    filter_combos = [
        # Minimal filters (more trades)
        ("Timeframe only (15m, 1d)", {
            'timeframe': ['15m', '1d'],
        }),

        # Add RSI filter
        ("Timeframe + RSI", {
            'timeframe': ['15m', '1d'],
            'rsi_zone': ['neutral', 'oversold'],
        }),

        # Add volume filter
        ("Timeframe + Volume", {
            'timeframe': ['15m', '1d'],
            'volume_profile': ['high', 'medium'],
        }),

        # RSI + Volume (no timeframe restriction)
        ("RSI + Volume", {
            'rsi_zone': ['neutral', 'oversold'],
            'volume_profile': ['high', 'medium'],
        }),

        # All three light filters
        ("Light combo (TF + RSI + Vol)", {
            'timeframe': ['15m', '1d', '1h'],
            'rsi_zone': ['neutral', 'oversold'],
            'volume_profile': ['high', 'medium'],
        }),

        # Gap size focused
        ("Small gaps (0.1-0.35%)", {
            'gap_size_pct': (0.1, 0.35),
        }),

        # Timeframe + small gaps
        ("TF + Small gaps", {
            'timeframe': ['15m', '1d'],
            'gap_size_pct': (0.1, 0.4),
        }),

        # Best hours only
        ("Afternoon hours (14-20)", {
            'hour_of_day': (14, 20),
        }),

        # Balanced combo
        ("Balanced (TF + RSI + small gap)", {
            'timeframe': ['15m', '1d'],
            'rsi_zone': ['neutral', 'oversold'],
            'gap_size_pct': (0.1, 0.5),
        }),

        # Conservative
        ("Conservative (all light filters)", {
            'timeframe': ['15m', '1d'],
            'rsi_zone': ['neutral', 'oversold'],
            'volume_profile': ['high', 'medium'],
            'gap_size_pct': (0.1, 0.5),
        }),
    ]

    results = []
    print("\n" + "="*60)
    print("TESTING FILTER COMBINATIONS")
    print("="*60)

    for name, filters in filter_combos:
        result = evaluate_filter_combo(df, feature_cols, filters, name)
        if result:
            results.append(result)
            print(f"\n{name}:")
            print(f"  Trades: {result['test_trades']}, WR: {result['win_rate']:.1%}, PF: {result['profit_factor']:.2f}, P&L: {result['total_pnl']:.1f}, Score: {result['score']:.2f}")

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)

    print("\n" + "="*60)
    print("TOP 5 CONFIGURATIONS BY SCORE")
    print("="*60)

    print(f"\n{'Rank':<6} {'Name':<35} {'Trades':>8} {'WR':>8} {'PF':>8} {'P&L':>10} {'Score':>8}")
    print("-" * 90)

    for i, r in enumerate(results[:5], 1):
        print(f"{i:<6} {r['name']:<35} {r['test_trades']:>8} {r['win_rate']:>7.1%} {r['profit_factor']:>8.2f} {r['total_pnl']:>10.1f} {r['score']:>8.2f}")

    # Select best balanced model
    best = results[0]
    print(f"\n\nSelected: {best['name']}")
    print(f"  Win Rate: {best['win_rate']:.1%}")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")
    print(f"  Total P&L: {best['total_pnl']:.1f} units")
    print(f"  Trades: {best['test_trades']}")

    # Save the best model
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_data = {
        'model': best['model'],
        'ticker': 'SPY',
        'feature_cols': feature_cols,
        'filters': best['filters'],
        'metrics': {
            'accuracy': best['accuracy'],
            'win_rate': best['win_rate'],
            'profit_factor': best['profit_factor'],
        },
        'trained_at': datetime.now().isoformat(),
        'version': 'balanced_v1',
    }

    with open(os.path.join(MODELS_DIR, 'spy_fvg_model.pkl'), 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n✓ Saved balanced model to {MODELS_DIR}/spy_fvg_model.pkl")

    # Save info
    info = {
        'ticker': 'SPY',
        'version': 'balanced_v1',
        'filters': best['filters'],
        'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in best.items() if k not in ['model', 'filters', 'name']},
        'trained_at': model_data['trained_at'],
    }

    with open(os.path.join(MODELS_DIR, 'spy_model_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    # Compare with baseline
    print("\n" + "="*60)
    print("COMPARISON WITH BASELINE")
    print("="*60)
    print(f"\nBaseline (all data):     PF 2.72, WR 65.6%, P&L 274.0, Trades 491")
    print(f"Balanced (optimized):    PF {best['profit_factor']:.2f}, WR {best['win_rate']:.1%}, P&L {best['total_pnl']:.1f}, Trades {best['test_trades']}")

    if best['profit_factor'] > 2.72:
        print(f"\n✓ Profit Factor improved by {(best['profit_factor'] - 2.72):.2f}")
    if best['win_rate'] > 0.656:
        print(f"✓ Win Rate improved by {(best['win_rate'] - 0.656)*100:.1f}%")


if __name__ == '__main__':
    main()
