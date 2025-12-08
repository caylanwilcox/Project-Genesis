"""
Verify SPY Improvement

Compare before/after optimization metrics on 2024 test data
"""

import json
import pandas as pd
import numpy as np
import pickle
import os

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


def load_test_data():
    with open(os.path.join(DATA_DIR, 'combined_large_dataset.json'), 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data.get('test_features', []))


def load_model():
    with open(os.path.join(MODELS_DIR, 'spy_fvg_model.pkl'), 'rb') as f:
        return pickle.load(f)


def calculate_pnl(outcome):
    pnl_map = {
        'tp1': 1.0, 'tp2': 1.5, 'tp3': 2.0,
        'stop_loss': -1.0, 'timeout': -0.5
    }
    return pnl_map.get(outcome, 0)


def analyze_with_filters(test_df, filters=None, name="Analysis"):
    """Analyze SPY performance with optional filters"""

    spy_df = test_df[test_df['ticker'].str.upper() == 'SPY'].copy()

    if filters:
        for col, values in filters.items():
            if col in spy_df.columns:
                if isinstance(values, list):
                    spy_df = spy_df[spy_df[col].isin(values)]
                elif isinstance(values, tuple):
                    spy_df = spy_df[(spy_df[col] >= values[0]) & (spy_df[col] <= values[1])]

    if len(spy_df) == 0:
        print(f"{name}: No trades after filtering")
        return None

    # Calculate metrics
    spy_df['pnl'] = spy_df['final_outcome'].apply(calculate_pnl)
    spy_df['win'] = spy_df['final_outcome'].isin(['tp1', 'tp2', 'tp3'])

    total_trades = len(spy_df)
    wins = spy_df['win'].sum()
    win_rate = wins / total_trades

    total_pnl = spy_df['pnl'].sum()
    avg_pnl = spy_df['pnl'].mean()

    gross_profit = spy_df[spy_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(spy_df[spy_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Sharpe
    if len(spy_df) > 1:
        sharpe = (spy_df['pnl'].mean() / spy_df['pnl'].std()) * np.sqrt(252) if spy_df['pnl'].std() > 0 else 0
    else:
        sharpe = 0

    return {
        'name': name,
        'trades': total_trades,
        'wins': wins,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'sharpe': sharpe,
    }


def main():
    print("="*60)
    print("SPY IMPROVEMENT VERIFICATION")
    print("2024 Out-of-Time Test Data")
    print("="*60)

    test_df = load_test_data()
    print(f"\nTotal test samples: {len(test_df)}")

    spy_count = len(test_df[test_df['ticker'].str.upper() == 'SPY'])
    print(f"SPY test samples: {spy_count}")

    # Load model to get filters
    model_data = load_model()
    filters = model_data.get('filters', {})
    print(f"\nOptimized filters: {filters}")

    # Baseline (all SPY trades)
    print("\n" + "="*60)
    print("BASELINE vs OPTIMIZED COMPARISON")
    print("="*60)

    baseline = analyze_with_filters(test_df, filters=None, name="Baseline (all trades)")

    # With optimized filters
    optimized = analyze_with_filters(test_df, filters=filters, name="Optimized (filtered)")

    # Print comparison
    print(f"\n{'Metric':<20} {'Baseline':>15} {'Optimized':>15} {'Change':>15}")
    print("-" * 65)

    if baseline and optimized:
        print(f"{'Trades':<20} {baseline['trades']:>15} {optimized['trades']:>15} {optimized['trades'] - baseline['trades']:>+15}")
        print(f"{'Win Rate':<20} {baseline['win_rate']:>14.1%} {optimized['win_rate']:>14.1%} {(optimized['win_rate'] - baseline['win_rate'])*100:>+14.1f}%")
        print(f"{'Profit Factor':<20} {baseline['profit_factor']:>15.2f} {optimized['profit_factor']:>15.2f} {optimized['profit_factor'] - baseline['profit_factor']:>+15.2f}")
        print(f"{'Total P&L':<20} {baseline['total_pnl']:>15.1f} {optimized['total_pnl']:>15.1f} {optimized['total_pnl'] - baseline['total_pnl']:>+15.1f}")
        print(f"{'Avg P&L/Trade':<20} {baseline['avg_pnl']:>15.3f} {optimized['avg_pnl']:>15.3f} {optimized['avg_pnl'] - baseline['avg_pnl']:>+15.3f}")
        print(f"{'Sharpe Ratio':<20} {baseline['sharpe']:>15.2f} {optimized['sharpe']:>15.2f} {optimized['sharpe'] - baseline['sharpe']:>+15.2f}")

    # Key insight
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)

    if baseline and optimized:
        pf_improvement = (optimized['profit_factor'] / baseline['profit_factor'] - 1) * 100
        wr_improvement = (optimized['win_rate'] - baseline['win_rate']) * 100

        print(f"""
PROFIT FACTOR: {baseline['profit_factor']:.2f} → {optimized['profit_factor']:.2f} ({pf_improvement:+.0f}% improvement)
WIN RATE:      {baseline['win_rate']:.1%} → {optimized['win_rate']:.1%} ({wr_improvement:+.1f}% improvement)

The optimized filters trade fewer opportunities ({optimized['trades']} vs {baseline['trades']})
but with significantly better quality:
  - Each trade is {optimized['avg_pnl']/baseline['avg_pnl']:.1f}x more profitable on average
  - Profit factor improved by {pf_improvement:.0f}%
  - Win rate improved by {wr_improvement:.1f} percentage points

RECOMMENDED USAGE:
When the prediction server receives a SPY FVG, apply these filters:
  Timeframe: {filters.get('timeframe', 'any')}
  RSI Zone: {filters.get('rsi_zone', 'any')}
  Volume Profile: {filters.get('volume_profile', 'any')}

If the FVG passes these filters, it's a high-quality trade.
If not, consider skipping or reducing position size.
""")

    # By timeframe breakdown for optimized
    print("\n" + "="*60)
    print("OPTIMIZED SPY BY TIMEFRAME")
    print("="*60)

    spy_test = test_df[test_df['ticker'].str.upper() == 'SPY'].copy()

    for tf in ['5m', '15m', '1h', '4h', '1d']:
        tf_filters = {**filters, 'timeframe': [tf]} if filters else {'timeframe': [tf]}
        result = analyze_with_filters(test_df, filters=tf_filters, name=tf)
        if result and result['trades'] > 0:
            print(f"  {tf}: {result['trades']} trades, WR {result['win_rate']:.1%}, PF {result['profit_factor']:.2f}, P&L {result['total_pnl']:.1f}")


if __name__ == '__main__':
    main()
