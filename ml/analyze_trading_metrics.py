"""
Comprehensive Trading Metrics Analysis

Analyzes the walk-forward backtest results to calculate:
- Win rate, profit factor, Sharpe ratio
- Max drawdown, recovery time
- Per-ticker and per-timeframe performance
- Model calibration analysis
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os

MODELS_DIR = './models'
DATA_DIR = './data'


def load_test_data():
    """Load the 2024 test data that was already generated"""
    combined_path = os.path.join(DATA_DIR, 'combined_large_dataset.json')

    if not os.path.exists(combined_path):
        print("ERROR: No test data found. Run generate_large_dataset.py first.")
        return None

    with open(combined_path, 'r') as f:
        data = json.load(f)

    test_df = pd.DataFrame(data.get('test_features', []))
    print(f"Loaded {len(test_df)} test samples (2024 data)")

    return test_df


def load_model(ticker):
    """Load trained model"""
    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_fvg_model.pkl')

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    # Fallback to combined
    combined_path = os.path.join(MODELS_DIR, 'combined_fvg_model.pkl')
    if os.path.exists(combined_path):
        with open(combined_path, 'rb') as f:
            return pickle.load(f)

    return None


# Categorical encoding
CATEGORICAL_MAPPINGS = {
    'fvg_type': {'bearish': 0, 'bullish': 1, 'unknown': 2},
    'volume_profile': {'high': 0, 'low': 1, 'medium': 2, 'unknown': 3},
    'market_structure': {'bearish': 0, 'bullish': 1, 'neutral': 2, 'unknown': 3},
    'rsi_zone': {'neutral': 0, 'overbought': 1, 'oversold': 2, 'unknown': 3},
    'macd_trend': {'bearish': 0, 'bullish': 1, 'neutral': 2, 'unknown': 3},
    'volatility_regime': {'high': 0, 'low': 1, 'medium': 2, 'unknown': 3},
}


def build_features(row, feature_cols):
    """Build feature vector from row data"""
    features = {}

    numeric_cols = [
        'gap_size_pct', 'validation_score', 'rsi_14', 'macd',
        'macd_signal', 'macd_histogram', 'atr_14', 'sma_20',
        'sma_50', 'ema_12', 'ema_26', 'bb_bandwidth', 'volume_ratio',
        'price_vs_sma20', 'price_vs_sma50', 'hour_of_day', 'day_of_week'
    ]

    for col in numeric_cols:
        features[col] = row.get(col, 0) or 0

    # Encode categorical
    for cat_col, mapping in CATEGORICAL_MAPPINGS.items():
        value = str(row.get(cat_col, 'unknown')).lower()
        encoded = mapping.get(value, mapping.get('unknown', 0))
        features[f'{cat_col}_encoded'] = encoded

    df = pd.DataFrame([features])

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols]


def calculate_pnl(outcome):
    """Calculate P&L based on outcome (in risk units)"""
    pnl_map = {
        'tp1': 1.0,      # 1:1 RR
        'tp2': 1.5,      # 1.5:1 RR
        'tp3': 2.0,      # 2:1 RR
        'stop_loss': -1.0,
        'timeout': -0.5   # Partial loss for timeout
    }
    return pnl_map.get(outcome, 0)


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """Calculate Sharpe Ratio"""
    if len(returns) < 2:
        return 0

    excess_returns = returns - risk_free_rate / periods_per_year
    if excess_returns.std() == 0:
        return 0

    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)


def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown"""
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def analyze_ticker(ticker, test_df):
    """Analyze performance for a single ticker"""
    print(f"\n{'='*60}")
    print(f"ANALYZING {ticker}")
    print(f"{'='*60}")

    # Filter for this ticker
    ticker_df = test_df[test_df['ticker'].str.upper() == ticker.upper()].copy()

    if len(ticker_df) == 0:
        print(f"No test data for {ticker}")
        return None

    print(f"Test samples: {len(ticker_df)}")

    # Load model
    model_data = load_model(ticker)
    if model_data is None:
        print(f"No model for {ticker}")
        return None

    # Make predictions
    predictions = []
    for idx, row in ticker_df.iterrows():
        features = build_features(row.to_dict(), model_data['feature_cols'])
        prob = model_data['model'].predict_proba(features)[0][1]
        predictions.append(prob)

    ticker_df['win_probability'] = predictions
    ticker_df['predicted_win'] = ticker_df['win_probability'] >= 0.5
    ticker_df['actual_win'] = ticker_df['final_outcome'].isin(['tp1', 'tp2', 'tp3'])
    ticker_df['correct'] = ticker_df['predicted_win'] == ticker_df['actual_win']
    ticker_df['pnl'] = ticker_df['final_outcome'].apply(calculate_pnl)

    # Basic metrics
    accuracy = ticker_df['correct'].mean()
    win_rate = ticker_df['actual_win'].mean()

    # Trading metrics
    total_trades = len(ticker_df)
    wins = ticker_df['actual_win'].sum()
    losses = total_trades - wins

    # P&L metrics
    total_pnl = ticker_df['pnl'].sum()
    avg_pnl = ticker_df['pnl'].mean()

    # Profit factor
    gross_profit = ticker_df[ticker_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(ticker_df[ticker_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe ratio (using individual trade returns)
    sharpe = calculate_sharpe_ratio(ticker_df['pnl'])

    # Max drawdown
    cumulative_pnl = ticker_df['pnl'].cumsum()
    max_dd = calculate_max_drawdown(cumulative_pnl + 100)  # Start with 100 units

    # Outcome distribution
    outcome_dist = ticker_df['final_outcome'].value_counts()

    # High-confidence trades only
    high_conf = ticker_df[ticker_df['win_probability'] >= 0.6]
    hc_win_rate = high_conf['actual_win'].mean() if len(high_conf) > 0 else 0
    hc_profit_factor = (
        high_conf[high_conf['pnl'] > 0]['pnl'].sum() /
        abs(high_conf[high_conf['pnl'] < 0]['pnl'].sum())
    ) if len(high_conf) > 0 and abs(high_conf[high_conf['pnl'] < 0]['pnl'].sum()) > 0 else 0

    # Print results
    print(f"\n--- MODEL PERFORMANCE ---")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Model trained accuracy: {model_data['metrics']['accuracy']:.1%}")
    print(f"  Difference: {(accuracy - model_data['metrics']['accuracy'])*100:+.1f}%")

    print(f"\n--- TRADING METRICS ---")
    print(f"  Total Trades: {total_trades}")
    print(f"  Wins: {wins} | Losses: {losses}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Total P&L: {total_pnl:.1f} units")
    print(f"  Avg P&L/Trade: {avg_pnl:.3f} units")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd:.1%}")

    print(f"\n--- HIGH CONFIDENCE TRADES (>60% prob) ---")
    print(f"  Count: {len(high_conf)}")
    print(f"  Win Rate: {hc_win_rate:.1%}")
    print(f"  Profit Factor: {hc_profit_factor:.2f}")

    print(f"\n--- OUTCOME DISTRIBUTION ---")
    for outcome, count in outcome_dist.items():
        pct = count / total_trades * 100
        print(f"  {outcome}: {count} ({pct:.1f}%)")

    # By timeframe
    if 'timeframe' in ticker_df.columns:
        print(f"\n--- BY TIMEFRAME ---")
        for tf, group in ticker_df.groupby('timeframe'):
            tf_wr = group['actual_win'].mean()
            tf_pf = (
                group[group['pnl'] > 0]['pnl'].sum() /
                abs(group[group['pnl'] < 0]['pnl'].sum())
            ) if abs(group[group['pnl'] < 0]['pnl'].sum()) > 0 else 0
            print(f"  {tf}: {len(group)} trades, WR {tf_wr:.1%}, PF {tf_pf:.2f}")

    return {
        'ticker': ticker,
        'total_trades': total_trades,
        'accuracy': accuracy,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'high_conf_count': len(high_conf),
        'high_conf_win_rate': hc_win_rate,
        'high_conf_profit_factor': hc_profit_factor,
        'outcome_distribution': outcome_dist.to_dict(),
    }


def main():
    print("="*60)
    print("COMPREHENSIVE TRADING METRICS ANALYSIS")
    print("Walk-Forward Validation on 2024 Data")
    print("="*60)

    # Load test data
    test_df = load_test_data()
    if test_df is None:
        return

    # Show test data summary
    print(f"\nTest Period: 2024-01-01 to today")
    print(f"Total samples: {len(test_df)}")
    if 'ticker' in test_df.columns:
        print(f"\nSamples per ticker:")
        print(test_df['ticker'].value_counts().to_string())

    # Analyze each ticker
    results = {}
    for ticker in ['SPY', 'QQQ', 'IWM']:
        result = analyze_ticker(ticker, test_df)
        if result:
            results[ticker] = result

    # Combined analysis
    print("\n" + "="*60)
    print("COMBINED ANALYSIS (ALL TICKERS)")
    print("="*60)
    combined = analyze_ticker('ALL', test_df)  # Uses combined model

    # Summary table
    print("\n" + "="*70)
    print("FINAL SUMMARY - OUT-OF-TIME PERFORMANCE (2024)")
    print("="*70)

    print("\n{:<8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Ticker", "Trades", "Win Rate", "PF", "Sharpe", "Max DD", "P&L"
    ))
    print("-" * 70)

    for ticker, r in results.items():
        print("{:<8} {:>8} {:>9.1%} {:>10.2f} {:>10.2f} {:>9.1%} {:>10.1f}".format(
            ticker,
            r['total_trades'],
            r['win_rate'],
            r['profit_factor'],
            r['sharpe_ratio'],
            r['max_drawdown'],
            r['total_pnl']
        ))

    # Key takeaways
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)

    total_pnl = sum(r['total_pnl'] for r in results.values())
    total_trades = sum(r['total_trades'] for r in results.values())
    avg_win_rate = np.mean([r['win_rate'] for r in results.values()])
    avg_pf = np.mean([r['profit_factor'] for r in results.values()])

    print(f"\n1. OVERALL PERFORMANCE:")
    print(f"   - Total trades tested: {total_trades}")
    print(f"   - Average win rate: {avg_win_rate:.1%}")
    print(f"   - Average profit factor: {avg_pf:.2f}")
    print(f"   - Total P&L: {total_pnl:.1f} units")

    print(f"\n2. MODEL RELIABILITY:")
    for ticker, r in results.items():
        model_data = load_model(ticker)
        if model_data:
            train_acc = model_data['metrics']['accuracy']
            test_acc = r['accuracy']
            diff = test_acc - train_acc
            print(f"   - {ticker}: Train {train_acc:.1%} → Test {test_acc:.1%} ({diff*100:+.1f}%)")

    print(f"\n3. BEST PERFORMER:")
    best = max(results.items(), key=lambda x: x[1]['profit_factor'])
    print(f"   - {best[0]} with PF {best[1]['profit_factor']:.2f}")

    print(f"\n4. HIGH CONFIDENCE TRADES:")
    for ticker, r in results.items():
        if r['high_conf_count'] > 0:
            print(f"   - {ticker}: {r['high_conf_count']} trades, WR {r['high_conf_win_rate']:.1%}, PF {r['high_conf_profit_factor']:.2f}")

    # Save results
    summary = {
        'analyzed_at': datetime.now().isoformat(),
        'test_period': '2024-01-01 to today',
        'total_trades': total_trades,
        'total_pnl': total_pnl,
        'results': {
            ticker: {k: v for k, v in r.items() if k != 'outcome_distribution'}
            for ticker, r in results.items()
        }
    }

    with open('trading_metrics_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to trading_metrics_analysis.json")

    return results


if __name__ == '__main__':
    results = main()
