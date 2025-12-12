"""
Backtest for BUY/SELL/HOLD Daily Signals

This script backtests the actionable trading signals:
- STRONG BUY (>=70% bullish)
- BUY (55-70% bullish)
- HOLD (45-55% bullish)
- SELL (30-45% bullish)
- STRONG SELL (<=30% bullish)

Metrics tracked:
- Win rate per signal type
- Average return per signal
- Risk/reward accuracy
- Profit factor
- Maximum drawdown
"""

import pandas as pd
import numpy as np
import pickle
import os
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from train_optimized_daily_model import calculate_optimized_features

# Polygon API
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
TICKERS = ['SPY', 'QQQ', 'IWM']


def fetch_polygon_data(ticker: str, days: int = 500) -> pd.DataFrame:
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
        'o': 'Open',
        'h': 'High',
        'l': 'Low',
        'c': 'Close',
        'v': 'Volume'
    })
    df = df.set_index('date')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    return df


def calculate_daily_features(df):
    """Use optimized feature calculation - wrapper for compatibility"""
    # Rename columns to lowercase for the optimized function
    df_lower = df.copy()
    df_lower.columns = [c.lower() for c in df_lower.columns]
    df_lower = calculate_optimized_features(df_lower)
    # Rename columns back to title case
    df_lower.columns = [c.title() if c in ['open', 'high', 'low', 'close', 'volume'] else c for c in df_lower.columns]
    return df_lower


def get_signal(bullish_prob):
    """Convert probability to signal"""
    if bullish_prob >= 0.70:
        return 'STRONG_BUY', 'STRONG'
    elif bullish_prob >= 0.60:
        return 'BUY', 'MODERATE'
    elif bullish_prob >= 0.55:
        return 'BUY', 'WEAK'
    elif bullish_prob <= 0.30:
        return 'STRONG_SELL', 'STRONG'
    elif bullish_prob <= 0.40:
        return 'SELL', 'MODERATE'
    elif bullish_prob <= 0.45:
        return 'SELL', 'WEAK'
    else:
        return 'HOLD', 'NEUTRAL'


def backtest_ticker(ticker: str, test_days: int = 60):
    """Backtest signals for a single ticker"""
    print(f"\n{'='*60}")
    print(f"Backtesting {ticker}")
    print('='*60)

    # Load model
    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_daily_model.pkl')
    if not os.path.exists(model_path):
        print(f"  No model found for {ticker}")
        return None

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    # Load high/low model for target predictions
    highlow_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_highlow_model.pkl')
    highlow_model = None
    if os.path.exists(highlow_path):
        with open(highlow_path, 'rb') as f:
            highlow_model = pickle.load(f)

    # Fetch data
    print("  Fetching data...")
    df = fetch_polygon_data(ticker, days=400)
    df = calculate_daily_features(df)

    feature_cols = model_data['feature_cols']
    weights = model_data['weights']

    # Drop NaN rows
    df_clean = df.dropna(subset=feature_cols)

    # Use last N days for backtesting
    df_test = df_clean.iloc[-test_days:]
    print(f"  Testing on {len(df_test)} days")

    results = []

    for i, (date, row) in enumerate(df_test.iterrows()):
        # Build features
        features = {col: row[col] if col in row else 0 for col in feature_cols}
        X = pd.DataFrame([features])[feature_cols]
        X_scaled = model_data['scaler'].transform(X)

        # Get ensemble probability
        rf_prob = model_data['models']['rf'].predict_proba(X_scaled)[0][1]
        gb_prob = model_data['models']['gb'].predict_proba(X_scaled)[0][1]
        lr_prob = model_data['models']['lr'].predict_proba(X_scaled)[0][1]

        bullish_prob = (
            rf_prob * weights['rf'] +
            gb_prob * weights['gb'] +
            lr_prob * weights['lr']
        )

        signal, strength = get_signal(bullish_prob)

        # Actual outcome
        actual_return = row['daily_return']  # This is today's return
        actual_direction = 'UP' if actual_return > 0 else 'DOWN'

        # Get today's actual high/low from open for target accuracy
        open_price = row['Open']
        high_price = row['High']
        low_price = row['Low']
        close_price = row['Close']

        actual_high_pct = ((high_price - open_price) / open_price) * 100
        actual_low_pct = ((open_price - low_price) / open_price) * 100

        # Determine if signal was correct
        if signal in ['STRONG_BUY', 'BUY']:
            # For BUY signals, we're correct if price went up
            is_correct = actual_return > 0
            # P/L is the actual return (long position)
            pnl = actual_return
        elif signal in ['STRONG_SELL', 'SELL']:
            # For SELL signals, we're correct if price went down
            is_correct = actual_return < 0
            # P/L is inverse of return (short position)
            pnl = -actual_return
        else:
            # HOLD - no position
            is_correct = None
            pnl = 0

        results.append({
            'date': date,
            'signal': signal,
            'strength': strength,
            'probability': bullish_prob,
            'actual_return': actual_return,
            'actual_direction': actual_direction,
            'is_correct': is_correct,
            'pnl': pnl,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'actual_high_pct': actual_high_pct,
            'actual_low_pct': actual_low_pct,
        })

    return pd.DataFrame(results)


def analyze_results(results_df, ticker):
    """Analyze backtest results"""
    print(f"\n{'='*60}")
    print(f"RESULTS FOR {ticker}")
    print('='*60)

    # Overall stats
    total_signals = len(results_df)
    print(f"\nTotal Trading Days: {total_signals}")

    # Signal distribution
    print("\n--- SIGNAL DISTRIBUTION ---")
    signal_counts = results_df['signal'].value_counts()
    for signal, count in signal_counts.items():
        pct = count / total_signals * 100
        print(f"  {signal}: {count} ({pct:.1f}%)")

    # Filter to actionable signals (not HOLD)
    trades = results_df[results_df['signal'] != 'HOLD'].copy()
    holds = results_df[results_df['signal'] == 'HOLD']

    print(f"\n--- TRADE SIGNALS (excluding HOLD) ---")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Days Held (no trade): {len(holds)}")

    if len(trades) == 0:
        print("  No actionable signals generated!")
        return None

    # Win rate by signal type
    print("\n--- WIN RATE BY SIGNAL ---")
    for signal in ['STRONG_BUY', 'BUY', 'SELL', 'STRONG_SELL']:
        signal_trades = trades[trades['signal'] == signal]
        if len(signal_trades) > 0:
            wins = int(signal_trades['is_correct'].fillna(False).astype(int).sum())
            win_rate = wins / len(signal_trades) * 100
            avg_pnl = signal_trades['pnl'].mean()
            print(f"  {signal}: {wins}/{len(signal_trades)} = {win_rate:.1f}% (avg P/L: {avg_pnl:+.2f}%)")

    # Win rate by strength
    print("\n--- WIN RATE BY STRENGTH ---")
    for strength in ['STRONG', 'MODERATE', 'WEAK']:
        strength_trades = trades[trades['strength'] == strength]
        if len(strength_trades) > 0:
            wins = int(strength_trades['is_correct'].fillna(False).astype(int).sum())
            win_rate = wins / len(strength_trades) * 100
            avg_pnl = strength_trades['pnl'].mean()
            print(f"  {strength}: {wins}/{len(strength_trades)} = {win_rate:.1f}% (avg P/L: {avg_pnl:+.2f}%)")

    # BUY vs SELL performance
    print("\n--- BUY vs SELL SIGNALS ---")
    buys = trades[trades['signal'].str.contains('BUY')]
    sells = trades[trades['signal'].str.contains('SELL')]

    if len(buys) > 0:
        buy_wins = int(buys['is_correct'].fillna(False).astype(int).sum())
        buy_wr = buy_wins / len(buys) * 100
        buy_pnl = buys['pnl'].sum()
        buy_avg = buys['pnl'].mean()
        print(f"  BUY signals: {buy_wins}/{len(buys)} = {buy_wr:.1f}% win rate")
        print(f"    Total P/L: {buy_pnl:+.2f}%, Avg: {buy_avg:+.2f}%")

    if len(sells) > 0:
        sell_wins = int(sells['is_correct'].fillna(False).astype(int).sum())
        sell_wr = sell_wins / len(sells) * 100
        sell_pnl = sells['pnl'].sum()
        sell_avg = sells['pnl'].mean()
        print(f"  SELL signals: {sell_wins}/{len(sells)} = {sell_wr:.1f}% win rate")
        print(f"    Total P/L: {sell_pnl:+.2f}%, Avg: {sell_avg:+.2f}%")

    # Overall trading stats
    print("\n--- OVERALL TRADING PERFORMANCE ---")
    total_wins = int(trades['is_correct'].fillna(False).astype(int).sum())
    overall_wr = total_wins / len(trades) * 100
    total_pnl = trades['pnl'].sum()
    avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if len(trades[trades['pnl'] > 0]) > 0 else 0
    avg_loss = trades[trades['pnl'] < 0]['pnl'].mean() if len(trades[trades['pnl'] < 0]) > 0 else 0

    print(f"  Overall Win Rate: {overall_wr:.1f}%")
    print(f"  Total P/L: {total_pnl:+.2f}%")
    print(f"  Average Win: {avg_win:+.2f}%")
    print(f"  Average Loss: {avg_loss:+.2f}%")

    # Profit factor
    gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    print(f"  Profit Factor: {profit_factor:.2f}")

    # Max drawdown (cumulative)
    cumulative_pnl = trades['pnl'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max()
    print(f"  Max Drawdown: {max_drawdown:.2f}%")

    # Sharpe-like ratio (simplified)
    if trades['pnl'].std() > 0:
        sharpe = trades['pnl'].mean() / trades['pnl'].std() * np.sqrt(252)
        print(f"  Annualized Sharpe: {sharpe:.2f}")

    return {
        'ticker': ticker,
        'total_days': total_signals,
        'trade_signals': len(trades),
        'hold_signals': len(holds),
        'overall_win_rate': overall_wr,
        'total_pnl': total_pnl,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
    }


def run_full_backtest(test_days=60):
    """Run backtest for all tickers"""
    print("\n" + "="*70)
    print("   DAILY SIGNALS BACKTEST - BUY/SELL/HOLD")
    print("="*70)
    print(f"\nTest Period: Last {test_days} trading days")
    print(f"Tickers: {', '.join(TICKERS)}")

    all_results = {}
    all_stats = []

    for ticker in TICKERS:
        try:
            results_df = backtest_ticker(ticker, test_days)
            if results_df is not None:
                all_results[ticker] = results_df
                stats = analyze_results(results_df, ticker)
                if stats:
                    all_stats.append(stats)
        except Exception as e:
            print(f"Error backtesting {ticker}: {e}")

    # Combined summary
    if all_stats:
        print("\n" + "="*70)
        print("   COMBINED SUMMARY")
        print("="*70)

        combined_df = pd.concat(all_results.values(), ignore_index=True)
        trades = combined_df[combined_df['signal'] != 'HOLD']

        if len(trades) > 0:
            total_wins = int(trades['is_correct'].fillna(False).astype(int).sum())
            overall_wr = total_wins / len(trades) * 100
            total_pnl = trades['pnl'].sum()

            print(f"\nAll Tickers Combined:")
            print(f"  Total Trade Signals: {len(trades)}")
            print(f"  Overall Win Rate: {overall_wr:.1f}%")
            print(f"  Total P/L: {total_pnl:+.2f}%")

            # By signal type
            print("\n  By Signal Type:")
            for signal in ['STRONG_BUY', 'BUY', 'SELL', 'STRONG_SELL']:
                sig_trades = trades[trades['signal'] == signal]
                if len(sig_trades) > 0:
                    wins = int(sig_trades['is_correct'].fillna(False).astype(int).sum())
                    wr = wins / len(sig_trades) * 100
                    pnl = sig_trades['pnl'].sum()
                    print(f"    {signal}: {wr:.1f}% WR ({len(sig_trades)} trades, {pnl:+.2f}% P/L)")

            # Summary table
            print("\n  Ticker Summary:")
            print(f"  {'Ticker':<8} {'WinRate':<10} {'Trades':<8} {'P/L':<10} {'PF':<8}")
            print(f"  {'-'*44}")
            for stat in all_stats:
                print(f"  {stat['ticker']:<8} {stat['overall_win_rate']:.1f}%{'':<5} {stat['trade_signals']:<8} {stat['total_pnl']:+.2f}%{'':<4} {stat['profit_factor']:.2f}")

    return all_results, all_stats


if __name__ == '__main__':
    results, stats = run_full_backtest(test_days=60)
