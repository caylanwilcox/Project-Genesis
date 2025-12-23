"""
Backtest Intraday Model - Last 3 Weeks (Dec 2-20, 2025)

Tests BOTH intraday model predictions:
  - Target A: Will close > open? (direction from open)
  - Target B: Will close > current price? (price will go higher)
"""

import pandas as pd
import numpy as np
import pickle
import os
import requests
from datetime import datetime

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def fetch_daily_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV data"""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}

    response = requests.get(url, params=params)
    data = response.json()

    if 'results' not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
    df = df.set_index('date')
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def load_intraday_model(ticker: str):
    """Load intraday model"""
    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_model.pkl')
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict_at_time(model_data, today_row, prev_row, prev_prev_row=None, time_pct=0.0):
    """Make prediction at a given time point during the day

    Returns:
        prob_open: probability close > open
        prob_current: probability close > current price (if model supports it)
        current_price: simulated current price at this time
    """

    feature_cols = model_data['feature_cols']
    scaler = model_data['scaler']
    models = model_data['models']
    weights = model_data['weights']

    today_open = today_row['Open']
    today_high = today_row['High']
    today_low = today_row['Low']
    today_close = today_row['Close']
    prev_close = prev_row['Close']
    prev_high = prev_row['High']
    prev_low = prev_row['Low']

    gap = (today_open - prev_close) / prev_close
    prev_return = (prev_close - prev_prev_row['Close']) / prev_prev_row['Close'] if prev_prev_row is not None else 0
    prev_range = (prev_high - prev_low) / prev_close

    # Simulate what we'd know at this time point
    if time_pct == 0:
        current_price = today_open
        high_so_far = today_open
        low_so_far = today_open
    else:
        # Simulate price progression (using actual high/low as bounds)
        progress = time_pct ** 0.8
        base_price = today_open + (today_close - today_open) * progress

        # High/low so far approximation
        high_so_far = max(today_open, min(today_high, today_open + (today_high - today_open) * min(1, time_pct * 1.5)))
        low_so_far = min(today_open, max(today_low, today_open + (today_low - today_open) * min(1, time_pct * 1.5)))
        current_price = np.clip(base_price, low_so_far, high_so_far)

    range_so_far = max(high_so_far - low_so_far, 0.0001)

    features = {
        'time_pct': time_pct,
        'time_remaining': 1 - time_pct,
        'gap': gap,
        'gap_direction': 1 if gap > 0 else (-1 if gap < 0 else 0),
        'gap_size': abs(gap),
        'prev_return': prev_return,
        'prev_range': prev_range,
        'current_vs_open': (current_price - today_open) / today_open,
        'current_vs_open_direction': 1 if current_price > today_open else (-1 if current_price < today_open else 0),
        'position_in_range': (current_price - low_so_far) / range_so_far if range_so_far > 0 else 0.5,
        'range_so_far_pct': range_so_far / today_open,
        'high_so_far_pct': (high_so_far - today_open) / today_open,
        'low_so_far_pct': (today_open - low_so_far) / today_open,
        'above_open': 1 if current_price > today_open else 0,
        'near_high': 1 if (high_so_far - current_price) < (current_price - low_so_far) else 0,
        'gap_filled': 1 if (gap > 0 and low_so_far <= prev_close) or (gap <= 0 and high_so_far >= prev_close) else 0,
    }

    X = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}])[feature_cols]
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    X_scaled = scaler.transform(X)

    # Target A: close > open
    prob_open = 0.0
    for model_name, model in models.items():
        prob_open += model.predict_proba(X_scaled)[0][1] * weights.get(model_name, 0.25)

    # Target B: close > current (if model has this)
    prob_current = None
    models_current = model_data.get('models_current')
    weights_current = model_data.get('weights_current')
    if models_current and weights_current:
        prob_current = 0.0
        for model_name, model in models_current.items():
            prob_current += model.predict_proba(X_scaled)[0][1] * weights_current.get(model_name, 0.25)

    return prob_open, prob_current, current_price


def predict_at_open(model_data, today_row, prev_row, prev_prev_row=None):
    """Make prediction at market open (time_pct = 0)"""
    prob_open, _, _ = predict_at_time(model_data, today_row, prev_row, prev_prev_row, time_pct=0.0)
    return prob_open


def backtest_ticker(ticker: str):
    """Backtest a single ticker for last 3 weeks at multiple time points

    Tests BOTH predictions:
    - Target A: close > open (bullish day)
    - Target B: close > current price (price will go higher from here)
    """

    print(f"\n{'='*80}")
    print(f"  BACKTEST: {ticker} (Dec 2-20, 2025)")
    print(f"{'='*80}")

    # Load model
    model_data = load_intraday_model(ticker)
    if not model_data:
        print(f"  No model found for {ticker}")
        return None

    # Check if model has "close > current" prediction
    has_current_model = 'models_current' in model_data and model_data['models_current']
    if has_current_model:
        print(f"  Model has BOTH targets: close>open AND close>current")
    else:
        print(f"  Model has only: close>open")

    # Fetch data (need extra days for prev_prev_row)
    df = fetch_daily_data(ticker, '2025-11-25', '2025-12-20')

    if len(df) < 5:
        print(f"  Not enough data")
        return None

    print(f"  Data range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

    # Filter to Dec 2-20 for testing
    test_start = '2025-12-02'
    test_df = df[df.index >= test_start].copy()

    # Test at multiple time points
    time_points = [
        (0.0, 'Open'),
        (0.1, '10% (~40min)'),
        (0.25, '25% (~1.5hr)'),
        (0.5, '50% (~3.25hr)'),
    ]

    all_results_open = {}  # Target A: close > open
    all_results_current = {}  # Target B: close > current

    for time_pct, label in time_points:
        results_open = []
        results_current = []

        for i, (date, row) in enumerate(test_df.iterrows()):
            full_idx = df.index.get_loc(date)
            if full_idx < 2:
                continue

            prev_row = df.iloc[full_idx - 1]
            prev_prev_row = df.iloc[full_idx - 2]

            prob_open, prob_current, current_price = predict_at_time(
                model_data, row, prev_row, prev_prev_row, time_pct
            )

            # Target A: close > open
            actual_bullish = 1 if row['Close'] > row['Open'] else 0
            predicted_bullish = 1 if prob_open > 0.5 else 0
            correct_open = predicted_bullish == actual_bullish
            high_conf_open = prob_open > 0.6 or prob_open < 0.4

            results_open.append({
                'date': date,
                'open': row['Open'],
                'close': row['Close'],
                'current_price': current_price,
                'change_pct': (row['Close'] - row['Open']) / row['Open'] * 100,
                'probability': prob_open,
                'predicted': 'BULL' if predicted_bullish else 'BEAR',
                'actual': 'BULL' if actual_bullish else 'BEAR',
                'correct': correct_open,
                'high_conf': high_conf_open,
            })

            # Target B: close > current (if model supports it)
            if prob_current is not None:
                actual_higher = 1 if row['Close'] > current_price else 0
                predicted_higher = 1 if prob_current > 0.5 else 0
                correct_current = predicted_higher == actual_higher
                high_conf_current = prob_current > 0.6 or prob_current < 0.4

                results_current.append({
                    'date': date,
                    'current_price': current_price,
                    'close': row['Close'],
                    'change_from_current': (row['Close'] - current_price) / current_price * 100,
                    'probability': prob_current,
                    'predicted': 'UP' if predicted_higher else 'DOWN',
                    'actual': 'UP' if actual_higher else 'DOWN',
                    'correct': correct_current,
                    'high_conf': high_conf_current,
                })

        all_results_open[label] = pd.DataFrame(results_open)
        if results_current:
            all_results_current[label] = pd.DataFrame(results_current)

    # ========== TARGET A: Close > Open ==========
    print(f"\n  {'='*76}")
    print(f"  TARGET A: Will close > open? (Bullish day prediction)")
    print(f"  {'='*76}")

    print(f"\n  === DETAILED RESULTS AT OPEN ===")
    print(f"  {'Date':<12} {'Open':>8} {'Close':>8} {'Chg%':>7} {'Prob':>6} {'Pred':>6} {'Actual':>6} {'Result':>6}")
    print(f"  {'-'*65}")

    for _, r in all_results_open['Open'].iterrows():
        result_str = '✓' if r['correct'] else '✗'
        chg_color = '+' if r['change_pct'] > 0 else ''
        print(f"  {r['date'].strftime('%Y-%m-%d'):<12} {r['open']:>8.2f} {r['close']:>8.2f} {chg_color}{r['change_pct']:>6.2f}% {r['probability']:>5.1%} {r['predicted']:>6} {r['actual']:>6} {result_str:>6}")

    print(f"\n  === ACCURACY BY TIME POINT (Close > Open) ===")
    print(f"  {'Time':<18} {'Accuracy':>12} {'High Conf':>12}")
    print(f"  {'-'*45}")

    summary_open = {}
    for label, results_df in all_results_open.items():
        total = len(results_df)
        correct = results_df['correct'].sum()
        accuracy = correct / total if total > 0 else 0

        hc_df = results_df[results_df['high_conf']]
        hc_total = len(hc_df)
        hc_correct = hc_df['correct'].sum() if hc_total > 0 else 0
        hc_accuracy = hc_correct / hc_total if hc_total > 0 else 0

        print(f"  {label:<18} {correct}/{total} = {accuracy:>5.1%}    {hc_correct}/{hc_total} = {hc_accuracy:>5.1%}")

        summary_open[label] = {
            'total': total, 'correct': correct, 'accuracy': accuracy,
            'hc_total': hc_total, 'hc_correct': hc_correct, 'hc_accuracy': hc_accuracy,
        }

    # ========== TARGET B: Close > Current ==========
    if all_results_current:
        print(f"\n  {'='*76}")
        print(f"  TARGET B: Will close > current price? (Price going higher)")
        print(f"  {'='*76}")

        # Show detailed at 25% time point (more interesting than open)
        detail_label = '25% (~1.5hr)' if '25% (~1.5hr)' in all_results_current else list(all_results_current.keys())[0]
        print(f"\n  === DETAILED RESULTS AT {detail_label.upper()} ===")
        print(f"  {'Date':<12} {'Current':>8} {'Close':>8} {'Chg%':>7} {'Prob':>6} {'Pred':>6} {'Actual':>6} {'Result':>6}")
        print(f"  {'-'*65}")

        for _, r in all_results_current[detail_label].iterrows():
            result_str = '✓' if r['correct'] else '✗'
            chg_color = '+' if r['change_from_current'] > 0 else ''
            print(f"  {r['date'].strftime('%Y-%m-%d'):<12} {r['current_price']:>8.2f} {r['close']:>8.2f} {chg_color}{r['change_from_current']:>6.2f}% {r['probability']:>5.1%} {r['predicted']:>6} {r['actual']:>6} {result_str:>6}")

        print(f"\n  === ACCURACY BY TIME POINT (Close > Current) ===")
        print(f"  {'Time':<18} {'Accuracy':>12} {'High Conf':>12}")
        print(f"  {'-'*45}")

        summary_current = {}
        for label, results_df in all_results_current.items():
            total = len(results_df)
            correct = results_df['correct'].sum()
            accuracy = correct / total if total > 0 else 0

            hc_df = results_df[results_df['high_conf']]
            hc_total = len(hc_df)
            hc_correct = hc_df['correct'].sum() if hc_total > 0 else 0
            hc_accuracy = hc_correct / hc_total if hc_total > 0 else 0

            print(f"  {label:<18} {correct}/{total} = {accuracy:>5.1%}    {hc_correct}/{hc_total} = {hc_accuracy:>5.1%}")

            summary_current[label] = {
                'total': total, 'correct': correct, 'accuracy': accuracy,
                'hc_total': hc_total, 'hc_correct': hc_correct, 'hc_accuracy': hc_accuracy,
            }
    else:
        summary_current = {}

    return {
        'ticker': ticker,
        'results_open': all_results_open,
        'results_current': all_results_current,
        'summary_open': summary_open,
        'summary_current': summary_current,
        'has_current_model': has_current_model,
    }


def main():
    print("="*80)
    print("  INTRADAY MODEL BACKTEST - BOTH TARGETS")
    print("  Period: December 2-20, 2025 (Last 3 Weeks)")
    print("  Testing at: Open, 10% (~40min), 25% (~1.5hr), 50% (~3.25hr)")
    print("="*80)
    print("  Target A: Will close > open? (Bullish day)")
    print("  Target B: Will close > current price? (Price going higher)")
    print("="*80)

    all_results = {}

    for ticker in ['SPY', 'QQQ', 'IWM']:
        result = backtest_ticker(ticker)
        if result:
            all_results[ticker] = result

    # Final summary
    print("\n" + "="*80)
    print("  FINAL SUMMARY - ALL TICKERS")
    print("="*80)

    time_labels = ['Open', '10% (~40min)', '25% (~1.5hr)', '50% (~3.25hr)']

    # Summary for Target A: Close > Open
    print("\n  " + "="*76)
    print("  TARGET A: Close > Open (Bullish Day)")
    print("  " + "="*76)

    for label in time_labels:
        print(f"\n  === {label.upper()} ===")
        print(f"  {'Ticker':<8} {'Accuracy':>15} {'High Conf':>15}")
        print(f"  {'-'*40}")

        total_correct = 0
        total_trades = 0
        hc_correct = 0
        hc_trades = 0

        for ticker, result in all_results.items():
            s = result['summary_open'].get(label, {})
            correct = s.get('correct', 0)
            total = s.get('total', 0)
            accuracy = s.get('accuracy', 0)
            hc_c = s.get('hc_correct', 0)
            hc_t = s.get('hc_total', 0)
            hc_acc = s.get('hc_accuracy', 0)
            print(f"  {ticker:<8} {correct}/{total} = {accuracy:>5.1%}    {hc_c}/{hc_t} = {hc_acc:>5.1%}")
            total_correct += correct
            total_trades += total
            hc_correct += hc_c
            hc_trades += hc_t

        if total_trades > 0:
            print(f"  {'-'*40}")
            hc_acc_total = hc_correct / hc_trades if hc_trades > 0 else 0
            print(f"  {'TOTAL':<8} {total_correct}/{total_trades} = {total_correct/total_trades:>5.1%}    {hc_correct}/{hc_trades} = {hc_acc_total:>5.1%}")

    # Summary for Target B: Close > Current
    has_any_current = any(r.get('has_current_model', False) for r in all_results.values())

    if has_any_current:
        print("\n  " + "="*76)
        print("  TARGET B: Close > Current (Price Going Higher)")
        print("  " + "="*76)

        for label in time_labels:
            print(f"\n  === {label.upper()} ===")
            print(f"  {'Ticker':<8} {'Accuracy':>15} {'High Conf':>15}")
            print(f"  {'-'*40}")

            total_correct = 0
            total_trades = 0
            hc_correct = 0
            hc_trades = 0

            for ticker, result in all_results.items():
                s = result['summary_current'].get(label, {})
                if not s:
                    print(f"  {ticker:<8} (no model)")
                    continue
                correct = s.get('correct', 0)
                total = s.get('total', 0)
                accuracy = s.get('accuracy', 0)
                hc_c = s.get('hc_correct', 0)
                hc_t = s.get('hc_total', 0)
                hc_acc = s.get('hc_accuracy', 0)
                print(f"  {ticker:<8} {correct}/{total} = {accuracy:>5.1%}    {hc_c}/{hc_t} = {hc_acc:>5.1%}")
                total_correct += correct
                total_trades += total
                hc_correct += hc_c
                hc_trades += hc_t

            if total_trades > 0:
                print(f"  {'-'*40}")
                hc_acc_total = hc_correct / hc_trades if hc_trades > 0 else 0
                print(f"  {'TOTAL':<8} {total_correct}/{total_trades} = {total_correct/total_trades:>5.1%}    {hc_correct}/{hc_trades} = {hc_acc_total:>5.1%}")


if __name__ == '__main__':
    main()
