"""
Walk-Forward Backtesting for FVG Models

This script performs proper out-of-time validation by:
1. Training on historical data (e.g., 2022-2023)
2. Testing on completely unseen future data (e.g., 2024-today)

This gives realistic performance metrics for actual trading.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configuration
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY') or os.environ.get('NEXT_PUBLIC_POLYGON_API_KEY')
MODELS_DIR = './models'
TICKERS = ['SPY', 'QQQ', 'IWM']

# Walk-forward periods
TRAIN_END = '2023-12-31'      # Train on data up to this date
TEST_START = '2024-01-01'     # Test on data from this date
TEST_END = datetime.now().strftime('%Y-%m-%d')  # Up to today


def load_model(ticker):
    """Load trained model for a ticker"""
    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_fvg_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    # Fallback to combined model
    combined_path = os.path.join(MODELS_DIR, 'combined_fvg_model.pkl')
    if os.path.exists(combined_path):
        with open(combined_path, 'rb') as f:
            return pickle.load(f)

    return None


def fetch_market_data(ticker, start_date, end_date, timeframe='1h'):
    """Fetch OHLCV data from Polygon API"""
    if not POLYGON_API_KEY:
        print("WARNING: No Polygon API key found. Using cached data only.")
        return None

    # Map timeframe to Polygon format
    tf_map = {
        '1m': ('minute', 1),
        '5m': ('minute', 5),
        '15m': ('minute', 15),
        '1h': ('hour', 1),
        '4h': ('hour', 4),
        '1d': ('day', 1),
    }

    timespan, multiplier = tf_map.get(timeframe, ('hour', 1))

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
    params = {
        'apiKey': POLYGON_API_KEY,
        'limit': 50000,
        'sort': 'asc'
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if data.get('status') == 'OK' and data.get('results'):
            df = pd.DataFrame(data['results'])
            df['time'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={
                'o': 'open', 'h': 'high', 'l': 'low',
                'c': 'close', 'v': 'volume'
            })
            return df[['time', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

    return None


def detect_fvg_patterns(df, min_gap_pct=0.15):
    """Detect FVG patterns in OHLCV data"""
    patterns = []

    for i in range(2, len(df)):
        candle1 = df.iloc[i-2]
        candle2 = df.iloc[i-1]
        candle3 = df.iloc[i]

        # Bullish FVG: candle3.low > candle1.high (gap up)
        if candle3['low'] > candle1['high']:
            gap_size = candle3['low'] - candle1['high']
            gap_pct = (gap_size / candle2['close']) * 100

            if gap_pct >= min_gap_pct:
                patterns.append({
                    'type': 'bullish',
                    'time': candle3['time'],
                    'gap_high': candle3['low'],
                    'gap_low': candle1['high'],
                    'gap_size': gap_size,
                    'gap_size_pct': gap_pct,
                    'entry_price': candle3['low'],
                    'candle2_close': candle2['close'],
                    'index': i
                })

        # Bearish FVG: candle3.high < candle1.low (gap down)
        elif candle3['high'] < candle1['low']:
            gap_size = candle1['low'] - candle3['high']
            gap_pct = (gap_size / candle2['close']) * 100

            if gap_pct >= min_gap_pct:
                patterns.append({
                    'type': 'bearish',
                    'time': candle3['time'],
                    'gap_high': candle1['low'],
                    'gap_low': candle3['high'],
                    'gap_size': gap_size,
                    'gap_size_pct': gap_pct,
                    'entry_price': candle3['high'],
                    'candle2_close': candle2['close'],
                    'index': i
                })

    return patterns


def calculate_technical_indicators(df, idx):
    """Calculate technical indicators at a given index"""
    if idx < 50:  # Need enough history
        return None

    window = df.iloc[max(0, idx-50):idx+1].copy()
    close = window['close']
    high = window['high']
    low = window['low']
    volume = window['volume']

    current_close = close.iloc[-1]

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_14 = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    # MACD
    ema_12 = close.ewm(span=12).mean().iloc[-1]
    ema_26 = close.ewm(span=26).mean().iloc[-1]
    macd = ema_12 - ema_26
    macd_signal = close.ewm(span=9).mean().iloc[-1]  # Simplified
    macd_histogram = macd - macd_signal

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean().iloc[-1]

    # SMAs and EMAs
    sma_20 = close.rolling(20).mean().iloc[-1]
    sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20

    # Bollinger Bands
    bb_std = close.rolling(20).std().iloc[-1]
    bb_bandwidth = (4 * bb_std / sma_20) * 100 if sma_20 > 0 else 0

    # Volume ratio
    avg_volume = volume.rolling(20).mean().iloc[-1]
    volume_ratio = volume.iloc[-1] / avg_volume if avg_volume > 0 else 1

    # Time features
    time = df.iloc[idx]['time']
    hour_of_day = time.hour if hasattr(time, 'hour') else 12
    day_of_week = time.dayofweek if hasattr(time, 'dayofweek') else 2

    # Categorical features
    rsi_zone = 'overbought' if rsi_14 > 70 else ('oversold' if rsi_14 < 30 else 'neutral')
    macd_trend = 'bullish' if macd > 0 else 'bearish'
    volatility_regime = 'high' if atr_14 > close.std() else ('low' if atr_14 < close.std() * 0.5 else 'medium')

    return {
        'rsi_14': rsi_14,
        'macd': macd,
        'macd_signal': macd_signal,
        'macd_histogram': macd_histogram,
        'atr_14': atr_14,
        'sma_20': sma_20,
        'sma_50': sma_50,
        'ema_12': ema_12,
        'ema_26': ema_26,
        'bb_bandwidth': bb_bandwidth,
        'volume_ratio': volume_ratio,
        'price_vs_sma20': (current_close - sma_20) / sma_20 * 100 if sma_20 > 0 else 0,
        'price_vs_sma50': (current_close - sma_50) / sma_50 * 100 if sma_50 > 0 else 0,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'rsi_zone': rsi_zone,
        'macd_trend': macd_trend,
        'volatility_regime': volatility_regime,
        'volume_profile': 'high' if volume_ratio > 1.5 else ('low' if volume_ratio < 0.5 else 'medium'),
        'market_structure': 'bullish' if current_close > sma_20 > sma_50 else ('bearish' if current_close < sma_20 < sma_50 else 'neutral'),
    }


def simulate_trade_outcome(df, pattern, max_bars=50):
    """
    Simulate what would have happened if we traded this FVG.

    For bullish FVG: Buy at gap_high, target = entry + 2*gap, stop = entry - gap
    For bearish FVG: Sell at gap_low, target = entry - 2*gap, stop = entry + gap
    """
    start_idx = pattern['index']
    gap_size = pattern['gap_size']

    if pattern['type'] == 'bullish':
        entry = pattern['gap_high']
        stop_loss = entry - gap_size
        take_profit_1 = entry + gap_size      # 1:1 RR
        take_profit_2 = entry + gap_size * 1.5  # 1.5:1 RR
        take_profit_3 = entry + gap_size * 2    # 2:1 RR

        for i in range(start_idx + 1, min(start_idx + max_bars, len(df))):
            bar = df.iloc[i]

            # Check stop loss first
            if bar['low'] <= stop_loss:
                return 'stop_loss', i - start_idx

            # Check take profits
            if bar['high'] >= take_profit_3:
                return 'tp3', i - start_idx
            if bar['high'] >= take_profit_2:
                return 'tp2', i - start_idx
            if bar['high'] >= take_profit_1:
                return 'tp1', i - start_idx

    else:  # bearish
        entry = pattern['gap_low']
        stop_loss = entry + gap_size
        take_profit_1 = entry - gap_size
        take_profit_2 = entry - gap_size * 1.5
        take_profit_3 = entry - gap_size * 2

        for i in range(start_idx + 1, min(start_idx + max_bars, len(df))):
            bar = df.iloc[i]

            # Check stop loss first
            if bar['high'] >= stop_loss:
                return 'stop_loss', i - start_idx

            # Check take profits
            if bar['low'] <= take_profit_3:
                return 'tp3', i - start_idx
            if bar['low'] <= take_profit_2:
                return 'tp2', i - start_idx
            if bar['low'] <= take_profit_1:
                return 'tp1', i - start_idx

    return 'timeout', max_bars


def build_prediction_features(pattern, indicators):
    """Build features for model prediction"""
    features = {
        'gap_size_pct': pattern['gap_size_pct'],
        'validation_score': 0.7,  # Default
        'fvg_type': pattern['type'],
        **indicators
    }
    return features


# Categorical encoding (same as training)
CATEGORICAL_MAPPINGS = {
    'fvg_type': {'bearish': 0, 'bullish': 1, 'unknown': 2},
    'volume_profile': {'high': 0, 'low': 1, 'medium': 2, 'unknown': 3},
    'market_structure': {'bearish': 0, 'bullish': 1, 'neutral': 2, 'unknown': 3},
    'rsi_zone': {'neutral': 0, 'overbought': 1, 'oversold': 2, 'unknown': 3},
    'macd_trend': {'bearish': 0, 'bullish': 1, 'neutral': 2, 'unknown': 3},
    'volatility_regime': {'high': 0, 'low': 1, 'medium': 2, 'unknown': 3},
}


def predict_with_model(model_data, features):
    """Make prediction using loaded model"""
    feature_cols = model_data['feature_cols']

    # Build feature vector
    feature_vec = {}

    # Numeric features
    numeric_cols = [
        'gap_size_pct', 'validation_score', 'rsi_14', 'macd',
        'macd_signal', 'macd_histogram', 'atr_14', 'sma_20',
        'sma_50', 'ema_12', 'ema_26', 'bb_bandwidth', 'volume_ratio',
        'price_vs_sma20', 'price_vs_sma50', 'hour_of_day', 'day_of_week'
    ]

    for col in numeric_cols:
        feature_vec[col] = features.get(col, 0) or 0

    # Encode categorical
    for cat_col, mapping in CATEGORICAL_MAPPINGS.items():
        value = str(features.get(cat_col, 'unknown')).lower()
        encoded = mapping.get(value, mapping.get('unknown', 0))
        feature_vec[f'{cat_col}_encoded'] = encoded

    # Create DataFrame
    df = pd.DataFrame([feature_vec])

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_cols]

    # Predict
    model = model_data['model']
    probability = model.predict_proba(df)[0][1]

    return probability


def run_backtest(ticker, timeframe='1h'):
    """Run walk-forward backtest for a ticker"""
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD BACKTEST: {ticker}")
    print(f"Test Period: {TEST_START} to {TEST_END}")
    print(f"Timeframe: {timeframe}")
    print(f"{'='*60}")

    # Load model
    model_data = load_model(ticker)
    if model_data is None:
        print(f"ERROR: No model found for {ticker}")
        return None

    print(f"Model loaded (trained accuracy: {model_data['metrics']['accuracy']:.1%})")

    # Fetch test data
    print(f"\nFetching market data...")
    df = fetch_market_data(ticker, TEST_START, TEST_END, timeframe)

    if df is None or len(df) < 100:
        print(f"ERROR: Insufficient data for {ticker}")
        return None

    print(f"Loaded {len(df)} bars from {df['time'].min()} to {df['time'].max()}")

    # Detect FVG patterns
    print(f"\nDetecting FVG patterns...")
    patterns = detect_fvg_patterns(df)
    print(f"Found {len(patterns)} FVG patterns")

    if len(patterns) == 0:
        print("No patterns to test")
        return None

    # Backtest each pattern
    results = []

    for pattern in patterns:
        # Get technical indicators
        indicators = calculate_technical_indicators(df, pattern['index'])
        if indicators is None:
            continue

        # Build features and predict
        features = build_prediction_features(pattern, indicators)
        win_probability = predict_with_model(model_data, features)
        predicted_win = win_probability >= 0.5

        # Simulate actual outcome
        outcome, bars_held = simulate_trade_outcome(df, pattern)
        actual_win = outcome in ['tp1', 'tp2', 'tp3']

        results.append({
            'time': pattern['time'],
            'type': pattern['type'],
            'gap_pct': pattern['gap_size_pct'],
            'predicted_win': predicted_win,
            'win_probability': win_probability,
            'actual_outcome': outcome,
            'actual_win': actual_win,
            'bars_held': bars_held,
            'correct': predicted_win == actual_win,
        })

    if len(results) == 0:
        print("No valid results")
        return None

    # Calculate metrics
    results_df = pd.DataFrame(results)

    y_true = results_df['actual_win'].astype(int)
    y_pred = results_df['predicted_win'].astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Trading metrics
    total_trades = len(results_df)
    wins = results_df['actual_win'].sum()
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0

    # Simulated P&L (assuming 1:2 risk/reward)
    # Win = +2 units, Loss = -1 unit
    pnl_per_trade = results_df['actual_outcome'].map({
        'tp1': 1.0, 'tp2': 1.5, 'tp3': 2.0,
        'stop_loss': -1.0, 'timeout': -0.5
    })
    total_pnl = pnl_per_trade.sum()
    avg_pnl = pnl_per_trade.mean()

    # Profit factor
    gross_profit = pnl_per_trade[pnl_per_trade > 0].sum()
    gross_loss = abs(pnl_per_trade[pnl_per_trade < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Print results
    print(f"\n{'='*50}")
    print(f"OUT-OF-TIME RESULTS: {ticker}")
    print(f"{'='*50}")
    print(f"\nPrediction Metrics:")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.1%}")

    print(f"\nTrading Metrics:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Wins: {wins} | Losses: {losses}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Total P&L: {total_pnl:.1f} units")
    print(f"  Avg P&L/Trade: {avg_pnl:.2f} units")

    print(f"\nOutcome Distribution:")
    print(results_df['actual_outcome'].value_counts().to_string())

    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Predicted:   Loss  Win")
    if len(cm) == 2:
        print(f"  Actual Loss: {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"  Actual Win:  {cm[1][0]:4d}  {cm[1][1]:4d}")

    return {
        'ticker': ticker,
        'test_period': f"{TEST_START} to {TEST_END}",
        'total_trades': total_trades,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'results_df': results_df,
    }


def main():
    print("="*60)
    print("WALK-FORWARD BACKTESTING")
    print("Testing FVG Models on Unseen Data")
    print("="*60)
    print(f"\nTrain Period: Before {TRAIN_END}")
    print(f"Test Period: {TEST_START} to {TEST_END} (TODAY)")

    all_results = {}

    for ticker in TICKERS:
        result = run_backtest(ticker, timeframe='1h')
        if result:
            all_results[ticker] = result

    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY - OUT-OF-TIME PERFORMANCE")
    print("="*60)

    print("\n{:<8} {:>10} {:>10} {:>10} {:>10} {:>12}".format(
        "Ticker", "Accuracy", "Win Rate", "PF", "Trades", "P&L"
    ))
    print("-" * 65)

    for ticker, r in all_results.items():
        print("{:<8} {:>9.1%} {:>9.1%} {:>10.2f} {:>10} {:>11.1f}".format(
            ticker,
            r['accuracy'],
            r['win_rate'],
            r['profit_factor'],
            r['total_trades'],
            r['total_pnl']
        ))

    # Save results
    summary = {
        'test_period': f"{TEST_START} to {TEST_END}",
        'run_at': datetime.now().isoformat(),
        'results': {
            ticker: {
                'accuracy': r['accuracy'],
                'precision': r['precision'],
                'recall': r['recall'],
                'f1': r['f1'],
                'win_rate': r['win_rate'],
                'profit_factor': r['profit_factor'],
                'total_trades': r['total_trades'],
                'total_pnl': r['total_pnl'],
            }
            for ticker, r in all_results.items()
        }
    }

    with open('backtest_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to backtest_results.json")

    return all_results


if __name__ == '__main__':
    results = main()
