"""
Generate Large FVG Training Dataset

Fetches 2+ years of data from Polygon API and generates thousands
of labeled FVG samples for proper ML training.

Target: 1000+ samples per ticker across multiple timeframes
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import time

# Configuration
POLYGON_API_KEY = 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O'
TICKERS = ['SPY', 'QQQ', 'IWM']
TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d']  # Multiple timeframes

# Date ranges for training/testing
TRAIN_START = '2023-01-01'
TRAIN_END = '2024-12-31'  # 2 years training
TEST_START = '2025-01-01'
TEST_END = datetime.now().strftime('%Y-%m-%d')  # Up to today (Dec 2025)

OUTPUT_DIR = './data'


def fetch_polygon_data(ticker, start_date, end_date, timeframe='1h', limit=50000):
    """Fetch OHLCV data from Polygon API"""

    # Map timeframe to Polygon format
    tf_map = {
        '1m': ('minute', 1),
        '5m': ('minute', 5),
        '15m': ('minute', 15),
        '30m': ('minute', 30),
        '1h': ('hour', 1),
        '2h': ('hour', 2),
        '4h': ('hour', 4),
        '1d': ('day', 1),
        '1w': ('week', 1),
    }

    timespan, multiplier = tf_map.get(timeframe, ('hour', 1))

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
    params = {
        'apiKey': POLYGON_API_KEY,
        'limit': limit,
        'sort': 'asc'
    }

    all_results = []

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if data.get('status') == 'OK' and data.get('results'):
            all_results.extend(data['results'])
            print(f"  Fetched {len(data['results'])} bars for {ticker} ({timeframe})")
        else:
            print(f"  No data: {data.get('status', 'unknown')}")

    except Exception as e:
        print(f"  Error: {e}")

    if not all_results:
        return None

    df = pd.DataFrame(all_results)
    df['time'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={
        'o': 'open', 'h': 'high', 'l': 'low',
        'c': 'close', 'v': 'volume', 'vw': 'vwap', 'n': 'trades'
    })

    return df[['time', 'open', 'high', 'low', 'close', 'volume']]


def detect_fvg_patterns(df, min_gap_pct=0.1, max_gap_pct=5.0):
    """Detect FVG patterns with various gap sizes"""
    patterns = []

    for i in range(2, len(df) - 50):  # Leave room for outcome simulation
        candle1 = df.iloc[i-2]
        candle2 = df.iloc[i-1]
        candle3 = df.iloc[i]

        # Bullish FVG: candle3.low > candle1.high
        if candle3['low'] > candle1['high']:
            gap_size = candle3['low'] - candle1['high']
            gap_pct = (gap_size / candle2['close']) * 100

            if min_gap_pct <= gap_pct <= max_gap_pct:
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

        # Bearish FVG: candle3.high < candle1.low
        elif candle3['high'] < candle1['low']:
            gap_size = candle1['low'] - candle3['high']
            gap_pct = (gap_size / candle2['close']) * 100

            if min_gap_pct <= gap_pct <= max_gap_pct:
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


def calculate_indicators(df, idx):
    """Calculate technical indicators at index"""
    if idx < 50:
        return None

    window = df.iloc[max(0, idx-50):idx+1].copy()
    close = window['close']
    high = window['high']
    low = window['low']
    volume = window['volume']
    current_close = close.iloc[-1]

    try:
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_14 = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

        # MACD
        ema_12 = float(close.ewm(span=12).mean().iloc[-1])
        ema_26 = float(close.ewm(span=26).mean().iloc[-1])
        macd = ema_12 - ema_26
        macd_signal = float(close.ewm(span=9).mean().iloc[-1])
        macd_histogram = macd - macd_signal

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr_14 = float(tr.rolling(14).mean().iloc[-1])

        # SMAs
        sma_20 = float(close.rolling(20).mean().iloc[-1])
        sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma_20

        # Bollinger Bands
        bb_std = float(close.rolling(20).std().iloc[-1])
        bb_bandwidth = (4 * bb_std / sma_20) * 100 if sma_20 > 0 else 0

        # Volume ratio
        avg_volume = float(volume.rolling(20).mean().iloc[-1])
        current_volume = float(volume.iloc[-1])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # Time features
        time_val = df.iloc[idx]['time']
        hour_of_day = time_val.hour if hasattr(time_val, 'hour') else 12
        day_of_week = time_val.dayofweek if hasattr(time_val, 'dayofweek') else 2

        # Categorical
        rsi_zone = 'overbought' if rsi_14 > 70 else ('oversold' if rsi_14 < 30 else 'neutral')
        macd_trend = 'bullish' if macd > 0 else ('bearish' if macd < 0 else 'neutral')

        std_val = float(close.std())
        volatility_regime = 'high' if atr_14 > std_val else ('low' if atr_14 < std_val * 0.5 else 'medium')
        volume_profile = 'high' if volume_ratio > 1.5 else ('low' if volume_ratio < 0.5 else 'medium')
        market_structure = 'bullish' if current_close > sma_20 > sma_50 else ('bearish' if current_close < sma_20 < sma_50 else 'neutral')

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
            'volume_profile': volume_profile,
            'market_structure': market_structure,
        }
    except Exception as e:
        return None


def simulate_outcome(df, pattern, max_bars=50):
    """Simulate trade outcome"""
    start_idx = pattern['index']
    gap_size = pattern['gap_size']

    if pattern['type'] == 'bullish':
        entry = pattern['gap_high']
        stop_loss = entry - gap_size
        tp1 = entry + gap_size
        tp2 = entry + gap_size * 1.5
        tp3 = entry + gap_size * 2

        for i in range(start_idx + 1, min(start_idx + max_bars, len(df))):
            bar = df.iloc[i]
            if bar['low'] <= stop_loss:
                return 'stop_loss', i - start_idx
            if bar['high'] >= tp3:
                return 'tp3', i - start_idx
            if bar['high'] >= tp2:
                return 'tp2', i - start_idx
            if bar['high'] >= tp1:
                return 'tp1', i - start_idx
    else:
        entry = pattern['gap_low']
        stop_loss = entry + gap_size
        tp1 = entry - gap_size
        tp2 = entry - gap_size * 1.5
        tp3 = entry - gap_size * 2

        for i in range(start_idx + 1, min(start_idx + max_bars, len(df))):
            bar = df.iloc[i]
            if bar['high'] >= stop_loss:
                return 'stop_loss', i - start_idx
            if bar['low'] <= tp3:
                return 'tp3', i - start_idx
            if bar['low'] <= tp2:
                return 'tp2', i - start_idx
            if bar['low'] <= tp1:
                return 'tp1', i - start_idx

    return 'timeout', max_bars


def generate_features_for_ticker(ticker, start_date, end_date, timeframes, dataset_name='train'):
    """Generate labeled features for a ticker across timeframes"""
    all_features = []

    for tf in timeframes:
        print(f"\nProcessing {ticker} - {tf} ({dataset_name})...")

        df = fetch_polygon_data(ticker, start_date, end_date, tf)
        if df is None or len(df) < 100:
            print(f"  Skipping - insufficient data")
            continue

        print(f"  Total bars: {len(df)}")

        # Detect patterns
        patterns = detect_fvg_patterns(df)
        print(f"  Found {len(patterns)} FVG patterns")

        # Process each pattern
        valid_count = 0
        for pattern in patterns:
            # Calculate indicators
            indicators = calculate_indicators(df, pattern['index'])
            if indicators is None:
                continue

            # Simulate outcome
            outcome, bars_held = simulate_outcome(df, pattern)

            # Build feature record
            feature = {
                'ticker': ticker,
                'timeframe': tf,
                'dataset': dataset_name,
                'detected_at': pattern['time'].isoformat() if hasattr(pattern['time'], 'isoformat') else str(pattern['time']),
                'fvg_type': pattern['type'],
                'gap_size_pct': pattern['gap_size_pct'],
                'validation_score': 0.7,  # Default
                'final_outcome': outcome,
                'bars_held': bars_held,
                **indicators
            }

            all_features.append(feature)
            valid_count += 1

        print(f"  Valid samples: {valid_count}")

        # Rate limiting
        time.sleep(0.5)

    return all_features


def main():
    print("="*60)
    print("GENERATING LARGE FVG DATASET")
    print("="*60)
    print(f"\nTraining Period: {TRAIN_START} to {TRAIN_END}")
    print(f"Testing Period: {TEST_START} to {TEST_END}")
    print(f"Tickers: {TICKERS}")
    print(f"Timeframes: {TIMEFRAMES}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_train_features = []
    all_test_features = []

    for ticker in TICKERS:
        print(f"\n{'='*40}")
        print(f"PROCESSING {ticker}")
        print(f"{'='*40}")

        # Generate training data
        train_features = generate_features_for_ticker(
            ticker, TRAIN_START, TRAIN_END, TIMEFRAMES, 'train'
        )
        all_train_features.extend(train_features)

        # Generate test data
        test_features = generate_features_for_ticker(
            ticker, TEST_START, TEST_END, TIMEFRAMES, 'test'
        )
        all_test_features.extend(test_features)

        # Save per-ticker files
        ticker_data = {
            'ticker': ticker,
            'train_samples': len(train_features),
            'test_samples': len(test_features),
            'features': train_features + test_features
        }

        with open(f'{OUTPUT_DIR}/{ticker.lower()}_large_features.json', 'w') as f:
            json.dump(ticker_data, f, indent=2)

        print(f"\n{ticker} Summary:")
        print(f"  Training samples: {len(train_features)}")
        print(f"  Testing samples: {len(test_features)}")

    # Save combined dataset
    combined = {
        'generated_at': datetime.now().isoformat(),
        'train_period': f"{TRAIN_START} to {TRAIN_END}",
        'test_period': f"{TEST_START} to {TEST_END}",
        'total_train_samples': len(all_train_features),
        'total_test_samples': len(all_test_features),
        'train_features': all_train_features,
        'test_features': all_test_features,
    }

    with open(f'{OUTPUT_DIR}/combined_large_dataset.json', 'w') as f:
        json.dump(combined, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)

    print(f"\nTotal Training Samples: {len(all_train_features)}")
    print(f"Total Testing Samples: {len(all_test_features)}")

    # Outcome distribution
    if all_train_features:
        train_df = pd.DataFrame(all_train_features)
        print(f"\nTraining Outcome Distribution:")
        print(train_df['final_outcome'].value_counts().to_string())

        win_rate = train_df['final_outcome'].isin(['tp1', 'tp2', 'tp3']).mean()
        print(f"\nActual Win Rate: {win_rate:.1%}")

    if all_test_features:
        test_df = pd.DataFrame(all_test_features)
        print(f"\nTest Outcome Distribution:")
        print(test_df['final_outcome'].value_counts().to_string())

        win_rate = test_df['final_outcome'].isin(['tp1', 'tp2', 'tp3']).mean()
        print(f"\nActual Win Rate (Test): {win_rate:.1%}")

    print(f"\nFiles saved to {OUTPUT_DIR}/")

    return combined


if __name__ == '__main__':
    dataset = main()
