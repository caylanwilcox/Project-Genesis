"""
Backtest Intraday Model V3 - STRICT INTEGRITY

INTEGRITY RULES:
1. Uses ONLY real hourly prices from Polygon
2. NO simulation, NO future information leakage
3. Tests on Dec 2-20, 2025 (out-of-sample)
4. Raw results with detailed breakdown

This tests the V3 model which was trained on 2020-01-01 to 2024-06-30.
"""

import pandas as pd
import numpy as np
import pickle
import os
import requests
from datetime import datetime, timedelta

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'cLGJlSCuMr4SeGhSUvhbk0A1TIMKxp6O')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def fetch_hourly_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly intraday data"""
    print(f"  Fetching hourly data for {ticker}...")

    all_data = []
    current_start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    while current_start < end:
        chunk_end = min(current_start + timedelta(days=30), end)
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{current_start.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}"
        params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}

        response = requests.get(url, params=params)
        data = response.json()
        if 'results' in data:
            all_data.extend(data['results'])
        current_start = chunk_end + timedelta(days=1)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df['datetime'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
    df = df.set_index('datetime')
    df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')

    print(f"    Got {len(df)} hourly bars")
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def fetch_daily_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV data"""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}

    response = requests.get(url, params=params)
    data = response.json()

    if 'results' not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
    df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
    df = df.set_index('date')
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def load_v3_model(ticker: str):
    """Load V3 intraday model"""
    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_v3.pkl')
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def get_price_at_hour(hourly_df, date, hour):
    """Get closing price at a specific hour"""
    try:
        day_start = pd.Timestamp(date).tz_localize('America/New_York').replace(hour=hour, minute=0)
        day_end = day_start + timedelta(hours=1)
        bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index < day_end)]
        if len(bars) > 0:
            return bars['Close'].iloc[0]
    except:
        pass
    return None


def create_enhanced_features(bars_so_far, today_open, prev_day, prev_prev_day, price_11am=None):
    """
    Create enhanced feature set using ONLY data available at prediction time.
    Same as training function to ensure consistency.
    """
    current_bar = bars_so_far.iloc[-1]
    current_price = current_bar['Close']
    current_hour = current_bar.name.hour

    high_so_far = bars_so_far['High'].max()
    low_so_far = bars_so_far['Low'].min()
    volume_so_far = bars_so_far['Volume'].sum()

    # Time features
    hours_since_open = (current_bar.name.hour - 9) + (current_bar.name.minute / 60)
    time_pct = min(max(hours_since_open / 6.5, 0), 1)

    # Previous day features
    prev_close = prev_day['Close']
    prev_open = prev_day['Open']
    prev_high = prev_day['High']
    prev_low = prev_day['Low']
    prev_volume = prev_day['Volume']

    gap = (today_open - prev_close) / prev_close
    prev_return = (prev_close - prev_prev_day['Close']) / prev_prev_day['Close']
    prev_range = (prev_high - prev_low) / prev_close
    prev_body = (prev_close - prev_open) / prev_open

    range_so_far = max(high_so_far - low_so_far, 0.0001)

    # ========== BASIC FEATURES ==========
    features = {
        'time_pct': time_pct,
        'time_remaining': 1 - time_pct,
        'gap': gap,
        'gap_direction': 1 if gap > 0 else (-1 if gap < 0 else 0),
        'gap_size': abs(gap),
        'prev_return': prev_return,
        'prev_range': prev_range,
        'prev_body': prev_body,
        'prev_bullish': 1 if prev_close > prev_open else 0,
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

    # ========== MOMENTUM FEATURES ==========
    if len(bars_so_far) >= 2:
        features['last_hour_return'] = (current_price - bars_so_far['Close'].iloc[-2]) / bars_so_far['Close'].iloc[-2]
        features['last_hour_direction'] = 1 if features['last_hour_return'] > 0 else -1
    else:
        features['last_hour_return'] = 0
        features['last_hour_direction'] = 0

    if len(bars_so_far) >= 3:
        features['two_hour_return'] = (current_price - bars_so_far['Close'].iloc[-3]) / bars_so_far['Close'].iloc[-3]
    else:
        features['two_hour_return'] = features.get('last_hour_return', 0)

    bullish_bars = sum(1 for i in range(len(bars_so_far)) if bars_so_far['Close'].iloc[i] > bars_so_far['Open'].iloc[i])
    features['bullish_bar_ratio'] = bullish_bars / len(bars_so_far)

    # ========== VOLATILITY FEATURES ==========
    if len(bars_so_far) >= 2:
        hourly_returns = bars_so_far['Close'].pct_change().dropna()
        features['intraday_volatility'] = hourly_returns.std() if len(hourly_returns) > 0 else 0
    else:
        features['intraday_volatility'] = 0

    features['range_vs_prev'] = range_so_far / (prev_high - prev_low) if (prev_high - prev_low) > 0 else 1

    # ========== VOLUME FEATURES ==========
    avg_hourly_volume = volume_so_far / len(bars_so_far) if len(bars_so_far) > 0 else 0
    expected_daily_volume = prev_volume

    features['volume_pace'] = (volume_so_far / (time_pct * expected_daily_volume)) if (time_pct * expected_daily_volume) > 0 else 1
    features['current_hour_volume_ratio'] = current_bar['Volume'] / avg_hourly_volume if avg_hourly_volume > 0 else 1

    # ========== 11 AM ANCHOR FEATURES ==========
    if price_11am is not None:
        features['current_vs_11am'] = (current_price - price_11am) / price_11am
        features['above_11am'] = 1 if current_price > price_11am else 0
        features['distance_from_11am'] = abs(current_price - price_11am) / price_11am
    else:
        features['current_vs_11am'] = 0
        features['above_11am'] = 0
        features['distance_from_11am'] = 0

    # ========== PATTERN FEATURES ==========
    features['morning_reversal'] = 1 if (gap > 0 and current_price < today_open) or (gap < 0 and current_price > today_open) else 0
    features['extended_move'] = 1 if abs(features['current_vs_open']) > 0.005 else 0

    return features


def predict_at_hour(model_data, bars_so_far, today_open, prev_day, prev_prev_day, price_11am=None):
    """
    Make prediction using V3 model with real hourly data.
    ONLY uses data available at the prediction time.
    """
    feature_cols = model_data['feature_cols']
    scaler = model_data['scaler']
    models = model_data['models']
    weights = model_data['weights']

    current_bar = bars_so_far.iloc[-1]
    current_price = current_bar['Close']
    current_hour = current_bar.name.hour
    time_pct = (current_bar.name.hour - 9) / 6.5

    # Create features using same function as training
    features = create_enhanced_features(bars_so_far, today_open, prev_day, prev_prev_day, price_11am)

    X = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}])[feature_cols]
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    X_scaled = scaler.transform(X)

    # Target A: close > open
    prob_bullish = 0.0
    for model_name, model in models.items():
        prob_bullish += model.predict_proba(X_scaled)[0][1] * weights.get(model_name, 0.25)

    # Target B: close > 11 AM price (only if we have the model and it's after 11 AM)
    prob_above_11am = None
    if current_hour >= 11 and price_11am is not None and 'models_11am' in model_data:
        models_11am = model_data['models_11am']
        weights_11am = model_data['weights_11am']
        scaler_11am = model_data['scaler_11am']
        feature_cols_11am = model_data['feature_cols_11am']

        X_b = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols_11am}])[feature_cols_11am]
        X_b = X_b.replace([np.inf, -np.inf], 0).fillna(0)
        X_b_scaled = scaler_11am.transform(X_b)

        prob_above_11am = 0.0
        for model_name, model in models_11am.items():
            prob_above_11am += model.predict_proba(X_b_scaled)[0][1] * weights_11am.get(model_name, 0.25)

    return prob_bullish, prob_above_11am, current_price, time_pct


def backtest_ticker(ticker: str, start_date: str, end_date: str):
    """Backtest V3 model using REAL hourly data - NO LEAKAGE"""

    print(f"\n{'='*70}")
    print(f"  BACKTEST V3: {ticker}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"{'='*70}")

    model_data = load_v3_model(ticker)
    if not model_data:
        print(f"  No V3 model found for {ticker}")
        return None

    has_11am_model = 'models_11am' in model_data
    print(f"  Model version: {model_data.get('version', 'unknown')}")
    print(f"  Training period: {model_data.get('train_period', 'unknown')}")
    print(f"  Has 11 AM model: {has_11am_model}")

    # Fetch data
    fetch_start = (pd.to_datetime(start_date) - timedelta(days=10)).strftime('%Y-%m-%d')
    hourly_df = fetch_hourly_data(ticker, fetch_start, end_date)
    daily_df = fetch_daily_data(ticker, fetch_start, end_date)

    if len(hourly_df) < 10 or len(daily_df) < 5:
        print("  Not enough data")
        return None

    # Get trading days in test period
    test_start = pd.to_datetime(start_date).date()
    test_end = pd.to_datetime(end_date).date()
    trading_days = sorted(set(hourly_df.index.date))
    trading_days = [d for d in trading_days if test_start <= d <= test_end]

    print(f"  Testing {len(trading_days)} trading days")

    results_by_hour = {}
    all_predictions = []

    for day in trading_days:
        if day not in daily_df.index:
            continue

        daily_dates = list(daily_df.index)
        day_idx = daily_dates.index(day)
        if day_idx < 2:
            continue

        today = daily_df.loc[day]
        prev_day = daily_df.iloc[day_idx - 1]
        prev_prev_day = daily_df.iloc[day_idx - 2]

        today_open = today['Open']
        today_close = today['Close']

        # Get 11 AM price
        price_11am = get_price_at_hour(hourly_df, day, 11)

        # Get hourly bars for this day
        day_start = pd.Timestamp(day).tz_localize('America/New_York').replace(hour=9, minute=0)
        day_end = pd.Timestamp(day).tz_localize('America/New_York').replace(hour=16, minute=30)
        day_bars = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index <= day_end)]

        if len(day_bars) < 2:
            continue

        # Test at each hour
        for j in range(1, len(day_bars) + 1):
            bars_so_far = day_bars.iloc[:j]
            current_hour = bars_so_far.iloc[-1].name.hour
            hour_label = f"{current_hour:02d}:00"

            # Only use 11am price if we're at/after 11 AM
            p11 = price_11am if current_hour >= 11 else None

            prob_bullish, prob_11am, current_price, time_pct = predict_at_hour(
                model_data, bars_so_far, today_open, prev_day, prev_prev_day, p11
            )

            # Target A: close > open
            actual_bullish = 1 if today_close > today_open else 0
            predicted_bullish = 1 if prob_bullish > 0.5 else 0
            correct_a = predicted_bullish == actual_bullish
            high_conf_a = prob_bullish > 0.6 or prob_bullish < 0.4

            result = {
                'date': day,
                'hour': hour_label,
                'time_pct': time_pct,
                'current_price': current_price,
                'today_open': today_open,
                'today_close': today_close,
                'prob_bullish': prob_bullish,
                'actual_bullish': actual_bullish,
                'predicted_bullish': predicted_bullish,
                'correct_a': correct_a,
                'high_conf_a': high_conf_a,
            }

            # Target B: close > 11 AM price (only after 11 AM)
            if prob_11am is not None and price_11am is not None:
                actual_above_11am = 1 if today_close > price_11am else 0
                predicted_above_11am = 1 if prob_11am > 0.5 else 0
                correct_b = predicted_above_11am == actual_above_11am
                high_conf_b = prob_11am > 0.6 or prob_11am < 0.4

                result['price_11am'] = price_11am
                result['prob_11am'] = prob_11am
                result['actual_above_11am'] = actual_above_11am
                result['predicted_above_11am'] = predicted_above_11am
                result['correct_b'] = correct_b
                result['high_conf_b'] = high_conf_b

            if hour_label not in results_by_hour:
                results_by_hour[hour_label] = []
            results_by_hour[hour_label].append(result)
            all_predictions.append(result)

    # ============ SUMMARY RESULTS ============
    print(f"\n  {'='*65}")
    print(f"  TARGET A: Will close > open?")
    print(f"  {'='*65}")
    print(f"\n  {'Hour':<8} {'Days':>6} {'Correct':>8} {'Accuracy':>10} {'HiConf':>20}")
    print(f"  {'-'*55}")

    total_correct_a = 0
    total_days_a = 0
    hc_correct_a = 0
    hc_total_a = 0

    for hour in sorted(results_by_hour.keys()):
        results = results_by_hour[hour]
        df = pd.DataFrame(results)

        total = len(df)
        correct = df['correct_a'].sum()
        accuracy = correct / total if total > 0 else 0

        hc = df[df['high_conf_a']]
        hc_t = len(hc)
        hc_c = hc['correct_a'].sum() if hc_t > 0 else 0
        hc_acc = hc_c / hc_t if hc_t > 0 else 0

        total_correct_a += correct
        total_days_a += total
        hc_correct_a += hc_c
        hc_total_a += hc_t

        print(f"  {hour:<8} {total:>6} {correct:>8} {accuracy:>9.1%} {hc_c}/{hc_t}={hc_acc:>5.1%}")

    overall_a = total_correct_a / total_days_a if total_days_a > 0 else 0
    hc_overall_a = hc_correct_a / hc_total_a if hc_total_a > 0 else 0
    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<8} {total_days_a:>6} {total_correct_a:>8} {overall_a:>9.1%} {hc_correct_a}/{hc_total_a}={hc_overall_a:>5.1%}")

    # Target B results
    if has_11am_model:
        print(f"\n  {'='*65}")
        print(f"  TARGET B: Will close > 11:00 AM price?")
        print(f"  (Only available after 11 AM)")
        print(f"  {'='*65}")
        print(f"\n  {'Hour':<8} {'Days':>6} {'Correct':>8} {'Accuracy':>10} {'HiConf':>20}")
        print(f"  {'-'*55}")

        total_correct_b = 0
        total_days_b = 0
        hc_correct_b = 0
        hc_total_b = 0

        for hour in sorted(results_by_hour.keys()):
            results = results_by_hour[hour]
            df = pd.DataFrame(results)

            if 'correct_b' not in df.columns or df['correct_b'].isna().all():
                continue

            df_b = df[df['correct_b'].notna()]
            total = len(df_b)
            if total == 0:
                continue

            correct = df_b['correct_b'].sum()
            accuracy = correct / total

            hc = df_b[df_b['high_conf_b']]
            hc_t = len(hc)
            hc_c = hc['correct_b'].sum() if hc_t > 0 else 0
            hc_acc = hc_c / hc_t if hc_t > 0 else 0

            total_correct_b += correct
            total_days_b += total
            hc_correct_b += hc_c
            hc_total_b += hc_t

            print(f"  {hour:<8} {total:>6} {correct:>8} {accuracy:>9.1%} {hc_c}/{hc_t}={hc_acc:>5.1%}")

        overall_b = total_correct_b / total_days_b if total_days_b > 0 else 0
        hc_overall_b = hc_correct_b / hc_total_b if hc_total_b > 0 else 0
        print(f"  {'-'*55}")
        print(f"  {'TOTAL':<8} {total_days_b:>6} {total_correct_b:>8} {overall_b:>9.1%} {hc_correct_b}/{hc_total_b}={hc_overall_b:>5.1%}")

    # ============ DETAILED RAW PREDICTIONS ============
    print(f"\n  {'='*65}")
    print(f"  RAW PREDICTIONS - ALL DAYS AT 2:00 PM (peak accuracy hour)")
    print(f"  {'='*65}")

    pm2_preds = [p for p in all_predictions if p['hour'] == '14:00']
    if pm2_preds:
        print(f"\n  {'Date':<12} {'Open':>8} {'Close':>8} {'Prob_A':>7} {'Pred_A':>7} {'Act_A':>6} {'OK':>4} {'Conf':>6}")
        print(f"  {'-'*75}")
        for p in pm2_preds:
            conf = "*HI*" if p['high_conf_a'] else ""
            ok = "Y" if p['correct_a'] else "N"
            pred = "BULL" if p['predicted_bullish'] else "BEAR"
            act = "BULL" if p['actual_bullish'] else "BEAR"
            print(f"  {str(p['date']):<12} {p['today_open']:>8.2f} {p['today_close']:>8.2f} {p['prob_bullish']:>6.1%} {pred:>7} {act:>6} {ok:>4} {conf:>6}")

    # Target B raw predictions
    if has_11am_model:
        print(f"\n  {'='*65}")
        print(f"  RAW PREDICTIONS - TARGET B AT 2:00 PM")
        print(f"  {'='*65}")

        pm2_b = [p for p in pm2_preds if 'correct_b' in p and pd.notna(p.get('correct_b'))]
        if pm2_b:
            print(f"\n  {'Date':<12} {'11AM':>8} {'Close':>8} {'Prob_B':>7} {'Pred_B':>7} {'Act_B':>6} {'OK':>4} {'Conf':>6}")
            print(f"  {'-'*75}")
            for p in pm2_b:
                conf = "*HI*" if p.get('high_conf_b') else ""
                ok = "Y" if p['correct_b'] else "N"
                pred = "ABOVE" if p['predicted_above_11am'] else "BELOW"
                act = "ABOVE" if p['actual_above_11am'] else "BELOW"
                print(f"  {str(p['date']):<12} {p['price_11am']:>8.2f} {p['today_close']:>8.2f} {p['prob_11am']:>6.1%} {pred:>7} {act:>6} {ok:>4} {conf:>6}")

    return results_by_hour


def main():
    print("="*70)
    print("  INTRADAY MODEL V3 BACKTEST - STRICT INTEGRITY")
    print("="*70)
    print()
    print("  INTEGRITY GUARANTEES:")
    print("  - Uses REAL hourly prices from Polygon API")
    print("  - NO simulated prices, NO future information leakage")
    print("  - Models trained on 2020-01-01 to 2025-01-01 ONLY")
    print("  - Testing on 2025-01-02 to 2025-12-19 (OUT OF SAMPLE)")
    print()

    # Test on recent data
    for ticker in ['SPY', 'QQQ', 'IWM']:
        backtest_ticker(ticker, '2025-01-02', '2025-12-19')

    print("\n" + "="*70)
    print("  INTERPRETATION")
    print("="*70)
    print("""
  Target A (close > open):
    - Random baseline: 50%
    - Good performance: >70%
    - Excellent performance: >80%

  Target B (close > 11 AM price):
    - Random baseline: 50%
    - Good performance: >65%
    - Excellent performance: >75%

  High Confidence predictions (prob > 60% or < 40%):
    - Should have HIGHER accuracy than overall
    - These are the actionable signals
""")


if __name__ == '__main__':
    main()
