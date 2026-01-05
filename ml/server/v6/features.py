"""
V6 Feature Engineering

Builds the 29 features required by V6 time-split model.
MUST match train_time_split.py exactly.
"""

import pandas as pd
import numpy as np


def build_v6_features(
    hourly_bars: list,
    daily_bars: list,
    today_open: float,
    current_hour: int
) -> dict:
    """Build V6 model features from market data

    Args:
        hourly_bars: List of hourly bar dicts with o, h, l, c, v, t
        daily_bars: List of daily bar dicts with o, h, l, c, v, t
        today_open: Today's open price (from daily bar, NOT hourly)
        current_hour: Current hour in ET

    Returns:
        Dict of feature values matching V6 model feature_cols
    """
    if len(hourly_bars) < 1 or len(daily_bars) < 3:
        return None

    current_close = hourly_bars[-1]['c']
    current_high = max(b['h'] for b in hourly_bars)
    current_low = min(b['l'] for b in hourly_bars)

    # Previous days
    prev_day = daily_bars[-2] if len(daily_bars) >= 2 else daily_bars[-1]
    prev_prev_day = daily_bars[-3] if len(daily_bars) >= 3 else prev_day

    # Get 11 AM price if available
    price_11am = None
    for bar in hourly_bars:
        bar_hour = pd.Timestamp(bar['t'], unit='ms', tz='America/New_York').hour
        if bar_hour == 11:
            price_11am = bar['c']
            break

    # Core calculations
    gap = (today_open - prev_day['c']) / prev_day['c']
    range_so_far = max(current_high - current_low, 0.0001)

    # Calculate time_pct like training: (hours since 9 AM) / 6.5
    last_bar_time = pd.Timestamp(hourly_bars[-1]['t'], unit='ms', tz='America/New_York')
    hours_since_open = (last_bar_time.hour - 9) + (last_bar_time.minute / 60)
    time_pct = min(max(hours_since_open / 6.5, 0), 1)

    features = {
        # Core features
        'gap': gap,
        'gap_size': abs(gap),
        'gap_direction': 1 if gap > 0 else (-1 if gap < 0 else 0),

        # Previous day features
        'prev_return': (prev_day['c'] - prev_prev_day['c']) / prev_prev_day['c'],
        'prev_range': (prev_day['h'] - prev_day['l']) / prev_day['c'],
        'prev_body': (prev_day['c'] - prev_day['o']) / prev_day['o'],
        'prev_bullish': 1 if prev_day['c'] > prev_day['o'] else 0,

        # Current position features
        'current_vs_open': (current_close - today_open) / today_open,
        'current_vs_open_direction': 1 if current_close > today_open else (-1 if current_close < today_open else 0),
        'above_open': 1 if current_close > today_open else 0,
        'position_in_range': (current_close - current_low) / range_so_far,
        'range_so_far_pct': range_so_far / today_open,

        # Near high - match training: (high - current) < (current - low)
        'near_high': 1 if (current_high - current_close) < (current_close - current_low) else 0,

        # Gap status
        'gap_filled': 1 if (gap > 0 and current_low <= prev_day['c']) or (gap <= 0 and current_high >= prev_day['c']) else 0,
        'morning_reversal': 1 if (gap > 0 and current_close < today_open) or (gap < 0 and current_close > today_open) else 0,

        # Time and momentum
        'time_pct': time_pct,
        'first_hour_return': (hourly_bars[0]['c'] - today_open) / today_open if len(hourly_bars) >= 1 else 0,
        'last_hour_return': (hourly_bars[-1]['c'] - hourly_bars[-2]['c']) / hourly_bars[-2]['c'] if len(hourly_bars) >= 2 else 0,
        'bullish_bar_ratio': sum(1 for b in hourly_bars if b['c'] > b['o']) / len(hourly_bars) if hourly_bars else 0.5,

        # Day of week
        'is_monday': 1 if last_bar_time.dayofweek == 0 else 0,
        'is_friday': 1 if last_bar_time.dayofweek == 4 else 0,

        # Mean reversion signal
        'mean_reversion_signal': -1 if gap > 0.01 else (1 if gap < -0.01 else 0),
    }

    # 11 AM features
    if price_11am is not None and current_hour >= 11:
        features['current_vs_11am'] = (current_close - price_11am) / price_11am
        features['above_11am'] = 1 if current_close > price_11am else 0
    else:
        features['current_vs_11am'] = 0
        features['above_11am'] = 0

    # Multi-day features
    if len(daily_bars) >= 7:
        features['return_3d'] = (daily_bars[-2]['c'] - daily_bars[-5]['c']) / daily_bars[-5]['c']
        features['return_5d'] = (daily_bars[-2]['c'] - daily_bars[-7]['c']) / daily_bars[-7]['c']
        returns = [(daily_bars[i]['c'] - daily_bars[i-1]['c']) / daily_bars[i-1]['c'] for i in range(-5, 0)]
        features['volatility_5d'] = np.std(returns) if returns else 0.01
    else:
        features['return_3d'] = 0
        features['return_5d'] = 0
        features['volatility_5d'] = 0.01

    # Consecutive days
    features['consecutive_up'] = 0
    features['consecutive_down'] = 0
    for i in range(1, min(4, len(daily_bars))):
        if daily_bars[-i]['c'] > daily_bars[-i]['o']:
            features['consecutive_up'] += 1
        else:
            break
    for i in range(1, min(4, len(daily_bars))):
        if daily_bars[-i]['c'] < daily_bars[-i]['o']:
            features['consecutive_down'] += 1
        else:
            break

    return features, price_11am
