"""
Level Calculation Module
Phase 1: Structural Context Layer

Calculates intraday and swing reference levels.
"""

from typing import List, Dict, Optional
from datetime import datetime, time, timedelta
import pytz


ET = pytz.timezone('America/New_York')


def calculate_opening_range(bars_1m: List[Dict], window_minutes: int = 30) -> Dict:
    """
    Calculate opening range (first N minutes) high/low.

    Args:
        bars_1m: 1-minute bars with 'timestamp', 'high', 'low'
        window_minutes: Opening range window (default 30)

    Returns:
        Dict with 'open_30m_high', 'open_30m_low', 'open_30m_mid'
    """
    if not bars_1m:
        return {
            'open_30m_high': None,
            'open_30m_low': None,
            'open_30m_mid': None
        }

    or_high = None
    or_low = None
    rth_start = time(9, 30)
    or_end = time(9, 30 + window_minutes - 1) if window_minutes < 30 else time(9, 59)

    for bar in bars_1m:
        timestamp = bar.get('timestamp', '')

        # Parse time from timestamp
        try:
            if 'T' in str(timestamp):
                time_str = str(timestamp).split('T')[1][:5]
            else:
                time_str = str(timestamp)[:5]
            hour, minute = map(int, time_str.split(':'))
            bar_time = time(hour, minute)
        except:
            continue

        # Check if within opening range
        if bar_time < rth_start:
            continue
        if bar_time > or_end:
            break

        high = bar.get('high', 0)
        low = bar.get('low', 0)

        if high > 0:
            or_high = max(or_high or 0, high)
        if low > 0:
            or_low = min(or_low or float('inf'), low)

    if or_low == float('inf'):
        or_low = None

    or_mid = (or_high + or_low) / 2 if or_high and or_low else None

    return {
        'open_30m_high': or_high,
        'open_30m_low': or_low,
        'open_30m_mid': or_mid
    }


def calculate_premarket(
    bars: List[Dict],
    start: str = "04:00",
    end: str = "09:29"
) -> Dict:
    """
    Calculate premarket high/low.

    Args:
        bars: Bars with 'timestamp', 'high', 'low'
        start: Premarket start time (default "04:00")
        end: Premarket end time (default "09:29")

    Returns:
        Dict with 'premarket_high', 'premarket_low'
    """
    if not bars:
        return {'premarket_high': None, 'premarket_low': None}

    start_hour, start_min = map(int, start.split(':'))
    end_hour, end_min = map(int, end.split(':'))
    pm_start = time(start_hour, start_min)
    pm_end = time(end_hour, end_min)

    pm_high = None
    pm_low = None

    for bar in bars:
        timestamp = bar.get('timestamp', '')

        try:
            if 'T' in str(timestamp):
                time_str = str(timestamp).split('T')[1][:5]
            else:
                time_str = str(timestamp)[:5]
            hour, minute = map(int, time_str.split(':'))
            bar_time = time(hour, minute)
        except:
            continue

        if bar_time < pm_start or bar_time > pm_end:
            continue

        high = bar.get('high', 0)
        low = bar.get('low', 0)

        if high > 0:
            pm_high = max(pm_high or 0, high)
        if low > 0:
            pm_low = min(pm_low or float('inf'), low)

    if pm_low == float('inf'):
        pm_low = None

    return {
        'premarket_high': pm_high,
        'premarket_low': pm_low
    }


def calculate_intraday_levels(
    bars_1m: List[Dict],
    prior_day: Optional[Dict] = None,
    vwap: Optional[float] = None
) -> Dict:
    """
    Calculate all intraday reference levels.

    Args:
        bars_1m: Today's 1-minute bars
        prior_day: Prior day OHLC dict
        vwap: Pre-calculated VWAP (or will calculate)

    Returns:
        Dict with all intraday levels
    """
    # Opening range
    or_levels = calculate_opening_range(bars_1m)

    # Premarket
    pm_levels = calculate_premarket(bars_1m)

    # RTH Open (first bar at/after 09:30)
    rth_open = None
    rth_high = None
    rth_low = None

    for bar in bars_1m:
        timestamp = bar.get('timestamp', '')
        try:
            if 'T' in str(timestamp):
                time_str = str(timestamp).split('T')[1][:5]
            else:
                time_str = str(timestamp)[:5]
            hour, minute = map(int, time_str.split(':'))
            bar_time = time(hour, minute)
        except:
            continue

        if bar_time >= time(9, 30):
            if rth_open is None:
                rth_open = bar.get('open', bar.get('close', 0))

            high = bar.get('high', 0)
            low = bar.get('low', 0)

            if high > 0:
                rth_high = max(rth_high or 0, high)
            if low > 0:
                rth_low = min(rth_low or float('inf'), low)

    if rth_low == float('inf'):
        rth_low = None

    # Prior day levels
    prior_high = prior_day.get('high') if prior_day else None
    prior_low = prior_day.get('low') if prior_day else None
    prior_close = prior_day.get('close') if prior_day else None

    return {
        'vwap': vwap,
        'rth_open': rth_open,
        'rth_high': rth_high,
        'rth_low': rth_low,
        'open_30m_high': or_levels['open_30m_high'],
        'open_30m_low': or_levels['open_30m_low'],
        'prior_day_high': prior_high,
        'prior_day_low': prior_low,
        'prior_day_close': prior_close,
        'premarket_high': pm_levels['premarket_high'],
        'premarket_low': pm_levels['premarket_low'],
    }


def calculate_swing_levels(daily_bars: List[Dict], as_of_date: Optional[str] = None) -> Dict:
    """
    Calculate HTF swing reference levels.

    Args:
        daily_bars: Daily OHLC bars (sorted oldest to newest)
        as_of_date: Date to calculate levels for (default: latest)

    Returns:
        Dict with week/month/quarter/year opens and prior period H/L
    """
    if not daily_bars:
        return {}

    # Sort by date
    sorted_bars = sorted(daily_bars, key=lambda x: x.get('date', x.get('timestamp', '')))

    # Get reference date
    if as_of_date:
        ref_date = datetime.strptime(as_of_date, '%Y-%m-%d')
    else:
        last_bar = sorted_bars[-1]
        date_str = last_bar.get('date', last_bar.get('timestamp', ''))[:10]
        ref_date = datetime.strptime(date_str, '%Y-%m-%d')

    # Find period starts
    week_start = ref_date - timedelta(days=ref_date.weekday())  # Monday
    month_start = ref_date.replace(day=1)
    quarter_month = ((ref_date.month - 1) // 3) * 3 + 1
    quarter_start = ref_date.replace(month=quarter_month, day=1)
    year_start = ref_date.replace(month=1, day=1)

    # Find prior periods
    prior_week_start = week_start - timedelta(days=7)
    prior_week_end = week_start - timedelta(days=1)

    if month_start.month == 1:
        prior_month_start = month_start.replace(year=month_start.year - 1, month=12)
    else:
        prior_month_start = month_start.replace(month=month_start.month - 1)
    prior_month_end = month_start - timedelta(days=1)

    # Initialize levels
    levels = {
        'week_open': None,
        'month_open': None,
        'quarter_open': None,
        'year_open': None,
        'prior_week_high': None,
        'prior_week_low': None,
        'prior_week_close': None,
        'prior_month_high': None,
        'prior_month_low': None,
    }

    prior_week_highs = []
    prior_week_lows = []
    prior_month_highs = []
    prior_month_lows = []

    for bar in sorted_bars:
        date_str = bar.get('date', bar.get('timestamp', ''))[:10]
        try:
            bar_date = datetime.strptime(date_str, '%Y-%m-%d')
        except:
            continue

        open_price = bar.get('open', 0)
        high = bar.get('high', 0)
        low = bar.get('low', 0)
        close = bar.get('close', 0)

        # Week open (first trading day of week)
        if bar_date >= week_start and bar_date <= ref_date:
            if levels['week_open'] is None and open_price > 0:
                levels['week_open'] = open_price

        # Month open
        if bar_date >= month_start and bar_date <= ref_date:
            if levels['month_open'] is None and open_price > 0:
                levels['month_open'] = open_price

        # Quarter open
        if bar_date >= quarter_start and bar_date <= ref_date:
            if levels['quarter_open'] is None and open_price > 0:
                levels['quarter_open'] = open_price

        # Year open
        if bar_date >= year_start and bar_date <= ref_date:
            if levels['year_open'] is None and open_price > 0:
                levels['year_open'] = open_price

        # Prior week
        if prior_week_start <= bar_date <= prior_week_end:
            if high > 0:
                prior_week_highs.append(high)
            if low > 0:
                prior_week_lows.append(low)
            levels['prior_week_close'] = close  # Last one wins

        # Prior month
        if prior_month_start <= bar_date <= prior_month_end:
            if high > 0:
                prior_month_highs.append(high)
            if low > 0:
                prior_month_lows.append(low)

    # Calculate prior period H/L
    if prior_week_highs:
        levels['prior_week_high'] = max(prior_week_highs)
    if prior_week_lows:
        levels['prior_week_low'] = min(prior_week_lows)
    if prior_month_highs:
        levels['prior_month_high'] = max(prior_month_highs)
    if prior_month_lows:
        levels['prior_month_low'] = min(prior_month_lows)

    return levels
