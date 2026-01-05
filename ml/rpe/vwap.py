"""
VWAP Calculation Module
Phase 1: Structural Context Layer
"""

from datetime import datetime, time
from typing import List, Dict, Optional
import pytz


ET = pytz.timezone('America/New_York')
RTH_START = time(9, 30)
RTH_END = time(16, 0)


def is_rth(timestamp: str) -> bool:
    """Check if timestamp is within RTH (09:30-16:00 ET)."""
    if isinstance(timestamp, str):
        # Parse various formats
        for fmt in ['%Y-%m-%dT%H:%M:%S', '%H:%M:%S', '%H:%M', '%Y-%m-%d %H:%M:%S']:
            try:
                dt = datetime.strptime(timestamp.split('.')[0].split('-04')[0].split('-05')[0], fmt)
                t = dt.time()
                return RTH_START <= t < RTH_END
            except ValueError:
                continue
        # Try just extracting time portion
        try:
            time_str = timestamp.split('T')[-1][:5]
            hour, minute = map(int, time_str.split(':'))
            t = time(hour, minute)
            return RTH_START <= t < RTH_END
        except:
            pass
    return True  # Default to including if can't parse


def calculate_vwap(bars: List[Dict], rth_only: bool = True) -> Optional[float]:
    """
    Calculate Volume Weighted Average Price.

    VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
    Typical Price = (High + Low + Close) / 3

    Args:
        bars: List of bar dicts with 'high', 'low', 'close', 'volume', optional 'timestamp'
        rth_only: If True, only use RTH bars (09:30-16:00 ET)

    Returns:
        VWAP as float, or None if no valid bars
    """
    if not bars:
        return None

    cumulative_tp_vol = 0.0
    cumulative_vol = 0.0

    for bar in bars:
        # Skip if RTH only and outside RTH
        if rth_only and 'timestamp' in bar:
            if not is_rth(bar['timestamp']):
                continue

        # Skip zero volume bars
        volume = bar.get('volume', 0)
        if volume <= 0:
            continue

        high = bar.get('high', 0)
        low = bar.get('low', 0)
        close = bar.get('close', 0)

        if high <= 0 or low <= 0 or close <= 0:
            continue

        typical_price = (high + low + close) / 3
        cumulative_tp_vol += typical_price * volume
        cumulative_vol += volume

    if cumulative_vol <= 0:
        return None

    return cumulative_tp_vol / cumulative_vol


def calculate_vwap_with_bands(bars: List[Dict], std_dev_multiplier: float = 2.0) -> Dict:
    """
    Calculate VWAP with standard deviation bands.

    Returns:
        Dict with 'vwap', 'upper_band', 'lower_band', 'std_dev'
    """
    vwap = calculate_vwap(bars)

    if vwap is None or len(bars) < 2:
        return {
            'vwap': vwap,
            'upper_band': None,
            'lower_band': None,
            'std_dev': None
        }

    # Calculate variance
    cumulative_vol = 0.0
    cumulative_sq_diff = 0.0

    for bar in bars:
        volume = bar.get('volume', 0)
        if volume <= 0:
            continue

        typical_price = (bar['high'] + bar['low'] + bar['close']) / 3
        cumulative_sq_diff += volume * (typical_price - vwap) ** 2
        cumulative_vol += volume

    if cumulative_vol <= 0:
        return {'vwap': vwap, 'upper_band': None, 'lower_band': None, 'std_dev': None}

    variance = cumulative_sq_diff / cumulative_vol
    std_dev = variance ** 0.5

    return {
        'vwap': vwap,
        'upper_band': vwap + (std_dev * std_dev_multiplier),
        'lower_band': vwap - (std_dev * std_dev_multiplier),
        'std_dev': std_dev
    }
