"""
Signal Caching

Prevents signal flip-flopping by locking signals for a period.
"""

from datetime import datetime
import pytz
from ..config import SIGNAL_LOCK_MINUTES

# Signal cache: "{ticker}_{date}_{hour}" -> {"action": "LONG", "locked_at": datetime, "data": {...}}
_signal_cache = {}

ET_TZ = pytz.timezone('America/New_York')


def get_cached_signal(ticker: str, hour: int) -> dict:
    """Get cached signal if it exists and is still valid

    Args:
        ticker: Stock symbol
        hour: Hour in ET

    Returns:
        Cached signal data or None if expired/not found
    """
    now = datetime.now(ET_TZ)
    cache_key = f"{ticker}_{now.strftime('%Y-%m-%d')}_{hour}"

    if cache_key in _signal_cache:
        cached = _signal_cache[cache_key]
        locked_at = cached.get('locked_at')
        if locked_at and (now - locked_at).total_seconds() < SIGNAL_LOCK_MINUTES * 60:
            return cached.get('data')

    return None


def cache_signal(ticker: str, hour: int, data: dict) -> None:
    """Cache a signal for the given ticker and hour

    Args:
        ticker: Stock symbol
        hour: Hour in ET
        data: Signal data to cache
    """
    now = datetime.now(ET_TZ)
    cache_key = f"{ticker}_{now.strftime('%Y-%m-%d')}_{hour}"
    _signal_cache[cache_key] = {
        'locked_at': now,
        'data': data
    }


def clear_signal_cache() -> None:
    """Clear all cached signals"""
    _signal_cache.clear()
