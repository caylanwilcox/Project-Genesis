"""
Market Hours and Session Utilities

Handles market open/close detection and session progress.
"""

from datetime import datetime, timedelta
import pytz
from ..config import (
    MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
    EARLY_SESSION_END_HOUR,
)


ET_TZ = pytz.timezone('America/New_York')


def get_now_et() -> datetime:
    """Get current time in Eastern timezone"""
    return datetime.now(ET_TZ)


def is_market_open() -> bool:
    """Check if US stock market is currently open

    SPEC:
    - 09:30:00 ET = OPEN (market opens)
    - 16:00:00 ET = CLOSED (market closes AT 4:00 PM, not after)

    Returns:
        True if market is open (9:30 AM - 4:00 PM ET, weekdays)
    """
    now = get_now_et()

    # Check if weekday
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    # Check market hours
    market_open = now.replace(
        hour=MARKET_OPEN_HOUR,
        minute=MARKET_OPEN_MINUTE,
        second=0,
        microsecond=0
    )
    market_close = now.replace(
        hour=MARKET_CLOSE_HOUR,
        minute=MARKET_CLOSE_MINUTE,
        second=0,
        microsecond=0
    )

    # SPEC: 9:30 AM is open, 4:00 PM is closed (use < not <=)
    return market_open <= now < market_close


def get_session_progress() -> float:
    """Calculate how far through the trading session we are

    Returns:
        Float from 0 to 1:
        - 0 = market open (9:30 ET)
        - 1 = market close (4:00 ET)
    """
    now = get_now_et()

    market_open = now.replace(
        hour=MARKET_OPEN_HOUR,
        minute=MARKET_OPEN_MINUTE,
        second=0,
        microsecond=0
    )
    market_close = now.replace(
        hour=MARKET_CLOSE_HOUR,
        minute=MARKET_CLOSE_MINUTE,
        second=0,
        microsecond=0
    )

    if now < market_open:
        return 0.0
    if now > market_close:
        return 1.0

    total_minutes = (market_close - market_open).total_seconds() / 60
    elapsed_minutes = (now - market_open).total_seconds() / 60

    return elapsed_minutes / total_minutes


def get_current_hour() -> int:
    """Get current hour in Eastern timezone"""
    return get_now_et().hour


def get_session() -> str:
    """Get current session: 'early' or 'late'

    SPEC: Early session = hour < 11, Late session = hour >= 11
    """
    hour = get_current_hour()
    return 'early' if hour < EARLY_SESSION_END_HOUR else 'late'


def get_next_trading_day() -> datetime:
    """Get the next trading day (skips weekends)

    Returns:
        datetime of next trading day at market open
    """
    now = get_now_et()
    next_day = now + timedelta(days=1)

    # Skip weekends
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)

    return next_day.replace(
        hour=MARKET_OPEN_HOUR,
        minute=MARKET_OPEN_MINUTE,
        second=0,
        microsecond=0
    )
