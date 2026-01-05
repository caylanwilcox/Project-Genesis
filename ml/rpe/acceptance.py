"""
Acceptance State Detection Module
Phase 1: Structural Context Layer

Acceptance requires:
1. Price CLOSES beyond the level (not just wick)
2. Price HOLDS beyond for minimum time window (bars)
"""

from typing import List, Dict, Optional
from enum import Enum


class AcceptanceStatus(Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    TESTING = "TESTING"
    UNTESTED = "UNTESTED"
    FAILED_ACCEPTANCE = "FAILED_ACCEPTANCE"


class AcceptanceStrength(Enum):
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


# Default configuration
DEFAULT_CONFIG = {
    "intraday": {
        "min_closes_weak": 3,      # 3-4 closes = WEAK (15-20 min on 5m)
        "min_closes_moderate": 5,   # 5-8 closes = MODERATE
        "min_closes_strong": 9,     # 9+ closes = STRONG
        "testing_threshold_pct": 0.10,  # Within 0.10% = TESTING
    },
    "swing": {
        "min_closes_weak": 1,       # 1 daily close = WEAK
        "min_closes_moderate": 2,   # 2 daily closes = MODERATE
        "min_closes_strong": 3,     # 3+ daily closes = STRONG
    }
}


def check_acceptance(
    level: float,
    bars: List[Dict],
    direction: str = "above",
    config: Optional[Dict] = None
) -> Dict:
    """
    Check if price has accepted a level (intraday).

    Args:
        level: Price level to check
        bars: List of bar dicts with 'high', 'low', 'close'
        direction: "above" or "below"
        config: Optional config override

    Returns:
        Dict with 'status', 'strength', 'closes_held', 'time_held_minutes'
    """
    if not bars or level <= 0:
        return {
            "status": AcceptanceStatus.UNTESTED.value,
            "strength": None,
            "closes_held": 0,
            "time_held_minutes": 0
        }

    cfg = config or DEFAULT_CONFIG["intraday"]
    threshold_pct = cfg.get("testing_threshold_pct", 0.10) / 100

    # Track consecutive closes beyond level
    consecutive_closes = 0
    max_consecutive = 0
    was_accepted = False
    currently_beyond = False

    for bar in bars:
        close = bar.get('close', 0)
        high = bar.get('high', 0)
        low = bar.get('low', 0)

        if direction == "above":
            beyond = close > level
            tested = high >= level
            distance_pct = abs(close - level) / level if level > 0 else 0
        else:  # below
            beyond = close < level
            tested = low <= level
            distance_pct = abs(close - level) / level if level > 0 else 0

        if beyond:
            consecutive_closes += 1
            max_consecutive = max(max_consecutive, consecutive_closes)
            if consecutive_closes >= cfg.get("min_closes_weak", 3):
                was_accepted = True
        else:
            consecutive_closes = 0

        currently_beyond = beyond

    # Determine final status
    min_weak = cfg.get("min_closes_weak", 3)
    min_mod = cfg.get("min_closes_moderate", 5)
    min_strong = cfg.get("min_closes_strong", 9)

    # Check last bar for current state
    last_bar = bars[-1]
    last_close = last_bar.get('close', 0)
    last_high = last_bar.get('high', 0)
    last_low = last_bar.get('low', 0)

    if direction == "above":
        last_distance_pct = abs(last_close - level) / level if level > 0 else 0
        last_beyond = last_close > level
        last_tested = last_high >= level
    else:
        last_distance_pct = abs(last_close - level) / level if level > 0 else 0
        last_beyond = last_close < level
        last_tested = last_low <= level

    # Determine status
    if was_accepted and not currently_beyond:
        status = AcceptanceStatus.FAILED_ACCEPTANCE.value
        strength = None
    elif max_consecutive >= min_weak:
        status = AcceptanceStatus.ACCEPTED.value
        if max_consecutive >= min_strong:
            strength = AcceptanceStrength.STRONG.value
        elif max_consecutive >= min_mod:
            strength = AcceptanceStrength.MODERATE.value
        else:
            strength = AcceptanceStrength.WEAK.value
    elif last_tested and not last_beyond:
        status = AcceptanceStatus.REJECTED.value
        strength = None
    elif last_distance_pct <= threshold_pct:
        status = AcceptanceStatus.TESTING.value
        strength = None
    else:
        status = AcceptanceStatus.UNTESTED.value
        strength = None

    return {
        "status": status,
        "strength": strength,
        "closes_held": max_consecutive,
        "time_held_minutes": max_consecutive * 5  # Assuming 5m bars
    }


def check_swing_acceptance(
    level: float,
    daily_bars: List[Dict],
    direction: str = "above",
    config: Optional[Dict] = None
) -> Dict:
    """
    Check if price has accepted a level (swing/daily).

    Args:
        level: Price level to check
        daily_bars: List of daily bar dicts with 'close'
        direction: "above" or "below"
        config: Optional config override

    Returns:
        Dict with 'status', 'strength', 'closes_held'
    """
    if not daily_bars or level <= 0:
        return {
            "status": AcceptanceStatus.UNTESTED.value,
            "strength": None,
            "closes_held": 0
        }

    cfg = config or DEFAULT_CONFIG["swing"]

    # Count consecutive closes from most recent
    consecutive_closes = 0
    for bar in reversed(daily_bars):
        close = bar.get('close', 0)

        if direction == "above":
            beyond = close > level
        else:
            beyond = close < level

        if beyond:
            consecutive_closes += 1
        else:
            break

    # Determine strength
    min_weak = cfg.get("min_closes_weak", 1)
    min_mod = cfg.get("min_closes_moderate", 2)
    min_strong = cfg.get("min_closes_strong", 3)

    if consecutive_closes >= min_strong:
        status = AcceptanceStatus.ACCEPTED.value
        strength = AcceptanceStrength.STRONG.value
    elif consecutive_closes >= min_mod:
        status = AcceptanceStatus.ACCEPTED.value
        strength = AcceptanceStrength.MODERATE.value
    elif consecutive_closes >= min_weak:
        status = AcceptanceStatus.ACCEPTED.value
        strength = AcceptanceStrength.WEAK.value
    else:
        # Check if tested
        last_bar = daily_bars[-1]
        if direction == "above":
            tested = last_bar.get('high', 0) >= level
        else:
            tested = last_bar.get('low', 0) <= level

        if tested:
            status = AcceptanceStatus.TESTING.value
        else:
            status = AcceptanceStatus.UNTESTED.value
        strength = None

    return {
        "status": status,
        "strength": strength,
        "closes_held": consecutive_closes
    }


def get_acceptance_side(current_price: float, level: float, threshold_pct: float = 0.001) -> str:
    """
    Determine which side of a level the current price is on.

    Returns:
        "ABOVE", "BELOW", or "TESTING"
    """
    if level <= 0:
        return "ABOVE"

    distance_pct = abs(current_price - level) / level

    if distance_pct <= threshold_pct:
        return "TESTING"
    elif current_price > level:
        return "ABOVE"
    else:
        return "BELOW"
