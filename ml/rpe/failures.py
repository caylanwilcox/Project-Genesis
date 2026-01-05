"""
Failure Signal Detection Module
Phase 1: Structural Context Layer

Detects:
- Failed breakouts
- Failed acceptance
- Unfinished auctions
"""

from typing import Dict, List
from enum import Enum


class FailureType(Enum):
    FAILED_BREAKOUT = "FAILED_BREAKOUT"
    FAILED_ACCEPTANCE = "FAILED_ACCEPTANCE"
    UNFINISHED_AUCTION = "UNFINISHED_AUCTION"
    FAILED_BREAKDOWN = "FAILED_BREAKDOWN"


def detect_failures(context: Dict) -> Dict:
    """
    Detect failure signals in session.

    Args:
        context: Dict containing:
            - open_30m_high/low: Opening range
            - high_of_day/low_of_day: Session extremes
            - current_price: Current price
            - close: Session close (if available)
            - acceptance_states: Dict of acceptance states
            - rotation_complete: Whether rotation completed

    Returns:
        Dict with 'present', 'types', 'notes'
    """
    failures = []
    notes = []

    or_high = context.get('open_30m_high', 0)
    or_low = context.get('open_30m_low', 0)
    high_of_day = context.get('high_of_day', 0)
    low_of_day = context.get('low_of_day', 0)
    current_price = context.get('current_price', 0)
    close = context.get('close', current_price)
    acceptance_states = context.get('acceptance_states', {})

    # FAILED BREAKOUT: Broke above OR high, now back inside
    if or_high > 0 and high_of_day > or_high:
        if current_price < or_high:
            failures.append(FailureType.FAILED_BREAKOUT.value)
            notes.append(f"Broke above OR high {or_high:.2f}, now at {current_price:.2f}")

    # FAILED BREAKDOWN: Broke below OR low, now back inside
    if or_low > 0 and low_of_day < or_low:
        if current_price > or_low:
            failures.append(FailureType.FAILED_BREAKDOWN.value)
            notes.append(f"Broke below OR low {or_low:.2f}, now at {current_price:.2f}")

    # FAILED ACCEPTANCE: Check acceptance states for FAILED_ACCEPTANCE status
    for level_name, state in acceptance_states.items():
        if state.get('status') == 'FAILED_ACCEPTANCE':
            failures.append(FailureType.FAILED_ACCEPTANCE.value)
            notes.append(f"Failed acceptance at {level_name}")

    # UNFINISHED AUCTION: Close near extreme without rotation
    if close > 0 and high_of_day > 0 and low_of_day > 0:
        range_size = high_of_day - low_of_day
        if range_size > 0:
            # Close in top 10% of range
            close_position = (close - low_of_day) / range_size
            rotation_complete = context.get('rotation_complete', False)

            if close_position >= 0.90 and not rotation_complete:
                failures.append(FailureType.UNFINISHED_AUCTION.value)
                notes.append("Close near high without rotation - potential gap fill tomorrow")
            elif close_position <= 0.10 and not rotation_complete:
                failures.append(FailureType.UNFINISHED_AUCTION.value)
                notes.append("Close near low without rotation - potential gap fill tomorrow")

    # Remove duplicates while preserving order
    seen = set()
    unique_failures = []
    for f in failures:
        if f not in seen:
            seen.add(f)
            unique_failures.append(f)

    return {
        'present': len(unique_failures) > 0,
        'types': unique_failures,
        'notes': notes
    }


def detect_swing_failures(context: Dict) -> Dict:
    """
    Detect failure signals in swing timeframe.

    Args:
        context: Dict containing HTF acceptance states

    Returns:
        Dict with 'present', 'types', 'notes'
    """
    failures = []
    notes = []

    htf_acceptance = context.get('htf_acceptance', {})

    for level_name, state in htf_acceptance.items():
        if state.get('status') == 'FAILED_ACCEPTANCE':
            failures.append(FailureType.FAILED_ACCEPTANCE.value)
            notes.append(f"Failed HTF acceptance at {level_name}")

    return {
        'present': len(failures) > 0,
        'types': failures,
        'notes': notes
    }
