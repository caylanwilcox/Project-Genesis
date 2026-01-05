"""
Auction State Classification Module
Phase 1: Structural Context Layer

Auction States:
- RESOLVED: Directional expansion with acceptance
- BALANCED: Rotational / mean-reverting
- FAILED_EXPANSION: Attempted breakout then re-entry / failure
"""

from typing import Dict, Optional
from enum import Enum


class AuctionState(Enum):
    RESOLVED = "RESOLVED"
    BALANCED = "BALANCED"
    FAILED_EXPANSION = "FAILED_EXPANSION"


class Direction(Enum):
    UP = "UP"
    DOWN = "DOWN"
    BALANCED = "BALANCED"


class ExpansionQuality(Enum):
    CLEAN = "CLEAN"    # Breakout + hold, no retest
    DIRTY = "DIRTY"    # Breakout + hold, but retested
    NONE = "NONE"      # No expansion


def classify_auction_state(context: Dict) -> Dict:
    """
    Classify the auction state based on session behavior.

    Args:
        context: Dict containing:
            - current_price: Current price
            - rth_open: RTH open price
            - open_30m_high: Opening range high
            - open_30m_low: Opening range low
            - high_of_day: Session high
            - low_of_day: Session low
            - vwap: Current VWAP
            - acceptance_states: Dict of level acceptance states
            - touched_or_high: Whether touched opening range high
            - touched_or_low: Whether touched opening range low
            - retested_breakout_level: Whether retested after breakout

    Returns:
        Dict with 'state', 'resolved_direction', 'rotation_complete', 'expansion_quality'
    """
    current_price = context.get('current_price', 0)
    rth_open = context.get('rth_open', 0)
    or_high = context.get('open_30m_high', 0)
    or_low = context.get('open_30m_low', 0)
    high_of_day = context.get('high_of_day', 0)
    low_of_day = context.get('low_of_day', 0)
    vwap = context.get('vwap', 0)
    acceptance_states = context.get('acceptance_states', {})

    # Check acceptance of key levels
    or_high_accepted = acceptance_states.get('open_30m_high', {}).get('status') == 'ACCEPTED'
    or_low_accepted = acceptance_states.get('open_30m_low', {}).get('status') == 'ACCEPTED'
    or_high_failed = acceptance_states.get('open_30m_high', {}).get('status') == 'FAILED_ACCEPTANCE'
    or_low_failed = acceptance_states.get('open_30m_low', {}).get('status') == 'FAILED_ACCEPTANCE'
    or_high_rejected = acceptance_states.get('open_30m_high', {}).get('status') == 'REJECTED'
    or_low_rejected = acceptance_states.get('open_30m_low', {}).get('status') == 'REJECTED'

    # Check if price broke out of opening range
    broke_above_or = high_of_day > or_high if or_high > 0 else False
    broke_below_or = low_of_day < or_low if or_low > 0 else False

    # Check for rotation
    touched_high = context.get('touched_or_high', broke_above_or or high_of_day >= or_high * 0.999)
    touched_low = context.get('touched_or_low', broke_below_or or low_of_day <= or_low * 1.001)
    touched_vwap = context.get('touched_vwap', True)  # Usually true intraday

    rotation_complete = touched_high and touched_low and touched_vwap

    # Check retest
    retested = context.get('retested_breakout_level', False)

    # Classify auction state
    state = AuctionState.BALANCED.value
    direction = Direction.BALANCED.value
    expansion_quality = ExpansionQuality.NONE.value

    # RESOLVED UP: Broke above OR and accepted
    if or_high_accepted and current_price > or_high:
        state = AuctionState.RESOLVED.value
        direction = Direction.UP.value
        expansion_quality = ExpansionQuality.DIRTY.value if retested else ExpansionQuality.CLEAN.value

    # RESOLVED DOWN: Broke below OR and accepted
    elif or_low_accepted and current_price < or_low:
        state = AuctionState.RESOLVED.value
        direction = Direction.DOWN.value
        expansion_quality = ExpansionQuality.DIRTY.value if retested else ExpansionQuality.CLEAN.value

    # FAILED EXPANSION: Broke out but failed acceptance
    elif or_high_failed or or_low_failed:
        state = AuctionState.FAILED_EXPANSION.value
        if or_high_failed:
            direction = Direction.DOWN.value  # Failed upside, likely going down
        else:
            direction = Direction.UP.value  # Failed downside, likely going up
        expansion_quality = ExpansionQuality.NONE.value

    # BALANCED: Rotational, rejected at extremes
    elif rotation_complete or (or_high_rejected and or_low_rejected):
        state = AuctionState.BALANCED.value
        direction = Direction.BALANCED.value
        expansion_quality = ExpansionQuality.NONE.value

    # Default: Still developing
    else:
        # Determine lean based on current position
        if current_price > vwap and current_price > rth_open:
            direction = Direction.UP.value
        elif current_price < vwap and current_price < rth_open:
            direction = Direction.DOWN.value
        else:
            direction = Direction.BALANCED.value

    return {
        "state": state,
        "resolved_direction": direction,
        "rotation_complete": rotation_complete,
        "expansion_quality": expansion_quality
    }


def classify_swing_auction_state(context: Dict) -> Dict:
    """
    Classify swing auction state from daily data.

    Args:
        context: Dict containing:
            - htf_acceptance: Dict of HTF level acceptance states
            - current_price: Current price
            - week_open, month_open, etc.

    Returns:
        Dict with 'state', 'resolved_direction', 'dominant_tf'
    """
    htf_acceptance = context.get('htf_acceptance', {})
    current_price = context.get('current_price', 0)

    # Count bullish vs bearish acceptances
    bullish_count = 0
    bearish_count = 0
    dominant_tf = "NONE"
    max_closes = 0

    # Weight by timeframe importance
    tf_weights = {
        'year_open': 4,
        'quarter_open': 3,
        'month_open': 2,
        'week_open': 1
    }

    for level_name, state in htf_acceptance.items():
        if state.get('status') != 'ACCEPTED':
            continue

        weight = tf_weights.get(level_name, 1)
        closes = state.get('closes_held', 1)

        # Track dominant timeframe
        if closes > max_closes:
            max_closes = closes
            if 'year' in level_name:
                dominant_tf = "YEARLY"
            elif 'quarter' in level_name:
                dominant_tf = "QUARTERLY"
            elif 'month' in level_name:
                dominant_tf = "MONTHLY"
            elif 'week' in level_name:
                dominant_tf = "WEEKLY"

        # Determine direction (above = bullish, below = bearish)
        level_price = context.get(level_name, 0)
        if current_price > level_price:
            bullish_count += weight
        else:
            bearish_count += weight

    # Determine state
    if bullish_count >= 6:
        state = AuctionState.RESOLVED.value
        direction = Direction.UP.value
    elif bearish_count >= 6:
        state = AuctionState.RESOLVED.value
        direction = Direction.DOWN.value
    elif bullish_count > 0 or bearish_count > 0:
        state = AuctionState.BALANCED.value
        direction = Direction.BALANCED.value
    else:
        state = AuctionState.BALANCED.value
        direction = Direction.BALANCED.value
        dominant_tf = "NONE"

    return {
        "state": state,
        "resolved_direction": direction,
        "dominant_tf": dominant_tf
    }
