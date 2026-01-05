"""
Main Compute Module
Phase 1: Structural Context Layer

Computes full Phase1IntradayContext and Phase1SwingContext.
"""

from typing import Dict, List, Optional
from datetime import datetime, time
import hashlib
import json

# Support both package imports and direct imports
try:
    from .vwap import calculate_vwap
    from .acceptance import check_acceptance, check_swing_acceptance, get_acceptance_side
    from .auction_state import classify_auction_state, classify_swing_auction_state
    from .levels import calculate_intraday_levels, calculate_swing_levels
    from .failures import detect_failures, detect_swing_failures
    from .beware import generate_beware_alerts, aggregate_risk_level
except ImportError:
    from vwap import calculate_vwap
    from acceptance import check_acceptance, check_swing_acceptance, get_acceptance_side
    from auction_state import classify_auction_state, classify_swing_auction_state
    from levels import calculate_intraday_levels, calculate_swing_levels
    from failures import detect_failures, detect_swing_failures
    from beware import generate_beware_alerts, aggregate_risk_level


VERSION = "3.0"


def get_session_label(current_time: str) -> str:
    """
    Get session label based on time.

    Returns:
        "EARLY" (9:30-11:30), "MID" (11:30-14:00), or "LATE" (14:00-16:00)
    """
    try:
        hour, minute = map(int, current_time.split(':'))
        t = time(hour, minute)

        if t < time(11, 30):
            return "EARLY"
        elif t < time(14, 0):
            return "MID"
        else:
            return "LATE"
    except:
        return "MID"


def compute_context_id(context: Dict) -> str:
    """Generate deterministic hash ID for context."""
    # Remove non-deterministic fields
    hashable = {k: v for k, v in context.items() if k not in ['context_id', 'as_of']}
    json_str = json.dumps(hashable, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def compute_intraday_context(
    ticker: str,
    bars_1m: List[Dict],
    bars_5m: Optional[List[Dict]] = None,
    prior_day: Optional[Dict] = None,
    swing_context: Optional[Dict] = None,
    current_time: Optional[str] = None,
    config: Optional[Dict] = None
) -> Dict:
    """
    Compute full Phase1IntradayContext.

    Args:
        ticker: Ticker symbol
        bars_1m: 1-minute bars for today
        bars_5m: 5-minute bars (or will aggregate from 1m)
        prior_day: Prior day OHLC dict
        swing_context: Pre-computed swing context
        current_time: Current time "HH:MM"
        config: Configuration overrides

    Returns:
        Phase1IntradayContext dict
    """
    if not bars_1m:
        return None

    # Calculate VWAP
    vwap = calculate_vwap(bars_1m, rth_only=True)

    # Calculate levels
    levels_dict = calculate_intraday_levels(bars_1m, prior_day, vwap)

    # Get current price
    current_price = bars_1m[-1].get('close', 0) if bars_1m else 0

    # Get session high/low
    high_of_day = max((b.get('high', 0) for b in bars_1m), default=0)
    low_of_day = min((b.get('low', float('inf')) for b in bars_1m if b.get('low', 0) > 0), default=0)
    if low_of_day == float('inf'):
        low_of_day = 0

    # Aggregate to 5m if not provided
    if not bars_5m:
        bars_5m = aggregate_to_5m(bars_1m)

    # Check acceptance states for each level
    acceptance_states = {}
    level_set = []

    for level_name, level_price in levels_dict.items():
        if level_price is None or level_price <= 0:
            continue

        # Determine direction
        if 'high' in level_name.lower() or level_name == 'vwap':
            direction = "above"
        elif 'low' in level_name.lower():
            direction = "below"
        else:
            direction = "above" if current_price > level_price else "below"

        acc_result = check_acceptance(level_price, bars_5m, direction, config)
        acceptance_states[level_name] = acc_result

        side = get_acceptance_side(current_price, level_price)

        level_set.append({
            'name': level_name,
            'price': level_price,
            'status': acc_result['status'],
            'side': side,
            'time_held_minutes': acc_result.get('time_held_minutes', 0),
            'closes_held': acc_result.get('closes_held', 0),
            'strength': acc_result.get('strength')
        })

    # Find nearest levels
    levels_above = [l for l in level_set if l['side'] == 'ABOVE' or l['side'] == 'TESTING']
    levels_below = [l for l in level_set if l['side'] == 'BELOW']

    nearest_above = min(levels_above, key=lambda x: x['price'] - current_price, default=None) if levels_above else None
    nearest_below = max(levels_below, key=lambda x: x['price'], default=None) if levels_below else None

    # Check for rotation
    or_high = levels_dict.get('open_30m_high', 0)
    or_low = levels_dict.get('open_30m_low', 0)
    touched_or_high = high_of_day >= or_high * 0.999 if or_high else False
    touched_or_low = low_of_day <= or_low * 1.001 if or_low else False

    # Classify auction state
    auction_context = {
        'current_price': current_price,
        'rth_open': levels_dict.get('rth_open', 0),
        'open_30m_high': or_high,
        'open_30m_low': or_low,
        'high_of_day': high_of_day,
        'low_of_day': low_of_day,
        'vwap': vwap,
        'acceptance_states': acceptance_states,
        'touched_or_high': touched_or_high,
        'touched_or_low': touched_or_low,
    }
    auction = classify_auction_state(auction_context)

    # Detect failures
    failure_context = {
        'open_30m_high': or_high,
        'open_30m_low': or_low,
        'high_of_day': high_of_day,
        'low_of_day': low_of_day,
        'current_price': current_price,
        'acceptance_states': acceptance_states,
        'rotation_complete': auction.get('rotation_complete', False)
    }
    failure = detect_failures(failure_context)

    # Build intraday context for beware
    intraday_for_beware = {
        'auction': auction,
        'failure': failure,
        'current_price': current_price,
        'levels': {'set': level_set}
    }

    # Generate BEWARE alerts
    beware_alerts = generate_beware_alerts(intraday_for_beware, swing_context, current_time)
    risk_level = aggregate_risk_level(beware_alerts)

    # Swing link
    swing_link = {
        'alignment': 'NEUTRAL',
        'swing_context_id': None,
        'swing_bias': 'NEUTRAL_CONTEXT'
    }

    if swing_context:
        swing_bias = swing_context.get('bias', {}).get('context', 'NEUTRAL_CONTEXT')
        swing_link['swing_context_id'] = swing_context.get('context_id')
        swing_link['swing_bias'] = swing_bias

        intraday_direction = auction.get('resolved_direction', 'BALANCED')

        if intraday_direction == 'UP' and swing_bias == 'BULLISH_CONTEXT':
            swing_link['alignment'] = 'ALIGNED'
        elif intraday_direction == 'DOWN' and swing_bias == 'BEARISH_CONTEXT':
            swing_link['alignment'] = 'ALIGNED'
        elif intraday_direction == 'UP' and swing_bias == 'BEARISH_CONTEXT':
            swing_link['alignment'] = 'CONFLICT'
        elif intraday_direction == 'DOWN' and swing_bias == 'BULLISH_CONTEXT':
            swing_link['alignment'] = 'CONFLICT'
        else:
            swing_link['alignment'] = 'NEUTRAL'

    # Session label
    session = get_session_label(current_time) if current_time else "MID"

    # Build context
    context = {
        'version': VERSION,
        'ticker': ticker,
        'as_of': datetime.now().isoformat(),
        'session': session,
        'current_price': current_price,

        'auction': auction,

        'levels': {
            'set': level_set,
            'nearest_above': {
                'name': nearest_above['name'],
                'price': nearest_above['price'],
                'distance_pct': round((nearest_above['price'] - current_price) / current_price * 100, 3)
            } if nearest_above else None,
            'nearest_below': {
                'name': nearest_below['name'],
                'price': nearest_below['price'],
                'distance_pct': round((current_price - nearest_below['price']) / current_price * 100, 3)
            } if nearest_below else None
        },

        'failure': failure,

        'beware': {
            'alerts': beware_alerts,
            'risk_level': risk_level
        },

        'swing_link': swing_link,

        'validity': {
            'expires_at': datetime.now().strftime('%Y-%m-%dT16:00:00-05:00'),
            'persistence': 'SESSION_ONLY'
        }
    }

    context['context_id'] = compute_context_id(context)

    return context


def compute_swing_context(
    ticker: str,
    daily_bars: List[Dict],
    as_of_date: Optional[str] = None,
    config: Optional[Dict] = None
) -> Dict:
    """
    Compute full Phase1SwingContext.

    Args:
        ticker: Ticker symbol
        daily_bars: Daily OHLC bars
        as_of_date: Date to compute for (default: latest)
        config: Configuration overrides

    Returns:
        Phase1SwingContext dict
    """
    if not daily_bars:
        return None

    # Calculate HTF levels
    htf_levels = calculate_swing_levels(daily_bars, as_of_date)

    # Get current price (latest close)
    sorted_bars = sorted(daily_bars, key=lambda x: x.get('date', x.get('timestamp', '')))
    current_price = sorted_bars[-1].get('close', 0) if sorted_bars else 0

    # Check acceptance for each HTF level
    htf_acceptance = {}
    level_set = []

    for level_name, level_price in htf_levels.items():
        if level_price is None or level_price <= 0:
            continue

        if 'high' in level_name.lower():
            direction = "above"
        elif 'low' in level_name.lower():
            direction = "below"
        else:
            direction = "above" if current_price > level_price else "below"

        acc_result = check_swing_acceptance(level_price, daily_bars, direction, config)
        htf_acceptance[level_name] = acc_result

        side = get_acceptance_side(current_price, level_price)

        level_set.append({
            'name': level_name,
            'price': level_price,
            'status': acc_result['status'],
            'side': side,
            'closes_held': acc_result.get('closes_held', 0),
            'strength': acc_result.get('strength')
        })

    # Classify swing auction state
    swing_auction_context = {
        'htf_acceptance': htf_acceptance,
        'current_price': current_price,
        **htf_levels
    }
    auction = classify_swing_auction_state(swing_auction_context)

    # Calculate bias
    bullish_count = sum(1 for l in level_set if l['side'] == 'ABOVE' and l['status'] == 'ACCEPTED')
    bearish_count = sum(1 for l in level_set if l['side'] == 'BELOW' and l['status'] == 'ACCEPTED')

    if bullish_count > bearish_count + 1:
        bias_context = 'BULLISH_CONTEXT'
    elif bearish_count > bullish_count + 1:
        bias_context = 'BEARISH_CONTEXT'
    else:
        bias_context = 'NEUTRAL_CONTEXT'

    # Strength based on acceptance count
    total_accepted = bullish_count + bearish_count
    if total_accepted >= 4:
        strength = 'STRONG'
    elif total_accepted >= 2:
        strength = 'MODERATE'
    else:
        strength = 'WEAK'

    # Invalidation level (most important accepted level)
    invalidation_level = None
    invalidation_desc = ""

    if bias_context == 'BULLISH_CONTEXT':
        # Invalidation is losing week open
        week_open = htf_levels.get('week_open', 0)
        if week_open > 0:
            invalidation_level = week_open
            invalidation_desc = f"Daily close below week_open ({week_open:.2f})"
    elif bias_context == 'BEARISH_CONTEXT':
        week_open = htf_levels.get('week_open', 0)
        if week_open > 0:
            invalidation_level = week_open
            invalidation_desc = f"Daily close above week_open ({week_open:.2f})"

    # Detect failures
    failure = detect_swing_failures({'htf_acceptance': htf_acceptance})

    # Build swing context for beware
    swing_for_beware = {
        'bias': {'context': bias_context},
        'failure': failure,
        'levels': {'set': level_set}
    }

    # Generate beware alerts (swing-only)
    beware_alerts = generate_beware_alerts({}, swing_for_beware)
    risk_level = aggregate_risk_level(beware_alerts)

    # Determine as_of_date
    if not as_of_date:
        as_of_date = sorted_bars[-1].get('date', sorted_bars[-1].get('timestamp', ''))[:10]

    context = {
        'version': VERSION,
        'ticker': ticker,
        'as_of_date': as_of_date,

        'auction': auction,

        'bias': {
            'context': bias_context,
            'strength': strength,
            'invalidation': {
                'description': invalidation_desc,
                'level': invalidation_level,
                'rule': invalidation_desc
            }
        },

        'levels': {
            'set': level_set
        },

        'failure': failure,

        'beware': {
            'alerts': beware_alerts,
            'risk_level': risk_level
        },

        'validity': {
            'valid_until': 'Next daily close',
            'persistence': 'MULTI_SESSION'
        }
    }

    context['context_id'] = compute_context_id(context)

    return context


def aggregate_to_5m(bars_1m: List[Dict]) -> List[Dict]:
    """Aggregate 1-minute bars to 5-minute bars."""
    if not bars_1m:
        return []

    bars_5m = []
    current_5m = None

    for bar in bars_1m:
        timestamp = bar.get('timestamp', '')

        try:
            if 'T' in str(timestamp):
                time_str = str(timestamp).split('T')[1][:5]
            else:
                time_str = str(timestamp)[:5]
            hour, minute = map(int, time_str.split(':'))
        except:
            continue

        # 5-minute bucket
        bucket_minute = (minute // 5) * 5
        bucket_key = f"{hour:02d}:{bucket_minute:02d}"

        if current_5m is None or current_5m.get('bucket') != bucket_key:
            if current_5m:
                bars_5m.append(current_5m)
            current_5m = {
                'bucket': bucket_key,
                'timestamp': bucket_key,
                'open': bar.get('open', bar.get('close', 0)),
                'high': bar.get('high', 0),
                'low': bar.get('low', float('inf')),
                'close': bar.get('close', 0),
                'volume': bar.get('volume', 0)
            }
        else:
            current_5m['high'] = max(current_5m['high'], bar.get('high', 0))
            low = bar.get('low', 0)
            if low > 0:
                current_5m['low'] = min(current_5m['low'], low)
            current_5m['close'] = bar.get('close', current_5m['close'])
            current_5m['volume'] += bar.get('volume', 0)

    if current_5m:
        if current_5m['low'] == float('inf'):
            current_5m['low'] = 0
        bars_5m.append(current_5m)

    return bars_5m
