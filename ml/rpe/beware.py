"""
BEWARE Alert Generation Module
Phase 1: Structural Context Layer

BEWARE alerts are structural hazards (not trade triggers).
"""

from typing import Dict, List, Optional
from enum import Enum
from datetime import time


class BewareType(Enum):
    # Intraday types
    SWING_CONFLICT = "SWING_CONFLICT"
    BALANCED_SESSION = "BALANCED_SESSION"
    FAILED_EXPANSION_RISK = "FAILED_EXPANSION_RISK"
    LOW_RANGE_EFFICIENCY = "LOW_RANGE_EFFICIENCY"
    NEAR_KEY_LEVEL = "NEAR_KEY_LEVEL"
    FAILED_ACCEPTANCE_PRESENT = "FAILED_ACCEPTANCE_PRESENT"
    LATE_SESSION = "LATE_SESSION"
    FAILED_BREAKOUT_PRESENT = "FAILED_BREAKOUT_PRESENT"

    # Swing types
    HTF_BALANCE = "HTF_BALANCE"
    HTF_CONFLICT_INTERNAL = "HTF_CONFLICT_INTERNAL"
    FAILED_HTF_ACCEPTANCE = "FAILED_HTF_ACCEPTANCE"
    RANGE_COMPRESSION = "RANGE_COMPRESSION"


class Severity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


def generate_beware_alerts(
    intraday_context: Dict,
    swing_context: Optional[Dict] = None,
    current_time: Optional[str] = None
) -> List[Dict]:
    """
    Generate BEWARE alerts from structural context.

    Args:
        intraday_context: Intraday context dict
        swing_context: Swing context dict (optional)
        current_time: Current time string "HH:MM" (optional)

    Returns:
        List of BewareAlert dicts
    """
    alerts = []

    # --- INTRADAY ALERTS ---

    # SWING CONFLICT
    if swing_context:
        intraday_direction = intraday_context.get('auction', {}).get('resolved_direction', 'BALANCED')
        swing_bias = swing_context.get('bias', {}).get('context', 'NEUTRAL_CONTEXT')

        conflict = (
            (intraday_direction == 'UP' and swing_bias == 'BEARISH_CONTEXT') or
            (intraday_direction == 'DOWN' and swing_bias == 'BULLISH_CONTEXT')
        )

        if conflict:
            alerts.append({
                'type': BewareType.SWING_CONFLICT.value,
                'severity': Severity.WARNING.value,
                'message': f"Intraday {intraday_direction} conflicts with swing {swing_bias}",
                'related_level': None,
                'distance_pct': None
            })

    # BALANCED SESSION
    auction_state = intraday_context.get('auction', {}).get('state', '')
    if auction_state == 'BALANCED':
        alerts.append({
            'type': BewareType.BALANCED_SESSION.value,
            'severity': Severity.INFO.value,
            'message': "Session is rotational/balanced - no clear direction",
            'related_level': None,
            'distance_pct': None
        })

    # FAILED EXPANSION RISK
    if auction_state == 'FAILED_EXPANSION':
        alerts.append({
            'type': BewareType.FAILED_EXPANSION_RISK.value,
            'severity': Severity.WARNING.value,
            'message': "Failed expansion detected - reversal risk elevated",
            'related_level': None,
            'distance_pct': None
        })

    # FAILED ACCEPTANCE PRESENT
    failure = intraday_context.get('failure', {})
    if failure.get('present', False):
        for failure_type in failure.get('types', []):
            if 'ACCEPTANCE' in failure_type:
                alerts.append({
                    'type': BewareType.FAILED_ACCEPTANCE_PRESENT.value,
                    'severity': Severity.WARNING.value,
                    'message': "Failed acceptance detected",
                    'related_level': None,
                    'distance_pct': None
                })
            if 'BREAKOUT' in failure_type or 'BREAKDOWN' in failure_type:
                alerts.append({
                    'type': BewareType.FAILED_BREAKOUT_PRESENT.value,
                    'severity': Severity.WARNING.value,
                    'message': f"{failure_type} detected",
                    'related_level': None,
                    'distance_pct': None
                })

    # NEAR KEY LEVEL
    current_price = intraday_context.get('current_price', 0)
    levels = intraday_context.get('levels', {}).get('set', [])

    for level in levels:
        level_price = level.get('price', 0)
        level_name = level.get('name', '')

        if level_price > 0 and current_price > 0:
            distance_pct = abs(current_price - level_price) / current_price * 100

            if distance_pct <= 0.30:  # Within 0.30%
                alerts.append({
                    'type': BewareType.NEAR_KEY_LEVEL.value,
                    'severity': Severity.INFO.value,
                    'message': f"Price within 0.3% of {level_name} ({level_price:.2f})",
                    'related_level': level_name,
                    'distance_pct': round(distance_pct, 3)
                })

    # LATE SESSION
    if current_time:
        try:
            hour, minute = map(int, current_time.split(':'))
            ct = time(hour, minute)

            if ct >= time(15, 30):  # After 3:30 PM
                alerts.append({
                    'type': BewareType.LATE_SESSION.value,
                    'severity': Severity.CRITICAL.value,
                    'message': "Late session - reduced time for moves to develop",
                    'related_level': None,
                    'distance_pct': None
                })
            elif ct >= time(15, 0):  # After 3:00 PM
                alerts.append({
                    'type': BewareType.LATE_SESSION.value,
                    'severity': Severity.WARNING.value,
                    'message': "Approaching close - monitor closely",
                    'related_level': None,
                    'distance_pct': None
                })
        except:
            pass

    # --- SWING ALERTS ---
    if swing_context:
        swing_bias = swing_context.get('bias', {})

        # HTF BALANCE
        if swing_bias.get('context') == 'NEUTRAL_CONTEXT':
            alerts.append({
                'type': BewareType.HTF_BALANCE.value,
                'severity': Severity.INFO.value,
                'message': "HTF structure is balanced - no clear swing bias",
                'related_level': None,
                'distance_pct': None
            })

        # HTF CONFLICT INTERNAL (week vs month conflict)
        htf_levels = swing_context.get('levels', {}).get('set', [])
        week_status = None
        month_status = None

        for level in htf_levels:
            if 'week' in level.get('name', '').lower():
                week_status = level.get('side', '')
            if 'month' in level.get('name', '').lower():
                month_status = level.get('side', '')

        if week_status and month_status and week_status != month_status:
            alerts.append({
                'type': BewareType.HTF_CONFLICT_INTERNAL.value,
                'severity': Severity.WARNING.value,
                'message': f"Week ({week_status}) vs Month ({month_status}) conflict",
                'related_level': None,
                'distance_pct': None
            })

        # FAILED HTF ACCEPTANCE
        swing_failure = swing_context.get('failure', {})
        if swing_failure.get('present', False):
            alerts.append({
                'type': BewareType.FAILED_HTF_ACCEPTANCE.value,
                'severity': Severity.WARNING.value,
                'message': "Failed HTF acceptance detected",
                'related_level': None,
                'distance_pct': None
            })

    return alerts


def aggregate_risk_level(alerts: List[Dict]) -> str:
    """
    Aggregate alerts into overall risk level.

    Args:
        alerts: List of BewareAlert dicts

    Returns:
        RiskLevel string: "LOW", "MEDIUM", or "HIGH"
    """
    if not alerts:
        return RiskLevel.LOW.value

    critical_count = sum(1 for a in alerts if a.get('severity') == Severity.CRITICAL.value)
    warning_count = sum(1 for a in alerts if a.get('severity') == Severity.WARNING.value)

    if critical_count > 0 or warning_count >= 3:
        return RiskLevel.HIGH.value
    elif warning_count >= 1:
        return RiskLevel.MEDIUM.value
    else:
        return RiskLevel.LOW.value
