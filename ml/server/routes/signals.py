"""
Trading Signals Endpoints

Main trading direction and signal endpoints.
"""

from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
import pytz

from ..config import SUPPORTED_TICKERS, POLYGON_API_KEY, EARLY_SESSION_END_HOUR
from ..data.polygon import fetch_hourly_bars, fetch_daily_bars
from ..data.market import is_market_open, get_now_et, get_current_hour
from ..v6.predictions import (
    get_v6_prediction,
    get_probability_bucket,
    get_time_multiplier,
    get_signal_agreement_multiplier,
    is_neutral_zone,
    get_trading_action,
)

bp = Blueprint('signals', __name__)


@bp.route('/trading_directions', methods=['GET'])
def trading_directions():
    """
    Main trading direction endpoint - returns LONG/SHORT/NO_TRADE for each ticker.

    Uses V6 time-split model predictions.
    SPEC compliant: Uses daily bar open, proper session boundaries.
    """
    now_et = get_now_et()
    current_hour = now_et.hour
    is_open = is_market_open()

    # Hard gate: Market must be open (9:30 AM - 4:00 PM ET)
    if not is_open:
        return jsonify({
            'generated_at': now_et.isoformat(),
            'market_open': False,
            'action': 'NO_TRADE',
            'reason': 'Market is closed (9:30 AM - 4:00 PM ET)',
            'tickers': {t: {'action': 'NO_TRADE', 'reason': 'Market closed'} for t in SUPPORTED_TICKERS}
        })

    today = now_et.strftime('%Y-%m-%d')
    yesterday = (now_et - timedelta(days=5)).strftime('%Y-%m-%d')

    result = {
        'generated_at': now_et.isoformat(),
        'market_open': is_open,
        'current_hour': current_hour,
        'session': 'early' if current_hour < EARLY_SESSION_END_HOUR else 'late',
        'tickers': {}
    }

    for ticker in SUPPORTED_TICKERS:
        try:
            # Fetch data
            hourly_bars = fetch_hourly_bars(ticker, today, today)
            daily_bars = fetch_daily_bars(ticker, yesterday, today)

            if not daily_bars:
                result['tickers'][ticker] = {
                    'action': 'NO_TRADE',
                    'reason': 'Daily open unavailable - abort to prevent skew'
                }
                continue

            # SPEC: Use daily bar open (9:30 AM regular market) - NEVER hourly_bars[0]['o']
            today_open = daily_bars[-1]['o']

            if len(hourly_bars) < 1:
                result['tickers'][ticker] = {
                    'action': 'NO_TRADE',
                    'reason': 'Insufficient hourly data'
                }
                continue

            # Get V6 prediction
            prob_a, prob_b, session, price_11am = get_v6_prediction(
                ticker, hourly_bars, daily_bars, current_hour
            )

            if prob_a is None:
                result['tickers'][ticker] = {
                    'action': 'NO_TRADE',
                    'reason': 'V6 model unavailable'
                }
                continue

            # Determine which probability to use
            if session == 'early':
                active_prob = prob_a
                active_target = 'A'
            else:
                # Late session: use higher confidence signal
                active_prob = prob_b if price_11am else prob_a
                active_target = 'B' if price_11am else 'A'

            # Get action
            action = get_trading_action(active_prob, session)

            # Calculate multipliers
            time_mult = get_time_multiplier(current_hour)
            agreement_mult = get_signal_agreement_multiplier(prob_a, prob_b) if session == 'late' else 1.0

            result['tickers'][ticker] = {
                'action': action,
                'active_target': active_target,
                'probability': round(active_prob, 3),
                'target_a_prob': round(prob_a, 3),
                'target_b_prob': round(prob_b, 3) if prob_b else None,
                'session': session,
                'price_11am': round(price_11am, 2) if price_11am else None,
                'today_open': round(today_open, 2),
                'bucket': get_probability_bucket(active_prob),
                'multipliers': {
                    'time': time_mult,
                    'agreement': agreement_mult,
                    'combined': round(time_mult * agreement_mult, 2)
                }
            }

        except Exception as e:
            result['tickers'][ticker] = {
                'action': 'NO_TRADE',
                'reason': f'Error: {str(e)}'
            }

    return jsonify(result)


@bp.route('/daily_signals', methods=['GET'])
def daily_signals():
    """Get daily trading signals summary for all tickers"""
    now_et = get_now_et()
    current_hour = now_et.hour
    today = now_et.strftime('%Y-%m-%d')
    yesterday = (now_et - timedelta(days=10)).strftime('%Y-%m-%d')

    result = {
        'generated_at': now_et.isoformat(),
        'session': 'early' if current_hour < EARLY_SESSION_END_HOUR else 'late',
        'tickers': {}
    }

    for ticker in SUPPORTED_TICKERS:
        try:
            hourly_bars = fetch_hourly_bars(ticker, today, today)
            daily_bars = fetch_daily_bars(ticker, yesterday, today)

            if not daily_bars or len(hourly_bars) < 1:
                result['tickers'][ticker] = {'error': 'Insufficient data'}
                continue

            prob_a, prob_b, session, price_11am = get_v6_prediction(
                ticker, hourly_bars, daily_bars, current_hour
            )

            if prob_a is None:
                result['tickers'][ticker] = {'error': 'V6 model unavailable'}
                continue

            result['tickers'][ticker] = {
                'target_a_prob': round(prob_a, 3),
                'target_b_prob': round(prob_b, 3) if prob_b else None,
                'session': session,
                'action': get_trading_action(prob_a if session == 'early' else prob_b or prob_a, session)
            }

        except Exception as e:
            result['tickers'][ticker] = {'error': str(e)}

    return jsonify(result)
