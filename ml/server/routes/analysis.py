"""
Analysis Endpoints

RPE, Northstar, and other analysis endpoints.
"""

from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
import pytz
import pandas as pd

from ..config import SUPPORTED_TICKERS, POLYGON_API_KEY, EARLY_SESSION_END_HOUR
from ..data.polygon import fetch_hourly_bars, fetch_daily_bars, fetch_minute_bars
from ..data.market import get_now_et, get_current_hour
from ..v6.predictions import get_v6_prediction

bp = Blueprint('analysis', __name__)


@bp.route('/northstar', methods=['GET'])
def northstar_analysis():
    """Northstar market analysis endpoint"""
    try:
        from rpe.northstar_pipeline import NorthstarPipeline, analyze_market
        NORTHSTAR_AVAILABLE = True
    except ImportError:
        NORTHSTAR_AVAILABLE = False

    if not NORTHSTAR_AVAILABLE:
        return jsonify({'error': 'Northstar pipeline not available'}), 503

    ticker = request.args.get('ticker', 'SPY').upper()

    if ticker not in SUPPORTED_TICKERS:
        return jsonify({'error': f'Unsupported ticker: {ticker}'}), 400

    now_et = get_now_et()
    today = now_et.strftime('%Y-%m-%d')
    yesterday = (now_et - timedelta(days=5)).strftime('%Y-%m-%d')

    try:
        # Fetch data
        minute_bars = fetch_minute_bars(ticker, today)
        daily_bars = fetch_daily_bars(ticker, yesterday, today)

        if len(minute_bars) < 30:
            return jsonify({
                'error': 'Insufficient minute data',
                'bars_available': len(minute_bars)
            }), 400

        # Convert to DataFrames
        bars_1m = pd.DataFrame(minute_bars)
        bars_1m = bars_1m.rename(columns={
            'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'
        })

        daily_df = None
        if len(daily_bars) >= 2:
            daily_df = pd.DataFrame(daily_bars)
            daily_df = daily_df.rename(columns={
                'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'
            })

        # Run Northstar
        pipeline = NorthstarPipeline()
        analysis = pipeline.run(
            symbol=ticker,
            bars_1m=bars_1m,
            daily_bars=daily_df
        )

        return jsonify({
            'ticker': ticker,
            'generated_at': now_et.isoformat(),
            'analysis': analysis
        })

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@bp.route('/rpe', methods=['GET'])
def rpe_analysis():
    """Reality Proof Engine (RPE) analysis endpoint"""
    try:
        from rpe.rpe_engine import RPEPipeline
        RPE_AVAILABLE = True
    except ImportError:
        RPE_AVAILABLE = False

    if not RPE_AVAILABLE:
        return jsonify({'error': 'RPE engine not available'}), 503

    tickers_param = request.args.get('tickers', ','.join(SUPPORTED_TICKERS))
    tickers = [t.strip().upper() for t in tickers_param.split(',')]

    now_et = get_now_et()
    current_hour = now_et.hour
    today = now_et.strftime('%Y-%m-%d')
    yesterday = (now_et - timedelta(days=5)).strftime('%Y-%m-%d')

    result = {
        'generated_at': now_et.isoformat(),
        'session': 'early' if current_hour < EARLY_SESSION_END_HOUR else 'late',
        'tickers': {}
    }

    pipeline = RPEPipeline()

    for ticker in tickers:
        try:
            # Fetch data
            minute_bars = fetch_minute_bars(ticker, today)
            daily_bars_raw = fetch_daily_bars(ticker, yesterday, today)
            hourly_bars = fetch_hourly_bars(ticker, today, today)

            if len(minute_bars) < 30:
                result['tickers'][ticker] = {
                    'error': 'Insufficient minute data',
                    'bars_available': len(minute_bars)
                }
                continue

            # Convert to DataFrames
            bars_1m = pd.DataFrame(minute_bars)
            bars_1m = bars_1m.rename(columns={
                'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'
            })

            daily_df = None
            if len(daily_bars_raw) >= 2:
                daily_df = pd.DataFrame(daily_bars_raw)
                daily_df = daily_df.rename(columns={
                    'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'
                })

            # Get V6 predictions
            v6_preds = None
            if len(hourly_bars) >= 1 and len(daily_bars_raw) >= 3:
                prob_a, prob_b, session, price_11am = get_v6_prediction(
                    ticker, hourly_bars, daily_bars_raw, current_hour
                )
                if prob_a is not None:
                    v6_preds = {
                        'target_a_prob': float(prob_a),
                        'target_b_prob': float(prob_b),
                        'session': session,
                        'price_11am': float(price_11am) if price_11am else None
                    }

            # Run RPE
            analysis = pipeline.run(
                symbol=ticker,
                bars_1m=bars_1m,
                daily_bars=daily_df,
                signals_last_10m=0,
                time_since_acceptance_minutes=0,
                v6_predictions=v6_preds
            )

            # Add price info - SPEC: Use daily bar open, NEVER minute_bars[0]['o']
            current_price = minute_bars[-1]['c']
            today_open = daily_bars_raw[-1]['o'] if daily_bars_raw else None
            today_change = ((current_price - today_open) / today_open * 100) if today_open else None

            analysis['current_price'] = round(current_price, 2)
            analysis['today_open'] = round(today_open, 2) if today_open else None
            analysis['today_change_pct'] = round(today_change, 2) if today_change else None

            if v6_preds:
                analysis['v6_signals'] = {
                    'target_a_prob': round(v6_preds['target_a_prob'], 3),
                    'target_b_prob': round(v6_preds['target_b_prob'], 3),
                    'session': v6_preds['session'],
                }

            result['tickers'][ticker] = analysis

        except Exception as e:
            import traceback
            result['tickers'][ticker] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    return jsonify(result)


@bp.route('/swing', methods=['GET'])
def swing_analysis():
    """SWING trade multi-timeframe analysis endpoint

    Returns V6 SWING predictions and RPE SWING structure analysis
    for multi-day/week swing trade decisions.
    """
    try:
        from rpe.swing_pipeline import SwingRPEPipeline
        SWING_RPE_AVAILABLE = True
    except ImportError:
        SWING_RPE_AVAILABLE = False

    tickers_param = request.args.get('tickers', ','.join(SUPPORTED_TICKERS))
    tickers = [t.strip().upper() for t in tickers_param.split(',')]

    now_et = get_now_et()
    today = now_et.strftime('%Y-%m-%d')
    lookback_start = (now_et - timedelta(days=90)).strftime('%Y-%m-%d')

    result = {
        'generated_at': now_et.isoformat(),
        'analysis_type': 'SWING',
        'timeframes': ['DAILY', 'WEEKLY'],
        'tickers': {}
    }

    for ticker in tickers:
        try:
            # Fetch daily data (90 days for indicators)
            daily_bars_raw = fetch_daily_bars(ticker, lookback_start, today)

            if len(daily_bars_raw) < 30:
                result['tickers'][ticker] = {
                    'error': 'Insufficient daily data',
                    'bars_available': len(daily_bars_raw)
                }
                continue

            # Convert to DataFrame
            daily_df = pd.DataFrame(daily_bars_raw)
            daily_df = daily_df.rename(columns={
                'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'
            })

            # Fetch weekly data
            weekly_df = None
            try:
                from ..data.polygon import fetch_weekly_bars
                weekly_bars_raw = fetch_weekly_bars(ticker, lookback_start, today)
                if weekly_bars_raw and len(weekly_bars_raw) >= 4:
                    weekly_df = pd.DataFrame(weekly_bars_raw)
                    weekly_df = weekly_df.rename(columns={
                        'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'
                    })
            except (ImportError, AttributeError):
                pass  # Weekly data optional

            # Get V6 SWING predictions if available
            v6_swing_preds = None
            try:
                from ..models.store import swing_v6_models, swing_3d_models, swing_1d_models
                if ticker in swing_v6_models:
                    from ..v6.swing_predictions import get_v6_swing_prediction, get_3d_swing_prediction, get_1d_swing_prediction
                    prob_5d, prob_10d = get_v6_swing_prediction(ticker, daily_df, weekly_df)
                    prob_3d = get_3d_swing_prediction(ticker, daily_df, weekly_df) if ticker in swing_3d_models else None
                    prob_1d = get_1d_swing_prediction(ticker, daily_df, weekly_df) if ticker in swing_1d_models else None
                    if prob_5d is not None:
                        v6_swing_preds = {
                            'prob_1d_up': round(float(prob_1d), 3) if prob_1d is not None else None,
                            'prob_3d_up': round(float(prob_3d), 3) if prob_3d is not None else None,
                            'prob_5d_up': round(float(prob_5d), 3),
                            'prob_10d_up': round(float(prob_10d), 3),
                            # SPEC: Swing model neutral zone is 20-80% based on V6_SWING_MODEL.md backtest data
                            # >80% or <20% = reliable signals (87%+ accuracy)
                            'signal_1d': 'BULLISH' if prob_1d and prob_1d > 0.8 else ('BEARISH' if prob_1d and prob_1d < 0.2 else 'NEUTRAL') if prob_1d else None,
                            'signal_3d': 'BULLISH' if prob_3d and prob_3d > 0.8 else ('BEARISH' if prob_3d and prob_3d < 0.2 else 'NEUTRAL') if prob_3d else None,
                            'signal_5d': 'BULLISH' if prob_5d > 0.8 else ('BEARISH' if prob_5d < 0.2 else 'NEUTRAL'),
                            'signal_10d': 'BULLISH' if prob_10d > 0.8 else ('BEARISH' if prob_10d < 0.2 else 'NEUTRAL')
                        }
            except (ImportError, KeyError):
                pass  # V6 SWING model optional

            # Run RPE SWING pipeline
            if SWING_RPE_AVAILABLE:
                pipeline = SwingRPEPipeline()
                analysis = pipeline.run(
                    symbol=ticker,
                    daily_bars=daily_df,
                    weekly_bars=weekly_df,
                    v6_swing_predictions=v6_swing_preds
                )
            else:
                # Fallback: just return basic info
                analysis = {
                    'error': 'RPE SWING pipeline not available',
                    'v6_swing': v6_swing_preds
                }

            # Add price info
            current_price = daily_bars_raw[-1]['c']
            analysis['current_price'] = round(current_price, 2)

            result['tickers'][ticker] = analysis

        except Exception as e:
            import traceback
            result['tickers'][ticker] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    return jsonify(result)


@bp.route('/mtf', methods=['GET'])
def multi_timeframe_analysis():
    """Multi-Timeframe Analysis endpoint

    Combines INTRADAY (V6 + RPE) and SWING (V6 SWING + RPE SWING)
    for comprehensive market view.
    """
    try:
        from rpe.northstar_pipeline import NorthstarPipeline
        INTRADAY_AVAILABLE = True
    except ImportError:
        INTRADAY_AVAILABLE = False

    try:
        from rpe.swing_pipeline import SwingRPEPipeline
        SWING_AVAILABLE = True
    except ImportError:
        SWING_AVAILABLE = False

    tickers_param = request.args.get('tickers', ','.join(SUPPORTED_TICKERS))
    tickers = [t.strip().upper() for t in tickers_param.split(',')]

    now_et = get_now_et()
    current_hour = now_et.hour
    today = now_et.strftime('%Y-%m-%d')
    lookback_start = (now_et - timedelta(days=90)).strftime('%Y-%m-%d')

    result = {
        'generated_at': now_et.isoformat(),
        'analysis_type': 'MULTI_TIMEFRAME',
        'session': 'early' if current_hour < EARLY_SESSION_END_HOUR else 'late',
        'tickers': {}
    }

    for ticker in tickers:
        try:
            ticker_result = {
                'intraday': None,
                'swing': None,
                'alignment': None
            }

            # Fetch data
            minute_bars = fetch_minute_bars(ticker, today)
            daily_bars_raw = fetch_daily_bars(ticker, lookback_start, today)
            hourly_bars = fetch_hourly_bars(ticker, today, today)

            # Daily DataFrame
            daily_df = None
            if len(daily_bars_raw) >= 2:
                daily_df = pd.DataFrame(daily_bars_raw)
                daily_df = daily_df.rename(columns={
                    'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'
                })

            # Weekly DataFrame
            weekly_df = None
            try:
                from ..data.polygon import fetch_weekly_bars
                weekly_bars_raw = fetch_weekly_bars(ticker, lookback_start, today)
                if weekly_bars_raw and len(weekly_bars_raw) >= 4:
                    weekly_df = pd.DataFrame(weekly_bars_raw)
                    weekly_df = weekly_df.rename(columns={
                        'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'
                    })
            except (ImportError, AttributeError):
                pass

            # ========== INTRADAY ANALYSIS ==========
            if INTRADAY_AVAILABLE and len(minute_bars) >= 30:
                bars_1m = pd.DataFrame(minute_bars)
                bars_1m = bars_1m.rename(columns={
                    'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'
                })

                # V6 intraday prediction
                v6_intraday = None
                if len(hourly_bars) >= 1 and len(daily_bars_raw) >= 3:
                    prob_a, prob_b, session, price_11am = get_v6_prediction(
                        ticker, hourly_bars, daily_bars_raw, current_hour
                    )
                    if prob_a is not None:
                        v6_intraday = {
                            'target_a_prob': round(float(prob_a), 3),
                            'target_b_prob': round(float(prob_b), 3),
                            'session': session,
                            # SPEC: Intraday neutral zone is 25-75% per 06_policy_risk_engine/CURRENT.md
                            'signal': 'BULLISH' if prob_a > 0.75 else ('BEARISH' if prob_a < 0.25 else 'NEUTRAL')
                        }

                # Northstar intraday
                pipeline = NorthstarPipeline()
                intraday_analysis = pipeline.run(
                    symbol=ticker,
                    bars_1m=bars_1m,
                    daily_bars=daily_df
                )
                intraday_analysis['v6'] = v6_intraday
                ticker_result['intraday'] = intraday_analysis

            # ========== SWING ANALYSIS ==========
            if SWING_AVAILABLE and daily_df is not None and len(daily_df) >= 30:
                # V6 SWING prediction
                v6_swing = None
                try:
                    from ..models.store import swing_v6_models, swing_3d_models, swing_1d_models
                    if ticker in swing_v6_models:
                        from ..v6.swing_predictions import get_v6_swing_prediction, get_3d_swing_prediction, get_1d_swing_prediction
                        prob_5d, prob_10d = get_v6_swing_prediction(ticker, daily_df, weekly_df)
                        prob_3d = get_3d_swing_prediction(ticker, daily_df, weekly_df) if ticker in swing_3d_models else None
                        prob_1d = get_1d_swing_prediction(ticker, daily_df, weekly_df) if ticker in swing_1d_models else None
                        if prob_5d is not None:
                            v6_swing = {
                                'prob_1d_up': round(float(prob_1d), 3) if prob_1d is not None else None,
                                'prob_3d_up': round(float(prob_3d), 3) if prob_3d is not None else None,
                                'prob_5d_up': round(float(prob_5d), 3),
                                'prob_10d_up': round(float(prob_10d), 3),
                                # SPEC: Swing model neutral zone is 20-80% based on V6_SWING_MODEL.md
                                'signal_1d': 'BULLISH' if prob_1d and prob_1d > 0.8 else ('BEARISH' if prob_1d and prob_1d < 0.2 else 'NEUTRAL') if prob_1d else None,
                                'signal_3d': 'BULLISH' if prob_3d and prob_3d > 0.8 else ('BEARISH' if prob_3d and prob_3d < 0.2 else 'NEUTRAL') if prob_3d else None,
                                'signal_5d': 'BULLISH' if prob_5d > 0.8 else ('BEARISH' if prob_5d < 0.2 else 'NEUTRAL'),
                                'signal_10d': 'BULLISH' if prob_10d > 0.8 else ('BEARISH' if prob_10d < 0.2 else 'NEUTRAL')
                            }
                except (ImportError, KeyError):
                    pass

                # RPE SWING analysis
                swing_pipeline = SwingRPEPipeline()
                swing_analysis = swing_pipeline.run(
                    symbol=ticker,
                    daily_bars=daily_df,
                    weekly_bars=weekly_df,
                    v6_swing_predictions=v6_swing
                )
                ticker_result['swing'] = swing_analysis

            # ========== ALIGNMENT CHECK ==========
            intraday_bias = None
            swing_bias = None

            if ticker_result['intraday']:
                phase4 = ticker_result['intraday'].get('phase4', {})
                intraday_bias = phase4.get('bias', 'NEUTRAL')

            if ticker_result['swing']:
                phase4 = ticker_result['swing'].get('phase4', {})
                swing_bias = phase4.get('bias', 'NEUTRAL')

            if intraday_bias and swing_bias:
                if intraday_bias == swing_bias and intraday_bias != 'NEUTRAL':
                    ticker_result['alignment'] = {
                        'status': 'ALIGNED',
                        'direction': intraday_bias,
                        'confidence': 'HIGH'
                    }
                elif intraday_bias == 'NEUTRAL' or swing_bias == 'NEUTRAL':
                    ticker_result['alignment'] = {
                        'status': 'PARTIAL',
                        'intraday': intraday_bias,
                        'swing': swing_bias,
                        'confidence': 'MEDIUM'
                    }
                else:
                    ticker_result['alignment'] = {
                        'status': 'CONFLICT',
                        'intraday': intraday_bias,
                        'swing': swing_bias,
                        'confidence': 'LOW',
                        'recommendation': 'REDUCE_SIZE or WAIT'
                    }

            # Current price
            if minute_bars:
                ticker_result['current_price'] = round(minute_bars[-1]['c'], 2)
            elif daily_bars_raw:
                ticker_result['current_price'] = round(daily_bars_raw[-1]['c'], 2)

            result['tickers'][ticker] = ticker_result

        except Exception as e:
            import traceback
            result['tickers'][ticker] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    return jsonify(result)


@bp.route('/replay', methods=['GET'])
def replay_mode():
    """Replay mode endpoint for historical analysis"""
    date_str = request.args.get('date')
    hour = request.args.get('hour', type=int)
    tickers_param = request.args.get('tickers', ','.join(SUPPORTED_TICKERS))
    tickers = [t.strip().upper() for t in tickers_param.split(',')]

    if not date_str:
        return jsonify({'error': 'date parameter required (YYYY-MM-DD)'}), 400

    if hour is None:
        return jsonify({'error': 'hour parameter required (9-15)'}), 400

    yesterday = (datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=10)).strftime('%Y-%m-%d')
    session = 'early' if hour < EARLY_SESSION_END_HOUR else 'late'

    result = {
        'replay_date': date_str,
        'replay_hour': hour,
        'session': session,
        'tickers': {}
    }

    for ticker in tickers:
        try:
            hourly_bars = fetch_hourly_bars(ticker, date_str, date_str)
            daily_bars = fetch_daily_bars(ticker, yesterday, date_str)

            # Filter hourly bars up to the specified hour
            filtered_bars = [
                b for b in hourly_bars
                if pd.Timestamp(b['t'], unit='ms', tz='America/New_York').hour <= hour
            ]

            if not daily_bars:
                result['tickers'][ticker] = {
                    'error': 'Daily open unavailable - cannot compute V6 features',
                    'v6_action': 'NO_TRADE'
                }
                continue

            # SPEC: Use daily bar open - NEVER use fallback
            today_open = daily_bars[-1]['o']

            if len(filtered_bars) < 1:
                result['tickers'][ticker] = {
                    'error': 'Insufficient hourly data',
                    'bars_available': len(filtered_bars)
                }
                continue

            prob_a, prob_b, _, price_11am = get_v6_prediction(
                ticker, filtered_bars, daily_bars, hour
            )

            if prob_a is None:
                result['tickers'][ticker] = {'error': 'V6 model unavailable'}
                continue

            result['tickers'][ticker] = {
                'target_a_prob': round(prob_a, 3),
                'target_b_prob': round(prob_b, 3) if prob_b else None,
                'session': session,
                'price_11am': round(price_11am, 2) if price_11am else None,
                'today_open': round(today_open, 2),
                'bars_analyzed': len(filtered_bars)
            }

        except Exception as e:
            result['tickers'][ticker] = {'error': str(e)}

    return jsonify(result)
