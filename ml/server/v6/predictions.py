"""
V6 Time-Split Model Predictions

Core prediction logic for V6 intraday model.
"""

import numpy as np
from ..models.store import intraday_v6_models
from ..config import EARLY_SESSION_END_HOUR, NEUTRAL_ZONE_LOW, NEUTRAL_ZONE_HIGH
from .features import build_v6_features


def get_v6_prediction(ticker: str, hourly_bars: list, daily_bars: list, current_hour: int):
    """Get prediction from V6 time-split model

    Args:
        ticker: Stock symbol (SPY, QQQ, IWM)
        hourly_bars: List of hourly bar dicts
        daily_bars: List of daily bar dicts
        current_hour: Current hour in ET

    Returns:
        Tuple of (prob_a, prob_b, session, price_11am) or (None, None, None, None)
    """
    ticker = ticker.upper()

    if ticker not in intraday_v6_models:
        return None, None, None, None

    model_data = intraday_v6_models[ticker]
    feature_cols = model_data['feature_cols']

    if len(hourly_bars) < 1 or len(daily_bars) < 3:
        return None, None, None, None

    # CRITICAL: Use daily bar open (9:30 AM regular market open) to match training
    today_open = daily_bars[-1]['o']

    # Build features
    result = build_v6_features(hourly_bars, daily_bars, today_open, current_hour)
    if result is None:
        return None, None, None, None

    features, price_11am = result

    # Create feature array
    X = np.array([[features.get(col, 0) for col in feature_cols]])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Determine session and get prediction
    # SPEC: Early session is current_hour < 11, Late session is current_hour >= 11
    if current_hour < EARLY_SESSION_END_HOUR:
        session = 'early'
        X_scaled = model_data['scaler_early'].transform(X)
        models = model_data['models_early']
        weights = model_data['weights_early']

        prob_a = sum(m.predict_proba(X_scaled)[0][1] * weights[name] for name, m in models.items())
        prob_b = 0.5  # No Target B in early session
    else:
        session = 'late'
        X_scaled = model_data['scaler_late'].transform(X)
        models_a = model_data['models_late_a']
        models_b = model_data['models_late_b']
        weights_a = model_data['weights_late_a']
        weights_b = model_data['weights_late_b']

        prob_a = sum(m.predict_proba(X_scaled)[0][1] * weights_a[name] for name, m in models_a.items())
        prob_b = sum(m.predict_proba(X_scaled)[0][1] * weights_b[name] for name, m in models_b.items())

    return prob_a, prob_b, session, price_11am


def get_probability_bucket(prob: float) -> str:
    """Classify probability into bucket

    SPEC (Updated 2026-01-03): Model accuracy only reliable at extremes
    - >= 90% or <= 10% → very_strong
    - >= 85% or <= 15% → strong
    - >= 80% or <= 20% → moderate
    - >= 75% or <= 25% → weak (boundary of neutral zone)
    - 25-75% → neutral (NO_TRADE - accuracy unreliable)

    Args:
        prob: Probability value (0-1)

    Returns:
        Bucket label
    """
    if prob >= 0.90:
        return 'very_strong_bull'
    elif prob >= 0.85:
        return 'strong_bull'
    elif prob >= 0.80:
        return 'moderate_bull'
    elif prob >= 0.75:
        return 'weak_bull'
    elif prob > 0.25:
        return 'neutral'  # 25-75% - NO_TRADE zone
    elif prob >= 0.20:
        return 'weak_bear'
    elif prob >= 0.15:
        return 'moderate_bear'
    elif prob >= 0.10:
        return 'strong_bear'
    else:
        return 'very_strong_bear'


def get_time_multiplier(hour: int) -> float:
    """Position size multiplier by time of day

    Args:
        hour: Current hour in ET

    Returns:
        Multiplier (0.5 to 1.2)
    """
    if hour < 12:
        return 0.7
    elif hour == 12:
        return 1.0
    elif hour in [13, 14]:
        return 1.2  # Peak accuracy
    elif hour == 15:
        return 0.8
    else:
        return 0.5


def get_signal_agreement_multiplier(prob_a: float, prob_b: float) -> float:
    """Multiplier when Target A and Target B agree

    SPEC (from TRADING_ENGINE_SPEC.md):
    - Both > 0.5: aligned_bullish => 1.2
    - Both < 0.5: aligned_bearish => 1.2
    - Conflicting (one > 0.5, one < 0.5): => 0.6

    Args:
        prob_a: Target A probability
        prob_b: Target B probability

    Returns:
        Multiplier (0.6 to 1.2)
    """
    if prob_a > 0.5 and prob_b > 0.5:
        return 1.2  # aligned_bullish
    elif prob_a < 0.5 and prob_b < 0.5:
        return 1.2  # aligned_bearish
    elif (prob_a > 0.5 and prob_b < 0.5) or (prob_a < 0.5 and prob_b > 0.5):
        return 0.6  # conflicting
    else:
        return 1.0  # neutral (exactly 0.5)


def is_neutral_zone(prob: float) -> bool:
    """Check if probability is in the neutral zone (NO_TRADE)

    SPEC (Updated 2026-01-03): Probabilities between 25-75% = NO_TRADE
    Model accuracy is only reliable at extremes (>75% or <25%)

    Args:
        prob: Probability value (0-1)

    Returns:
        True if in neutral zone
    """
    return NEUTRAL_ZONE_LOW <= prob <= NEUTRAL_ZONE_HIGH


def get_trading_action(prob: float, session: str) -> str:
    """Determine trading action from probability

    SPEC (Updated 2026-01-03):
    - prob > 0.75 → LONG (bullish)
    - prob < 0.25 → SHORT (bearish)
    - 0.25 <= prob <= 0.75 → NO_TRADE (neutral zone)

    Args:
        prob: Probability value (0-1)
        session: 'early' or 'late'

    Returns:
        Action string: 'LONG', 'SHORT', or 'NO_TRADE'
    """
    if is_neutral_zone(prob):
        return 'NO_TRADE'
    elif prob > NEUTRAL_ZONE_HIGH:
        return 'LONG'
    else:
        return 'SHORT'
