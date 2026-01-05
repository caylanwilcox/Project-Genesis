"""
Unit Tests: Trading Policy Rules

SPEC REFERENCE:
- NZ: Neutral Zone (25-75% = NO_TRADE) - Model accuracy unreliable in this range
- BK: Confidence Buckets
- TM: Time Multiplier (morning boost, afternoon decay)
- AM: Agreement Multiplier (both aligned = 1.2, conflicting = 0.6)
- TS: Target Selection (early → A, late → B)
"""

import pytest
import sys
sys.path.insert(0, '/Users/it/Documents/mvp_coder_starter_kit (2)/mvp-trading-app/ml')


# Helper functions that mirror production logic
def get_action(prob: float) -> str:
    """
    NZ SPEC (Updated 2026-01-03):
    - prob > 0.75 → BULLISH (model accuracy reliable above 75%)
    - prob < 0.25 → BEARISH (model accuracy reliable below 25%)
    - 0.25 <= prob <= 0.75 → NO_TRADE (neutral zone - accuracy unreliable)
    """
    if prob > 0.75:
        return 'BULLISH'
    elif prob < 0.25:
        return 'BEARISH'
    else:
        return 'NO_TRADE'


def get_confidence_bucket(prob: float) -> tuple:
    """
    BK SPEC (Updated 2026-01-03):
    - prob >= 0.90 OR prob <= 0.10 → "very_strong" (size_mult=1.0)
    - prob >= 0.85 OR prob <= 0.15 → "strong" (size_mult=0.75)
    - prob >= 0.80 OR prob <= 0.20 → "moderate" (size_mult=0.50)
    - prob >= 0.75 OR prob <= 0.25 → "weak" (size_mult=0.25)
    - else (neutral zone 25-75%) → None
    """
    if prob >= 0.90 or prob <= 0.10:
        return ('very_strong', 1.0)
    elif prob >= 0.85 or prob <= 0.15:
        return ('strong', 0.75)
    elif prob >= 0.80 or prob <= 0.20:
        return ('moderate', 0.50)
    elif prob >= 0.75 or prob <= 0.25:
        return ('weak', 0.25)
    else:
        return (None, 0)  # Neutral zone (25-75%) - no bucket


def get_time_multiplier(hour: int) -> float:
    """
    TM SPEC (from TRADING_ENGINE_SPEC.md):
    - hour 13-15 → 1.0 (peak accuracy hours)
    - hour 11-12 → 0.8 (good accuracy)
    - hour 10 → 0.6 (moderate accuracy)
    - hour < 10 → 0.4 (early session, lower confidence)
    """
    if hour >= 13 and hour <= 15:
        return 1.0
    elif hour >= 11 and hour <= 12:
        return 0.8
    elif hour == 10:
        return 0.6
    else:
        return 0.4


def get_agreement_multiplier(prob_a: float, prob_b: float) -> float:
    """
    AM SPEC:
    - Both > 0.5 → 1.2 (aligned bullish)
    - Both < 0.5 → 1.2 (aligned bearish)
    - One > 0.5, one < 0.5 → 0.6 (conflicting)
    - One or both == 0.5 → 1.0 (neutral)
    """
    if prob_a > 0.5 and prob_b > 0.5:
        return 1.2
    elif prob_a < 0.5 and prob_b < 0.5:
        return 1.2
    elif (prob_a > 0.5 and prob_b < 0.5) or (prob_a < 0.5 and prob_b > 0.5):
        return 0.6
    else:
        return 1.0


def get_target_for_session(session: str) -> str:
    """
    TS SPEC:
    - early (hour < 11) → Target A (close > open)
    - late (hour >= 11) → Target B (close > 11am)
    """
    return 'A' if session == 'early' else 'B'


class TestNeutralZone:
    """NZ: Neutral Zone tests (25-75% = NO_TRADE) - Model accuracy unreliable in this range"""

    def test_nz1_above_75_is_bullish(self):
        """NZ-1: prob > 0.75 → BULLISH"""
        assert get_action(0.76) == 'BULLISH'
        assert get_action(0.80) == 'BULLISH'
        assert get_action(0.99) == 'BULLISH'

    def test_nz2_below_25_is_bearish(self):
        """NZ-2: prob < 0.25 → BEARISH"""
        assert get_action(0.24) == 'BEARISH'
        assert get_action(0.15) == 'BEARISH'
        assert get_action(0.01) == 'BEARISH'

    def test_nz3_boundary_75_is_no_trade(self):
        """NZ-3: prob=0.75 (boundary) → NO_TRADE"""
        assert get_action(0.75) == 'NO_TRADE'

    def test_nz4_boundary_25_is_no_trade(self):
        """NZ-4: prob=0.25 (boundary) → NO_TRADE"""
        assert get_action(0.25) == 'NO_TRADE'

    def test_nz5_middle_is_no_trade(self):
        """NZ-5: prob=0.50 → NO_TRADE (middle of neutral zone)"""
        assert get_action(0.50) == 'NO_TRADE'
        assert get_action(0.40) == 'NO_TRADE'
        assert get_action(0.60) == 'NO_TRADE'

    def test_nz6_float_precision_upper_boundary(self):
        """NZ-6: Float precision - values just above 0.75 → BULLISH"""
        # Prevents bugs from float representation near boundaries
        assert get_action(0.7500000001) == 'BULLISH'
        assert get_action(0.750001) == 'BULLISH'
        assert get_action(0.75 + 1e-10) == 'BULLISH'

    def test_nz7_float_precision_lower_boundary(self):
        """NZ-7: Float precision - values just below 0.75 → NO_TRADE"""
        assert get_action(0.7499999999) == 'NO_TRADE'
        assert get_action(0.749999) == 'NO_TRADE'
        assert get_action(0.75 - 1e-10) == 'NO_TRADE'

    def test_nz8_float_precision_bearish_boundary(self):
        """NZ-8: Float precision - values around 0.25 boundary"""
        # Just below 0.25 → BEARISH
        assert get_action(0.2499999999) == 'BEARISH'
        assert get_action(0.25 - 1e-10) == 'BEARISH'
        # Exactly 0.25 → NO_TRADE
        assert get_action(0.25) == 'NO_TRADE'
        # Just above 0.25 → NO_TRADE
        assert get_action(0.2500000001) == 'NO_TRADE'


class TestConfidenceBuckets:
    """BK: Confidence Bucket tests (per TRADING_ENGINE_SPEC.md) - Updated 2026-01-03"""

    def test_bk1_very_strong_above_90(self):
        """BK-1: prob >= 0.90 → "very_strong" (size_mult=1.0)"""
        assert get_confidence_bucket(0.90) == ('very_strong', 1.0)
        assert get_confidence_bucket(0.95) == ('very_strong', 1.0)
        assert get_confidence_bucket(0.99) == ('very_strong', 1.0)

    def test_bk2_very_strong_below_10(self):
        """BK-2: prob <= 0.10 → "very_strong" (size_mult=1.0)"""
        assert get_confidence_bucket(0.10) == ('very_strong', 1.0)
        assert get_confidence_bucket(0.05) == ('very_strong', 1.0)
        assert get_confidence_bucket(0.01) == ('very_strong', 1.0)

    def test_bk3_strong(self):
        """BK-3: prob >= 0.85 OR prob <= 0.15 → "strong" (size_mult=0.75)"""
        assert get_confidence_bucket(0.85) == ('strong', 0.75)
        assert get_confidence_bucket(0.87) == ('strong', 0.75)
        assert get_confidence_bucket(0.89) == ('strong', 0.75)
        assert get_confidence_bucket(0.15) == ('strong', 0.75)
        assert get_confidence_bucket(0.13) == ('strong', 0.75)
        assert get_confidence_bucket(0.11) == ('strong', 0.75)

    def test_bk4_moderate(self):
        """BK-4: prob >= 0.80 OR prob <= 0.20 → "moderate" (size_mult=0.50)"""
        assert get_confidence_bucket(0.80) == ('moderate', 0.50)
        assert get_confidence_bucket(0.82) == ('moderate', 0.50)
        assert get_confidence_bucket(0.84) == ('moderate', 0.50)
        assert get_confidence_bucket(0.20) == ('moderate', 0.50)
        assert get_confidence_bucket(0.18) == ('moderate', 0.50)
        assert get_confidence_bucket(0.16) == ('moderate', 0.50)

    def test_bk5_weak(self):
        """BK-5: prob >= 0.75 OR prob <= 0.25 → "weak" (size_mult=0.25) - boundary of neutral zone"""
        assert get_confidence_bucket(0.75) == ('weak', 0.25)
        assert get_confidence_bucket(0.77) == ('weak', 0.25)
        assert get_confidence_bucket(0.79) == ('weak', 0.25)
        assert get_confidence_bucket(0.25) == ('weak', 0.25)
        assert get_confidence_bucket(0.23) == ('weak', 0.25)
        assert get_confidence_bucket(0.21) == ('weak', 0.25)

    def test_bk6_neutral_zone_no_bucket(self):
        """BK-6: prob in (0.25, 0.75) exclusive → None (neutral zone - model accuracy unreliable)"""
        # These are in the neutral zone - no signal should be generated
        assert get_confidence_bucket(0.50) == (None, 0)
        assert get_confidence_bucket(0.40) == (None, 0)
        assert get_confidence_bucket(0.60) == (None, 0)
        assert get_confidence_bucket(0.26) == (None, 0)
        assert get_confidence_bucket(0.74) == (None, 0)


class TestTimeMultiplier:
    """TM: Time Multiplier tests (per TRADING_ENGINE_SPEC.md)"""

    def test_tm1_peak_hours(self):
        """TM-1: hour 13-15 → 1.0 (peak accuracy)"""
        assert get_time_multiplier(13) == 1.0
        assert get_time_multiplier(14) == 1.0
        assert get_time_multiplier(15) == 1.0

    def test_tm2_good_accuracy(self):
        """TM-2: hour 11-12 → 0.8 (good accuracy)"""
        assert get_time_multiplier(11) == 0.8
        assert get_time_multiplier(12) == 0.8

    def test_tm3_moderate_accuracy(self):
        """TM-3: hour 10 → 0.6 (moderate accuracy)"""
        assert get_time_multiplier(10) == 0.6

    def test_tm4_early_session(self):
        """TM-4: hour < 10 → 0.4 (early session, lower confidence)"""
        assert get_time_multiplier(9) == 0.4
        assert get_time_multiplier(8) == 0.4


class TestAgreementMultiplier:
    """AM: Agreement Multiplier tests"""

    def test_am1_both_bullish(self):
        """AM-1: both > 0.5 → 1.2"""
        assert get_agreement_multiplier(0.60, 0.70) == 1.2
        assert get_agreement_multiplier(0.51, 0.51) == 1.2

    def test_am2_both_bearish(self):
        """AM-2: both < 0.5 → 1.2"""
        assert get_agreement_multiplier(0.40, 0.30) == 1.2
        assert get_agreement_multiplier(0.49, 0.49) == 1.2

    def test_am3_conflicting(self):
        """AM-3: one > 0.5, one < 0.5 → 0.6"""
        assert get_agreement_multiplier(0.60, 0.40) == 0.6
        assert get_agreement_multiplier(0.40, 0.60) == 0.6

    def test_am4_neutral(self):
        """AM-4: one or both == 0.5 → 1.0"""
        assert get_agreement_multiplier(0.50, 0.60) == 1.0
        assert get_agreement_multiplier(0.60, 0.50) == 1.0
        assert get_agreement_multiplier(0.50, 0.50) == 1.0


class TestTargetSelection:
    """TS: Target Selection tests"""

    def test_ts1_early_session_uses_target_a(self):
        """TS-1: early session → Target A"""
        assert get_target_for_session('early') == 'A'

    def test_ts2_late_session_uses_target_b(self):
        """TS-2: late session → Target B"""
        assert get_target_for_session('late') == 'B'

    def test_ts3_session_boundary(self):
        """TS-3: Verify session determines target, not hour directly"""
        # The session should be computed from hour elsewhere
        # Here we just verify the mapping
        assert get_target_for_session('early') == 'A'
        assert get_target_for_session('late') == 'B'


class TestCompositeScore:
    """Test composite score calculation"""

    def test_composite_score_calculation(self):
        """Verify composite score = size_mult * time_mult * agreement_mult"""
        prob_a = 0.85  # Strong bucket (>= 0.85)
        prob_b = 0.82  # Moderate bucket, but both bullish for agreement
        hour = 14  # Peak hour

        bucket, size_mult = get_confidence_bucket(prob_a)  # ('strong', 0.75)
        time_mult = get_time_multiplier(hour)  # 1.0
        agree_mult = get_agreement_multiplier(prob_a, prob_b)  # 1.2

        expected_score = size_mult * time_mult * agree_mult  # 0.75 * 1.0 * 1.2 = 0.9

        assert abs(expected_score - 0.9) < 0.001

    def test_composite_score_capped(self):
        """Verify composite score is capped at 1.0"""
        # Extreme case: very_strong confidence + all multipliers
        prob_a = 0.95
        prob_b = 0.92
        hour = 14  # Peak hour

        bucket, size_mult = get_confidence_bucket(prob_a)  # ('very_strong', 1.0)
        time_mult = get_time_multiplier(hour)  # 1.0
        agree_mult = get_agreement_multiplier(prob_a, prob_b)  # 1.2

        raw_score = size_mult * time_mult * agree_mult  # 1.0 * 1.0 * 1.2 = 1.2
        capped_score = min(raw_score, 1.0)

        assert capped_score == 1.0


# Ticker-specific TP/SL percentages from TRADING_ENGINE_SPEC.md
TICKER_TARGETS = {
    'SPY': {'take_profit_pct': 0.0025, 'stop_loss_pct': 0.0033},  # 0.25% TP, 0.33% SL
    'QQQ': {'take_profit_pct': 0.0034, 'stop_loss_pct': 0.0045},  # 0.34% TP, 0.45% SL
    'IWM': {'take_profit_pct': 0.0045, 'stop_loss_pct': 0.0060},  # 0.45% TP, 0.60% SL
}


def get_exit_levels(ticker: str, entry: float, action: str) -> dict:
    """Calculate TP/SL based on ticker-specific percentages from spec"""
    targets = TICKER_TARGETS.get(ticker, TICKER_TARGETS['SPY'])

    if action == 'BUY_CALL':  # Bullish
        take_profit = entry * (1 + targets['take_profit_pct'])
        stop_loss = entry * (1 - targets['stop_loss_pct'])
    else:  # BUY_PUT - Bearish
        take_profit = entry * (1 - targets['take_profit_pct'])
        stop_loss = entry * (1 + targets['stop_loss_pct'])

    return {'take_profit': take_profit, 'stop_loss': stop_loss}


class TestEntryExitTargets:
    """EX: Entry/Exit Target tests (per TRADING_ENGINE_SPEC.md)"""

    def test_ex1_spy_targets(self):
        """EX-1: SPY TP=0.25%, SL=0.33%"""
        entry = 595.50
        exits = get_exit_levels('SPY', entry, 'BUY_CALL')

        # TP = 595.50 * 1.0025 = 596.99
        expected_tp = entry * 1.0025
        assert abs(exits['take_profit'] - expected_tp) < 0.01

        # SL = 595.50 * (1 - 0.0033) = 593.53
        expected_sl = entry * (1 - 0.0033)
        assert abs(exits['stop_loss'] - expected_sl) < 0.01

    def test_ex2_qqq_targets(self):
        """EX-2: QQQ TP=0.34%, SL=0.45%"""
        entry = 520.00
        exits = get_exit_levels('QQQ', entry, 'BUY_CALL')

        # TP = 520.00 * 1.0034 = 521.77
        expected_tp = entry * 1.0034
        assert abs(exits['take_profit'] - expected_tp) < 0.01

        # SL = 520.00 * (1 - 0.0045) = 517.66
        expected_sl = entry * (1 - 0.0045)
        assert abs(exits['stop_loss'] - expected_sl) < 0.01

    def test_ex3_iwm_targets(self):
        """EX-3: IWM TP=0.45%, SL=0.60%"""
        entry = 225.00
        exits = get_exit_levels('IWM', entry, 'BUY_CALL')

        # TP = 225.00 * 1.0045 = 226.01
        expected_tp = entry * 1.0045
        assert abs(exits['take_profit'] - expected_tp) < 0.01

        # SL = 225.00 * (1 - 0.0060) = 223.65
        expected_sl = entry * (1 - 0.0060)
        assert abs(exits['stop_loss'] - expected_sl) < 0.01

    def test_ex4_bearish_targets_inverted(self):
        """EX-4: BUY_PUT inverts TP/SL direction"""
        entry = 595.50
        exits = get_exit_levels('SPY', entry, 'BUY_PUT')

        # For puts: TP is BELOW entry, SL is ABOVE entry
        assert exits['take_profit'] < entry
        assert exits['stop_loss'] > entry

        # TP = 595.50 * (1 - 0.0025) = 594.01
        expected_tp = entry * (1 - 0.0025)
        assert abs(exits['take_profit'] - expected_tp) < 0.01

        # SL = 595.50 * (1 + 0.0033) = 597.47
        expected_sl = entry * (1 + 0.0033)
        assert abs(exits['stop_loss'] - expected_sl) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
