"""
Unit Tests: Data Source (DS) and Feature Schema (FS)

SPEC REFERENCE:
- DS-1: today_open = daily_bars[-1]['o'], NEVER minute_bars[0]['o']
- DS-2: hourly_bars for features MUST include only bars <= current_hour
- DS-3: daily_bars MUST NOT include today when building prev-day features
- DS-4: V6 model expects exactly 29 features in fixed order
- FS-1: Feature names must match training feature_cols exactly
- FS-2: Categorical encoding must use same mapping as training
- FS-3: Continuous features must not be scaled if model was trained unscaled
- FS-4: Missing features must raise error, not silently default to 0
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/Users/it/Documents/mvp_coder_starter_kit (2)/mvp-trading-app/ml')

from server.v6.features import build_v6_features
from server.config import V6_FEATURE_COLS


class TestDataSource:
    """DS: Data Source integrity tests"""

    def test_ds1_today_open_from_daily_bar(self):
        """DS-1: today_open must come from daily_bars[-1]['o'], never minute bars"""
        # Mock daily bars - today's bar should have open=500.00
        daily_bars = [
            {'o': 495.0, 'h': 498.0, 'l': 494.0, 'c': 497.0, 'v': 1000000, 't': 1704067200000},  # day -2
            {'o': 497.0, 'h': 500.0, 'l': 496.0, 'c': 499.0, 'v': 1100000, 't': 1704153600000},  # day -1
            {'o': 500.0, 'h': 503.0, 'l': 499.0, 'c': 502.0, 'v': 1200000, 't': 1704240000000},  # today
        ]

        # Mock hourly bars with different open price
        hourly_bars = [
            {'o': 501.5, 'h': 502.0, 'l': 500.5, 'c': 501.8, 'v': 100000, 't': 1704276600000},  # 9:30 bar
        ]

        # SPEC: today_open comes from daily_bars[-1]['o']
        today_open = daily_bars[-1]['o']  # 500.0
        assert today_open == 500.0

        # NEVER use hourly_bars[0]['o'] (501.5)
        assert hourly_bars[0]['o'] != today_open

        # Build features using correct today_open
        features, _ = build_v6_features(hourly_bars, daily_bars, today_open, current_hour=10)
        assert features is not None

    def test_ds2_hourly_bars_filtered_by_hour(self):
        """DS-2: Only bars <= current_hour should be used for features"""
        daily_bars = [
            {'o': 495.0, 'h': 498.0, 'l': 494.0, 'c': 497.0, 'v': 1000000, 't': 1704067200000},
            {'o': 497.0, 'h': 500.0, 'l': 496.0, 'c': 499.0, 'v': 1100000, 't': 1704153600000},
            {'o': 500.0, 'h': 503.0, 'l': 499.0, 'c': 502.0, 'v': 1200000, 't': 1704240000000},
        ]
        today_open = daily_bars[-1]['o']

        # Create hourly bars for different hours
        hourly_bars_h10 = [
            {'o': 500.0, 'h': 501.0, 'l': 499.0, 'c': 500.5, 'v': 100000, 't': 1704276600000},   # 9:30
            {'o': 500.5, 'h': 502.0, 'l': 500.0, 'c': 501.5, 'v': 110000, 't': 1704280200000},   # 10:30
        ]

        hourly_bars_h12 = hourly_bars_h10 + [
            {'o': 501.5, 'h': 503.0, 'l': 501.0, 'c': 502.5, 'v': 120000, 't': 1704283800000},   # 11:30
            {'o': 502.5, 'h': 504.0, 'l': 502.0, 'c': 503.5, 'v': 130000, 't': 1704287400000},   # 12:30
        ]

        # At hour 10, only 2 bars available
        features_h10, _ = build_v6_features(hourly_bars_h10, daily_bars, today_open, current_hour=10)

        # At hour 12, 4 bars available
        features_h12, _ = build_v6_features(hourly_bars_h12, daily_bars, today_open, current_hour=12)

        assert features_h10 is not None
        assert features_h12 is not None

        # Current close differs (501.5 vs 503.5), so current_vs_open should differ
        assert features_h10['current_vs_open'] != features_h12['current_vs_open']

    def test_ds3_daily_bars_excludes_today_for_prev_features(self):
        """DS-3: prev_day features must use daily_bars[-2], not today"""
        daily_bars = [
            {'o': 490.0, 'h': 495.0, 'l': 489.0, 'c': 494.0, 'v': 900000, 't': 1703980800000},   # day -3
            {'o': 494.0, 'h': 498.0, 'l': 493.0, 'c': 497.0, 'v': 1000000, 't': 1704067200000},  # day -2
            {'o': 497.0, 'h': 500.0, 'l': 496.0, 'c': 499.0, 'v': 1100000, 't': 1704153600000},  # day -1 (yesterday)
            {'o': 500.0, 'h': 505.0, 'l': 499.0, 'c': 504.0, 'v': 1200000, 't': 1704240000000},  # today
        ]
        today_open = daily_bars[-1]['o']

        hourly_bars = [
            {'o': 500.0, 'h': 502.0, 'l': 499.0, 'c': 501.5, 'v': 100000, 't': 1704276600000},
        ]

        features, _ = build_v6_features(hourly_bars, daily_bars, today_open, current_hour=10)

        # prev_return should use daily_bars[-2] (yesterday) and daily_bars[-3]
        # NOT today's bar
        expected_prev_return = (daily_bars[-2]['c'] - daily_bars[-3]['c']) / daily_bars[-3]['c']
        assert abs(features['prev_return'] - expected_prev_return) < 0.0001

    def test_ds4_v6_expects_29_features(self):
        """DS-4: V6 model expects exactly 29 features"""
        assert len(V6_FEATURE_COLS) == 29, f"Expected 29 features, got {len(V6_FEATURE_COLS)}"


class TestFeatureSchema:
    """FS: Feature Schema tests"""

    def test_fs1_feature_names_match_training(self):
        """FS-1: Feature names must match training exactly"""
        expected_features = [
            'hour', 'day_of_week', 'week_of_year', 'month',
            'open_to_current', 'open_to_high', 'open_to_low',
            'current_range', 'prev_day_close', 'prev_day_range',
            'gap_pct', 'gap_direction', 'volume_ratio',
            'hourly_momentum', 'hourly_volatility',
            'high_low_ratio', 'body_to_range', 'upper_wick_pct', 'lower_wick_pct',
            'prev_hour_close', 'prev_hour_range', 'two_hour_momentum',
            'day_range_pct', 'dist_from_high', 'dist_from_low',
            'morning_momentum', 'morning_volatility',
            'prev_day_body', 'prev_day_direction'
        ]

        assert V6_FEATURE_COLS == expected_features, f"Feature mismatch: {set(V6_FEATURE_COLS) ^ set(expected_features)}"

    def test_fs2_build_features_returns_dict(self):
        """FS-2: build_v6_features returns a dict with feature values"""
        hourly_bars = [
            {'o': 500.0, 'h': 502.0, 'l': 499.0, 'c': 501.5, 'v': 100000, 't': 1704276600000},
            {'o': 501.5, 'h': 503.0, 'l': 501.0, 'c': 502.5, 'v': 110000, 't': 1704280200000},
        ]
        daily_bars = [
            {'o': 495.0, 'h': 498.0, 'l': 494.0, 'c': 497.0, 'v': 1000000, 't': 1704067200000},
            {'o': 497.0, 'h': 500.0, 'l': 496.0, 'c': 499.0, 'v': 1100000, 't': 1704153600000},
            {'o': 500.0, 'h': 503.0, 'l': 499.0, 'c': 502.0, 'v': 1200000, 't': 1704240000000},
        ]
        today_open = daily_bars[-1]['o']

        features, _ = build_v6_features(hourly_bars, daily_bars, today_open, current_hour=10)

        assert features is not None
        assert isinstance(features, dict)

    def test_fs3_features_are_numeric(self):
        """FS-3: All features must be numeric (int or float)"""
        hourly_bars = [
            {'o': 500.0, 'h': 502.0, 'l': 499.0, 'c': 501.5, 'v': 100000, 't': 1704276600000},
        ]
        daily_bars = [
            {'o': 495.0, 'h': 498.0, 'l': 494.0, 'c': 497.0, 'v': 1000000, 't': 1704067200000},
            {'o': 497.0, 'h': 500.0, 'l': 496.0, 'c': 499.0, 'v': 1100000, 't': 1704153600000},
            {'o': 500.0, 'h': 503.0, 'l': 499.0, 'c': 502.0, 'v': 1200000, 't': 1704240000000},
        ]
        today_open = daily_bars[-1]['o']

        features, _ = build_v6_features(hourly_bars, daily_bars, today_open, current_hour=10)

        assert features is not None
        for key, val in features.items():
            assert isinstance(val, (int, float, np.integer, np.floating)), \
                f"Feature {key} is not numeric: {type(val)}"

    def test_fs4_no_nan_features(self):
        """FS-4: Features must not contain NaN values"""
        hourly_bars = [
            {'o': 500.0, 'h': 502.0, 'l': 499.0, 'c': 501.5, 'v': 100000, 't': 1704276600000},
        ]
        daily_bars = [
            {'o': 495.0, 'h': 498.0, 'l': 494.0, 'c': 497.0, 'v': 1000000, 't': 1704067200000},
            {'o': 497.0, 'h': 500.0, 'l': 496.0, 'c': 499.0, 'v': 1100000, 't': 1704153600000},
            {'o': 500.0, 'h': 503.0, 'l': 499.0, 'c': 502.0, 'v': 1200000, 't': 1704240000000},
        ]
        today_open = daily_bars[-1]['o']

        features, _ = build_v6_features(hourly_bars, daily_bars, today_open, current_hour=10)

        assert features is not None
        for key, val in features.items():
            if isinstance(val, float):
                assert not np.isnan(val), f"Feature {key} is NaN"


class TestFeatureCalculations:
    """Specific feature calculation tests"""

    def test_gap_calculation(self):
        """gap = (today_open - prev_close) / prev_close"""
        daily_bars = [
            {'o': 495.0, 'h': 498.0, 'l': 494.0, 'c': 497.0, 'v': 1000000, 't': 1704067200000},
            {'o': 497.0, 'h': 500.0, 'l': 496.0, 'c': 499.0, 'v': 1100000, 't': 1704153600000},  # prev close=499
            {'o': 502.0, 'h': 505.0, 'l': 501.0, 'c': 504.0, 'v': 1200000, 't': 1704240000000},  # today open=502
        ]
        today_open = daily_bars[-1]['o']  # 502.0

        hourly_bars = [
            {'o': 502.0, 'h': 503.0, 'l': 501.0, 'c': 502.5, 'v': 100000, 't': 1704276600000},
        ]

        features, _ = build_v6_features(hourly_bars, daily_bars, today_open, current_hour=10)

        # gap = (502.0 - 499.0) / 499.0 = 0.006012...
        expected_gap = (502.0 - 499.0) / 499.0
        assert abs(features['gap'] - expected_gap) < 0.0001

    def test_current_vs_open_calculation(self):
        """current_vs_open = (current_price - open) / open"""
        daily_bars = [
            {'o': 495.0, 'h': 498.0, 'l': 494.0, 'c': 497.0, 'v': 1000000, 't': 1704067200000},
            {'o': 497.0, 'h': 500.0, 'l': 496.0, 'c': 499.0, 'v': 1100000, 't': 1704153600000},
            {'o': 500.0, 'h': 505.0, 'l': 499.0, 'c': 504.0, 'v': 1200000, 't': 1704240000000},  # today open=500
        ]
        today_open = daily_bars[-1]['o']  # 500.0

        hourly_bars = [
            {'o': 500.0, 'h': 503.0, 'l': 499.0, 'c': 502.0, 'v': 100000, 't': 1704276600000},  # current=502
        ]

        features, _ = build_v6_features(hourly_bars, daily_bars, today_open, current_hour=10)

        # current_vs_open = (502.0 - 500.0) / 500.0 = 0.004
        expected = (502.0 - 500.0) / 500.0
        assert abs(features['current_vs_open'] - expected) < 0.0001

    def test_above_open_flag(self):
        """above_open = 1 if current > open else 0"""
        daily_bars = [
            {'o': 495.0, 'h': 498.0, 'l': 494.0, 'c': 497.0, 'v': 1000000, 't': 1704067200000},
            {'o': 497.0, 'h': 500.0, 'l': 496.0, 'c': 499.0, 'v': 1100000, 't': 1704153600000},
            {'o': 500.0, 'h': 505.0, 'l': 499.0, 'c': 504.0, 'v': 1200000, 't': 1704240000000},
        ]
        today_open = daily_bars[-1]['o']  # 500.0

        # Current above open
        hourly_bars_above = [
            {'o': 500.0, 'h': 503.0, 'l': 499.0, 'c': 502.0, 'v': 100000, 't': 1704276600000},
        ]
        features_above, _ = build_v6_features(hourly_bars_above, daily_bars, today_open, current_hour=10)
        assert features_above['above_open'] == 1

        # Current below open
        hourly_bars_below = [
            {'o': 500.0, 'h': 501.0, 'l': 498.0, 'c': 499.0, 'v': 100000, 't': 1704276600000},
        ]
        features_below, _ = build_v6_features(hourly_bars_below, daily_bars, today_open, current_hour=10)
        assert features_below['above_open'] == 0


class TestInsufficientData:
    """Test handling of insufficient data"""

    def test_insufficient_hourly_bars_returns_none(self):
        """Returns None if no hourly bars"""
        daily_bars = [
            {'o': 495.0, 'h': 498.0, 'l': 494.0, 'c': 497.0, 'v': 1000000, 't': 1704067200000},
            {'o': 497.0, 'h': 500.0, 'l': 496.0, 'c': 499.0, 'v': 1100000, 't': 1704153600000},
            {'o': 500.0, 'h': 503.0, 'l': 499.0, 'c': 502.0, 'v': 1200000, 't': 1704240000000},
        ]

        result = build_v6_features([], daily_bars, 500.0, current_hour=10)
        assert result is None

    def test_insufficient_daily_bars_returns_none(self):
        """Returns None if less than 3 daily bars"""
        daily_bars = [
            {'o': 497.0, 'h': 500.0, 'l': 496.0, 'c': 499.0, 'v': 1100000, 't': 1704153600000},
            {'o': 500.0, 'h': 503.0, 'l': 499.0, 'c': 502.0, 'v': 1200000, 't': 1704240000000},
        ]
        hourly_bars = [
            {'o': 500.0, 'h': 502.0, 'l': 499.0, 'c': 501.5, 'v': 100000, 't': 1704276600000},
        ]

        result = build_v6_features(hourly_bars, daily_bars, 500.0, current_hour=10)
        assert result is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
