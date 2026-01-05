"""
Pytest Configuration and Shared Fixtures

Provides common fixtures for all tests.
"""

import pytest
import sys
from datetime import datetime
from unittest.mock import MagicMock
import pytz

# Add ml directory to path for imports
sys.path.insert(0, '/Users/it/Documents/mvp_coder_starter_kit (2)/mvp-trading-app/ml')

ET_TZ = pytz.timezone('America/New_York')


@pytest.fixture
def sample_daily_bars():
    """Sample daily bars for testing (3 days)"""
    return [
        {'o': 495.0, 'h': 498.0, 'l': 494.0, 'c': 497.0, 'v': 1000000, 't': 1704067200000},  # day -2
        {'o': 497.0, 'h': 500.0, 'l': 496.0, 'c': 499.0, 'v': 1100000, 't': 1704153600000},  # day -1
        {'o': 500.0, 'h': 503.0, 'l': 499.0, 'c': 502.0, 'v': 1200000, 't': 1704240000000},  # today
    ]


@pytest.fixture
def sample_hourly_bars():
    """Sample hourly bars for testing (4 hours)"""
    return [
        {'o': 500.0, 'h': 501.0, 'l': 499.0, 'c': 500.5, 'v': 100000, 't': 1704276600000},   # 9:30
        {'o': 500.5, 'h': 502.0, 'l': 500.0, 'c': 501.5, 'v': 110000, 't': 1704280200000},   # 10:30
        {'o': 501.5, 'h': 503.0, 'l': 501.0, 'c': 502.5, 'v': 120000, 't': 1704283800000},   # 11:30
        {'o': 502.5, 'h': 504.0, 'l': 502.0, 'c': 503.5, 'v': 130000, 't': 1704287400000},   # 12:30
    ]


@pytest.fixture
def mock_market_open():
    """Fixture for mocking market as open (Monday 10:30 AM ET)"""
    return ET_TZ.localize(datetime(2025, 1, 6, 10, 30, 0))


@pytest.fixture
def mock_market_closed():
    """Fixture for mocking market as closed (Saturday 10:00 AM ET)"""
    return ET_TZ.localize(datetime(2025, 1, 4, 10, 0, 0))


@pytest.fixture
def mock_v6_model():
    """Mock V6 model for testing"""
    model = MagicMock()
    model.predict_proba.return_value = [[0.35, 0.65]]  # 65% bullish
    return {
        'model_a': model,
        'model_b': model,
        'feature_cols': [
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
    }


@pytest.fixture
def trading_directions_response():
    """Sample /trading_directions response for testing"""
    return {
        'ticker': 'SPY',
        'action': 'BULLISH',
        'target_a_prob': 0.65,
        'target_b_prob': 0.60,
        'session': 'early',
        'confidence': 0.15,
        'confidence_bucket': 'medium',
        'generated_at': '2025-01-06T10:30:00-05:00',
        'entry': 500.0,
        'stop': 497.5,
        'target': 505.0,
        'time_multiplier': 1.1,
        'agreement_multiplier': 1.2,
    }
