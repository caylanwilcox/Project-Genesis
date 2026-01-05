"""
Unit Tests: Session Classification (SC) and Market Hours (MH)

SPEC REFERENCE:
- MH-1: 09:30:00 ET = OPEN
- MH-2: 15:59:59 ET = OPEN
- MH-3: 16:00:00 ET = CLOSED
- MH-4: Weekend = CLOSED
- SC-1: hour < 11 → "early"
- SC-2: hour >= 11 → "late"
- SC-3: session boundaries are 11:00:00 sharp
"""

import pytest
from datetime import datetime
from unittest.mock import patch
import pytz

# Import from server module
import sys
sys.path.insert(0, '/Users/it/Documents/mvp_coder_starter_kit (2)/mvp-trading-app/ml')

from server.data.market import (
    is_market_open,
    get_session,
    get_current_hour,
    get_session_progress,
    ET_TZ
)


class TestMarketHours:
    """MH: Market Hours boundary tests"""

    @pytest.fixture
    def mock_time(self):
        """Fixture to mock get_now_et()"""
        def _mock_time(year=2025, month=1, day=6, hour=10, minute=0, second=0):
            # January 6, 2025 is a Monday
            return ET_TZ.localize(datetime(year, month, day, hour, minute, second))
        return _mock_time

    def test_mh1_market_open_at_930(self, mock_time):
        """MH-1: 09:30:00 ET should be OPEN"""
        with patch('server.data.market.get_now_et', return_value=mock_time(hour=9, minute=30, second=0)):
            assert is_market_open() is True

    def test_mh2_market_open_at_1559(self, mock_time):
        """MH-2: 15:59:59 ET should be OPEN"""
        with patch('server.data.market.get_now_et', return_value=mock_time(hour=15, minute=59, second=59)):
            assert is_market_open() is True

    def test_mh3_market_closed_at_1600(self, mock_time):
        """MH-3: 16:00:00 ET should be CLOSED (critical boundary)"""
        with patch('server.data.market.get_now_et', return_value=mock_time(hour=16, minute=0, second=0)):
            assert is_market_open() is False

    def test_mh4_market_closed_on_saturday(self, mock_time):
        """MH-4: Weekend (Saturday) should be CLOSED"""
        # January 4, 2025 is a Saturday
        saturday = ET_TZ.localize(datetime(2025, 1, 4, 10, 0, 0))
        with patch('server.data.market.get_now_et', return_value=saturday):
            assert is_market_open() is False

    def test_mh4_market_closed_on_sunday(self, mock_time):
        """MH-4: Weekend (Sunday) should be CLOSED"""
        # January 5, 2025 is a Sunday
        sunday = ET_TZ.localize(datetime(2025, 1, 5, 10, 0, 0))
        with patch('server.data.market.get_now_et', return_value=sunday):
            assert is_market_open() is False

    def test_market_closed_before_930(self, mock_time):
        """Pre-market: 09:29:59 ET should be CLOSED"""
        with patch('server.data.market.get_now_et', return_value=mock_time(hour=9, minute=29, second=59)):
            assert is_market_open() is False

    def test_market_closed_after_1600(self, mock_time):
        """After-hours: 16:01:00 ET should be CLOSED"""
        with patch('server.data.market.get_now_et', return_value=mock_time(hour=16, minute=1, second=0)):
            assert is_market_open() is False


class TestSessionClassification:
    """SC: Session Classification tests"""

    @pytest.fixture
    def mock_time(self):
        """Fixture to mock get_now_et()"""
        def _mock_time(hour):
            return ET_TZ.localize(datetime(2025, 1, 6, hour, 0, 0))
        return _mock_time

    def test_sc1_early_session_hour_10(self, mock_time):
        """SC-1: hour=10 → "early" """
        with patch('server.data.market.get_now_et', return_value=mock_time(10)):
            assert get_session() == 'early'

    def test_sc1_early_session_hour_9(self, mock_time):
        """SC-1: hour=9 → "early" """
        with patch('server.data.market.get_now_et', return_value=mock_time(9)):
            assert get_session() == 'early'

    def test_sc2_late_session_hour_11(self, mock_time):
        """SC-2: hour=11 → "late" (boundary)"""
        with patch('server.data.market.get_now_et', return_value=mock_time(11)):
            assert get_session() == 'late'

    def test_sc2_late_session_hour_14(self, mock_time):
        """SC-2: hour=14 → "late" """
        with patch('server.data.market.get_now_et', return_value=mock_time(14)):
            assert get_session() == 'late'

    def test_sc3_boundary_at_1100(self):
        """SC-3: Session boundary is at 11:00:00 sharp"""
        # 10:59:59 should be early
        time_1059 = ET_TZ.localize(datetime(2025, 1, 6, 10, 59, 59))
        with patch('server.data.market.get_now_et', return_value=time_1059):
            # get_session uses hour, so 10:59:59 has hour=10 → early
            assert get_session() == 'early'

        # 11:00:00 should be late
        time_1100 = ET_TZ.localize(datetime(2025, 1, 6, 11, 0, 0))
        with patch('server.data.market.get_now_et', return_value=time_1100):
            assert get_session() == 'late'


class TestSessionProgress:
    """Session progress calculation tests"""

    def test_progress_at_open(self):
        """Progress should be 0.0 at market open"""
        time_930 = ET_TZ.localize(datetime(2025, 1, 6, 9, 30, 0))
        with patch('server.data.market.get_now_et', return_value=time_930):
            assert get_session_progress() == 0.0

    def test_progress_at_close(self):
        """Progress should be 1.0 at or after market close"""
        time_1600 = ET_TZ.localize(datetime(2025, 1, 6, 16, 0, 0))
        with patch('server.data.market.get_now_et', return_value=time_1600):
            assert get_session_progress() == 1.0

    def test_progress_midday(self):
        """Progress should be ~0.5 at midday (12:45 PM)"""
        # 9:30 to 4:00 = 6.5 hours = 390 minutes
        # 12:45 = 3h15m from open = 195 minutes
        # 195/390 = 0.5
        time_1245 = ET_TZ.localize(datetime(2025, 1, 6, 12, 45, 0))
        with patch('server.data.market.get_now_et', return_value=time_1245):
            progress = get_session_progress()
            assert 0.49 <= progress <= 0.51  # Allow small float variance


class TestCurrentHour:
    """Current hour utility tests"""

    def test_current_hour_returns_et(self):
        """get_current_hour should return hour in ET timezone"""
        time_1400 = ET_TZ.localize(datetime(2025, 1, 6, 14, 30, 0))
        with patch('server.data.market.get_now_et', return_value=time_1400):
            assert get_current_hour() == 14


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
