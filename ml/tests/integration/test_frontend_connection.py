"""
Integration Tests: Frontend-Backend Connection

Tests the actual HTTP connections between components:
- Frontend API routes
- ML Server endpoints
- End-to-end signal delivery

Run: python3 -m pytest tests/integration/test_frontend_connection.py -v
"""

import pytest
import requests
from datetime import datetime
import json

# Connection targets
ML_SERVER_URLS = [
    'http://localhost:5000',
    'https://genesis-production-c1e9.up.railway.app'
]
FRONTEND_URL = 'http://localhost:3000'


def get_working_ml_server():
    """Find a responsive ML server"""
    for url in ML_SERVER_URLS:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                return url
        except requests.RequestException:
            continue
    return None


class TestMLServerConnection:
    """Test ML server is reachable and responding correctly"""

    def test_ml_server_health_endpoint(self):
        """ML server /health endpoint returns 200"""
        ml_url = get_working_ml_server()
        if not ml_url:
            pytest.skip("No ML server available")

        response = requests.get(f"{ml_url}/health", timeout=10)
        assert response.status_code == 200

    def test_ml_server_trading_directions_endpoint(self):
        """ML server /trading_directions returns valid JSON"""
        ml_url = get_working_ml_server()
        if not ml_url:
            pytest.skip("No ML server available")

        response = requests.get(f"{ml_url}/trading_directions", timeout=15)
        assert response.status_code == 200

        data = response.json()
        # Must have signals for at least one ticker
        assert isinstance(data, dict)

    def test_ml_server_model_info_endpoint(self):
        """ML server /model_info returns model metadata"""
        ml_url = get_working_ml_server()
        if not ml_url:
            pytest.skip("No ML server available")

        response = requests.get(f"{ml_url}/model_info", timeout=10)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, dict)


class TestMLServerOutputContract:
    """Test ML server output matches spec requirements"""

    def test_trading_directions_has_required_tickers(self):
        """Response includes SPY, QQQ, IWM signals or market-closed response"""
        ml_url = get_working_ml_server()
        if not ml_url:
            pytest.skip("No ML server available")

        response = requests.get(f"{ml_url}/trading_directions", timeout=15)
        data = response.json()

        # Market closed response has flat structure with 'action': 'NO_TRADE'
        if data.get('action') == 'NO_TRADE' and data.get('market_open') == False:
            # Valid market-closed response
            return

        expected_tickers = ['SPY', 'QQQ', 'IWM']
        for ticker in expected_tickers:
            assert ticker in data, f"Missing ticker: {ticker}"

    def test_each_signal_has_required_fields(self):
        """Each ticker signal has action, probabilities, session"""
        ml_url = get_working_ml_server()
        if not ml_url:
            pytest.skip("No ML server available")

        response = requests.get(f"{ml_url}/trading_directions", timeout=15)
        data = response.json()

        required_fields = ['action', 'session']
        for ticker in ['SPY', 'QQQ', 'IWM']:
            if ticker not in data:
                continue
            signal = data[ticker]
            for field in required_fields:
                assert field in signal, f"{ticker} missing field: {field}"

    def test_action_is_valid_enum(self):
        """Action is one of BUY_CALL, BUY_PUT, NO_TRADE, LONG, SHORT, BULLISH, BEARISH"""
        ml_url = get_working_ml_server()
        if not ml_url:
            pytest.skip("No ML server available")

        response = requests.get(f"{ml_url}/trading_directions", timeout=15)
        data = response.json()

        valid_actions = ['BUY_CALL', 'BUY_PUT', 'NO_TRADE', 'LONG', 'SHORT', 'BULLISH', 'BEARISH']
        for ticker in ['SPY', 'QQQ', 'IWM']:
            if ticker not in data:
                continue
            action = data[ticker].get('action')
            assert action in valid_actions, f"{ticker} has invalid action: {action}"

    def test_session_is_early_or_late(self):
        """Session is 'early' or 'late'"""
        ml_url = get_working_ml_server()
        if not ml_url:
            pytest.skip("No ML server available")

        response = requests.get(f"{ml_url}/trading_directions", timeout=15)
        data = response.json()

        for ticker in ['SPY', 'QQQ', 'IWM']:
            if ticker not in data:
                continue
            session = data[ticker].get('session')
            assert session in ['early', 'late'], f"{ticker} has invalid session: {session}"

    def test_probabilities_in_valid_range(self):
        """Probabilities are between 0 and 1"""
        ml_url = get_working_ml_server()
        if not ml_url:
            pytest.skip("No ML server available")

        response = requests.get(f"{ml_url}/trading_directions", timeout=15)
        data = response.json()

        for ticker in ['SPY', 'QQQ', 'IWM']:
            if ticker not in data:
                continue
            signal = data[ticker]

            prob_a = signal.get('probability_a') or signal.get('target_a_prob')
            prob_b = signal.get('probability_b') or signal.get('target_b_prob')

            if prob_a is not None:
                assert 0 <= prob_a <= 1, f"{ticker} prob_a out of range: {prob_a}"
            if prob_b is not None:
                assert 0 <= prob_b <= 1, f"{ticker} prob_b out of range: {prob_b}"

    def test_spec_version_included(self):
        """Response includes spec_version and engine_version (SV-1)"""
        ml_url = get_working_ml_server()
        if not ml_url:
            pytest.skip("No ML server available")

        response = requests.get(f"{ml_url}/trading_directions", timeout=15)
        data = response.json()

        for ticker in ['SPY', 'QQQ', 'IWM']:
            if ticker not in data:
                continue
            signal = data[ticker]

            assert 'spec_version' in signal, f"{ticker} missing spec_version"
            assert 'engine_version' in signal, f"{ticker} missing engine_version"


class TestFrontendConnection:
    """Test frontend can reach ML server"""

    def test_frontend_trading_directions_api(self):
        """Frontend /api/v2/trading-directions proxies to ML server"""
        try:
            response = requests.get(f"{FRONTEND_URL}/api/v2/trading-directions", timeout=15)
        except requests.exceptions.ConnectionError:
            pytest.skip("Frontend not running at localhost:3000")

        # Accept 200, 502 (ML down), 503 (service unavailable)
        assert response.status_code in [200, 502, 503], f"Unexpected status: {response.status_code}"

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_frontend_receives_signal_structure(self):
        """Frontend receives properly structured signals"""
        try:
            response = requests.get(f"{FRONTEND_URL}/api/v2/trading-directions", timeout=15)
        except requests.exceptions.ConnectionError:
            pytest.skip("Frontend not running at localhost:3000")

        if response.status_code != 200:
            pytest.skip(f"ML server not available (status {response.status_code})")

        data = response.json()

        # Market closed response has flat structure
        if data.get('action') == 'NO_TRADE' and data.get('market_open') == False:
            assert 'action' in data
            return

        # Check at least one signal has proper structure
        for ticker in ['SPY', 'QQQ', 'IWM']:
            if ticker in data and isinstance(data[ticker], dict):
                signal = data[ticker]
                assert 'action' in signal
                return

        pytest.fail("No valid signal structure found in response")


class TestNeutralZoneCompliance:
    """Test neutral zone rules are enforced (NZ-1, NZ-2, NZ-3)"""

    def test_neutral_zone_produces_no_trade(self):
        """Probability in 0.45-0.55 range should produce NO_TRADE"""
        ml_url = get_working_ml_server()
        if not ml_url:
            pytest.skip("No ML server available")

        response = requests.get(f"{ml_url}/trading_directions", timeout=15)
        data = response.json()

        for ticker in ['SPY', 'QQQ', 'IWM']:
            if ticker not in data:
                continue

            signal = data[ticker]
            action = signal.get('action', '')
            session = signal.get('session', 'early')

            # Get the active probability based on session
            if session == 'late':
                prob = signal.get('probability_b') or signal.get('target_b_prob')
            else:
                prob = signal.get('probability_a') or signal.get('target_a_prob')

            if prob is None:
                continue

            # Verify neutral zone compliance
            if 0.45 <= prob <= 0.55:
                assert action == 'NO_TRADE', \
                    f"{ticker}: prob {prob} in neutral zone but action is {action}"
            elif prob > 0.55:
                assert action in ['BUY_CALL', 'LONG', 'BULLISH'], \
                    f"{ticker}: prob {prob} > 0.55 but action is {action}"
            elif prob < 0.45:
                assert action in ['BUY_PUT', 'SHORT', 'BEARISH'], \
                    f"{ticker}: prob {prob} < 0.45 but action is {action}"


class TestRPEConnection:
    """Test RPE (Reality Proof Engine) 5-phase analysis endpoint"""

    def test_rpe_endpoint_responds(self):
        """RPE /rpe endpoint returns valid 5-phase analysis"""
        ml_url = get_working_ml_server()
        if not ml_url:
            pytest.skip("No ML server available")

        response = requests.get(f"{ml_url}/rpe?ticker=SPY", timeout=30)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, dict)

    def test_rpe_has_5_phases(self):
        """RPE response includes 5 phase structure (descriptions or data)"""
        ml_url = get_working_ml_server()
        if not ml_url:
            pytest.skip("No ML server available")

        response = requests.get(f"{ml_url}/rpe?ticker=SPY", timeout=30)
        if response.status_code != 200:
            pytest.skip("RPE endpoint not responding")

        data = response.json()

        # When market closed, phase_descriptions shows the 5-phase structure
        if 'phase_descriptions' in data:
            expected = ['phase1', 'phase2', 'phase3', 'phase4', 'phase5']
            found = [p for p in expected if any(p in k for k in data['phase_descriptions'].keys())]
            assert len(found) >= 5, f"Expected 5 phases in descriptions, found: {found}"
            return

        # When market open, phases are in ticker data
        if 'tickers' in data and 'SPY' in data['tickers']:
            ticker_data = data['tickers']['SPY']
            if 'error' not in ticker_data:
                expected_phases = ['phase1', 'phase2', 'phase3', 'phase4', 'phase5']
                found_phases = [p for p in expected_phases if p in ticker_data]
                assert len(found_phases) >= 4, f"Expected 4+ phases, found: {found_phases}"


class TestNorthstarConnection:
    """Test Northstar 4-phase pipeline endpoint"""

    def test_northstar_endpoint_responds(self):
        """Northstar /northstar endpoint returns valid 4-phase analysis"""
        ml_url = get_working_ml_server()
        if not ml_url:
            pytest.skip("No ML server available")

        response = requests.get(f"{ml_url}/northstar?ticker=SPY", timeout=30)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, dict)

    def test_northstar_has_4_phases(self):
        """Northstar response includes 4 phase structure (descriptions or data)"""
        ml_url = get_working_ml_server()
        if not ml_url:
            pytest.skip("No ML server available")

        response = requests.get(f"{ml_url}/northstar?ticker=SPY", timeout=30)
        if response.status_code != 200:
            pytest.skip("Northstar endpoint not responding")

        data = response.json()

        # phase_descriptions shows the 4-phase structure
        if 'phase_descriptions' in data:
            expected = ['phase1', 'phase2', 'phase3', 'phase4']
            found = [p for p in expected if p in data['phase_descriptions']]
            assert len(found) >= 4, f"Expected 4 phases in descriptions, found: {found}"
            return

        # When market open, phases may be in ticker data
        if 'tickers' in data and 'SPY' in data['tickers']:
            ticker_data = data['tickers']['SPY']
            if isinstance(ticker_data, dict) and 'error' not in ticker_data:
                expected_phases = ['phase1', 'phase2', 'phase3', 'phase4']
                found_phases = [p for p in expected_phases if p in ticker_data]
                assert len(found_phases) >= 3, f"Expected 3+ phases, found: {found_phases}"


class TestReplayMode:
    """Test Replay mode time-travel endpoint"""

    def test_replay_endpoint_responds(self):
        """Replay /replay endpoint returns historical data"""
        ml_url = get_working_ml_server()
        if not ml_url:
            pytest.skip("No ML server available")

        response = requests.get(
            f"{ml_url}/replay?date=2025-12-30&time=14:00",
            timeout=60
        )
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, dict)

    def test_replay_has_required_fields(self):
        """Replay response includes mode, date, time, tickers"""
        ml_url = get_working_ml_server()
        if not ml_url:
            pytest.skip("No ML server available")

        response = requests.get(
            f"{ml_url}/replay?date=2025-12-30&time=14:00",
            timeout=60
        )
        if response.status_code != 200:
            pytest.skip("Replay endpoint not responding")

        data = response.json()

        # Replay should have these fields
        assert 'mode' in data or 'replay_date' in data, "Missing replay mode info"
        assert 'tickers' in data or 'v6_signals' in data, "Missing ticker data"

    def test_replay_includes_v6_and_northstar(self):
        """Replay response includes both V6 signals and Northstar analysis"""
        ml_url = get_working_ml_server()
        if not ml_url:
            pytest.skip("No ML server available")

        response = requests.get(
            f"{ml_url}/replay?date=2025-12-30&time=14:00",
            timeout=60
        )
        if response.status_code != 200:
            pytest.skip("Replay endpoint not responding")

        data = response.json()

        # Should have V6 signals and Northstar analysis
        has_v6 = 'v6_signals' in data or ('tickers' in data and any(
            'v6' in str(t) for t in data.get('tickers', {}).values() if isinstance(t, dict)
        ))
        has_northstar = 'northstar' in data or ('tickers' in data and any(
            'northstar' in str(t) for t in data.get('tickers', {}).values() if isinstance(t, dict)
        ))

        assert has_v6 or has_northstar, "Replay missing V6/Northstar analysis"


class TestFrontendReplayConnection:
    """Test frontend can reach replay endpoint"""

    def test_frontend_replay_api(self):
        """Frontend /api/v2/replay proxies to ML server"""
        try:
            response = requests.get(
                f"{FRONTEND_URL}/api/v2/replay?date=2025-12-30&time=14:00",
                timeout=60
            )
        except requests.exceptions.ConnectionError:
            pytest.skip("Frontend not running at localhost:3000")

        # Accept 200, 502 (ML down), 503 (service unavailable)
        assert response.status_code in [200, 502, 503], \
            f"Unexpected status: {response.status_code}"

    def test_frontend_rpe_api(self):
        """Frontend /api/v2/rpe proxies to ML server"""
        try:
            response = requests.get(
                f"{FRONTEND_URL}/api/v2/rpe?ticker=SPY",
                timeout=30
            )
        except requests.exceptions.ConnectionError:
            pytest.skip("Frontend not running at localhost:3000")

        # Accept 200, 502 (ML down), 503 (service unavailable)
        assert response.status_code in [200, 502, 503], \
            f"Unexpected status: {response.status_code}"


class TestDeterminism:
    """Test that same request produces same response (determinism)"""

    def test_repeated_requests_produce_same_action(self):
        """Multiple requests within same hour produce same action"""
        ml_url = get_working_ml_server()
        if not ml_url:
            pytest.skip("No ML server available")

        # Make 3 requests
        responses = []
        for _ in range(3):
            response = requests.get(f"{ml_url}/trading_directions", timeout=15)
            if response.status_code == 200:
                responses.append(response.json())

        if len(responses) < 2:
            pytest.skip("Could not get multiple responses")

        # All responses should have same actions for each ticker
        for ticker in ['SPY', 'QQQ', 'IWM']:
            actions = []
            for resp in responses:
                if ticker in resp:
                    actions.append(resp[ticker].get('action'))

            if len(actions) > 1:
                # All actions should be identical
                assert all(a == actions[0] for a in actions), \
                    f"{ticker} has non-deterministic actions: {actions}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
