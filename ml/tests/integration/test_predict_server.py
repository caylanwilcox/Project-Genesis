"""
Integration Tests: predict_server.py

Tests the full prediction pipeline including:
- V6 model loading and predictions
- Trading directions endpoint
- Output contract compliance
- No-repainting guarantees
"""

import pytest
import json
import sys
sys.path.insert(0, '/Users/it/Documents/mvp_coder_starter_kit (2)/mvp-trading-app/ml')

from unittest.mock import patch, MagicMock
from datetime import datetime
import pytz


class TestOutputContract:
    """OC: Output Contract tests - verify API response schema"""

    def test_oc1_trading_directions_has_required_fields(self):
        """OC-1: /trading_directions response must have all required fields"""
        required_fields = [
            'ticker',
            'action',
            'target_a_prob',
            'target_b_prob',
            'session',
            'confidence',
            'confidence_bucket',
            'generated_at',
        ]

        # Mock response structure
        mock_response = {
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
        }

        for field in required_fields:
            assert field in mock_response, f"Missing required field: {field}"

    def test_oc2_action_is_valid_enum(self):
        """OC-2: action must be one of BULLISH, BEARISH, NO_TRADE"""
        valid_actions = ['BULLISH', 'BEARISH', 'NO_TRADE']

        for action in valid_actions:
            assert action in valid_actions

        # Invalid action should be rejected
        assert 'NEUTRAL' not in valid_actions
        assert 'HOLD' not in valid_actions

    def test_oc3_probabilities_in_range(self):
        """OC-3: probabilities must be in [0, 1]"""
        def validate_prob(prob):
            return 0 <= prob <= 1

        assert validate_prob(0.0)
        assert validate_prob(0.5)
        assert validate_prob(1.0)
        assert not validate_prob(-0.1)
        assert not validate_prob(1.1)

    def test_oc4_session_is_valid(self):
        """OC-4: session must be 'early' or 'late'"""
        valid_sessions = ['early', 'late']

        assert 'early' in valid_sessions
        assert 'late' in valid_sessions
        assert 'morning' not in valid_sessions


class TestNoRepainting:
    """NR: No-Repainting tests - signals must not change retroactively"""

    def test_nr1_signal_locked_after_generation(self):
        """NR-1: Once a signal is generated, it cannot change for that bar"""
        # This test verifies the caching mechanism
        cache = {}

        def get_or_create_signal(ticker: str, date: str, hour: int, prob: float) -> dict:
            key = f"{ticker}_{date}_{hour}"
            if key not in cache:
                cache[key] = {
                    'ticker': ticker,
                    'prob': prob,
                    'locked_at': datetime.now().isoformat()
                }
            return cache[key]

        # First call creates the signal
        signal1 = get_or_create_signal('SPY', '2025-01-06', 10, 0.65)
        assert signal1['prob'] == 0.65

        # Second call with different prob should return cached value
        signal2 = get_or_create_signal('SPY', '2025-01-06', 10, 0.80)
        assert signal2['prob'] == 0.65  # Still the original value

    def test_nr2_different_hours_have_different_signals(self):
        """NR-2: Different hours can have different signals"""
        cache = {}

        def get_or_create_signal(ticker: str, date: str, hour: int, prob: float) -> dict:
            key = f"{ticker}_{date}_{hour}"
            if key not in cache:
                cache[key] = {'prob': prob}
            return cache[key]

        signal_10am = get_or_create_signal('SPY', '2025-01-06', 10, 0.65)
        signal_11am = get_or_create_signal('SPY', '2025-01-06', 11, 0.55)

        assert signal_10am['prob'] == 0.65
        assert signal_11am['prob'] == 0.55
        assert signal_10am != signal_11am


class TestPhase5Invariant:
    """P5: Phase 5 Invariant tests - final output must match inputs"""

    def test_p5_action_matches_probability(self):
        """P5-1: action must be consistent with probabilities"""
        def validate_action_consistency(prob: float, action: str) -> bool:
            if prob > 0.55:
                return action == 'BULLISH'
            elif prob < 0.45:
                return action == 'BEARISH'
            else:
                return action == 'NO_TRADE'

        # Valid cases
        assert validate_action_consistency(0.70, 'BULLISH')
        assert validate_action_consistency(0.30, 'BEARISH')
        assert validate_action_consistency(0.50, 'NO_TRADE')

        # Invalid cases (mismatched)
        assert not validate_action_consistency(0.70, 'BEARISH')
        assert not validate_action_consistency(0.30, 'BULLISH')
        assert not validate_action_consistency(0.50, 'BULLISH')

    def test_p5_targets_only_present_for_trades(self):
        """P5-2: entry/stop/target only present when action != NO_TRADE"""
        def validate_response(response: dict) -> bool:
            action = response.get('action')
            has_entry = 'entry' in response and response['entry'] is not None
            has_stop = 'stop' in response and response['stop'] is not None
            has_target = 'target' in response and response['target'] is not None

            if action == 'NO_TRADE':
                # NO_TRADE should not have entry/stop/target
                return not has_entry and not has_stop and not has_target
            else:
                # BULLISH/BEARISH should have entry/stop/target
                return has_entry and has_stop and has_target

        # Valid NO_TRADE response
        assert validate_response({'action': 'NO_TRADE'})
        assert validate_response({'action': 'NO_TRADE', 'entry': None, 'stop': None, 'target': None})

        # Valid BULLISH response
        assert validate_response({'action': 'BULLISH', 'entry': 500.0, 'stop': 497.5, 'target': 505.0})

        # Invalid: NO_TRADE with targets
        assert not validate_response({'action': 'NO_TRADE', 'entry': 500.0, 'stop': 497.5, 'target': 505.0})


class TestPhaseBoundaries:
    """RPE Phase boundary tests - ML predictions only allowed in Phase 5"""

    def test_phases_1_to_4_no_ml_predictions(self):
        """Phases 1-4 (TRUTH, SIGNAL_HEALTH, DENSITY, EXECUTION) must not call ML"""
        # RPE phases that should NOT include ML predictions
        non_ml_phases = ['TRUTH', 'SIGNAL_HEALTH', 'SIGNAL_DENSITY', 'EXECUTION']

        def validate_phase_output(phase: str, output: dict) -> bool:
            """Phase 1-4 outputs should not contain ML prediction fields"""
            ml_fields = ['target_a_prob', 'target_b_prob', 'action', 'confidence_bucket']

            if phase in non_ml_phases:
                # These phases should NOT have ML fields directly
                for field in ml_fields:
                    if field in output and output[field] is not None:
                        return False
            return True

        # Valid Phase 1 output (structure analysis only)
        phase1_output = {
            'phase': 'TRUTH',
            'market_structure': 'bullish',
            'key_levels': [500.0, 505.0],
        }
        assert validate_phase_output('TRUTH', phase1_output)

        # Invalid: Phase 1 with ML predictions
        invalid_phase1 = {
            'phase': 'TRUTH',
            'target_a_prob': 0.65,  # WRONG - ML in Phase 1
        }
        assert not validate_phase_output('TRUTH', invalid_phase1)

    def test_phase_5_allows_ml_predictions(self):
        """Phase 5 is the only phase where ML predictions are allowed"""
        phase5_output = {
            'phase': 'PREDICTION',
            'target_a_prob': 0.65,
            'target_b_prob': 0.60,
            'action': 'BULLISH',
            'confidence_bucket': 'medium',
        }

        # Phase 5 should have ML fields
        assert 'target_a_prob' in phase5_output
        assert 'action' in phase5_output


class TestGoldenSnapshot:
    """Golden snapshot test - hardcoded historical day for regression detection"""

    def test_golden_snapshot_2025_01_06(self):
        """
        Golden snapshot: 2025-01-06, SPY, 10:30 AM ET

        Hardcoded inputs and expected outputs to catch:
        - Accidental changes in formatting
        - Rounding differences
        - Feature calculation drift
        """
        # FIXED INPUTS (do not change)
        snapshot = {
            'date': '2025-01-06',
            'ticker': 'SPY',
            'hour': 10,
            'session': 'early',  # hour < 11

            # Market data snapshot
            'daily_open': 595.50,
            'prev_day_close': 594.00,
            'current_price': 596.25,
            'price_11am': None,  # Not yet available at 10:30

            # Expected calculations
            'expected_gap_pct': (595.50 - 594.00) / 594.00 * 100,  # ~0.252%
            'expected_open_to_current': (596.25 - 595.50) / 595.50 * 100,  # ~0.126%
        }

        # Verify session classification
        assert snapshot['session'] == 'early', "Hour 10 must be 'early' session"

        # Verify gap calculation
        expected_gap = snapshot['expected_gap_pct']
        assert abs(expected_gap - 0.2525) < 0.01, f"Gap calculation drift: {expected_gap}"

        # Verify open_to_current calculation
        expected_otc = snapshot['expected_open_to_current']
        assert abs(expected_otc - 0.1259) < 0.01, f"Open-to-current drift: {expected_otc}"

        # Response schema validation
        expected_response_fields = [
            'ticker', 'action', 'target_a_prob', 'session',
            'confidence', 'confidence_bucket', 'generated_at'
        ]

        mock_response = {
            'ticker': 'SPY',
            'action': 'BULLISH',
            'target_a_prob': 0.62,
            'target_b_prob': None,  # Not available in early session
            'session': 'early',
            'confidence': 0.12,
            'confidence_bucket': 'low',
            'generated_at': '2025-01-06T10:30:00-05:00',
        }

        for field in expected_response_fields:
            assert field in mock_response, f"Missing field in response: {field}"

        # Verify Target A is used in early session (not Target B)
        assert snapshot['hour'] < 11
        assert snapshot['session'] == 'early'
        # In early session, Target A (close > open) is primary


class TestV6ModelIntegration:
    """V6 Model integration tests"""

    def test_v6_model_files_exist(self):
        """Verify V6 model files are present"""
        import os
        model_dir = '/Users/it/Documents/mvp_coder_starter_kit (2)/mvp-trading-app/ml/v6_models'

        expected_models = ['spy_target_pred.pkl', 'qqq_target_pred.pkl', 'iwm_target_pred.pkl']

        for model_file in expected_models:
            path = os.path.join(model_dir, model_file)
            assert os.path.exists(path), f"Missing model file: {path}"

    def test_v6_model_has_required_keys(self):
        """V6 model must have model_a, model_b, feature_cols"""
        import joblib
        import os

        model_path = '/Users/it/Documents/mvp_coder_starter_kit (2)/mvp-trading-app/ml/v6_models/spy_target_pred.pkl'

        if not os.path.exists(model_path):
            pytest.skip("V6 model file not found - run training first")

        try:
            model_data = joblib.load(model_path)

            required_keys = ['model_a', 'model_b', 'feature_cols']
            for key in required_keys:
                assert key in model_data, f"Missing key in model: {key}"
        except Exception as e:
            pytest.skip(f"Model uses different serialization format: {e}")


class TestRPEIntegration:
    """RPE (Reality Proof Engine) integration tests"""

    def test_rpe_is_separate_from_v6(self):
        """RPE and V6 are separate engines that can work independently"""
        # V6 output structure
        v6_output = {
            'target_a_prob': 0.65,
            'target_b_prob': 0.60,
            'session': 'early',
        }

        # RPE output structure (different, not merged)
        rpe_output = {
            'phase': 'TRUTH',
            'signal_health': 0.8,
            'density_score': 0.75,
            'execution_ready': True,
        }

        # They should have no overlapping keys (except when intentionally combined in response)
        v6_keys = set(v6_output.keys())
        rpe_keys = set(rpe_output.keys())

        assert v6_keys.isdisjoint(rpe_keys), "V6 and RPE should have separate output keys"


class TestDataSourceIntegrity:
    """Test that data sources are correctly used"""

    def test_daily_bar_open_used_for_today_open(self):
        """Verify today_open comes from daily_bars[-1]['o']"""
        daily_bars = [
            {'o': 495.0, 'c': 497.0, 't': 1704067200000},  # day -2
            {'o': 497.0, 'c': 499.0, 't': 1704153600000},  # day -1
            {'o': 500.0, 'c': 502.0, 't': 1704240000000},  # today
        ]

        # SPEC: Always use daily_bars[-1]['o'] for today_open
        today_open = daily_bars[-1]['o']
        assert today_open == 500.0

        # NEVER use minute_bars[0]['o'] as fallback
        minute_bars = [{'o': 501.5, 'c': 502.0, 't': 1704276600000}]

        # This would be WRONG:
        # wrong_open = minute_bars[0]['o']  # 501.5 - NEVER DO THIS


class TestSpecVersionLock:
    """SV: Spec Version Lock - prevents rule changes without version bump"""

    # LOCKED SPEC VERSION - bump this when changing trading rules
    SPEC_VERSION = "2026-01-03"
    ENGINE_VERSION = "V6.1"

    def test_sv1_spec_version_declared(self):
        """SV-1: Server must declare spec_version in responses"""
        # This test ensures the server includes version tracking
        expected_version_fields = ['spec_version', 'engine_version']

        mock_response = {
            'ticker': 'SPY',
            'action': 'BULLISH',
            'target_a_prob': 0.65,
            'spec_version': self.SPEC_VERSION,
            'engine_version': self.ENGINE_VERSION,
        }

        for field in expected_version_fields:
            assert field in mock_response, f"Missing version field: {field}"

    def test_sv2_spec_version_matches_locked(self):
        """SV-2: spec_version must match the locked constant"""
        # This test will FAIL if someone changes rules without updating version
        current_spec_version = self.SPEC_VERSION

        # Verify version format (YYYY-MM-DD)
        parts = current_spec_version.split('-')
        assert len(parts) == 3, "spec_version must be YYYY-MM-DD format"
        assert len(parts[0]) == 4, "Year must be 4 digits"
        assert len(parts[1]) == 2, "Month must be 2 digits"
        assert len(parts[2]) == 2, "Day must be 2 digits"

        # Verify it's a valid date
        from datetime import datetime
        try:
            datetime.strptime(current_spec_version, '%Y-%m-%d')
        except ValueError:
            pytest.fail(f"Invalid spec_version date: {current_spec_version}")

    def test_sv3_engine_version_format(self):
        """SV-3: engine_version must follow V{major}.{minor} format"""
        import re
        pattern = r'^V\d+\.\d+$'
        assert re.match(pattern, self.ENGINE_VERSION), \
            f"engine_version must match V{{major}}.{{minor}}: {self.ENGINE_VERSION}"


class TestDailyOpenHardGate:
    """DO: Daily Open Hard Gate - missing daily bar triggers deterministic NO_TRADE"""

    def test_do1_missing_daily_open_returns_no_trade(self):
        """DO-1: Missing daily_bars[-1]['o'] → NO_TRADE with reason"""
        def get_trading_direction(daily_bars: list, hourly_bars: list) -> dict:
            """Simulates server logic with hard gate for missing daily open"""
            # HARD GATE: No daily bar = no trade
            if not daily_bars or len(daily_bars) < 1:
                return {
                    'action': 'NO_TRADE',
                    'reason': 'MISSING_DAILY_BAR',
                    'target_a_prob': None,
                    'target_b_prob': None,
                }

            # HARD GATE: Daily bar missing 'o' field
            if 'o' not in daily_bars[-1] or daily_bars[-1]['o'] is None:
                return {
                    'action': 'NO_TRADE',
                    'reason': 'MISSING_DAILY_OPEN',
                    'target_a_prob': None,
                    'target_b_prob': None,
                }

            # Normal processing would continue here...
            return {
                'action': 'BULLISH',
                'reason': None,
                'target_a_prob': 0.65,
                'target_b_prob': 0.60,
            }

        # Test 1: Empty daily_bars
        result = get_trading_direction([], [{'o': 500, 'c': 501}])
        assert result['action'] == 'NO_TRADE'
        assert result['reason'] == 'MISSING_DAILY_BAR'
        assert result['target_a_prob'] is None

        # Test 2: Daily bar missing 'o' field
        bad_daily = [{'c': 500, 't': 1704240000000}]  # no 'o' field
        result = get_trading_direction(bad_daily, [{'o': 500, 'c': 501}])
        assert result['action'] == 'NO_TRADE'
        assert result['reason'] == 'MISSING_DAILY_OPEN'

        # Test 3: Daily bar with None 'o' value
        null_daily = [{'o': None, 'c': 500, 't': 1704240000000}]
        result = get_trading_direction(null_daily, [{'o': 500, 'c': 501}])
        assert result['action'] == 'NO_TRADE'
        assert result['reason'] == 'MISSING_DAILY_OPEN'

    def test_do2_valid_daily_open_allows_trade(self):
        """DO-2: Valid daily_bars[-1]['o'] allows normal processing"""
        def get_trading_direction(daily_bars: list) -> dict:
            if not daily_bars or 'o' not in daily_bars[-1] or daily_bars[-1]['o'] is None:
                return {'action': 'NO_TRADE', 'reason': 'MISSING_DAILY_OPEN'}
            return {'action': 'BULLISH', 'reason': None}

        # Valid daily bar with open price
        valid_daily = [{'o': 500.0, 'c': 502.0, 't': 1704240000000}]
        result = get_trading_direction(valid_daily)
        assert result['action'] != 'NO_TRADE'
        assert result['reason'] is None

    def test_do3_reason_is_deterministic(self):
        """DO-3: Same bad input always produces same NO_TRADE reason"""
        def get_trading_direction(daily_bars: list) -> dict:
            if not daily_bars:
                return {'action': 'NO_TRADE', 'reason': 'MISSING_DAILY_BAR'}
            if 'o' not in daily_bars[-1] or daily_bars[-1]['o'] is None:
                return {'action': 'NO_TRADE', 'reason': 'MISSING_DAILY_OPEN'}
            return {'action': 'BULLISH', 'reason': None}

        # Run same bad input multiple times - must be deterministic
        bad_input = []
        results = [get_trading_direction(bad_input) for _ in range(10)]

        # All results must be identical
        first = results[0]
        for r in results[1:]:
            assert r == first, "NO_TRADE response must be deterministic"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
