"""
Phase 1 Structure Tests
Verifies deterministic output and correctness of all components.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import hashlib
from typing import Dict, List

# Import Phase 1 modules directly (not as package)
from vwap import calculate_vwap
from acceptance import check_acceptance, check_swing_acceptance, get_acceptance_side
from auction_state import classify_auction_state, classify_swing_auction_state
from levels import calculate_opening_range, calculate_intraday_levels, calculate_swing_levels
from failures import detect_failures, detect_swing_failures
from beware import generate_beware_alerts, aggregate_risk_level
from compute import compute_intraday_context, compute_swing_context


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

def create_1m_bars_fixture() -> List[Dict]:
    """Create realistic 1-minute bars for testing."""
    bars = []
    base_price = 600.0
    base_time = "09:30"

    # Simulate opening 30 minutes
    for i in range(30):
        hour = 9
        minute = 30 + i
        if minute >= 60:
            hour = 10
            minute -= 60

        # Price action: small rally then pullback
        if i < 15:
            price_offset = i * 0.10  # Rally
        else:
            price_offset = 1.50 - (i - 15) * 0.05  # Pullback

        bars.append({
            'timestamp': f'{hour:02d}:{minute:02d}',
            'open': base_price + price_offset,
            'high': base_price + price_offset + 0.20,
            'low': base_price + price_offset - 0.15,
            'close': base_price + price_offset + 0.05,
            'volume': 100000 + i * 5000
        })

    return bars


def create_5m_bars_fixture() -> List[Dict]:
    """Create 5-minute bars for acceptance testing."""
    return [
        {'timestamp': '09:30', 'open': 600.0, 'high': 600.50, 'low': 599.80, 'close': 600.30, 'volume': 500000},
        {'timestamp': '09:35', 'open': 600.30, 'high': 600.80, 'low': 600.10, 'close': 600.60, 'volume': 450000},
        {'timestamp': '09:40', 'open': 600.60, 'high': 601.00, 'low': 600.40, 'close': 600.90, 'volume': 400000},
        {'timestamp': '09:45', 'open': 600.90, 'high': 601.20, 'low': 600.70, 'close': 601.10, 'volume': 380000},
        {'timestamp': '09:50', 'open': 601.10, 'high': 601.50, 'low': 600.90, 'close': 601.30, 'volume': 350000},
        {'timestamp': '09:55', 'open': 601.30, 'high': 601.40, 'low': 600.80, 'close': 600.95, 'volume': 320000},
        {'timestamp': '10:00', 'open': 600.95, 'high': 601.10, 'low': 600.60, 'close': 600.70, 'volume': 300000},
    ]


def create_daily_bars_fixture() -> List[Dict]:
    """Create daily bars for swing context testing."""
    return [
        {'date': '2025-01-02', 'open': 595.0, 'high': 598.0, 'low': 594.0, 'close': 597.5, 'volume': 50000000},
        {'date': '2025-01-03', 'open': 597.5, 'high': 600.0, 'low': 596.0, 'close': 599.0, 'volume': 48000000},
        {'date': '2025-01-06', 'open': 599.0, 'high': 602.0, 'low': 598.0, 'close': 601.5, 'volume': 52000000},
        {'date': '2025-01-07', 'open': 601.5, 'high': 603.0, 'low': 600.0, 'close': 602.0, 'volume': 47000000},
        {'date': '2025-01-08', 'open': 602.0, 'high': 604.0, 'low': 601.0, 'close': 603.5, 'volume': 51000000},
    ]


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

def test_vwap_determinism():
    """VWAP calculation must be deterministic."""
    bars = create_1m_bars_fixture()

    result1 = calculate_vwap(bars)
    result2 = calculate_vwap(bars)
    result3 = calculate_vwap(bars)

    assert result1 == result2 == result3, "VWAP is not deterministic!"
    print(f"  VWAP Determinism: PASS (value={result1:.4f})")
    return True


def test_acceptance_determinism():
    """Acceptance state must be deterministic."""
    bars = create_5m_bars_fixture()
    level = 600.50

    result1 = check_acceptance(level, bars, direction="above")
    result2 = check_acceptance(level, bars, direction="above")
    result3 = check_acceptance(level, bars, direction="above")

    assert result1 == result2 == result3, "Acceptance is not deterministic!"
    print(f"  Acceptance Determinism: PASS (status={result1['status']})")
    return True


def test_auction_state_determinism():
    """Auction state must be deterministic."""
    context = {
        'current_price': 601.00,
        'rth_open': 600.00,
        'open_30m_high': 601.50,
        'open_30m_low': 599.50,
        'high_of_day': 602.00,
        'low_of_day': 599.00,
        'vwap': 600.50,
        'acceptance_states': {},
        'touched_or_high': True,
        'touched_or_low': False,
    }

    result1 = classify_auction_state(context)
    result2 = classify_auction_state(context)
    result3 = classify_auction_state(context)

    assert result1 == result2 == result3, "Auction state is not deterministic!"
    print(f"  Auction State Determinism: PASS (state={result1['state']})")
    return True


def test_intraday_context_determinism():
    """Full intraday context must be deterministic."""
    bars_1m = create_1m_bars_fixture()
    prior_day = {'high': 602.0, 'low': 598.0, 'close': 600.0}

    result1 = compute_intraday_context('SPY', bars_1m, prior_day=prior_day, current_time='10:00')
    result2 = compute_intraday_context('SPY', bars_1m, prior_day=prior_day, current_time='10:00')

    # Remove non-deterministic fields for comparison
    for r in [result1, result2]:
        if r:
            r.pop('as_of', None)
            r.pop('context_id', None)

    assert result1 == result2, "Intraday context is not deterministic!"
    print(f"  Intraday Context Determinism: PASS")
    return True


def test_swing_context_determinism():
    """Full swing context must be deterministic."""
    daily_bars = create_daily_bars_fixture()

    result1 = compute_swing_context('SPY', daily_bars)
    result2 = compute_swing_context('SPY', daily_bars)

    # Remove non-deterministic fields
    for r in [result1, result2]:
        if r:
            r.pop('context_id', None)

    assert result1 == result2, "Swing context is not deterministic!"
    print(f"  Swing Context Determinism: PASS")
    return True


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_vwap_calculation():
    """Test VWAP calculation correctness."""
    print("\n--- VWAP Tests ---")

    # Test 1: Single bar
    bar = [{"high": 100, "low": 98, "close": 99, "volume": 1000}]
    expected = (100 + 98 + 99) / 3
    result = calculate_vwap(bar)
    assert abs(result - expected) < 0.01, f"Single bar VWAP failed: {result} != {expected}"
    print(f"  Single bar: PASS")

    # Test 2: Multiple bars weighted by volume
    bars = [
        {"high": 100, "low": 98, "close": 99, "volume": 1000},
        {"high": 102, "low": 100, "close": 101, "volume": 2000},
    ]
    expected = (99 * 1000 + 101 * 2000) / 3000
    result = calculate_vwap(bars)
    assert abs(result - expected) < 0.01, f"Multi-bar VWAP failed: {result} != {expected}"
    print(f"  Multi-bar weighted: PASS")

    # Test 3: Empty bars
    result = calculate_vwap([])
    assert result is None, "Empty bars should return None"
    print(f"  Empty bars: PASS")

    # Test 4: Zero volume bar skipped
    bars = [
        {"high": 100, "low": 98, "close": 99, "volume": 1000},
        {"high": 102, "low": 100, "close": 101, "volume": 0},
    ]
    result = calculate_vwap(bars)
    expected = 99.0
    assert abs(result - expected) < 0.01, f"Zero volume VWAP failed"
    print(f"  Zero volume skip: PASS")

    return True


def test_acceptance_states():
    """Test acceptance state detection."""
    print("\n--- Acceptance Tests ---")

    # Test 1: ACCEPTED (3+ closes above)
    level = 100.0
    bars = [{"close": 100.50, "high": 100.60, "low": 100.40}] * 4
    result = check_acceptance(level, bars, direction="above")
    assert result["status"] == "ACCEPTED", f"Expected ACCEPTED, got {result['status']}"
    assert result["strength"] == "WEAK", f"Expected WEAK, got {result['strength']}"
    print(f"  Accepted (weak): PASS")

    # Test 2: MODERATE strength (5-8 closes)
    bars = [{"close": 100.50, "high": 100.60, "low": 100.40}] * 6
    result = check_acceptance(level, bars, direction="above")
    assert result["strength"] == "MODERATE", f"Expected MODERATE, got {result['strength']}"
    print(f"  Accepted (moderate): PASS")

    # Test 3: STRONG strength (9+ closes)
    bars = [{"close": 100.50, "high": 100.60, "low": 100.40}] * 10
    result = check_acceptance(level, bars, direction="above")
    assert result["strength"] == "STRONG", f"Expected STRONG, got {result['strength']}"
    print(f"  Accepted (strong): PASS")

    # Test 4: REJECTED (wick above, close below)
    bars = [{"close": 99.80, "high": 100.20, "low": 99.70}]
    result = check_acceptance(level, bars, direction="above")
    assert result["status"] == "REJECTED", f"Expected REJECTED, got {result['status']}"
    print(f"  Rejected: PASS")

    # Test 5: UNTESTED (never approached)
    bars = [{"close": 95.0, "high": 96.0, "low": 94.0}]
    result = check_acceptance(level, bars, direction="above")
    assert result["status"] == "UNTESTED", f"Expected UNTESTED, got {result['status']}"
    print(f"  Untested: PASS")

    # Test 6: FAILED_ACCEPTANCE (accepted then lost)
    bars = [
        {"close": 100.50, "high": 100.60, "low": 100.40},
        {"close": 100.30, "high": 100.40, "low": 100.20},
        {"close": 100.20, "high": 100.30, "low": 100.10},
        {"close": 99.80, "high": 100.00, "low": 99.70},  # Lost
    ]
    result = check_acceptance(level, bars, direction="above")
    assert result["status"] == "FAILED_ACCEPTANCE", f"Expected FAILED_ACCEPTANCE, got {result['status']}"
    print(f"  Failed acceptance: PASS")

    return True


def test_auction_states():
    """Test auction state classification."""
    print("\n--- Auction State Tests ---")

    # Test 1: RESOLVED UP (breakout above OR high, held)
    context = {
        'current_price': 602.00,
        'rth_open': 600.00,
        'open_30m_high': 601.00,
        'open_30m_low': 599.00,
        'high_of_day': 603.00,
        'low_of_day': 599.50,
        'vwap': 601.00,
        'acceptance_states': {'open_30m_high': {'status': 'ACCEPTED'}},
        'touched_or_high': True,
        'touched_or_low': False,
    }
    result = classify_auction_state(context)
    assert result['state'] == 'RESOLVED', f"Expected RESOLVED, got {result['state']}"
    assert result['resolved_direction'] == 'UP', f"Expected UP, got {result['resolved_direction']}"
    print(f"  Resolved UP: PASS")

    # Test 2: BALANCED (no clear direction)
    context = {
        'current_price': 600.50,
        'rth_open': 600.00,
        'open_30m_high': 601.00,
        'open_30m_low': 599.00,
        'high_of_day': 601.20,
        'low_of_day': 599.20,
        'vwap': 600.20,
        'acceptance_states': {},
        'touched_or_high': True,
        'touched_or_low': True,
    }
    result = classify_auction_state(context)
    assert result['state'] == 'BALANCED', f"Expected BALANCED, got {result['state']}"
    print(f"  Balanced: PASS")

    return True


def test_failure_detection():
    """Test failure signal detection."""
    print("\n--- Failure Detection Tests ---")

    # Test 1: FAILED_BREAKOUT (broke OR high, now back inside)
    context = {
        'open_30m_high': 601.00,
        'open_30m_low': 599.00,
        'high_of_day': 602.00,  # Broke above OR high
        'low_of_day': 599.50,
        'current_price': 600.50,  # Now back inside
        'acceptance_states': {},
    }
    result = detect_failures(context)
    assert result['present'] == True, "Expected failure present"
    assert 'FAILED_BREAKOUT' in result['types'], f"Expected FAILED_BREAKOUT, got {result['types']}"
    print(f"  Failed breakout: PASS")

    # Test 2: FAILED_BREAKDOWN
    context = {
        'open_30m_high': 601.00,
        'open_30m_low': 599.00,
        'high_of_day': 600.50,
        'low_of_day': 598.00,  # Broke below OR low
        'current_price': 600.00,  # Now back inside
        'acceptance_states': {},
    }
    result = detect_failures(context)
    assert 'FAILED_BREAKDOWN' in result['types'], f"Expected FAILED_BREAKDOWN, got {result['types']}"
    print(f"  Failed breakdown: PASS")

    # Test 3: No failure
    context = {
        'open_30m_high': 601.00,
        'open_30m_low': 599.00,
        'high_of_day': 600.80,  # Never broke OR high
        'low_of_day': 599.20,  # Never broke OR low
        'current_price': 600.00,
        'acceptance_states': {},
    }
    result = detect_failures(context)
    assert result['present'] == False, "Expected no failure"
    print(f"  No failure: PASS")

    return True


def test_beware_alerts():
    """Test BEWARE alert generation."""
    print("\n--- BEWARE Alert Tests ---")

    # Test 1: BALANCED_SESSION alert
    intraday_context = {
        'auction': {'state': 'BALANCED', 'resolved_direction': 'BALANCED'},
        'failure': {'present': False, 'types': []},
        'current_price': 600.00,
        'levels': {'set': []}
    }
    alerts = generate_beware_alerts(intraday_context, None, '12:00')
    balanced_alerts = [a for a in alerts if a['type'] == 'BALANCED_SESSION']
    assert len(balanced_alerts) > 0, "Expected BALANCED_SESSION alert"
    print(f"  Balanced session alert: PASS")

    # Test 2: LATE_SESSION alert
    alerts = generate_beware_alerts(intraday_context, None, '15:35')
    late_alerts = [a for a in alerts if a['type'] == 'LATE_SESSION']
    assert len(late_alerts) > 0, "Expected LATE_SESSION alert"
    print(f"  Late session alert: PASS")

    # Test 3: Risk level aggregation
    alerts = [
        {'severity': 'WARNING'},
        {'severity': 'WARNING'},
        {'severity': 'WARNING'},
    ]
    risk = aggregate_risk_level(alerts)
    assert risk == 'HIGH', f"Expected HIGH risk, got {risk}"
    print(f"  Risk aggregation: PASS")

    return True


def test_full_intraday_context():
    """Test full intraday context computation."""
    print("\n--- Full Intraday Context Test ---")

    bars_1m = create_1m_bars_fixture()
    prior_day = {'high': 602.0, 'low': 598.0, 'close': 600.0}

    context = compute_intraday_context(
        ticker='SPY',
        bars_1m=bars_1m,
        prior_day=prior_day,
        current_time='10:00'
    )

    assert context is not None, "Context should not be None"
    assert 'version' in context, "Missing version"
    assert 'ticker' in context, "Missing ticker"
    assert 'auction' in context, "Missing auction"
    assert 'levels' in context, "Missing levels"
    assert 'failure' in context, "Missing failure"
    assert 'beware' in context, "Missing beware"

    print(f"  Version: {context['version']}")
    print(f"  Auction State: {context['auction']['state']}")
    print(f"  Current Price: {context['current_price']:.2f}")
    print(f"  Levels Count: {len(context['levels']['set'])}")
    print(f"  Risk Level: {context['beware']['risk_level']}")
    print(f"  Full Intraday Context: PASS")

    return context


def test_full_swing_context():
    """Test full swing context computation."""
    print("\n--- Full Swing Context Test ---")

    daily_bars = create_daily_bars_fixture()

    context = compute_swing_context(
        ticker='SPY',
        daily_bars=daily_bars
    )

    assert context is not None, "Context should not be None"
    assert 'version' in context, "Missing version"
    assert 'ticker' in context, "Missing ticker"
    assert 'auction' in context, "Missing auction"
    assert 'bias' in context, "Missing bias"
    assert 'levels' in context, "Missing levels"

    print(f"  Version: {context['version']}")
    print(f"  Bias: {context['bias']['context']}")
    print(f"  Strength: {context['bias']['strength']}")
    print(f"  Levels Count: {len(context['levels']['set'])}")
    print(f"  Full Swing Context: PASS")

    return context


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all Phase 1 tests."""
    print("=" * 60)
    print("PHASE 1 STRUCTURE TESTS")
    print("=" * 60)

    passed = 0
    failed = 0

    # Determinism tests
    print("\n### DETERMINISM TESTS ###")
    tests = [
        ("VWAP Determinism", test_vwap_determinism),
        ("Acceptance Determinism", test_acceptance_determinism),
        ("Auction State Determinism", test_auction_state_determinism),
        ("Intraday Context Determinism", test_intraday_context_determinism),
        ("Swing Context Determinism", test_swing_context_determinism),
    ]

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            failed += 1

    # Unit tests
    print("\n### UNIT TESTS ###")
    unit_tests = [
        ("VWAP Calculation", test_vwap_calculation),
        ("Acceptance States", test_acceptance_states),
        ("Auction States", test_auction_states),
        ("Failure Detection", test_failure_detection),
        ("BEWARE Alerts", test_beware_alerts),
    ]

    for name, test_func in unit_tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            failed += 1

    # Integration tests
    print("\n### INTEGRATION TESTS ###")
    try:
        intraday_ctx = test_full_intraday_context()
        passed += 1
    except Exception as e:
        print(f"  Full Intraday Context: FAILED - {e}")
        failed += 1

    try:
        swing_ctx = test_full_swing_context()
        passed += 1
    except Exception as e:
        print(f"  Full Swing Context: FAILED - {e}")
        failed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\nALL TESTS PASSED - Phase 1 is deterministic and correct!")
    else:
        print(f"\n{failed} TESTS FAILED - Review errors above")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
