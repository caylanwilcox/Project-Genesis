# Spec → Test Traceability Matrix

This document maps every spec rule to its corresponding test(s), ensuring full coverage and enabling quick audits.

## Legend
- **Spec ID**: Unique identifier from TRADING_ENGINE_SPEC.md
- **Test Name**: pytest test function name
- **File**: Test file location
- **Status**: ✅ Covered | ⚠️ Partial | ❌ Missing

---

## Market Hours (MH)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| MH-1 | 09:30:00 ET = OPEN | `test_mh1_market_open_at_930` | tests/unit/test_session.py | ✅ |
| MH-2 | 15:59:59 ET = OPEN | `test_mh2_market_open_at_1559` | tests/unit/test_session.py | ✅ |
| MH-3 | 16:00:00 ET = CLOSED | `test_mh3_market_closed_at_1600` | tests/unit/test_session.py | ✅ |
| MH-4 | Weekend = CLOSED | `test_mh4_market_closed_on_saturday`, `test_mh4_market_closed_on_sunday` | tests/unit/test_session.py | ✅ |

---

## Session Classification (SC)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| SC-1 | hour < 11 → "early" | `test_sc1_early_session_hour_10`, `test_sc1_early_session_hour_9` | tests/unit/test_session.py | ✅ |
| SC-2 | hour >= 11 → "late" | `test_sc2_late_session_hour_11`, `test_sc2_late_session_hour_14` | tests/unit/test_session.py | ✅ |
| SC-3 | Boundary at 11:00:00 | `test_sc3_boundary_at_1100` | tests/unit/test_session.py | ✅ |

---

## Data Source (DS)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| DS-1 | today_open = daily_bars[-1]['o'] | `test_ds1_today_open_from_daily_bar`, `test_daily_bar_open_used_for_today_open` | tests/unit/test_schema.py, tests/integration/test_predict_server.py | ✅ |
| DS-2 | hourly_bars filtered by hour | `test_ds2_hourly_bars_filtered_by_hour` | tests/unit/test_schema.py | ✅ |
| DS-3 | daily_bars excludes today for prev-day | `test_ds3_daily_bars_excludes_today_for_prev_features` | tests/unit/test_schema.py | ✅ |
| DS-4 | V6 expects 29 features | `test_ds4_v6_expects_29_features` | tests/unit/test_schema.py | ✅ |

---

## Feature Schema (FS)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| FS-1 | Feature names match training | `test_fs1_feature_names_match_training` | tests/unit/test_schema.py | ✅ |
| FS-2 | Returns dict with features | `test_fs2_build_features_returns_dict` | tests/unit/test_schema.py | ✅ |
| FS-3 | Features are numeric | `test_fs3_features_are_numeric` | tests/unit/test_schema.py | ✅ |
| FS-4 | No NaN features | `test_fs4_no_nan_features` | tests/unit/test_schema.py | ✅ |

---

## Neutral Zone (NZ)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| NZ-1 | prob > 0.55 → BULLISH | `test_nz1_above_55_is_bullish` | tests/unit/test_policy.py | ✅ |
| NZ-2 | prob < 0.45 → BEARISH | `test_nz2_below_45_is_bearish` | tests/unit/test_policy.py | ✅ |
| NZ-3 | prob = 0.55 → NO_TRADE | `test_nz3_boundary_55_is_no_trade` | tests/unit/test_policy.py | ✅ |
| NZ-4 | prob = 0.45 → NO_TRADE | `test_nz4_boundary_45_is_no_trade` | tests/unit/test_policy.py | ✅ |
| NZ-5 | prob = 0.50 → NO_TRADE | `test_nz5_middle_is_no_trade` | tests/unit/test_policy.py | ✅ |
| NZ-6 | Float precision (0.5500000001 → BULLISH) | `test_nz6_float_precision_upper_boundary` | tests/unit/test_policy.py | ✅ |
| NZ-7 | Float precision (0.5499999999 → NO_TRADE) | `test_nz7_float_precision_lower_boundary` | tests/unit/test_policy.py | ✅ |
| NZ-8 | Float precision at 0.45 boundary | `test_nz8_float_precision_bearish_boundary` | tests/unit/test_policy.py | ✅ |

---

## Confidence Buckets (BK)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| BK-1 | prob >= 0.90 → "very_strong" (size_mult=1.0) | `test_bk1_very_strong_above_90` | tests/unit/test_policy.py | ✅ |
| BK-2 | prob <= 0.10 → "very_strong" (size_mult=1.0) | `test_bk2_very_strong_below_10` | tests/unit/test_policy.py | ✅ |
| BK-3 | prob >= 0.70 OR <= 0.30 → "strong" (size_mult=0.75) | `test_bk3_strong` | tests/unit/test_policy.py | ✅ |
| BK-4 | prob >= 0.60 OR <= 0.40 → "moderate" (size_mult=0.50) | `test_bk4_moderate` | tests/unit/test_policy.py | ✅ |
| BK-5 | prob >= 0.55 OR <= 0.45 → "weak" (size_mult=0.25) | `test_bk5_weak` | tests/unit/test_policy.py | ✅ |
| BK-6 | prob in (0.45, 0.55) → None (neutral zone) | `test_bk6_neutral_zone_no_bucket` | tests/unit/test_policy.py | ✅ |

---

## Time Multiplier (TM)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| TM-1 | hour 13-15 → 1.0 (peak accuracy) | `test_tm1_peak_hours` | tests/unit/test_policy.py | ✅ |
| TM-2 | hour 11-12 → 0.8 (good accuracy) | `test_tm2_good_accuracy` | tests/unit/test_policy.py | ✅ |
| TM-3 | hour 10 → 0.6 (moderate accuracy) | `test_tm3_moderate_accuracy` | tests/unit/test_policy.py | ✅ |
| TM-4 | hour < 10 → 0.4 (early session) | `test_tm4_early_session` | tests/unit/test_policy.py | ✅ |

---

## Agreement Multiplier (AM)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| AM-1 | Both > 0.5 → 1.2 | `test_am1_both_bullish` | tests/unit/test_policy.py | ✅ |
| AM-2 | Both < 0.5 → 1.2 | `test_am2_both_bearish` | tests/unit/test_policy.py | ✅ |
| AM-3 | Conflicting → 0.6 | `test_am3_conflicting` | tests/unit/test_policy.py | ✅ |
| AM-4 | One = 0.5 → 1.0 | `test_am4_neutral` | tests/unit/test_policy.py | ✅ |

---

## Target Selection (TS)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| TS-1 | early → Target A | `test_ts1_early_session_uses_target_a` | tests/unit/test_policy.py | ✅ |
| TS-2 | late → Target B | `test_ts2_late_session_uses_target_b` | tests/unit/test_policy.py | ✅ |
| TS-3 | Session determines target | `test_ts3_session_boundary` | tests/unit/test_policy.py | ✅ |

---

## Entry/Exit Targets (EX)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| EX-1 | SPY: TP=0.25%, SL=0.33% | `test_ex1_spy_targets` | tests/unit/test_policy.py | ✅ |
| EX-2 | QQQ: TP=0.34%, SL=0.45% | `test_ex2_qqq_targets` | tests/unit/test_policy.py | ✅ |
| EX-3 | IWM: TP=0.45%, SL=0.60% | `test_ex3_iwm_targets` | tests/unit/test_policy.py | ✅ |
| EX-4 | BUY_PUT inverts TP/SL direction | `test_ex4_bearish_targets_inverted` | tests/unit/test_policy.py | ✅ |

---

## Output Contract (OC)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| OC-1 | Required response fields | `test_oc1_trading_directions_has_required_fields` | tests/integration/test_predict_server.py | ✅ |
| OC-2 | action enum valid | `test_oc2_action_is_valid_enum` | tests/integration/test_predict_server.py | ✅ |
| OC-3 | prob in [0,1] | `test_oc3_probabilities_in_range` | tests/integration/test_predict_server.py | ✅ |
| OC-4 | session valid | `test_oc4_session_is_valid` | tests/integration/test_predict_server.py | ✅ |

---

## No-Repainting (NR)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| NR-1 | Signal locked after generation | `test_nr1_signal_locked_after_generation` | tests/integration/test_predict_server.py | ✅ |
| NR-2 | Different hours = different signals | `test_nr2_different_hours_have_different_signals` | tests/integration/test_predict_server.py | ✅ |

---

## Phase 5 Invariant (P5)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| P5-1 | action matches probability | `test_p5_action_matches_probability` | tests/integration/test_predict_server.py | ✅ |
| P5-2 | targets only for trades | `test_p5_targets_only_present_for_trades` | tests/integration/test_predict_server.py | ✅ |
| P5-3 | Phases 1-4 no ML | `test_phases_1_to_4_no_ml_predictions` | tests/integration/test_predict_server.py | ✅ |
| P5-4 | Phase 5 allows ML | `test_phase_5_allows_ml_predictions` | tests/integration/test_predict_server.py | ✅ |

---

## Golden Snapshot (GS)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| GS-1 | Historical regression test | `test_golden_snapshot_2025_01_06` | tests/integration/test_predict_server.py | ✅ |

---

## Spec Version Lock (SV)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| SV-1 | Response includes spec_version + engine_version | `test_sv1_spec_version_declared` | tests/integration/test_predict_server.py | ✅ |
| SV-2 | spec_version matches locked YYYY-MM-DD | `test_sv2_spec_version_matches_locked` | tests/integration/test_predict_server.py | ✅ |
| SV-3 | engine_version follows V{major}.{minor} | `test_sv3_engine_version_format` | tests/integration/test_predict_server.py | ✅ |

---

## Daily Open Hard Gate (DO)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| DO-1 | Missing daily bar → NO_TRADE + reason | `test_do1_missing_daily_open_returns_no_trade` | tests/integration/test_predict_server.py | ✅ |
| DO-2 | Valid daily_bars[-1]['o'] allows trade | `test_do2_valid_daily_open_allows_trade` | tests/integration/test_predict_server.py | ✅ |
| DO-3 | NO_TRADE reason is deterministic | `test_do3_reason_is_deterministic` | tests/integration/test_predict_server.py | ✅ |

---

## Frontend Connection (FC)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| FC-1 | ML server /health responds | `test_ml_server_health_endpoint` | tests/integration/test_frontend_connection.py | ✅ |
| FC-2 | ML server /trading_directions responds | `test_ml_server_trading_directions_endpoint` | tests/integration/test_frontend_connection.py | ✅ |
| FC-3 | ML server /model_info responds | `test_ml_server_model_info_endpoint` | tests/integration/test_frontend_connection.py | ✅ |
| FC-4 | Response has SPY/QQQ/IWM or market-closed | `test_trading_directions_has_required_tickers` | tests/integration/test_frontend_connection.py | ✅ |
| FC-5 | Each signal has action + session | `test_each_signal_has_required_fields` | tests/integration/test_frontend_connection.py | ✅ |
| FC-6 | Action is valid enum | `test_action_is_valid_enum` | tests/integration/test_frontend_connection.py | ✅ |
| FC-7 | Session is 'early' or 'late' | `test_session_is_early_or_late` | tests/integration/test_frontend_connection.py | ✅ |
| FC-8 | Probabilities in [0,1] | `test_probabilities_in_valid_range` | tests/integration/test_frontend_connection.py | ✅ |
| FC-9 | Response includes spec_version | `test_spec_version_included` | tests/integration/test_frontend_connection.py | ✅ |
| FC-10 | Frontend API proxies to ML | `test_frontend_trading_directions_api` | tests/integration/test_frontend_connection.py | ✅ |
| FC-11 | Frontend receives signal structure | `test_frontend_receives_signal_structure` | tests/integration/test_frontend_connection.py | ✅ |
| FC-12 | Neutral zone enforced | `test_neutral_zone_produces_no_trade` | tests/integration/test_frontend_connection.py | ✅ |
| FC-13 | Repeated requests = same action | `test_repeated_requests_produce_same_action` | tests/integration/test_frontend_connection.py | ✅ |

---

## RPE Connection (RPE)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| RPE-1 | RPE /rpe endpoint responds | `test_rpe_endpoint_responds` | tests/integration/test_frontend_connection.py | ✅ |
| RPE-2 | RPE includes 5 phases | `test_rpe_has_5_phases` | tests/integration/test_frontend_connection.py | ✅ |

---

## Northstar Connection (NS)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| NS-1 | Northstar /northstar endpoint responds | `test_northstar_endpoint_responds` | tests/integration/test_frontend_connection.py | ✅ |
| NS-2 | Northstar includes 4 phases | `test_northstar_has_4_phases` | tests/integration/test_frontend_connection.py | ✅ |

---

## Replay Mode (RM)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| RM-1 | Replay /replay endpoint responds | `test_replay_endpoint_responds` | tests/integration/test_frontend_connection.py | ✅ |
| RM-2 | Replay has mode, date, time, tickers | `test_replay_has_required_fields` | tests/integration/test_frontend_connection.py | ✅ |
| RM-3 | Replay includes V6 + Northstar | `test_replay_includes_v6_and_northstar` | tests/integration/test_frontend_connection.py | ✅ |
| RM-4 | Frontend /api/v2/replay proxies | `test_frontend_replay_api` | tests/integration/test_frontend_connection.py | ✅ |
| RM-5 | Frontend /api/v2/rpe proxies | `test_frontend_rpe_api` | tests/integration/test_frontend_connection.py | ✅ |

---

## System Health (SH)

| Spec ID | Rule | Test Name | File | Status |
|---------|------|-----------|------|--------|
| SH-1 | Governance docs exist | `check_governance_documents()` | ml/system_check.py | ✅ |
| SH-2 | ML models loaded | `check_ml_models()` | ml/system_check.py | ✅ |
| SH-3 | ML server endpoints respond | `check_ml_server()` | ml/system_check.py | ✅ |
| SH-4 | Frontend→ML connection works | `check_frontend_connection()` | ml/system_check.py | ✅ |
| SH-5 | Spec compliance verified | `check_spec_compliance()` | ml/system_check.py | ✅ |
| SH-6 | Test files exist | `check_test_coverage()` | ml/system_check.py | ✅ |

---

## Summary

| Category | Total Rules | Covered | Partial | Missing |
|----------|-------------|---------|---------|---------|
| Market Hours (MH) | 4 | 4 | 0 | 0 |
| Session Classification (SC) | 3 | 3 | 0 | 0 |
| Data Source (DS) | 4 | 4 | 0 | 0 |
| Feature Schema (FS) | 4 | 4 | 0 | 0 |
| Neutral Zone (NZ) | 8 | 8 | 0 | 0 |
| Confidence Buckets (BK) | 6 | 6 | 0 | 0 |
| Time Multiplier (TM) | 4 | 4 | 0 | 0 |
| Agreement Multiplier (AM) | 4 | 4 | 0 | 0 |
| Target Selection (TS) | 3 | 3 | 0 | 0 |
| Entry/Exit (EX) | 4 | 4 | 0 | 0 |
| Output Contract (OC) | 4 | 4 | 0 | 0 |
| No-Repainting (NR) | 2 | 2 | 0 | 0 |
| Phase 5 Invariant (P5) | 4 | 4 | 0 | 0 |
| Golden Snapshot (GS) | 1 | 1 | 0 | 0 |
| Spec Version Lock (SV) | 3 | 3 | 0 | 0 |
| Daily Open Hard Gate (DO) | 3 | 3 | 0 | 0 |
| Frontend Connection (FC) | 13 | 13 | 0 | 0 |
| RPE Connection (RPE) | 2 | 2 | 0 | 0 |
| Northstar Connection (NS) | 2 | 2 | 0 | 0 |
| Replay Mode (RM) | 5 | 5 | 0 | 0 |
| System Health (SH) | 6 | 6 | 0 | 0 |
| **TOTAL** | **89** | **89** | **0** | **0** |

---

## Running Tests

```bash
# Run all tests
cd ml && python3 -m pytest tests/ -v

# Run specific category
python3 -m pytest tests/unit/test_session.py -v  # MH + SC
python3 -m pytest tests/unit/test_policy.py -v   # NZ + BK + TM + AM
python3 -m pytest tests/unit/test_schema.py -v   # DS + FS

# Run integration tests (requires running servers)
python3 -m pytest tests/integration/test_predict_server.py -v    # OC + NR + P5 + GS + SV + DO
python3 -m pytest tests/integration/test_frontend_connection.py -v  # FC (live connection tests)

# Run with coverage
python3 -m pytest tests/ --cov=server --cov-report=html

# Run system health check
python3 system_check.py  # SH-1 through SH-6
```

---

*Last updated: 2026-01-03*
