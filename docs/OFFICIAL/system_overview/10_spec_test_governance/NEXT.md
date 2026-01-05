# Spec & Test Governance Layer - Future Development

## Planned Improvements

### 1. Phase 2-4 Unit Test Coverage

**Current Gap**: Phases 2-4 (Health, Density, Posture) have no dedicated unit tests.

**Improvement**:
```python
# tests/unit/test_phase2.py
class TestPhase2Health:
    def test_p2_no_acceptance_penalty(self):
        reality = RealityState(acceptance=Acceptance(accepted=False))
        health = Phase2Engine().calculate(reality)
        assert health.structural_integrity <= 75  # -25 penalty

    def test_p2_stand_down_threshold(self):
        # health_score < 45 → stand_down = True
        ...

# tests/unit/test_phase3.py
class TestPhase3Density:
    def test_p3_same_level_spam(self):
        # >3 signals in 10m, <=1 level → -40 penalty
        ...

# tests/unit/test_phase4.py
class TestPhase4Posture:
    def test_p4_denied_when_not_structural_edge(self):
        ...
```

**Impact**: Complete spec coverage for RPE pipeline.

---

### 2. Automated Spec Validation

**Current Gap**: Manual verification that code matches spec.

**Improvement**: Automated spec extraction and validation.

```python
# Extract specs from markdown
specs = parse_spec_document('TRADING_ENGINE_SPEC.md')

# Generate test stubs
for spec in specs:
    if not has_test(spec.id):
        print(f"Missing test for {spec.id}: {spec.rule}")

# Validate implementation
for spec in specs:
    result = validate_implementation(spec)
    assert result.compliant, f"{spec.id} non-compliant"
```

**Impact**: Automated drift detection.

---

### 3. Continuous Spec Verification

**Current Gap**: Specs verified only in test suite.

**Improvement**: Runtime spec verification.

```python
# Runtime assertion in production
def get_trading_direction(ticker, prob):
    action = determine_action(prob)

    # Runtime spec verification
    assert_spec('NZ-1', prob > 0.55 and action == 'LONG')
    assert_spec('NZ-2', prob < 0.45 and action == 'SHORT')
    assert_spec('NZ-3', 0.45 <= prob <= 0.55 and action == 'NO_TRADE')

    return action
```

**Impact**: Immediate detection of spec violations in production.

---

### 4. Spec Change Management

**Current Gap**: No formal process for spec changes.

**Improvement**:
```
Spec Change Process:
1. Propose change in RFC document
2. Impact analysis on existing tests
3. Review and approval by 2+ engineers
4. Implement code changes
5. Update TRADING_ENGINE_SPEC.md
6. Update SPEC_TEST_TRACE.md
7. Update tests
8. Bump spec_version
9. Deploy with version verification
```

---

### 5. Property-Based Testing

**Current Gap**: Only example-based tests.

**Improvement**: Hypothesis-based property testing.

```python
from hypothesis import given, strategies as st

class TestNeutralZoneProperties:
    @given(prob=st.floats(min_value=0.45, max_value=0.55))
    def test_neutral_zone_always_no_trade(self, prob):
        action = determine_action(prob)
        assert action == 'NO_TRADE'

    @given(prob=st.floats(min_value=0.551, max_value=1.0))
    def test_above_neutral_always_bullish(self, prob):
        action = determine_action(prob)
        assert action in ['LONG', 'BUY_CALL']
```

**Impact**: Discover edge cases automatically.

---

### 6. Visual Spec Documentation

**Current Gap**: Specs are text-only.

**Improvement**: Mermaid diagrams embedded in specs.

```markdown
## Neutral Zone Decision Flow

graph TD
    A[prob input] --> B{prob > 0.55?}
    B -->|Yes| C[BULLISH]
    B -->|No| D{prob < 0.45?}
    D -->|Yes| E[BEARISH]
    D -->|No| F[NO_TRADE]
```

**Impact**: Clearer spec visualization.

---

### 7. Mutation Testing

**Current Gap**: Test coverage doesn't guarantee test quality.

**Improvement**: Mutation testing with mutmut.

```bash
mutmut run --paths-to-mutate=server/

# Results:
# Mutants killed: 95%
# Mutants survived: 5% (need more tests)
```

**Impact**: Verify tests actually catch bugs.

---

## Priority Matrix

| Improvement | Complexity | Impact | Priority |
|-------------|------------|--------|----------|
| Phase 2-4 unit tests | Low | High | P0 |
| Automated spec validation | Medium | High | P1 |
| Runtime verification | Medium | Medium | P2 |
| Spec change management | Low | Medium | P1 |
| Property-based testing | Medium | Medium | P2 |
| Visual documentation | Low | Low | P3 |
| Mutation testing | Medium | Medium | P2 |

---

## Dependencies

| Improvement | Requires |
|-------------|----------|
| Phase 2-4 tests | Test framework (existing) |
| Automated validation | Spec parser |
| Runtime verification | Logging infrastructure |
| Spec change management | Process documentation |
| Property-based testing | Hypothesis library |
| Visual documentation | Mermaid renderer |
| Mutation testing | mutmut or similar |

---

## Test Gap Analysis

| Category | Current Tests | Missing Tests |
|----------|---------------|---------------|
| Phase 1 | 1 (test_phase1.py) | Level calculation |
| Phase 2 | 0 | All scoring logic |
| Phase 3 | 0 | All density logic |
| Phase 4 | 0 | All posture logic |
| Northstar | 0 | Pipeline integration |
| RPE Engine | 0 | Full engine tests |

---

## Spec Evolution Strategy

### Version Numbering

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| Bug fix | spec_version date | 2026-01-03 → 2026-01-04 |
| Minor feature | engine minor | V6.1 → V6.2 |
| Breaking change | engine major | V6.1 → V7.0 |

### Backward Compatibility

```python
# Support old clients
if client_version < "V6.0":
    return legacy_response(signal)
else:
    return modern_response(signal)
```

---

*Spec & Test Governance improvements focus on coverage, automation, and process.*
