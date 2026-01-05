# Reality Proof Engine (RPE) - Architectural Roadmap

## Purpose

This document describes structural improvements required for the Reality Proof Engine to achieve production-grade reliability, maintainability, and extensibility.

---

## Gap 1: Phase 2-4 Test Coverage

### Current State
Phase 1 has integration tests via the `/replay` endpoint. Phases 2, 3, and 4 execute in production but lack dedicated unit tests that verify their invariants in isolation.

### Why This Matters
Without isolated tests, regressions in health scoring, density throttling, or posture classification can propagate undetected. A failure in Phase 2 (signal health) could silently corrupt Phase 4 (execution posture) without triggering any test failure.

### Required Improvement
Each phase must have dedicated unit tests that:
- Verify invariants (determinism, no repainting)
- Test boundary conditions (edge thresholds)
- Validate output contracts (required fields, valid ranges)

---

## Gap 2: Module Documentation

### Current State
The RPE modules (`acceptance.py`, `auction_state.py`, `levels.py`, etc.) have type annotations but lack architectural documentation explaining:
- Why each module exists
- What market concepts it encodes
- How it relates to adjacent modules

### Why This Matters
New engineers cannot understand the system without reading thousands of lines of code. The market concepts (acceptance, auction state, VWAP bands) are domain-specific and non-obvious.

### Required Improvement
Each module needs a header docstring explaining:
- Market concept being modeled
- Input/output contract
- Invariants maintained
- Relationship to the phase pipeline

---

## Gap 3: northstar_pipeline.py Complexity

### Current State
The `northstar_pipeline.py` file is 34KB (~1,100 lines) and serves as the integration layer between the core RPE engine and the predict server. It handles:
- Data transformation
- Phase orchestration
- Output formatting
- Error handling

### Why This Matters
A 1,100-line integration layer is difficult to maintain and test. Logic that belongs in specific phases may have migrated here over time.

### Required Improvement
The pipeline should be decomposed into:
- A thin orchestration layer (phase sequencing only)
- Phase-specific formatters (output transformation)
- A dedicated error handling module

---

## Gap 4: Configuration Externalization

### Current State
Thresholds and parameters (e.g., health score tiers, density throttle levels) are embedded in code as magic numbers.

### Why This Matters
Tuning the system requires code changes. A/B testing different configurations is not possible without code deployment.

### Required Improvement
Extract all tunable parameters to a configuration layer that:
- Defines parameter schemas with validation
- Supports environment-based overrides
- Logs active configuration at startup

---

## Gap 5: Phase Output Versioning

### Current State
Phase outputs do not include a version identifier. When the output format changes, downstream consumers have no way to detect or handle the change.

### Why This Matters
The Policy Engine and Dashboard UI depend on specific Phase output structures. A breaking change in Phase 1 output could silently break position sizing or UI rendering.

### Required Improvement
Each phase output must include:
- `phase_version: str` (semantic version)
- `schema_version: str` (output format version)
- Breaking changes require major version bump

---

## Gap 6: Replay Mode Parity

### Current State
The `/replay` endpoint supports historical replay for Phase 1 and Phase 4. Phases 2 and 3 execute but are not individually inspectable in replay mode.

### Why This Matters
Debugging historical signals requires understanding what each phase computed. Without Phase 2/3 visibility, diagnosing "why did the signal health degrade?" requires manual reconstruction.

### Required Improvement
The replay endpoint should return:
- Full Phase 2 output (health dimensions, tier, reasons)
- Full Phase 3 output (density mode, throttle, budget)
- Timestamps for each phase computation

---

## Gap 7: Failure Pattern Library

### Current State
Failure patterns (failed auctions, excess, trapped traders) are detected but the pattern library is not externalized. Adding new patterns requires code changes.

### Why This Matters
Market structure evolves. New failure patterns emerge. The system cannot adapt without engineering effort.

### Required Improvement
Create a failure pattern DSL (domain-specific language) that allows:
- Pattern definition in configuration
- Backtesting new patterns against historical data
- A/B testing pattern effectiveness

---

## Priority Matrix

| Gap | Severity | Effort | Priority |
|-----|----------|--------|----------|
| Phase 2-4 Test Coverage | High | Medium | P1 |
| Module Documentation | Medium | Low | P2 |
| northstar_pipeline Decomposition | Medium | High | P2 |
| Configuration Externalization | Low | Medium | P3 |
| Phase Output Versioning | Medium | Low | P2 |
| Replay Mode Parity | Low | Low | P3 |
| Failure Pattern Library | Low | High | P4 |

---

*This is an architectural roadmap, not a task list. Implementation details and timelines are not specified.*
