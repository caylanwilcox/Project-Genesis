# Reality Proof Engine (RPE) - Current State

## Overview

The Reality Proof Engine is the market structure observation and execution posture system. It provides a 5-phase pipeline that transforms raw market data into actionable execution context without making probabilistic predictions.

## Core Responsibility

RPE observes and classifies market conditions. It does not predict. It answers: "What is the market doing right now?" not "What will the market do?"

## What RPE Owns

| Ownership | Description |
|-----------|-------------|
| Market structure classification | Trend, range, balance states |
| Acceptance state at key levels | Whether price is accepted or rejected |
| Multi-timeframe continuity | Alignment across 1m, 5m, 15m, 1h |
| Participation metrics | Volume conviction, effort/result match |
| Failure pattern detection | Failed auctions, trapped traders |
| Signal health assessment | Data integrity, trend invalidation |
| Execution posture framing | Bias, play type, risk state |

## What RPE Does NOT Own

| Exclusion | Belongs To |
|-----------|------------|
| Probability predictions | V6 ML Model (Phase 5 only) |
| Position sizing calculations | Policy Engine |
| Entry/exit price targets | Policy Engine |
| Trading action decisions | Policy Engine |
| Data fetching from APIs | Predict Server |

## 5-Phase Pipeline Architecture

```
Phase 1: TRUTH (Market Structure Observation)
    ↓
Phase 2: SIGNAL_HEALTH (Data Integrity & Risk Assessment)
    ↓
Phase 3: SIGNAL_DENSITY (Spam/Clustering Control)
    ↓
Phase 4: EXECUTION_POSTURE (Trade Permission Framing)
    ↓
Phase 5: LEARNING_FORECASTING (ML Predictions - ONLY phase with ML)
```

## Inputs

| Input | Source | Format |
|-------|--------|--------|
| Minute bars | Polygon API | `[{o, h, l, c, v, t}, ...]` |
| Daily bars | Polygon API | `[{o, h, l, c, v, t}, ...]` |
| Current timestamp | System clock | ET timezone |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| Phase1Truth | Immutable struct | Direction, acceptance, range state, MTF continuity |
| Phase2Health | Immutable struct | Health score (0-100), tier, stand_down flag |
| Phase3Density | Immutable struct | Throttle level, density score, allowed signals |
| Phase4Posture | Immutable struct | Bias, execution mode, risk state, allowed flag |

## Guarantees and Invariants

### Invariant 1: No ML in Phases 1-4
Phases 1 through 4 are pure observation. They use only deterministic calculations on market data. No machine learning models are invoked until Phase 5.

### Invariant 2: No Repainting
All calculations use only bars where `timestamp < current_time`. Future data is never accessed. Once a phase output is computed for a given bar, it does not change.

### Invariant 3: Truth First
Phase 1 output depends only on market data, never on decisions or predictions. There are no circular dependencies.

### Invariant 4: Deterministic
Given the same input bars and timestamp, RPE produces identical output. There is no randomness in Phases 1-4.

## Component Dependencies

| Component | Depends On RPE For |
|-----------|-------------------|
| V6 ML Model | Session classification, market context |
| Policy Engine | Execution posture, risk state |
| Predict Server | Signal gating, stand_down decisions |
| Dashboard UI | Phase visualizations, health indicators |

## Current Implementation

| Module | Location | Lines | Purpose |
|--------|----------|-------|---------|
| rpe_engine.py | `/ml/rpe/` | ~2,400 | Core 5-phase engine |
| northstar_pipeline.py | `/ml/rpe/` | ~1,100 | Integration layer |
| acceptance.py | `/ml/rpe/` | ~230 | Phase 1: Acceptance logic |
| auction_state.py | `/ml/rpe/` | ~240 | Phase 1: Auction detection |
| levels.py | `/ml/rpe/` | ~310 | Phase 1: Key price levels |
| vwap.py | `/ml/rpe/` | ~115 | Phase 1: VWAP calculations |
| failures.py | `/ml/rpe/` | ~130 | Phase 2: Failure detection |
| beware.py | `/ml/rpe/` | ~260 | Phase 3: Risk conditions |
| compute.py | `/ml/rpe/` | ~480 | Phase 4: Execution computation |

## Production Status

- Phase 1: Fully implemented, tested via `/replay` endpoint
- Phase 2: Implemented, integrated via northstar_pipeline
- Phase 3: Implemented, integrated via northstar_pipeline
- Phase 4: Implemented, returns execution posture
- Phase 5: Delegates to V6 ML model

---

*Last updated: 2026-01-03*
