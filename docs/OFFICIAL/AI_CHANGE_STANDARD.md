# AI Change Standard - Governing Document

**Document Type:** Constitutional / Mandatory
**Last Updated:** 2026-01-03
**Authority:** This document governs ALL AI-assisted changes to this repository.

---

## Purpose

This document defines the rules, constraints, and procedures that any AI agent must follow when proposing changes to this codebase. It ensures that:

1. Human-authored truth is preserved
2. Changes are reviewable, not automatic
3. The system gets smarter without getting dangerous
4. Spec drift is impossible

---

## Hierarchy of Truth

| Priority | Document | Role | Override Rule |
|----------|----------|------|---------------|
| 0 (North Star) | `SYSTEM_VISION.md` | WHY - What work is worth doing | Chooses direction |
| 1 (Highest) | `TRADING_ENGINE_SPEC.md` | WHAT - Rules and invariants | Chooses safety (overrides P0 for behavior) |
| 2 | `SPEC_TEST_TRACE.md` | PROOF - Rule → Test mapping | Verifies P1 |
| 3 | `AI_CHANGE_STANDARD.md` | HOW - Change protocol | Governs process |
| 4 | `system_overview/` docs | UNDERSTANDING - Architecture | Advisory only |

**The North Star Goal:**
> Connect high-probability ML tested signals × frontend × tickers → high options win %

**Conflict Resolution:**
- **Vision** decides *what work is worth doing* (direction, prioritization)
- **Spec** decides *how the system behaves* (safety, runtime behavior)
- **Spec wins for runtime behavior. Vision wins for prioritization.**

Every change must serve the goal. If it doesn't improve win rate, it doesn't belong.

---

## Authored Baseline (Immutable)

The following are **human-authored truth** and may NEVER be modified by automation:

### Protected Files
```
docs/OFFICIAL/SYSTEM_VISION.md          (Priority 0 - WHY)
docs/OFFICIAL/TRADING_ENGINE_SPEC.md    (Priority 1 - WHAT)
docs/OFFICIAL/SPEC_TEST_TRACE.md        (Priority 2 - PROOF)
docs/OFFICIAL/AI_CHANGE_STANDARD.md     (Priority 3 - HOW)
docs/OFFICIAL/AI_REPO_GUARDIAN_PROMPT.md (DRIVER - Runs the loop)
ml/models/*.pkl                         (trained model artifacts)
```

### Protected Invariants
- V6 model weights and structure
- Spec rule definitions (NZ-*, BK-*, TM-*, etc.)
- Test-to-spec mappings
- Version lock constants

### If AI Wants to Change Protected Content
It MUST produce a **Spec Change Proposal** document instead:
```
docs/OFFICIAL/proposals/YYYY-MM-DD_proposal_<name>.md
```

The proposal must include:
- Current rule/behavior
- Proposed change
- Rationale
- Impact analysis
- Required test changes
- Spec version bump plan

Human approval required before any protected content changes.

---

## Change Categories

### Category A: Documentation Updates (Allowed)
- `system_overview/**/CURRENT.md` - If provable with evidence
- `system_overview/**/NEXT.md` - Improvement plans
- `system_overview/**/OUTLET.md` - Interface documentation
- `ARCHITECT_LOG.md` - Loop history

### Category B: Code Improvements (Patch-Only)
- Refactoring for clarity
- Bug fixes with test proof
- Dead code identification

**Output:** `.patch` file or proposal, never direct commit

### Category C: Spec-Impacting Changes (Proposal-Only)
- Any change that affects behavior
- New features
- Rule modifications

**Output:** Spec Change Proposal document

### Category D: Deprecation (Procedure Required)
- Moving unused files to `Deprecated/YYYY-MM/`
- Must follow deprecation procedure exactly

---

## Mandatory Change Process

For EVERY proposed change, the AI must explicitly produce:

### 1. INTENT
```
- Problem being solved: <description>
- Why necessary now: <urgency/impact>
- Spec rules affected: <IDs or "none">
```

### 2. SCOPE
```
- Files/modules touched:
  - path/to/file.py (reason)
  - path/to/other.py (reason)
- Dependencies affected: <list>
```

### 3. INVARIANTS
```
- Rules that MUST remain true:
  - NZ-1: prob > 0.55 → BULLISH (preserved)
  - DS-1: today_open = daily_bars[-1]['o'] (preserved)
- Invariants at risk: <list or "none">
```

### 4. FAILURE MODES
```
- On missing input: <behavior>
- On invalid data: <behavior>
- Must fail CLOSED: YES/NO
- Deterministic reason string: <example>
```

### 5. TESTS
```
- New tests required: <list>
- Existing tests affected: <list>
- SPEC_TEST_TRACE updates: <list>
```

### 6. EVIDENCE
```
- Files proving implementation: <paths>
- Tests proving behavior: <test names>
- Spec IDs verified: <IDs>
```

**No step may be skipped.**

---

## CURRENT.md Rules (Evidence-Based)

A fact may only appear in CURRENT.md if it can cite:

| Requirement | Example |
|-------------|---------|
| File path(s) | `ml/server/v6/predictions.py` |
| Function/class/endpoint | `get_v6_prediction()` |
| Spec rule ID (if applicable) | `NZ-1` |
| Test that proves it | `test_nz1_above_55_is_bullish` |

If ANY requirement is missing, the fact belongs in NEXT.md instead.

---

## NEXT.md Rules (Improvement Plans)

Each item in NEXT.md must include:

| Field | Description |
|-------|-------------|
| Intent | Why this improvement matters |
| Scope | Files/modules affected |
| Invariants | Spec IDs that must be preserved |
| Failure modes | How system fails if done wrong |
| Tests to add | Specific test names/descriptions |
| Completion criteria | Objective definition of "done" |
| Priority | P0 (critical) to P3 (nice-to-have) |

---

## OUTLET.md Rules (Interface Documentation)

Must include:

| Section | Content |
|---------|---------|
| Inputs | Schema/type of data received |
| Outputs | Schema/type of data produced |
| Upstream dependencies | What this component needs |
| Downstream consumers | What uses this component |
| Failure boundary | What happens on bad input |
| Determinism notes | Why same input → same output |
| Connection ladder | Component → Layer → Server → Output |

---

## Stop-the-Line Conditions

AI must STOP and request human guidance if:

- [ ] Invariant impact is unclear
- [ ] Tests cannot be added or updated
- [ ] Ambiguity increases (not decreases)
- [ ] Determinism is weakened
- [ ] Guardrails are removed or bypassed
- [ ] Protected content would be modified
- [ ] Spec version would need to bump
- [ ] Breaking change is required

When stopped, AI must produce:
```
STOP-THE-LINE REPORT
====================
Reason: <why stopped>
Missing information: <what's needed>
Recommended action: <what human should do>
```

---

## Deprecation Procedure

When a file/module is unused or replaced:

### Step 1: Verify Zero References
```bash
# Check for imports
grep -r "from old_module import" ml/
grep -r "import old_module" ml/

# Check for string references
grep -r "old_module" ml/
```

### Step 2: Identify Replacement
- New path (if replaced)
- "NONE" (if simply removed)

### Step 3: Move File
```
FROM: ml/policy/old_logic.py
TO:   Deprecated/2026-01/ml/policy/old_logic.py
```

### Step 4: Add Deprecation Header
```python
"""
DEPRECATED: 2026-01-03
REPLACED_BY: ml/server/v6/predictions.py
REASON: Logic consolidated into V6 prediction module
"""
```

### Step 5: Update Deprecation Log
Append to `docs/OFFICIAL/DEPRECATION_LOG.md`:
```markdown
| Date | Old Path | New Path | Reason |
|------|----------|----------|--------|
| 2026-01-03 | ml/policy/old_logic.py | ml/server/v6/predictions.py | Consolidation |
```

### Step 6: Verify Tests Pass
```bash
cd ml && python3 -m pytest tests/ -v
```

---

## Version Governance

### When spec_version Must Bump
- Any change to behavior (not just code structure)
- New rules added to TRADING_ENGINE_SPEC.md
- Rules modified or removed
- Output schema changes

### When engine_version Must Bump
- Major: Breaking changes, architectural shifts
- Minor: New features, improvements

### Format
```
spec_version: "YYYY-MM-DD"
engine_version: "V{major}.{minor}"
```

---

## Output Modes

### Default: Patch-First
All code changes should be delivered as `.patch` files:
```bash
git diff > patches/YYYY-MM-DD_description.patch
```

### Alternative: Proposal Document
For spec-impacting changes:
```
docs/OFFICIAL/proposals/YYYY-MM-DD_proposal_<name>.md
```

### Never: Direct Commit
AI agents must NEVER:
- `git commit`
- `git push`
- Modify files without producing review artifacts

---

## Architect Log

Every loop run should append to `docs/OFFICIAL/ARCHITECT_LOG.md`:

```markdown
## YYYY-MM-DD HH:MM - Run #N

### Analyzed
- <what was examined>

### Proposed
- <what changes were suggested>

### Verified
- <what tests passed>

### Remaining
- <what still needs attention>

### Artifacts Produced
- <patches, proposals, doc updates>
```

---

## Final Output Format

Every AI response must include these sections in order:

```
1) Change Plan
2) Deprecation Candidates (if any)
3) Move Map (from → to)
4) Shim Plan (if needed)
5) Tests to Run
6) Risks & Mitigations
```

No meta commentary. No philosophy summaries. Decisive, precise, architectural action.

---

*This document is the third pillar of system governance. Compliance is mandatory.*
