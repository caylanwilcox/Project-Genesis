# Deprecation Log

**Purpose:** Track all deprecated files, their replacements, and reasons for deprecation.

**Last Updated:** 2026-01-03

---

## Active Deprecations

| Date | Old Path | New Path | Reason | Status |
|------|----------|----------|--------|--------|
| - | - | - | No active deprecations | - |

---

## Deprecation Procedure Reference

When deprecating a file:

1. Verify ZERO active references
2. Move to `Deprecated/YYYY-MM/<original_path>/`
3. Add header comment with DEPRECATED, REPLACED_BY, REASON
4. Update this log
5. Verify tests pass

See `AI_CHANGE_STANDARD.md` for full procedure.

---

## Historical Deprecations

*No historical deprecations recorded yet.*

---

## Notes

- Files in `Deprecated/` are preserved for reference
- Do not delete deprecated files without explicit approval
- Deprecated folder structure mirrors original paths
