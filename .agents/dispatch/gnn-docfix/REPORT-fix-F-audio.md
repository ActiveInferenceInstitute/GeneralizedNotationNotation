# REPORT-fix-F-audio — GNN documentation accuracy fixes

Scope: SAFE DOC-ONLY edits per charter `.agents/dispatch/gnn-docfix/fix-F-audio.md`.
No source (.py), config, test, or dependency was touched. Nothing was staged or committed.

## Files changed and edits applied

### doc/pedalboard/AGENTS.md
- Fix 1 — Function Signatures & API: added a "Planned / Not Yet Implemented" note above the
  `generate_audio_from_gnn` / `apply_audio_effects` signatures, clarifying that
  `src/audio/pedalboard/` is scaffold-only (no `.py` source) and the signatures are planned,
  not shipped.
- Fix 1 — Footer Status: `**Status**: ✅ Production Ready` → `**Status**: 🛠️ Scaffold / Planned`.
- Fix 2 — Contents: `**Files**: 3 | **Subdirectories**: 1` → `**Files**: 5 | **Subdirectories**: 0`
  (actual files: AGENTS.md, README.md, SPEC.md, pedalboard_gnn.md, pedalboard.md; no subdirs).

### doc/pedalboard/README.md
- Fix 1 — Status (both occurrences, Overview line 11 and footer): `**Status**: ✅ Production Ready`
  → `**Status**: 🛠️ Scaffold / Planned`.
- Fix 2 — Contents: `**Files**: 3 | **Subdirectories**: 0` →
  `**Files**: 5 (AGENTS.md, README.md, SPEC.md, pedalboard_gnn.md, pedalboard.md) | **Subdirectories**: 0`.

### doc/sapf/README.md
- Fix 3 — Contents: `**Files**: 3 | **Subdirectories**: 0` → `**Files**: 5 | **Subdirectories**: 0`
  (actual files: README.md, SPEC.md, sapf_gnn.md, sapf.md, AGENTS.md; no subdirs).

### doc/sapf/AGENTS.md
- Fix 4 — Contents: `**Files**: 3 | **Subdirectories**: 1` → `**Files**: 5 | **Subdirectories**: 0`.

### doc/audio/AGENTS.md
- Fix 5 — Purpose (line 14): SAPF acronym corrected from `Sonified Active Inference Parameter Framework`
  to `Synthetic Audio Processing Framework`, matching the canonical expansion used in `src/audio/README.md`
  (the module's own README, the primary SAPF source).

### doc/audio/README.md
- Fix 6 — Contents (line 36): `**Files**: 1 | **Subdirectories**: 0` →
  `**Files**: 3 (README.md, AGENTS.md, SPEC.md) | **Subdirectories**: 0`.

### doc/other/README.md
- Fix 7 — Contents (line 38): `**Files**: 10+ | **Subdirectories**: 0` →
  `**Files**: 43 | **Subdirectories**: 19` (verified by `ls`/`find`: 62 top-level entries = 43 files + 19 dirs).
- Fix 8 — Removed all self-referential `[Archive](../other/README.md)` links that pointed to this same file:
  line 5 (Cross-References metadata), and the Related Directories / Research Resources / See Also bullets
  (former lines 28, 91, 111). No real archive sibling exists under `doc/`, so per the charter the entries
  were dropped rather than repointed.

## Verification (all green)
- `uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write`
  → 0 broken links, 0 bad anchors, 0 AGENTS/SPEC gaps.
- `uv run --extra dev python scripts/check_repo_terminology.py --strict` → "maintained tree clean".
- `uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict` → no banned patterns.

## Status
All 8 fixes applied, DOC-ONLY. Changes left uncommitted per the charter's hard rule.