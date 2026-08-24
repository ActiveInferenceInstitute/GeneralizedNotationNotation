# REPORT — fix-A-dev: GNN documentation accuracy fixes (DOC-ONLY)

Repo: /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation

## Summary
Applied all 12 documentation fixes from the charter. Changes are **documentation-only**
(no `.py`, config, tests, or dependencies touched) and were left **uncommitted / unstaged**
(no `git add`/`commit`/`push`), per the HARD RULE.

Files changed (only `.md`):
1. `doc/development/README.md`
2. `doc/configuration/examples.md`
3. `doc/configuration/README.md`
4. `SKILL.md`

## Per-file edits applied

### `doc/development/README.md`
- **Fix 1** (line ~307): `from src.gnn.parser import parse_gnn_file` → `from src.gnn.processor import parse_gnn_file` (verified `parse_gnn_file` is defined at `src/gnn/processor.py:214`).
- **Fix 2** (line ~168): `from utils.logging_utils import setup_standalone_logging` → `from utils.logging.logging_utils import setup_standalone_logging` (verified path `src/utils/logging/logging_utils.py`).
- **Fix 3** (line ~362): pytest path `src/tests/unit/` → `src/tests/gnn/`.
- **Fix 4** (line ~369): `src/tests/performance/ --benchmark-only` → real path `src/tests/pipeline/ -v`; comment retitled "Run pipeline orchestration tests"; dropped `--benchmark-only`.
- **Fix 5** (line ~471): `src/tests/unit/test_specific.py` → `src/tests/gnn/test_gnn_overall.py` (verified file exists).
- **Fix 6** (line ~120): "171 pytest files" → "~323 pytest source files" (verified `find src/tests -name '*.py' | wc -l` = 323).
- **Fix 7** (lines ~287-300 tree; also line ~378 prose): rewrote the test-organization tree to the real layout (`src/tests/<module>/`, `src/tests/integration/`, `src/tests/helpers/`, `src/tests/test_data/`, `conftest.py`). Also updated the "Add tests" reference from `tests/unit/` and `tests/integration/` to `src/tests/<module>/` and `src/tests/integration/`.

### `doc/configuration/examples.md`
- **Fix 8**: added a prominent `> ⚠️ Illustrative examples only` blockquote at the top stating the YAML blocks are illustrative/aspirational and NOT auto-loaded; the only auto-loaded config is `input/config.yaml`, whose schema is described in `doc/configuration/README.md`. No longer claims loadability.

### `doc/configuration/README.md`
- **Fix 9** (line ~65): added `uv` to the enumerated config sections → "...contains `uv`, `testing_matrix`, `io`, `logging`, `validation`, `performance`, and `security` sections." Verified `uv:` is a real top-level key in `input/config.yaml` (line 43).

### `SKILL.md`
- **Fix 10** (line ~25): "Check all 8 renderer backends" → "Check all 9 renderer backends" (verified `src/render/framework_registry.py` and `src/render/SPEC.md` list 9: pymdp, rxinfer, activeinference_jl, jax, discopy, pytorch, numpyro, stan, bnlearn).
- **Fix 11** (line ~14): added `bnlearn` to the framework list in "When to Use This Skill".
- **Fix 12** (line ~22): `just   # List all 21 recipes` → `just   # List recipes` (dropped hardcoded count).

report.

## Verification results
All three charter-mandated gates pass:

| Gate | Command | Result |
|------|---------|--------|
| 1 | `uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write` | Exit 0 — Broken links 0, Bad anchors 0, AGENTS/SPEC gaps 0, all counts 0 |
| 2 | `uv run --extra dev python scripts/check_repo_terminology.py --strict` | Exit 0 — "maintained tree clean." (no banned words) |
| 3 | `uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict` | Exit 0 — "no banned patterns in doc/ and src/gnn/" |

Git state: The 4 edited files show as modified (`M`) and uncommitted. No `add`/`commit`/`push` was performed. (Other `M` files in the worktree — e.g. `doc/gnn/modules/16_analysis.md` — were already modified before this task and were NOT touched by these fixes.)