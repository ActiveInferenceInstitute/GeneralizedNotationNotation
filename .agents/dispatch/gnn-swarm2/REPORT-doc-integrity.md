# REPORT — doc-integrity (gnn-swarm2)

## Scope owned
- `doc/troubleshooting/` (add/edit pages only under this tree)
- `doc/quickstart.md`, `doc/START_HERE.md`, `doc/INDEX.md`, `doc/CROSS_REFERENCE_INDEX.md`

## Baseline
Confirmed clean before touching anything:
- `docs_audit.py --strict --check-anchors --no-write` → 0 broken links, 0 bad anchors, 0 gaps.
- `check_gnn_doc_patterns.py --strict` → no banned patterns.
- `check_repo_terminology.py --strict` → tree clean.
- Independently re-verified every relative link in the four owned top-level docs + all `doc/troubleshooting/` markdown: 187 local links, all resolve to existing targets.

## Changes made (broken cross-file references, all verified against the repo)

Root cause: several `doc/troubleshooting/*.md` pages referenced a retired/nonexistent
`src/gnn_type_checker` module, a nonexistent Python `GNNModel` class, and a nonexistent
`src/gnn.exceptions` module. Verified real targets: `validate_gnn_file`/`parse_gnn_file`
are exported from `src/gnn/__init__.py` (return dicts); `GNSSyntaxError` lives in
`src/gnn/types.py`; `gnn validate <file> --strict [--json]` is a real CLI subcommand;
`run_pipeline(target_dir=…, output_dir=…, steps=…, verbose=…) -> dict` is exported from
`src/pipeline`. Repaired every broken reference to a real, runnable surface:

1. **doc/troubleshooting/common_errors.md** — "Step 4: Use Validation Tools" now uses
   `from src.gnn import validate_gnn_file` with the real dict return shape
   (`result["is_valid"]`, `result["errors"]`, string errors) and `.md` path.
2. **doc/troubleshooting/debugging_workflows.md** —
   - "Step 1: Dimension Consistency Check" → real `parse_gnn_file` + `validate_gnn_file`.
   - "Step 3: Connection Consistency" → authoritative `uv run gnn validate … --strict [--json]`.
   - Resource-estimation block → `uv run python src/5_type_checker.py --target-dir … --estimate-resources --verbose`.
   - Two interactive-debugging snippets (GNNModel/TypeChecker) → real `parse_gnn_file`.
3. **doc/troubleshooting/error_taxonomy.md** — GNNModel debug snippet → real `parse_gnn_file`/`validate_gnn_file`.
4. **doc/troubleshooting/api_error_reference.md** —
   - "Basic Error Handling" `load_and_validate_model` → real `parse_gnn_file`/`validate_gnn_file`/`GNSSyntaxError`; removed fabricated GNNException/GNNValidationError/GNNRuntimeError/GNNModel/TypeChecker.
   - "Pipeline Error Handling" (subclassed nonexistent `src.pipeline.PipelineExecutor`, imported `src.gnn.exceptions.PipelineExecutionError`) → real `run_pipeline` retry wrapper keyed on returned dict.
   - "Context Manager" (fabricated `GNNModel`/`TypeChecker`/exception tree) → real `load_context` on `GNSSyntaxError`/`validate_gnn_file`.

Removed all references to `src.gnn_type_checker`, `src.gnn.exceptions`, `GNNModel`,
and retired `TypeChecker()` construction from the troubleshooting tree.

## No new page added
The mission permitted ONE genuine gap page but only if a real gap exists. Reviewing
coverage: cross-framework run guidance is already in eigen `quickstart.md` and
`doc/execution/FRAMEWORK_AVAILABILITY.md`; no export-duplication gap surfaced in my
scope. Adding a page would be padding, so none was added (per "do not pad").

## Verification (scoped)
- `uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write` → **green**: 0 broken links / 0 bad anchors / 0 gaps.
- `uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict` → **green**: no banned patterns.
- `uv run --extra dev python scripts/check_repo_terminology.py --strict` → **NOT currently green**, but exclusively due to **out-of-scope sibling-agent work**: 11 "stale-version wording" violations in `src/validation/consistency_checker.py` (modified in the working tree) and the untracked `src/tests/analysis/test_generate_cross_model_report.py`. Both paths are owned by other agents (validation module / tests-analysis), not by this scope. None of my four edited files contains the flagged token, and my baseline run of this gate reported the tree clean minutes earlier — the violations appeared from sibling uncommitted edits after baseline. Per the disjoint-scope HARD RULE I did not touch those files.

## Tests
No regression tests added: this was a documentation cross-reference repair with no
behavior pinned in `src/`; the scoped doc-integrity gates above are the enforcement
surface and remain green for every path in my ownership. All 4 edited files left
uncommitted.

## Uncommitted
All changes left uncommitted (no git add / commit / push / index write), per mission rule. Only the four `doc/troubleshooting/*.md` files in my ownership were modified.