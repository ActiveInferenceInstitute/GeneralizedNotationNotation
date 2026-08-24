# REPORT — fix-E-gui: GNN documentation accuracy fixes (DOC-ONLY)

Repo: /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation
Charter: .agents/dispatch/gnn-docfix/fix-E-gui.md
Date: 2026-08-24
Scope: Documentation-only edits to `.md` files. No source, test, config, or dependency changes.
Git: Changes left uncommitted and unstaged (hard rule honored).

## Files changed

### 1. doc/gui_oxdraw/gnn_oxdraw.md

- FIX 1 (was line 78): Module path `src/gnn/mermaid_converter.py` -> `src/gui/oxdraw/mermaid_converter.py`
- FIX 2 (was line 355): Module path `src/gnn/mermaid_parser.py` -> `src/gui/oxdraw/mermaid_parser.py`
- FIX 3 (was line 95): Import `from gnn.parser import parse_gnn_file, ParsedGNN` -> `from gnn.processor import parse_gnn_file` (verified `parse_gnn_file` lives on `gnn.processor`)
- FIX 4 (was line 371 and surrounding): Rewrote the fabricated fielded-`ParsedGNN` API to the real dict-based API used by `src/gui/oxdraw/mermaid_converter.py` / `mermaid_parser.py`:
  - `gnn_to_mermaid(gnn_model: Dict[str, Any], include_metadata: bool = True) -> str` (was `ParsedGNN`), with body converted from attribute access (`gnn_model.model_name`, etc.) to dict access (`gnn_model["model_name"]`, etc.)
  - `mermaid_to_gnn(...) -> Dict[str, Any]` (was `-> ParsedGNN`)
  - `convert_mermaid_file_to_gnn(...) -> Dict[str, Any]` (was `-> ParsedGNN`)
  - Dropped the fabricated fielded `ParsedGNN(...)` constructor; now returns a plain dict. Updated docstrings that said "ParsedGNN model" -> "GNN model dict".
- FIX 5 (was lines 823, 864, 888 and 845, 853, 878): repointed the fabricated `src/oxdraw_integration/` package section to the actual `src/gui/oxdraw/` integration:
  - `### Module: src/oxdraw_integration/` -> `src/gui/oxdraw/`
  - `#### File: src/oxdraw_integration/__init__.py` -> `src/gui/oxdraw/__init__.py`
  - `#### File: src/oxdraw_integration/processor.py` -> `src/gui/oxdraw/processor.py`
  - Import `from oxdraw_integration.processor import process_oxdraw_gui` -> `from gui.oxdraw.processor import process_oxdraw`; `processing_function=process_oxdraw`; `def process_oxdraw(`; `"process_oxdraw"` in `__all__` (verified `process_oxdraw` is the real function in `src/gui/oxdraw/processor.py`).

### 2. doc/gui_oxdraw/VERIFICATION.md

- FIX 6 (was line 25): `### Core Implementation (src/oxdraw/)` -> `src/gui/oxdraw/`
- FIX 7 (was lines 29-36): refreshed module line counts to verified `wc -l` values — `__init__.py` 162, `processor.py` 367, `mermaid_converter.py` 369, `mermaid_parser.py` 430, `utils.py` 257, `mcp.py` 328, `AGENTS.md` 565, `README.md` 315. Updated the derived "Total Module Code" line (Python total now 1,913; docs 880).
- FIX 8 (was line 299): nonexistent `24_oxdraw.py` -> `22_gui.py` (the real GUI orchestrator).
- FIX 9 (was lines 197-198): `src/gnn/mermaid_converter.py` / `src/gnn/mermaid_parser.py` -> `src/gui/oxdraw/mermaid_converter.py` / `src/gui/oxdraw/mermaid_parser.py`
- FIX 10 (was lines 61, 63-65): corrected test file paths and counts (verified against the tree):
  - `src/tests/gui/test_oxdraw_integration.py` — 465 lines, 15 test functions
  - `src/tests/visualization/test_mermaid_converter.py` — 399 lines, 26 test functions
  - `src/tests/visualization/test_mermaid_parser.py` — 402 lines, 27 test functions
- FIX 11 (was line 67): "Total 69 test cases" -> 68 (15+26+27). Also updated all downstream derived counts for consistency: category headers (Integration 15/15, Converter 25/26, Parser 24/27), "Minor Test Issues 3/68", lead summary "65/68", Verification Checklist counts, module doc line counts in the Documentation section (AGENTS.md 565, README.md 315), and the closing Quantitative Metrics ("68 test cases").

### 3. doc/pymdp/pymdp_pomdp/INTEGRATION_SUMMARY.md

- FIX 12: No change required. The import already reads the verified-correct `from src.execute.pymdp.execute_pymdp import batch_execute_pymdp`, matching the real symbol in `src/execute/pymdp/execute_pymdp.py`. (File shows a pre-existing unrelated modification in the worktree from a sibling agent; not touched by me.)

## Verification (all from charter)

- `uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write` -> **GREEN** (0 broken links, 0 bad anchors, 0 gaps)
- `uv run --extra dev python scripts/check_repo_terminology.py --strict` -> **CLEAN**
- `uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict` -> **CLEAN** (no banned patterns)

No `.py`, config, or test files were modified. No commit/stage/push performed.