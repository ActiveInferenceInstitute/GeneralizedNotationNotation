# E-gui — GNN documentation accuracy fixes (SAFE DOC-ONLY EDITS)

Repo: /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation

## Mission
Apply the documentation corrections listed below. These are SAFE, DOCUMENTATION-ONLY
edits: change markdown prose, code examples, import references, file paths, counts,
and doc metadata. Do NOT change any source code (.py), test logic, config, or
dependency. The fixes were verified against the current tree (imports resolve; paths
are tracked); your job is to apply them to the .md files exactly.

## Rules
- Edit ONLY the .md files named below. Do NOT touch src/**, scripts/**, pyproject.toml,
  tests/**, or any .py.
- Preserve surrounding formatting/markdown. Make the minimal change (one line / one
  token) per fix.
- For imports: replace the broken module path with the verified-correct one below.
- For counts: set the number to the value stated below (verified by git ls-files/wc).
- For prose claims that are wrong/unverifiable and no exact replacement exists: reword
  minimally to be accurate (e.g. mark as illustrative, or remove the fabricated claim).
- HARD RULE: do NOT commit, stage, or push. Leave changes uncommitted.

## Specific fixes to apply
FIXES:
1. doc/gui_oxdraw/gnn_oxdraw.md:78 — `src/gnn/mermaid_converter.py` -> `src/gui/oxdraw/mermaid_converter.py`
2. doc/gui_oxdraw/gnn_oxdraw.md:355 — `src/gnn/mermaid_parser.py` -> `src/gui/oxdraw/mermaid_parser.py`
3. doc/gui_oxdraw/gnn_oxdraw.md:95 — `from gnn.parser import parse_gnn_file, ParsedGNN` -> `from gnn.processor import parse_gnn_file` (ParsedGNN from gnn.types/__init__)
4. doc/gui_oxdraw/gnn_oxdraw.md:371 — ParsedGNN positional-constructor example -> rewrite to the dict-based API actually used: `gnn_to_mermaid(gnn_model: Dict[str, Any])` / `mermaid_to_gnn(...) -> Dict[str, Any]` (src/gui/oxdraw/mermaid_converter.py). Drop the fabricated fielded dataclass signature.
5. doc/gui_oxdraw/gnn_oxdraw.md:823,864,888 — package `src/oxdraw_integration/` does not exist -> repoint/remove the section; actual integration is src/gui/oxdraw/.
6. doc/gui_oxdraw/VERIFICATION.md:25 — "Core Implementation (src/oxdraw/)" -> src/gui/oxdraw/
7. doc/gui_oxdraw/VERIFICATION.md:29-36 — refresh line counts to actual wc -l (given: __init__.py 162, processor.py 367, mermaid_converter.py 369, mermaid_parser.py 430, utils.py 257, mcp.py 328, AGENTS.md 565, README.md 315)
8. doc/gui_oxdraw/VERIFICATION.md:299 — "24_oxdraw.py" does not exist -> reference `22_gui.py` (the GUI orchestrator) or drop
9. doc/gui_oxdraw/VERIFICATION.md:197-198 — src/gnn/mermaid_converter.py / mermaid_parser.py -> src/gui/oxdraw/ paths
10. doc/gui_oxdraw/VERIFICATION.md:61,63-65 — test file paths/counts: test_oxdraw_integration.py is in src/tests/gui/ (15 def test_), test_mermaid_converter.py / test_mermaid_parser.py are in src/tests/visualization/ (26 / 27 def test_). Update paths and counts.
11. doc/gui_oxdraw/VERIFICATION.md:67 — "Total 69 test cases" -> 68 (15+26+27)
12. doc/pymdp/pymdp_pomdp/INTEGRATION_SUMMARY.md (if also present) — fix same import as batch C item 6 (from src.execute.pymdp.execute_pymdp import batch_execute_pymdp)


## Verification
After editing, run:
- uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write   (must stay green)
- uv run --extra dev python scripts/check_repo_terminology.py --strict   (must stay clean — do NOT use banned words: legacy/stub/placeholder/deprecated)
- uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict

## Report
Write a concise report to /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation/.agents/dispatch/gnn-docfix/REPORT-fix-E-gui.md listing each file you changed and the specific
edit(s) applied. Reply with only the absolute path to your report.
