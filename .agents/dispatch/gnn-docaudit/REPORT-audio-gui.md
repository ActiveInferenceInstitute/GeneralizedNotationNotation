# REPORT-audio-gui — GNN Docs-vs-Code Audit

Region: doc/audio/, doc/sapf/, doc/pedalboard/, doc/gui_oxdraw/, doc/other/
Scope: REPORT-ONLY (no file modified). Verification via `git ls-files`, `wc -l`, `grep`, `ls`.

## Verified-ok baseline (not findings)
These documented repo paths/APIs exist and are well-formed:
- `gnn.processor.parse_gnn_file` (src/gnn/processor.py:214), `gnn.processor.discover_gnn_files` — real.
- `ontology.processor.load_defined_ontology_terms` (src/ontology/processor.py:355), `validate_annotations` (462) — real.
- `utils.pipeline_template.create_standardized_pipeline_script` (src/utils/pipeline_template.py:203), `pipeline.config.get_output_dir_for_script` (src/pipeline/config.py:136) — real.
- src/audio/ module: `process_audio`, `generate_audio_from_gnn`, `create_sonification`, `generate_audio_from_sapf`, `analyze_audio_characteristics`, `generate_audio_summary` all exist (src/audio/processor.py, __init__.py).
- src/sapf/ re-exports match doc claims: `convert_gnn_to_sapf`, `process_gnn_to_audio`, `generate_audio_from_sapf`, `validate_sapf_code`, `get_module_info` (src/sapf/__init__.py).
- src/audio/sapf/ implements `generate_sapf_audio`, `process_gnn_to_audio`, `create_sapf_visualization`, `generate_sapf_report`, `convert_gnn_to_sapf`, `validate_sapf_code` (processor.py, sapf_gnn_processor.py).
- oxdraw MCP registers 5 tools — matches doc count.
- sapf/apf MCP registers 4 tools.
- doc/pedalboard/pedalboard_gnn.md and doc/sapf/sapf_gnn.md and pedalbook.md/sapf.md/oxdraw.md use the EXTERNAL library/language APIs (pedalboard, SAPF language, cargo/oxdraw CLI) — correct, no repo path claims to flag.
- doc/other/ prose dialogs reference "SAPF"/"sonify" only as creative prose, no repo-paths.

---

## doc/gui_oxdraw/gnn_oxdraw.md

`doc/gui_oxdraw/gnn_oxdraw.md:78 | ERROR | Document claims module "src/gnn/mermaid_converter.py"; git ls-files shows NO src/gnn/mermaid* file exists. Actual implementation is src/gui/oxdraw/mermaid_converter.py (tracked).`
-> Fix: point the module path (and all `src/gnn/mermaid_*` refs) to `src/gui/oxdraw/mermaid_converter.py`.

`doc/gui_oxdraw/gnn_oxdraw.md:355 | ERROR | Documented module "src/gnn/mermaid_parser.py" does not exist; actual is src/gui/oxdraw/mermaid_parser.py (tracked).`
-> Fix: correct the documented path.

`doc/gui_oxdraw/gnn_oxdraw.md:95 | ERROR | Code listing imports "from gnn.parser import parse_gnn_file, ParsedGNN"; `parse_gnn_file` is NOT defined in src/gnn/parser.py (grep "def parse_gnn_file" in parser.py = empty); it lives in src/gnn/processor.py. Import would fail.`
-> Fix: change to `from gnn.processor import parse_gnn_file` (and import ParsedGNN from gnn.types/__init__).

`doc/gui_oxdraw/gnn_oxdraw.md:371 | ERROR | Code listing imports "from gnn.parser import ParsedGNN" and constructs `ParsedGNN(model_name=..., version=..., variables=..., connections=..., parameters=..., ontology_mappings=...)` positionally. Actual src/gui/oxdraw API uses `gnn_to_mermaid(gnn_model: Dict[str, Any])` dict-based (confirmed src/gui/oxdraw/mermaid_converter.py:16) and `mermaid_to_gnn(...) -> Dict[str, Any]`; there is no such fielded dataclass constructor in this module's API. Listing is aspirational/mismatched.`
` -> Fix: rewrite listing to the dict-based API actually implemented in src/gui/oxdraw/, or drop the fabricated signature.`

`doc/gui_oxdraw/gnn_oxdraw.md:823,864,888 | ERROR | Documents package "src/oxdraw_integration/" (`__init__.py`, `processor.py`). No `src/oxdraw_integration/` directory exists (git ls-files empty). Actual integration lives under src/gui/oxdraw/.`
`-> Fix: remove the src/oxdraw_integration section or repoint to src/gui/oxdraw/processor.py + __init__.py.`

`doc/gui_oxdraw/gnn_oxdraw.md:904 | INFO | `from gnn.processor import discover_gnn_files, parse_gnn_file` is correct and exists; only the surrounding illustration uses the nonexistent src/oxdraw_integration package.`
`(context note; not a defect by itself)`

## doc/gui/oxdraw/VERIFICATION.md

`doc/gui_oxdraw/VERIFICATION.md:25 | WARNING | Header "Core Implementation (src/oxdraw/)" — actual source path is src/gui/oxdraw/ (git ls-files confirms).`
-> Fix: correct the path prefix to src/gui/oxdraw/.

`doc/gui_oxdraw/VERIFICATION.md:29-36 | WARNING | Per-file line counts do not match actual files: __init__.py claimed 72 vs actual 162; processor.py 230 vs 367; mermaid_converter.py 345 vs 369; mermaid_parser.py 430 vs 430 (ok); utils.py 283 vs 257; mcp.py 185 vs 328; AGENTS.md 520 vs 565; README.md 290 vs 315 (verified wc -l).`
-> Fix: refresh all "Lines" cells with actual `wc -l`.

`doc/gui_oxdraw/VERIFICATION.md:299 | ERROR | Checklist claims "[x] Thin orchestrator script (24_oxdraw.py)"; no `src/24_oxdraw.py` exists (git ls-files shows the GUI orchestrator is src/22_gui.py, 96 lines).`
-> Fix: reference `22_gui.py` (oxdraw option) or drop the claim.

`doc/gui_oxdraw/VERIFICATION.md:197-198 | ERROR | Repeats `src/gnn/mermaid_converter.py` / `src/gnn/mermaid_parser.py` (full implementation) — neither exists at src/gnn/.`
-> Fix: correct to src/gui/oxdraw/ paths.

`doc/gui_oxdraw/VERIFICATION.md:61,63-65 | WARNING | Test file claims are wrong. `test_oxdraw_integration.py` is under src/tests/gui/ (not "src/tests/"); `test_mermaid_converter.py` and `test_mermaid_parser.py` are under src/tests/visualization/. Stated test counts also diverge: test_oxdraw_integration.py actual 465 lines / 15 "def test_", claimed 380/14; test_mermaid_converter.py actual 399/26, claimed 360/39; test_mermaid_parser.py actual 402/27, claimed 390/16.`
-> Fix: correct test-file paths and recount.

`doc/gui_oxdraw/VERIFICATION.md:67 | INFO | "Total 69 test cases" — actual 15+26+27 = 68 `def test_` (grep -c).`
-> Fix: update the total.

`doc/gui_oxdraw/VERIFICATION.md:143-146,252-256 | OK | Claimed real methods gnn.processor.parse_gnn_file, discover_gnn_files, ontology load_defined_ontology_terms, validate_annotations, utils.pipeline_template.create_standardized_pipeline_script, pipeline.config.get_output_dir_for_script — all exist (verified). No finding.`

## doc/pedalboard/

`doc/pedalboard/AGENTS.md:130 (and README.md:11 "Production Ready") | WARNING | Doc declares Pedalboard integration "Production Ready" and lists module-level functions `generate_audio_from_gnn` (L69) and `apply_audio_effects` (L82) as if implemented by the repo's audio.pedalboard module. Actual `src/audio/pedalboard/` is scaffold-only — it contains NO `.py` files (git ls-files: only AGENTS.md, README.md, SPEC.md), and its own README states "scaffolded but not yet implemented... no Python code". The doc-agents/advertised .py API does not exist in the tree.`
-> Fix: mark the doc Status as scaffold/planned and drop or re-label the module-level function signatures, or implement src/audio/pedalboard/.

`doc/pedalboard/README.md:37 | INFO | "Files: 3 | Subdirectories: 1" — actual dir has 5 .md files (AGENTS.md, README.md, SPEC.md, pedalboard_gnn.md, pedalboard.md); no subdir.`
-> Fix: update the file/subdir count.

`doc/pedalboard/README.md:38-53 | OK | Core Files list is accurate (pedalboard_gnn.md, AGENTS.md, README.md).`

## doc/sapf/

`doc/sapf/README.md:38 | INFO | "Files: 3 | Subdirectories: 0" — actual dir has 5 .md files (AGENTS.md, README.md, SPEC.md, sapf_gnn.md, sapf.md).`
-> Fix: update the count and list.

`doc/sapf/AGENTS.md (metadata "Contents Files: 3 | Subdirectories: 1") | INFO | Same count drift; actual files 5, subdirs 0.`
-> Fix: update.

## doc/audio/

`doc/audio/AGENTS.md:14 | INFO | SAPF acronym expansion "Sonified Active Inference Parameter Framework" conflicts with the source module's own expansion. src/audio/README.md uses "Synthetic Audio Processing Framework"; src/audio/AGENTS.md uses "Structured Audio Processing Format". No matching expansion in src for "Sonified Active Inference Parameter Framework".`
-> Fix: align the acronym expansion with src/audio (pick one canonical expansion).

`doc/audio/README.md:15,36 | INFO | "Contents: Files 1 | Subdirectories 0" — actual doc/audio has 3 .md files (README.md, AGENTS.md, SPEC.md). No subdir.`
-> Fix: update count to include AGENTS.md/SPEC.md (or clarify the convention of counting only content files).

## doc/other/

`doc/other/README.md:38 | INFO | "Contents: Files 10+ | Subdirectories 0" — actual doc/other has 43 .md files and 19 subdirectories (ls). AGENTS.md says "Files 9 | Subdirectories 1", also stale.`
-> Fix: recount the directory contents.

`doc/other/README.md:28,91,111 | INFO | Self-referential "Archive ../other/README.md" points back to the same file (not to a distinct "Archive" target). Purely cosmetic; does not break.`
-> Fix: point to a real sibling or drop the entry.

## Regions that are clean
- doc/audio/README.md: all cross-references (src/audio/README.md, ../sapf/, ../pedalboard/, ../CROSS_REFERENCE_INDEX.md, ../gnn/) resolve; no commands/imports to flag.
- doc/sapf/sapf.md (external SAPF-language overview): env vars and code are the external sapf language; no repo-path/import claims to verify. No internal findings.
- doc/pedalboard/pedalboard.md (external Spotify library overview): external API only; no repo-path claims.
- doc/gui_oxdraw/oxdraw.md (external oxdraw CLI/Rust tool): external tool docs; no repo-path claims.
- doc/sapf/sapf_gnn.md: SAPF-language/GNN mapping prose; no repo paths.
- doc/pedalboard/pedalboard_gnn.md: external pedalboard API examples; no repo paths.
- doc/gui_oxdraw/README + AGENTS: reference src/gui/README.md (exists) and doc siblings (exist); no command/import claims to fail.

## Summary
- doc/gui_oxdraw/gnn_oxdraw.md and VERIFICATION.md carry the majority of genuine defects: fabricated module locations (src/gnn/mermaid_*, src/oxdraw_integration/), a nonexistent orchestrator (24_oxdraw.py), a wrong `gnn.parser` import, and stale line/test counts. These are documentation from an earlier "integration proposal" that was never placed at the documented paths (actual home: src/gui/oxdraw/).
- doc/pedalboard/ overstates a scaffold-only subsystem as Production Ready with nonexistent module functions.
- doc/sapf/, doc/audio/, doc/other/ have minor count/acronym-drift only.