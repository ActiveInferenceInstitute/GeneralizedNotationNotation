# gnn-worker REPORT — src/gnn module fleet3 (2026-09-04)

Worker scope: `src/gnn/` (+ `src/3_gnn.py`, unchanged — already 32 lines, thin-orchestrator compliant).

## Files changed + why

### Deletions (dead code, zero importers verified by repo-wide grep before removal)
- **`src/gnn/roundtrip_processor.py` (1047 LOC deleted)** — entire module dead: its three public functions (`process_gnn_folder`, `run_gnn_round_trip_tests`, `validate_gnn_cross_format_consistency`) were reimplemented in `processors.py`, which `mcp.py` actually imports. All live references resolve to `gnn.processors`.

### Refactor (dedup / composability)
- **`src/gnn/parsers/common.py`** — `BaseGNNParser` now owns the embedded `MODEL_DATA` machinery: `EMBEDDED_JSON_PATTERNS: ClassVar[list[str]]` (per-parser comment regexes), shared `_extract_embedded_json_data` (delegates to existing `common.extract_embedded_json_data`), shared strict `_parse_from_embedded_data`, `EMBEDDED_LENIENT_MODEL_NAME: ClassVar[str]` + shared lenient `_parse_embedded_data_lenient`.
- **7 strict-variant parsers migrated** (~620 LOC of verbatim copies deleted): `scala_parser.py`, `lean_parser.py`, `coq_parser.py`, `isabelle_parser.py`, `python_parser.py`, `functional_parser.py`, `temporal_parser.py` (TLA + Agda). Each now declares `EMBEDDED_JSON_PATTERNS` only. Migrated via workpool subagents after per-file semantic-identity verification (all reported verbatim-identical; zero external references to the deleted private methods).
- **3 lenient-variant parsers migrated** (~210 LOC): `grammar_parser.py` (BNF + EBNF classes — EBNF's copy is live via dynamic dispatch, both classes got their own `EMBEDDED_LENIENT_MODEL_NAME`), `maxima_parser.py`. Call sites renamed to `_parse_embedded_data_lenient`.
- **`xml_parser.py`** — same-name `_parse_from_embedded_data` with a *different* contract (lenient, returns model) renamed to `_build_model_from_embedded_data` — fixes mypy LSP override clash created by the new base methods; zero external callers (grep-verified).
- **`xml_parser.py` note**: its `_extract_embedded_json_data` keeps a local XML-comment pattern; behavior unchanged.
- **`src/gnn/parsers/unified_parser.py`** — 27-branch `_get_parser_class` if/elif replaced by `_PARSER_CLASS_PATHS: Dict[GNNFormat, tuple[str, str]]` table + `importlib` (also covers PKL, previously a hard `ParseError` — parity with `PARSER_REGISTRY` now 23/23, test-pinned); `clear_parser_cache` typed `-> None`; the content-sniffing heuristic extracted as public `detect_gnn_format_from_content` (logic byte-identical; the method delegates).
- **`src/gnn/pomdp_extractor.py`** — `POMDPExtractor._nested_shape` now delegates to module-level `_shape_of` (single implementation of the twin).
- **`src/gnn/multi_format_processor.py`** — fixed latent `NameError` on `ext` in the per-format failure path (computed `ext` outside the `try`).
- **`src/gnn/processor.py`** — `generate_gnn_report` now honors its documented-but-ignored `output_path` parameter (write + parent mkdir); no caller passed it (grep-verified), so zero behavior change for existing callers.
- **`src/gnn/parsers/maxima_parser.py`** — same lenient migration; unused `extract_embedded_json_data` import dropped.
- **`src/gnn/extract.py`, 3 pre-existing-unformatted test files** — `ruff format` applied (unformatted at HEAD; they are in-scope files failing the format gate).

### Additive functionality
- **`GNNParsingSystem.convert_file`** (`parsers/system.py`) — one-call file-to-file conversion with extension-inferred formats, explicit overrides, parent-dir creation, typed error contract. Smoke-tested md→json/md→yaml.
- **`detect_gnn_format_from_content(content) -> GNNFormat`** (`parsers/unified_parser.py`) — public pure content sniffing, MARKDOWN fallback, never raises.
- **PKL dispatch parity** — `UnifiedGNNParser` now resolves PKL instead of raising.

### Tests
- **`src/tests/gnn/test_gnn_convert.py` (new, 17 tests)** — pins `detect_gnn_format_from_content` (7: xml/pnml/json/markdown/unknown-fallback/empty/coq), `convert_file` (8: success, reparse model-name equality, explicit target format override, unknown extension ValueError, missing input FileNotFoundError, failed-parse ParseError, parse-only target PNML ValueError, nested parent creation), `_PARSER_CLASS_PATHS` parity with `PARSER_REGISTRY` (2). Deterministic, tmp_path-based, no network.

### Docs of record
- **`src/gnn/AGENTS.md`** — new "Format Detection and File Conversion" section (public detector + `convert_file` contract) and "Shared Embedded-Model-Data Mechanism" section (class attrs + strict/lenient variants + which parsers keep specialized implementations); Last Updated bumped.
- **`src/gnn/README.md`** — phantom API `run_comprehensive_gnn_testing` (never existed) replaced with the real `process_gnn_folder` example; new "Format Conversion" snippet.
- `src/gnn/SPEC.md`, `parsers/` subdocs — checked; no stale references to removed/changed symbols (grep-verified).

## API deltas (all additive or private; no existing public entry point changed behavior)
| Surface | Delta |
|---|---|
| `GNNParsingSystem.convert_file` | NEW method |
| `gnn.parsers.unified_parser.detect_gnn_format_from_content` | NEW public function (also in module `__all__`) |
| `BaseGNNParser._extract_embedded_json_data` / `_parse_from_embedded_data` / `_parse_embedded_data_lenient` | NEW shared implementations (private-by-convention) |
| `BaseGNNParser.EMBEDDED_JSON_PATTERNS` / `EMBEDDED_LENIENT_MODEL_NAME` | NEW ClassVar extension points |
| `XMLGNNParser._parse_from_embedded_data` | renamed `_build_model_from_embedded_data` (private, 0 external callers) |
| `generate_gnn_report(..., output_path=X)` | now writes the report to X (previously silently ignored) |
| `UnifiedGNNParser` PKL parsing | previously `ParseError`, now works (parity fix) |
| `roundtrip_processor` module | REMOVED (dead) |

## Verification output tails (canonical commands)
```
uv run ruff check src/gnn src/tests/gnn     → All checks passed!
uv run --extra dev mypy src/gnn --config-file pyproject.toml
                                            → Success: no issues found in 82 source files
uv run pytest src/tests/gnn/ -q             → 404 passed in 1.47s
                                              (baseline 387 + 17 new; `just` is not
                                              installed on this host — ran the
                                              Justfile's exact recipe command
                                              `uv run pytest src/tests/gnn/ -q` instead)
uv run ruff format --check src/gnn src/tests/gnn → 107 files already formatted
```
Baseline before work: ruff+mypy clean, 387 tests passing at HEAD f64ac9085.

## Follow-ups for doc/ and manuscript/ owners (other workers)
- `doc/gnn/modules/03_gnn.md` documents `generate_gnn_report(processing_results, output_path=None)` — behavior changed slightly (output_path now honored); worth a one-line note.
- `src/gnn/parsers/` subdocs don't mention the embedded-data mechanism; if you keep per-parser docs, add the `EMBEDDED_JSON_PATTERNS` extension point.
- `src/gnn/parse_cache.py` and `contracts.py` have zero production callers (ParseCache is test-hardened, kept; `contracts.validate_rendered_output` has no production callers) — needs an owner decision (wire up or remove).
- `README.md:488` phantom-API fix is inside my scope, but doc/gnn/modules/03_gnn.md mirrors parts of the README — cross-check for the same stale examples there.

## Follow-up ideas (ranked, from the 8-lens audit — largest were executed this session)
1. **Serializer dedup (~1300 LOC)**: 15 serializers copy-paste the ~55-line embedded-dict block; `base_serializer._create_embedded_model_data`/`_add_embedded_model_data` already exist. Also remove `datetime.now()` from markdown/python/pkl serializers for deterministic golden outputs (check pinned "Generated:" tests first).
2. **Discovery unification**: four ad-hoc policies (processor.discover_gnn_files, processors.py glob, discovery.FileDiscoveryStrategy, roundtrip-era globs) → one parameterized helper in `parsers/common.py`.
3. **Validation funnel**: three divergent format-sniffers (validation.py, simple_validator.py, schema_validator.py) and five result shapes; converge on `schema.py`'s coded errors.
4. **`schema_validator.py` split (1609 LOC)**: GNNParser vs GNNValidator into separate modules; rename `parsers/validators.GNNValidator` to kill the three-way name collision.
5. **unified_parser._detect_format_from_content**: 20-arm if/elif → ordered predicate table.
6. **pomdp_extractor**: split `extract_from_gnn_content` (~260 lines) into assemble_discrete/continuous; make `_ExtractionContext` replace instance-mutable parse state (reentrancy).
7. **Latent defects spotted by audit, not in executed scope**: `roundtrip_processor` dead code carried a salted-`hash()` content_hash (now gone with the file); `parse_gnn_content(format_hint)` MCP param accepted but unused (schema-visible; needs manifest-coordinated change).

## Environment notes for the coordinator
- `just` is not installed (`just test-mod gnn` unrunnable); the equivalent recipe command was run.
- A stale non-editable `gnn` copy exists in `.venv/lib/python3.11/site-packages/gnn` — harmless under pytest (conftest prepends `src/`) but `python -c "import gnn"` resolves the stale copy. Worth re-running `uv sync` when the fleet drains (I did NOT touch uv.lock per fleet rules).
- `src/utils/pipeline_validator.py` had a transient syntax error mid-session (peer in-flight edit); resolved by the time of final verification.
- mcp.py lazy-capability refactor was attempted and **reverted to HEAD** per advisory: `validate-mcp-manifest`/`mcp-selftest` gates cannot be run concurrently, payoff unproven. Tracked as follow-up idea #0 for a solo session: mcp.py:26-46 import-time try/except + warning is the only remaining import-time side effect in the module.
