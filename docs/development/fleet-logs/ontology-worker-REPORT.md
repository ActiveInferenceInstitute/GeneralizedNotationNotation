# ontology-worker REPORT — GNN module fleet 3 (2026-09-04)

**Scope owned:** `src/ontology/` entirely (incl. `AGENTS.md`, `README.md`, `SKILL.md`, `SPEC.md`) + numbered orchestrator `src/10_ontology.py`.
**Repo:** `/Users/hum/Documents/GitHub/HumOS/projects/outside_of_hum/GeneralizedNotationNotation` (branch `main`, HEAD `f64ac9085`).
**Worker:** ontology-worker (one of 35 herdr tabs, wave dispatch in 7s).

## Deepening pass (post-report extension, same turn)

Following the advisory sweep, a second improvement pass landed on top of the work above:

1. **`validate_annotations` call site reads named fields** — the loop now consumes `classification.matched/.match_info/.key/.value/.comment/.reason` from the `TermClassification` NamedTuple instead of positional unpacking. The cross-annotation conflicting-key check ("annotation key maps to multiple ontology terms") stays in `validate_annotations` (documented inline: a per-annotation helper cannot see previously matched keys).
2. **`list_standard_ontology_terms_mcp` derives from the real vocabulary** — it claimed to be "canonical" but was a hard-coded 14-term dict. Now built from `load_defined_ontology_terms()` (64 terms, real descriptions), so it cannot drift from what `validate_annotations`/`extract_ontology_annotations_mcp` accept. Pins hold (`count > 10`; every description a non-empty string).
3. **`build_ontology_terms` rejects case-folded duplicates** — `["A", "a"]` now raises `ValueError: ontology terms are ambiguous when case-folded: 'A' and 'a'`, matching the `_build_term_lookup` invariant (`["A","a"]` previously collapsed silently in the casefolded lookup).
4. **New public class `OntologyTermIndex`** — prebuilt case-insensitive index for batch callers: construct via `OntologyTermIndex(terms)` / `.from_file(path)` / `.from_names(...)`; methods `lookup(value)` (O(1) case-insensitive membership, returns `{"name", **metadata}` or `None`), `known_terms()` (sorted canonical names), `validate(annotations)` (same contract as `validate_annotations`), `suggest(annotations)`, `len()`, `in`. Immutable after construction; delegates to the module-level pure functions. Exported from `ontology/__init__.py` and listed in `__all__`.
5. **`load_defined_ontology_terms(..., *, search_paths: Sequence[Path] | None = None)`** — dependency-injection hook over the hard-coded file lookup. Precedence: explicit `ontology_terms_file` (authoritative, fails closed with `FileNotFoundError`) → caller `search_paths` (warn-and-continue on misses) → built-in module-relative paths → built-in default term set. Default call signature unchanged.
6. **Tests +16** (now 102 total in `src/tests/ontology/`): `TestOntologyTermIndex` (9: construction, build-rule propagation, case-insensitive lookup/miss, `contains` incl. non-str, sorted `known_terms`, validate-contract parity with `validate_annotations`, suggest delegation, real-vocabulary `from_file`), `TestLoadSearchPathsDI` (3: custom paths resolve, explicit file leads over search paths, missing path falls back to defaults), `TestBuildCasefoldedDuplicates` (2: casefolded dup, stripped dup), `TestMcpCanonicalTermsList` (2: derived-from-vocabulary equality, non-empty string descriptions).
7. **Docs updated again** — AGENTS.md (`search_paths` precedence, casefold-dup rule, `OntologyTermIndex` class entry, MCP canonical-derivation note), README.md (loader signature, build rule, `OntologyTermIndex` in Convenience Classes), SKILL.md (import + Key Exports), SPEC.md (Features: batch index + injectable search paths).

### Verification (deepening pass)

```
$ uv run ruff check src/ontology src/tests/ontology
All checks passed!
$ uv run --extra dev mypy src/ontology --config-file pyproject.toml
Success: no issues found in 4 source files
$ uv run pytest src/tests/ontology/ -q
102 passed in 0.25s
$ # cross-module MCP-wrapper consumers
18 passed in 9.69s
```

Sanity script (functional, not just structural):
```
index size: 64 | hiddenstate in idx: True
lookup('hidenstate') -> None            # exact lookup only; suggestions handle typos
known[:3]: ['Action', 'ActionFactor1', 'BetaParameter']
mcp count: 64 | PolicyVector desc: 'A vector representing a specific sequence...'
casefold dup rejected: YES: ontology terms are ambiguous when case-folded: 'A' and 'a'
```

### Advisory resolution record

All advisory-reported breakages from mid-turn were verified fixed **before** this extension ran, with fresh evidence (`ast.parse` OK, targeted greps, test runs) — none required rework: `__all__` contains both `analyze_ontology_content` and `suggest_terms` (lines 148-149); SKILL.md has no orphaned code outside fences; the JSON dedup test counts only top-level keys (`root_keys`); processor.py has no stale loop tail or positional unpack; `known_terms()` restored after a partial-edit mishap and verified present.

### Additional follow-up for the fleet (new)

- **MCP `suggest_ontology_terms` tool**: deliberately NOT added — `src/mcp/validate_tools.py:106` and the doc tables pin the 4-tool name set and are outside ontology scope. `suggest_terms` remains a Python public API; adding the MCP wrapper is a one-line `register_tool` call if the MCP/docs owners approve it.

## Summary

Raised composability, functionality, and internal quality of `src/ontology` while preserving every existing public entry point's behaviour (verified: 50 pre-existing tests + 18 cross-module MCP-wrapper consumer tests still pass). Refactored the buried suggestion/term-index logic into pure, typed, reusable units; added four additive public functions; deduped the vocabulary JSON (canonical URIs restored); and fixed the MCP wrapper that validated against a stale hard-coded 10-term subset instead of the real 64-term vocabulary.

## Files changed + why

| File | Change | Why |
|---|---|---|
| `src/ontology/processor.py` | Added `ParsedAnnotation` NamedTuple, `TermClassification` NamedTuple, pure helpers `_build_term_lookup` / `_term_matches`, public `suggest_terms`, `analyze_ontology_content`, `summarise_coverage`, `build_ontology_terms`, `SUGGESTION_MAX_DISTANCE` constant. `process_gnn_ontology` now delegates to `analyze_ontology_content` (dedup of parse+load+validate). `validate_annotations` delegates to the pure helpers, reads named `TermClassification` fields in its loop, and preserves its output dict contract. `parse_annotation` returns the typed `ParsedAnnotation` (still a 3-tuple). `load_defined_ontology_terms` uses the module logger and gained the keyword-only `search_paths` DI (explicit file still fails closed → caller search paths warn-and-continue → built-in paths → default terms). `build_ontology_terms` additionally rejects case-folded duplicates. **New public class `OntologyTermIndex`** (prebuilt case-insensitive index: `OntologyTermIndex(terms)` / `from_file` / `from_names`, `lookup`, `known_terms`, `validate`, `suggest`, `len`, `in`). | Composability: pure functions, explicit typed interfaces, dependency injection, deduplication, cohesive small units. |
| `src/ontology/__init__.py` | `OntologyProcessor.process_ontology` and `OntologyValidator.validate` delegate to the new shared helpers (no duplicated pipeline). `__version__` 1.6.0 → 1.7.0. New exports in `__all__`: `analyze_ontology_content`, `suggest_terms`, `summarise_coverage`, `build_ontology_terms`, `ParsedAnnotation`, `SUGGESTION_MAX_DISTANCE`, `OntologyTermIndex`. | Single source of truth for parse→load→validate; additive typed API surface. |
| `src/ontology/mcp.py` | `extract_ontology_annotations_mcp` validates against `load_defined_ontology_terms()` case-folded instead of a hard-coded 10-term set (was diverging from the real 64-term vocabulary). `list_standard_ontology_terms_mcp` now derives its name→description map from the real 64-term vocabulary instead of a hard-coded 14-term dict, so it cannot drift from what `validate_annotations` accepts. Log line "5 tools" → "4 tools" (it registers exactly 4). | Correctness: MCP wrappers now in sync with the real vocabulary; log matches reality (pinned by `test_all_register_tools_have_logger_info`). |
| `src/ontology/utils.py` | `get_module_info()["version"]` "1.0.0" → "1.7.0" (synced with `__version__`). | Removed version drift. |
| `src/ontology/act_inf_ontology_terms.json` | Removed 2 duplicate keys (`Time` TEMP_000066, `VariationalFreeEnergy` TEMP_000065) that shadowed the canonical entries. 66 entries → 64 unique; canonical URIs restored (`VariationalFreeEnergy` → `obo:ACTO_000012`, `Time` → `obo:TEMP_000019`). | Data-quality fix: `json.load` silently keeps the last occurrence, so the TEMP duplicates were overriding canonical ACTO URIs in every loaded vocabulary. |
| `src/tests/ontology/test_ontology_composability.py` | **New file** (52 tests after the deepening pass): `ParsedAnnotation`/`parse_annotation` contract, `suggest_terms` (ranking/determinism/case-insensitivity/substring-distance-0), `analyze_ontology_content` (3-key shape, terms pass-through, parity with `process_gnn_ontology`), `summarise_coverage` (full/partial/pluralisation/real-result), `build_ontology_terms` (validate-compatible shape, rejects empty/duplicate/case-folded-duplicate names, case-insensitive validation), internal helpers (`_term_matches`/`TermClassification`/`_build_term_lookup`), vocabulary dedup invariant (top-level dup detection via `object_pairs_hook`, canonical URIs, 64 unique terms), `extract_ontology_annotations_mcp` real-vocabulary behaviour, `OntologyTermIndex` (9 tests), `search_paths` DI (3), casefolded-duplicate rejection (2), MCP canonical-terms derivation (2), public surface (`__all__`, version, utils sync, `SUGGESTION_MAX_DISTANCE`). | Pin real behaviour of the helpers, the data fix, and the deepening additions; deterministic, isolated, no network. |
| `src/ontology/AGENTS.md` | API Reference lists the new public functions + typed `parse_annotation`; `search_paths` DI precedence documented; `build_ontology_terms` casefold-dup rule; `OntologyTermIndex` in Public Classes; MCP `list_standard_ontology_terms` derivation note; version line "3.2.0 (module `__version__` 1.7.0)"; Last Updated 2026-09-04; Test Files lists all 4 test files. | Docs of record kept in sync with API. |
| `src/ontology/README.md` | New functions in the API list (incl. `search_paths` loader signature); **fixed Troubleshooting error** ("term matching is case-sensitive" → "case-insensitive (case-folded)"); `OntologyTermIndex` in Convenience Classes; test list updated. | Correctness of user-facing docs; `validate_annotations` has always matched case-insensitively. |
| `src/ontology/SKILL.md` | API import block + Key Exports expanded with all new symbols (incl. `ParsedAnnotation` and `OntologyTermIndex`); removed an orphaned duplicate code block. | Capability API reflects the real surface. |
| `src/ontology/SPEC.md` | Components lists `processor.py`/`utils.py`/`mcp.py`/`act_inf_ontology_terms.json`; Features adds suggestion, in-memory vocab construction, batch index, injectable search paths; Key Exports block expanded. | Spec reflects current components. |
| `src/10_ontology.py` | **Unchanged.** Already a thin orchestrator (57 lines, delegates to `process_ontology`). | No work needed; thin-orchestrator pattern already satisfied. |
| `docs/development/fleet-logs/ontology-worker.md` | Created; one checkpoint line per phase (audit/refactor/functionality/tests/docs). | Mandatory fleet checkpoint cadence. |

## API deltas

### New public functions (additive, typed, documented)
- `analyze_ontology_content(content: str, ontology_terms: Dict[str, Any] | None = None) -> Dict[str, Any]` — single pure parse→load→validate entry point; returns `{"ontology_data", "validation_result", "ontology_terms"}`.
- `suggest_terms(annotations: List[str], ontology_terms: Dict[str, Any] | None = None, *, max_distance: int = 3) -> List[Dict[str, Any]]` — nearest-term suggestions ranked closest-first; each `{"annotation", "suggested_term", "description", "distance"}`.
- `summarise_coverage(validation_result: Dict[str, Any]) -> str` — compact human-readable coverage line for report/LLM consumers.
- `build_ontology_terms(terms: List[str], *, descriptions: Dict[str, str] | None = None, uris: Dict[str, str] | None = None) -> Dict[str, Any]` — in-memory vocabulary builder (complement of `_normalise_ontology_terms`); rejects empty names, exact duplicates, and case-folded duplicates (same ambiguity message as the loader).

### New public class
- `OntologyTermIndex` — prebuilt case-insensitive index for batch callers: `OntologyTermIndex(terms)` / `.from_file(path)` / `.from_names(...)`, then `lookup(value)` (O(1), returns `{"name", **metadata}` or `None`), `known_terms()` (sorted canonical), `validate(annotations)` (identical contract to `validate_annotations`), `suggest(annotations)`, `len()`, `in`. Immutable after construction.

### Extended signature (additive, keyword-only)
- `load_defined_ontology_terms(ontology_terms_file: Path | None = None, *, search_paths: Sequence[Path] | None = None)` — DI hook over the file lookup; precedence: explicit file (fails closed) → caller search paths (warn-and-continue) → built-in paths → built-in default terms.

### New public types/constants
- `ParsedAnnotation` (NamedTuple: `key`, `value`, `comment`) — still a 3-tuple; `parse_annotation` now returns this.
- `SUGGESTION_MAX_DISTANCE = 3` — the Levenshtein threshold used by `suggest_terms`/`validate_annotations`.

### Behaviour-preserving internal refactor
- `validate_annotations` output dict shape **unchanged** (`valid_annotations`, `invalid_annotations`, `matched_terms` with `description`+`uri`, `suggestions` with `{annotation, suggested_term, description}`, `coverage_score`, `invalid_details`). `src/llm/processor.py` injects this verbatim into LLM prompts — contract preserved.
- `process_gnn_ontology` return keys **unchanged** (`success`, `file_path`, `ontology_data`, `validation_result`, `ontology_terms`).
- `OntologyProcessor.process_ontology` / `OntologyValidator.validate_ontology` / `check_consistency` — behaviour preserved (verified by 50 pre-existing tests).
- `extract_ontology_annotations_mcp` — output keys unchanged; validation widened from a 10-term hard-coded set to the real 64-term vocabulary (case-insensitive). Pinned test `valid_count == 2` still passes; real terms like `Policy` now validate correctly instead of landing in `unknown_terms`.

### Version
- `ontology.__version__` 1.6.0 → 1.7.0; `utils.get_module_info()["version"]` synced.

## Verification output tails

```
$ uv run ruff check src/ontology src/tests/ontology
All checks passed!

$ uv run --extra dev mypy src/ontology --config-file pyproject.toml
Success: no issues found in 4 source files

$ uv run pytest src/tests/ontology/ -q
102 passed   (50 pre-existing + 52 new; final deepening-pass run: "102 passed in 2.18s")
```

Cross-module consumer confirmation (outside `src/ontology`, run to prove no downstream regression from the MCP/version changes):
```
$ uv run --extra dev python -m pytest \
    src/tests/report/test_report_mcp_wrappers.py \
    src/tests/intelligent_analysis/test_intelligent_analysis_mcp_wrappers.py -q
18 passed in 6.99s
```

### `just test-mod ontology` note
The `just` binary is **not installed** on this host (`just: command not found`). The recipe (per `justfile:30-32`) is `uv run pytest src/tests/ontology/ -v`, which I ran directly and is shown above (102 passed). I did not install `just` (fleet rule: no dependency installs).

### Broader-sweep caveat (not in scope)
A wider run including `src/tests/test_core_modules.py` and `src/tests/pipeline/test_pipeline_scripts.py` showed 2 failures, both caused by a **`visualization`/`analysis` circular import** in `16_analysis.py` (`visualization.analysis.combined_analysis` partially-initialized module) when those tests shell out to step 16. This is entirely outside `src/ontology` — a peer-worker module — and is independent of my changes (the traceback contains no `ontology` frame). Flagging it here so it is not mistaken for an ontology regression.

## Doc / manuscript follow-ups needed (other workers own these)

1. **`doc/gnn/modules/10_ontology.md`** (docs worker) — per `doc/gnn/modules/SPEC.md:33`, the module doc version must stay in sync with `src/<module>/__init__.py` `__version__`. I bumped `ontology.__version__` to `1.7.0`; if `10_ontology.md` carries a version header it should be synced. (I could not edit `doc/` — outside my scope.)
2. **`doc/gnn/advanced/gnn_ontology.md:120-121`** documents the `validate_annotations` suggestion element shape `{annotation, suggested_term, description}` — still accurate; no change needed, but worth a glance since I touched that code path.
3. **`doc/gnn/mcp/tool_reference.md:58-61` / `doc/mcp/README.md:55-59`** list the 4 ontology MCP tools — still accurate (I did not add/rename/remove any MCP tool).
4. **Generated artifacts** (`output/21_mcp_output/registered_tools.json`, `src/mcp/audit_report.json`, `output/20_website_output/mcp.html`) are regenerated by step 21 and pin the 4 tool names/category — unaffected by my changes; they'll regenerate cleanly on the next pipeline run.
5. **`src/mcp/validate_tools.py:106`** hard-codes `list_standard_ontology_terms` in a callability spot-check — unaffected (name unchanged).

## Follow-up ideas (not done; out of scope or deferred)

- `load_defined_ontology_terms` still hard-codes a cwd-relative fallback path `Path("src/ontology/act_inf_ontology_terms.json")`. The packaged-file path (`Path(__file__).parent / ...`) already wins, so the fallback is dead in normal installs; could be removed for clarity, but it is a behaviour-preserving no-op and touching it risks the "zero regressions" contract. Left as-is.
- `validate_annotations` catches broad `Exception` and returns an error dict. This is a boundary function for the pipeline step and the contract is documented; narrowing the catch risks behaviour change. Left as-is.
- ~~A `TermIndex` class that pre-builds the case-folded lookup once for batch processing~~ — **DONE in the deepening pass** as `OntologyTermIndex` (public, exported, tested; though `validate`/`suggest` still delegate to the module functions — only `lookup`/`known_terms`/`in` reuse the prebuilt lookup).
- The `extract_ontology_annotations_mcp` parser only recognises the exact `## ActInfOntologyAnnotation` header (case-sensitive), while `parse_gnn_ontology_section` also accepts `## Ontology` case-insensitively. Aligning the two would widen extraction; deferred because the MCP wrapper test pins the exact-header behaviour and the divergence is pre-existing.

## 5-line terminal summary

- Refactored `src/ontology/processor.py` into pure, typed, composable units (`ParsedAnnotation`, `TermClassification`, `_build_term_lookup`, `_term_matches`, public `suggest_terms` + `analyze_ontology_content`); `process_gnn_ontology` and the `OntologyProcessor`/`OntologyValidator` classes delegate to one shared pipeline; the validate loop reads named `TermClassification` fields.
- Added public API: 4 functions (`analyze_ontology_content`, `suggest_terms`, `summarise_coverage`, `build_ontology_terms`), `OntologyTermIndex` batch class (lookup/known_terms/validate/suggest), `search_paths` DI on `load_defined_ontology_terms`; bumped `__version__` 1.6.0→1.7.0 (utils synced).
- Fixed 4 latent bugs: vocabulary JSON duplicate keys shadowing canonical ACTO URIs (64 unique, canonical restored); `extract_ontology_annotations_mcp` validated against a stale 10-term set; `list_standard_ontology_terms_mcp` claimed "canonical" from a hard-coded 14-term dict (now derived from the real 64-term vocabulary); MCP log "5 tools"→4.
- Added `src/tests/ontology/test_ontology_composability.py` (52 tests); **verify green**: `ruff` clean, `mypy --strict` clean (4 files), `pytest src/tests/ontology/` 102 passed, cross-module MCP-wrapper consumers 18 passed.
- Updated all 4 docs of record in scope (AGENTS/README/SKILL/SPEC) twice (base + deepening); flagged `doc/gnn/modules/10_ontology.md` version-sync and an opt-in `suggest_ontology_terms` MCP wrapper as follow-ups for their owners.