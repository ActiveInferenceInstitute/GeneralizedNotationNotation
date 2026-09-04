# render-worker REPORT — GNN module fleet 3 (2026-09-04)

Worker scope: `src/render/` (+ `src/11_render.py` orchestrator). Repo HEAD at start: `f64ac9085` (green).

## Files changed + why

| File | Change | Why |
|---|---|---|
| `src/render/naming.py` | **NEW** — `safe_output_stem()` (was duplicated verbatim in processor.py + pomdp_processor.py), `atomic_write_text()` (temp-file + `os.replace` pattern, was inlined per renderer), `MAX_STEM_LENGTH` | Composability: single source of truth for output naming + atomic writes |
| `src/render/spec_matrices.py` | **NEW** — `extract_abcd_matrices()`, `parse_gnn_matrix_value()`, `format_array_literal()` | PyTorch and NumPyro renderers carried a verbatim ~50-line duplicate extractor + twin literal formatters; now one shared typed module |
| `src/render/processor.py` | Removed no-op duplicate ImportError fallback (dead code, imports were identical in try/except); `_safe_output_stem` now delegates to `render.naming`; removed stale `AVAILABLE_RENDERERS` module-level snapshot (last consumer: none; registry call is cheap); extracted inline `--frameworks` parsing into public pure `parse_frameworks_selection()`; `"lite"` preset now resolved via `get_lite_frameworks()` | Dead code, dedup, single-source framework inventory, testability of CLI parsing policy |
| `src/render/framework_registry.py` | Removed dead `_UNSET` sentinel (defined, never used); added `LITE_FRAMEWORKS` constant + `get_lite_frameworks()` | Registry is now the single source for the lite preset (was a hardcoded literal in `process_render`) |
| `src/render/pomdp_processor.py` | Table-driven renderer dispatch: frozen `RendererRoute` table + one shared `_invoke_renderer()` skeleton (lazy import → stem path → pre-validation abort → options building → call → post-validation → artifact assembly → ImportError fallback). All 9 `_call_*_renderer` methods survive as thin delegations; bnlearn branch byte-identical (dedicated path); `_safe_output_stem` delegates to `render.naming`; removed duplicate `_safe_output_stem` body + unused `import re` | The 9 methods repeated the same ~60-line skeleton differing only in module/extension/flags; adding a framework required editing both dispatch and registry |
| `src/render/pytorch/pytorch_renderer.py` + `src/render/numpyro/numpyro_renderer.py` | `_extract_matrices` bodies → `render.spec_matrices.extract_abcd_matrices`; 4 inline tempfile+os.replace blocks → `render.naming.atomic_write_text`; `_format_tensor`/`_format_jnp_array` → `format_array_literal` delegations (names/signatures kept) | Verbatim cross-file duplication (~150 lines); atomic write pattern unified |
| `src/render/generators.py` | Deleted unreachable dead code: embedded Julia template blocks after unconditional `return code` in `generate_activeinference_jl_code` (lines 310–735) and `generate_rxinfer_code` (lines 1132–1557). **1564 → 713 lines (−851)** | Dead code verified unreachable (return precedes both blocks); live bnlearn/discopy/pymdp paths and test-imported helpers (`_to_pascal_case`, `_matrix_to_julia`, ...) untouched |
| `src/render/render.py` | Removed 5 vestigial module-level try-imports + `*_AVAILABLE` flags (nothing in `main()` or anywhere else read them; `main()` dispatches via `render_gnn_spec`) | Dead state; availability is owned by `health.py` + `framework_registry` |
| `src/render/mcp.py` | **NEW tool** `render_spec_to_format` (5th tool) — renders one GNN file to exactly one framework via the canonical `render_gnn_spec` dispatch; `Optional` import added | The existing `render_gnn_to_format` tool documents (in code) that it never filters by framework; this fills the gap additively without changing it |
| `src/render/AGENTS.md` | API Reference: added `parse_frameworks_selection` + "Shared helpers" section; MCP section: replaced 6 fictional tool names (`render.generate_pymdp` etc.) with the 5 real registered tools; Testing section: removed 3 refs to nonexistent test files, added real files incl. new contracts test | Docs of record must match code (audit found the fiction) |
| `src/render/README.md` | Module tree completed: added `naming.py`, `spec_matrices.py` + previously unlisted `health.py`, `pomdp_math.py`, `matrix_utils.py`, `multi_agent_common.py`, `continuous_common.py`, `continuous_script.py`, `generators.py`; `render.py` relabeled as standalone CLI entry | Tree was missing ~1/3 of the package |
| `src/render/SKILL.md` | MCP tool list: added `render_spec_to_format` | Consistency |
| `src/tests/render/test_render_contracts.py` | **NEW** — 36 tests pinning: stem sanitization/truncation/fallback, atomic write (nested parents, overwrite, no temp litter), shared A/B/C/D extraction (precedence chain, defaults, normalization, safe-literal parsing), array literal formatting (1D/2D/3D exact strings), lite preset = registry constant, `parse_frameworks_selection` all paths, `_render_succeeded` policy (exit-2, strict vs 80% aggregate, file-counting fallback), `validate_render` contract, unknown-framework dispatch message, MCP single-spec tool end-to-end + error shapes | Pinned previously-untested policy logic (audit gap #1/#5/#7) |
| `src/tests/render/test_render_mcp_wiring.py` | Expected MCP tool set: 4 → 5 (added `render_spec_to_format`) | Existing test pinned the tool-set contract; the additive tool legitimately changes it |

`src/11_render.py`: read, verified thin (<150 lines, 82 lines), unchanged — already conforms.

## API deltas (all additive or internal; zero public entry points removed)

**New public API:**
- `render.processor.parse_frameworks_selection(frameworks) -> tuple[Optional[List[str]], bool]` — pure normalization of the `--frameworks` selection (returns `(list_or_None, explicit_request)`).
- `render.naming.safe_output_stem / atomic_write_text / MAX_STEM_LENGTH`
- `render.spec_matrices.extract_abcd_matrices / parse_gnn_matrix_value / format_array_literal`
- `render.framework_registry.LITE_FRAMEWORKS` + `get_lite_frameworks()`
- MCP tool `render.render_spec_to_format` (registered; schema enum = registry frameworks)

**Deliberate behavior refinement (documented, test-pinned):** `parse_frameworks_selection` strips whitespace before matching `"all"`/`"lite"`, so `" all "` now resolves to the all-frameworks preset instead of the original's bogus explicit list `[" all "]` (which no framework matched and would fail every render). The original's strictness policy already stripped (explicit_framework_request), so the original behavior was internally inconsistent; the new behavior is the documented intent. Pinned by `test_all_keyword_normalizes_to_none` in `test_render_contracts.py`.

**Behavior-preserving internals:** `_safe_output_stem` (both processors) → `render.naming`; `pomdp_processor._call_*` → route-table skeleton (all messages/artifact contracts byte-identical, verified against `git show HEAD:src/render/pomdp_processor.py` incl. the bnlearn failure dict); `processor.get_available_renderers()` now delegates live to the registry instead of a frozen import-time snapshot (staleness fix; return shape unchanged). Shared `spec_matrices.extract_abcd_matrices` verified line-equivalent to the pre-refactor PyTorch/NumPyro `_extract_matrices`.

**Removed (grep-verified zero external consumers):** dead `_UNSET` sentinel, no-op duplicate import block, `AVAILABLE_RENDERERS` snapshot global, vestigial `PYMDP_AVAILABLE`-style flags + `render_gnn_to_*` fallbacks in `render/render.py`, unreachable generator template blocks.

## Verification output tails

```
uv run ruff check src/render src/tests/render
  → All checks passed!

uv run --extra dev mypy src/render --config-file pyproject.toml
  → Success: no issues found in 53 source files

uv run pytest src/tests/render/ -q        (just test-mod render; `just` not installed → direct pytest)
  → 320 passed, 1 skipped in 117.17s
  (skip = pre-existing cmdstanpy toolchain probe in test_continuous_renderers.py)
```

Behavioral spot-checks (live, during the pass): route dispatch renders pymdp/pytorch/jax/bnlearn artifacts with correct names; unknown-framework message `No renderer implemented for <name>` preserved; bnlearn ImportError message contract preserved; `parse_frameworks_selection` probed for None/all/lite/comma/list; new MCP tool renders the committed sample GNN end-to-end (bnlearn artifact written).

Fleet interference note: three times the tree was transiently broken by concurrent peers (`utils/pipeline.py`, `gnn/parsers/unified_parser.py`, `gnn/parsers/common.py`, `utils/arg_parsing.py` — never render files). All mypy/pytest runs above were taken on a settled tree.

## Follow-ups needed (other workers own these)

- **doc/ workers**: `doc/gnn/modules/11_render.md` and `doc/gnn/integration/gnn_implementation.md` document the CLI target list and processor API — worth syncing with `parse_frameworks_selection`/lite-preset wording (they are not wrong today, just less specific).
- **manuscript/ workers**: none needed; no user-facing pipeline behavior changed (exit codes, summary JSON schema, artifact names all preserved).
- **mcp/ workers**: `src/mcp/validate_tools.py` pins expected tool names for listed modules — `render` wasn't in its list, but if the fleet touches it, `render_spec_to_format` should be added to whatever registry they maintain.

## Follow-up ideas (next pass on this module)

1. Unify the ≥6 column-normalization implementations (`matrix_utils.normalize_columns`, `pomdp_math._normalise_columns`, `pomdp_contract.normalise_matrix_columns`, stan `_discrete_parameters`, pymdp template `_norm_cols`, multi_agent_common) behind one function parameterized by zero-column policy (keep | uniform | raise) — semantics genuinely differ per callsite, so this needs care + a decision.
2. Collapse `render_gnn_spec`'s hardcoded target set + file suffixes onto `framework_registry` (registry already stores `file_extension`; the POMDP-target set could derive from `pomdp_compatible`).
3. `_rehydrate_file_backed_parse_summary` + `_internal_representation_to_mapping` are processor concerns duplicated conceptually in rxinfer's `render_file`; consider one `spec_adapter` module.
4. `discopy/translator.py` availability-shim globals + lazy init; `rxinfer/toml_generator.py` is production-retired but test-pinned — migrate those parser tests, then retire the module.
5. `JAXRenderer` facade class in `__init__` has zero external consumers (grep) and no tests — candidate for removal next pass (kept this pass; removal is API-surface change).

## Checkpoint log

See `docs/development/fleet-logs/render-worker.md` (audit → refactor+functionality → tests+docs → verify).
