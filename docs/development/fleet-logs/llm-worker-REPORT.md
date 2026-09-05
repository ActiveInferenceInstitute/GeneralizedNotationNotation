# llm-worker REPORT — GNN module fleet 3, 2026-09-04

Scope: `src/llm/` (all files) + `src/13_llm.py`. Branch main @ f64ac9085, edits in place, no git ops, no dependency changes.

## Files changed + why

| File | Change | Why |
|---|---|---|
| `src/llm/processor.py` (987→~970 lines) | Extracted shared async funnel `_execute_prompt()` + pure helpers `_classify_auth_error()` / `_prompt_fallback_text()`; both prompt loops (structured `PromptType` + custom) now delegate; hoisted loop-invariant `max_prompt_timeout` (fixes latent NameError when budget dies before first prompt); module-level `"llm"`-channel logger. | The two loops were ~120 lines of copy-paste (cache/timeout/empty-content/auth/fallback/write). Logs claimed "skipping future calls" for auth-failed providers but nothing skipped — fail-fast is now actually enforced (peek via `get_default_provider`, duck-typed so stubs without routing introspection skip it). |
| `src/llm/llm_processor.py` (1025→~1015 lines) | `_error_result()` shared failure payload; `generate_explanation`/`enhance_model` now delegate to `analyze_gnn_model(..., "summary"/"enhancement")`; `analyze_gnn` trips the per-provider circuit breaker on failure and `_try_fallback_analysis` resets it on success (parity with `get_response`); fallback excludes the *actually-failed* provider; new public `get_default_provider()`; module-level `import asyncio` (was 3× local). | Dedup + routing-consistency. Previously only `get_response` honored breakers. |
| `src/llm/llm_operations.py` (507→~460 lines) | New `_async_analyze()` funnel; 5 duplicated `_async_*` ops (summarize/structure/questions/enhance/validate) delegate to it; module-level `summarize_gnn(..., ollama_model=None)` kwarg added. | Identical init-check + `analyze_gnn` + content-return repeated 5×. |
| `src/llm/prompts.py` (396→~450 lines) | Added missing prompt configs for `PromptType.COMPARE_MODELS` and `PromptType.VALIDATE_SYNTAX`. | `get_all_prompt_types()` advertised them but `get_prompt` raised `ValueError` — a trap for any caller iterating the enum. |
| `src/llm/analyzer.py` | New public `variable_type_counts()`; `perform_semantic_analysis` uses it. | Shared census with generator (was duplicated inline). |
| `src/llm/generator.py` | Uses shared `variable_type_counts` (import from `.analyzer`). | Dedup; no behavior change. |
| `src/llm/cache.py` | `summary() -> dict[str, Any]` (was bare `dict`). | Typing only. |
| `src/llm/mcp.py` (470→~505 lines) | `analyze_gnn_with_llm_mcp` now honors its advertised params: unknown `analysis_type` → `{"success": False, error listing allowed values}` (was silently ignored); `provider="ollama"` pins `DEFAULT_OLLAMA_MODEL` for the summary call (was ignored); both params echoed in the payload; `get_llm_providers_mcp` adds openrouter/perplexity (missing entries); `register_tools` is table-driven (5 near-identical blocks → `tool_specs` tuple + loop; schema enum now derives from `_ANALYSIS_TYPES`). | Dead advertised params + incomplete provider matrix + registration boilerplate. |
| `src/llm/__init__.py` | `get_available_providers()` env/config-driven via `load_api_keys_from_env()` (ollama unless `OLLAMA_DISABLED`; cloud only with keys; perplexity now included) — was import-probe based (always listed openai/openrouter, never perplexity); `analyze_gnn_model` compat helper dedup'd (removed duplicated except-branch); unused imports (`os`, `List`, `Optional`, `Tuple`) removed. | Misleading "availability" + dead code. |
| `src/llm/llm_system_demo.py`, `src/llm/demo_llm_features.py` | **Deleted.** | Zero references repo-wide (grep incl. doc/). The former "tested" local stubs (fabricated assertions, ~60% no-op); the latter called live LLM providers when run. Usage examples live in AGENTS.md/README. |
| `src/llm/AGENTS.md`, `src/llm/README.md`, `src/llm/SKILL.md` | New "2026-09-04 Composability Refactor (API deltas)" section + test-file list + footer date; README mermaid diagrams corrected (Anthropic nodes → Perplexity/OpenRouter reality, matching the prose at line 6); SKILL.md `get_available_providers` comment updated. | Docs of record must match API/behavior. |
| `src/13_llm.py` | Unchanged (55 lines). | Already thin; contract preserved. |
| `src/tests/llm/` (+6 files, ~28KB) | `test_llm_analyzer_extractors.py`, `test_llm_prompts_registry.py` (regression: registry covers every enum member), `test_llm_cache.py` (key sensitivity/roundtrip/corruption/clear), `test_llm_generator.py`, `test_llm_processor_helpers.py` (auth classifier, budget resolvers, sort key, config merge, `get_default_provider`, full `_execute_prompt` funnel: cache hit / fail-fast / success-caches / empty-response / auth-recorded-once / timeout / non-auth error), `test_llm_sync_wrappers.py` (never-raises contracts, offline). | Audit showed extractors/generator/cache/prompts had zero direct behavioral tests. |

## API deltas

- **Additive**: `LLMProcessor.get_default_provider()`, `analyzer.variable_type_counts()`, `processor._classify_auth_error` / `_prompt_fallback_text` / `_execute_prompt`, `llm_processor._error_result`, `llm_operations.summarize_gnn(..., ollama_model=None)`, MCP `openrouter`/`perplexity` provider-matrix entries.
- **Behavior-tightening (consumer grep proved safe)**: MCP unknown `analysis_type` now rejected (was ignored); `provider="ollama"` honored; `get_available_providers()` reflects actual env usability (perplexity included; import-probe branches were dead). `test_llm_overall` only pins `len > 0` — holds.
- **Removed**: two dead demo scripts (no consumers).
- **Unchanged contracts**: `process_llm` signature/return, `llm_results.json` / `llm_summary.md` output shape, prompt `.md` file naming/content format, log channel (`"llm"`) and per-prompt message wording, exit codes via `13_llm.py`, all `__init__` exports (incl. facade `LLMProcessor`/`LLMAnalyzer`).

## Verification output tails

```
uv run ruff check src/llm src/tests/llm            → All checks passed!
uv run ruff format --check src/llm src/tests/llm   → 32 files already formatted
uv run --extra dev mypy src/llm --config-file pyproject.toml
                                                   → Success: no issues found in 16 source files
uv run pytest src/tests/llm/ -q                    → 179 passed, 4 skipped in 1.76s
  (skips = network-guarded Ollama tests, no daemon on this host)
mypy (same strict config) over the 7 NEW test files
                                                   → Success: no issues found in 7 source files
consumer regression: test_core_modules + test_coverage_overall +
  api/test_comprehensive_api + pipeline/test_pipeline_recovery +
  pipeline/test_pipeline_scripts + llm/test_llm_mcp_security
                                                   → 149 passed in 19.71s
```
`just` is not installed on this host; `just test-mod llm` was run as its exact recipe body (`uv run pytest src/tests/llm/`). Full suite / uv lock / pipeline intentionally not run (fleet rules).

## Doc/manuscript follow-ups (other workers own these)

- `doc/api/comprehensive_api_reference.md:426` still references a fictional `gnn.llm` module path (pre-existing).
- `doc/llm/README.md` + `doc/mcp/tool_reference.md` may want the MCP `analysis_type` validation + `provider` honoring semantics mirrored.
- `doc/api/README.md:90` / provider tables still say "Anthropic provider" in places outside my scope.

## Follow-up ideas

- Providers still share ~4 copies of the sync `analyze()` bridge and aiohttp init/stream loops (openrouter/perplexity are near-identical) — a `providers/_http_common.py` would remove ~150 lines; left alone to keep this pass behavior-frozen.
- `LLMConfig.timeout` is accepted but ignored by all four providers (audit finding).
- `_select_best_ollama_model` + `_start_ollama_if_needed` are well-tested but still tightly subprocess-coupled; an injectable runner would make step-13 fully fake-able.
- `FEATURES["anthropic_integration"]` in `__init__.py` is misleading (no Anthropic provider exists); left untouched to avoid `get_module_info` contract drift.
