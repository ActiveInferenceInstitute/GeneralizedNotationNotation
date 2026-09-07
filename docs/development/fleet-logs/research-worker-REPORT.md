# research-worker REPORT — fleet 3, 2026-09-04

Scope: `src/gnn/research/` (processor.py, mcp.py, `__init__.py`, AGENTS/README/SPEC/SKILL) + `src/gnn/19_research.py`. All edits in place; no git ops; no dependency changes.

## Files changed + why

| File | Change | Why |
|---|---|---|
| `src/gnn/research/processor.py` | Decomposed the 190-line `process_research` monolith into pure units; added `_iter_section_lines()` deduplicating the section-scanning state machine previously copy-pasted in `extract_state_space_dims`/`count_connections`; hoisted `asyncio`/`tempfile` imports to module level and removed inline `import os as _os`/`import tempfile as _tempfile`; normalized annotations to modern builtin generics (`dict[str, list[int]]` etc., dropped unused `Optional`/`cast`); removed the never-used `content` param from private `_generate_llm_hypotheses` | Composability + internal quality; every output byte preserved |
| `src/gnn/research/processor.py` | **Additive API**: frozen `@dataclass ModelAnalysis` + `analyze_gnn(content)` one-call static analysis; `MODEL_FAMILIES` constant; `summarize_hypotheses()` (priority/type counts); public `render_research_report()` (pure markdown renderer extracted from the writer); `write_research_outputs()` (atomic JSON + report writes, explicit `encoding="utf-8"`); `discover_gnn_files()` (sorted, scoped); `merge_llm_hypotheses()` (extracted from inline merge) | Genuinely useful typed surface for non-pipeline consumers; single source of truth for report rendering |
| `tests/research/test_research_analysis.py` | **New**: 13 tests pinning the analysis API (bundle consistency, empty-content, `MODEL_FAMILIES` coverage, section-boundary isolation, symbolic/nonpositive dim rejection, undirected dash counting, discovery sort/scope/missing-dir, merge dedup order, summary counts/determinism, renderer purity + byte parity with the written report) | Pin real behavior of the new surface |
| `src/gnn/research/README.md`, `AGENTS.md` | Fixed **stale `generate_rule_based_hypotheses(content, model_name, output_dir, logger) -> Tuple[List[Dict], str]`** signature (docs of record contradicted the actual `(content, model_family, dims, connections) -> list[dict]` since before this fleet); documented the new API; test list updated | Docs were provably wrong |
| `src/gnn/research/SPEC.md`, `SKILL.md` | New components/capability lines | Docs of record in lock-step |
| `src/gnn/research/__init__.py` | `__version__` 1.6.0 → 1.7.0 | API addition |

Untouched by design: `src/gnn/19_research.py` (55 lines, already thin), `mcp.py` (4 tools unchanged; test uses subset check so no break either way).

## API deltas

- **Added** (all additive, typed, documented): `analyze_gnn`, `ModelAnalysis`, `MODEL_FAMILIES`, `summarize_hypotheses`, `render_research_report`, `write_research_outputs`, `discover_gnn_files`, `merge_llm_hypotheses`.
- **Behavior-preserving**: `process_research`, `detect_model_family`, `extract_state_space_dims`, `count_connections`, `generate_rule_based_hypotheses`, `_validate_llm_hypotheses`, and the four MCP handlers are call- and output-compatible; JSON payloads and markdown report are byte-identical (pinned by new parity test `test_rendered_report_matches_written_report`).
- **Removed**: `_generate_llm_hypotheses(content, ...)` → `_generate_llm_hypotheses(model_family, dims, logger)` — private, single caller, `content` was never read (Pyright hint).
- **Changed**: JSON writes now explicitly `encoding="utf-8"` (was locale default); report write already atomic, JSONs now deterministic across locales.

## Verification (tails)

```
uv run ruff check src/gnn/research tests/research   -> All checks passed!
uv run --extra dev mypy src/gnn/research --config-file pyproject.toml
                                                    -> Success: no issues found in 3 source files
uv run pytest tests/research/ -v (= just test-mod research; `just` binary absent on host)
                                                    -> 40 passed in 0.07s  (27 pre-existing + 13 new)
uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict
                                                    -> no banned patterns
19_research.py smoke (tmp target with pomdp model)  -> exit=0; 4 artifacts written;
                                                       family=pomdp; hypotheses={connectivity_enrichment,
                                                       ontology_annotation, parameter_learning,
                                                       parameterization, precision_modulation};
                                                       report has '## model.md (pomdp model)' + priority grouping
```

`docs/development/docs_audit.py --strict --check-anchors --no-write` exits 1 on **pre-existing** issue `tests/tests` (dir with .py but no AGENTS.md, created 10:59 by a fleet peer before this worker started — not in my scope). My doc surface is clean under it.

## Follow-ups for other owners

1. `tests/tests/` (peer): needs an `AGENTS.md` or relocation, else `docs_audit --strict` stays red repo-wide.
2. `docs/` or `manuscript/` workers: none required — output contracts (`research_results.json`, `research_summary.json`, `research_processing_summary.json`, `research_report.md`) unchanged, so `pipeline_validation`/`report/analyzer`/`gui` consumers are unaffected.
3. Repo-wide convention note: module `FEATURES` exists in both `__init__.py` and `processor.py` with different keys (drift by convention, same in audio/analysis/... modules). Left alone deliberately; a fleet-level decision would be needed to unify.

## Follow-up ideas (out of scope today)

- Promote `summarize_hypotheses` into the MCP surface (e.g. enrich `read_research_results_mcp` with per-priority counts) — additive, would need docs/test updates.
- `detect_model_family` could return a `Literal` type derived from `MODEL_FAMILIES` once Python 3.12 `TypeAliasType` ergonomics are acceptable here.
- LLM enrichment path (`FEATURES["llm_hypothesis_generation"]`) is dead in production (gate is False); consider an env-var override or removal in a future minor.

## Post-report advisory sweep (same turn)

- **FEATURES consumers**: repo-wide grep for `research.FEATURES` / `from research import *FEATURES` / `processor.FEATURES` outside `src/gnn/research/` → **zero consumers**. The `__init__`/processor two-dict split stays (repo-wide convention: audio, analysis, cli, api all do this); the advisory's unification condition (both surfaces consumed) is false.
- **Version pins**: grep for `1.6.0` pins across `tests/research/` and the api test → none reference `research.__version__`; the `1.6.0` strings belong to `src/gnn/__init__.py` and `src/gnn/mcp/__init__.py` (independent dual-versioning policy, `pyproject.toml` 3.2.0 authoritative). The 1.7.0 module bump is safe.
- **git-diff parity scan** (`git diff src/gnn/research/processor.py`, removed-vs-added long-string comparison): **parity scan: pass, 2 explained artifacts** — of 13 removed long strings, 11 reappear verbatim; the 2 non-verbatim are (a) the inline `rglob/glob` discovery one-liner, extracted intact into `discover_gnn_files`, and (b) the verbose log line, now interpolated as `{analysis.model_family}` inside the same f-string — identical rendered output. No hypothesis description/rationale/priority text drifted. Module tests (including the renderer/writer byte-parity test) re-run green after formatting.
- **black** (configured in pyproject, dev dep): reformatted my two in-scope Python files (cosmetic: blank lines, line rewrapping — no string/logic changes); `black --check` now clean on them. Note: `tests/research/test_research_functional.py` has **pre-existing** black drift (0 diff lines from me — untouched, owner is the repo/pre-existing state).
- **Re-verification after black**: ruff clean, mypy clean (3 files), `pytest tests/research/` 40/40.
- SKILL.md capability bullets re-verified on disk (5 bullets intact); README example imports all three names it calls.
