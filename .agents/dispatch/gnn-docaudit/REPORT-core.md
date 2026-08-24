# REPORT-core — Index/Region docs-vs-code audit (REPORT-ONLY)

Repo: `/home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation`
Region audited: root docs (README.md, AGENTS.md, ARCHITECTURE.md, DOCS.md, SETUP_GUIDE.md, SPEC.md, SKILL.md, CONTRIBUTING.md, CLAUDE.md) + `doc/pipeline/`, `doc/execution/`, `doc/deployment/`, `doc/configuration/`, `doc/development/` (incl. `doc/PIPELINE_SCRIPTS.md`).

Verification method: `git ls-files` for tracked paths; grep for `^(def|class)` / import-resolution in `src/`; `justfile` recipe listing; `sed` of `pyproject.toml` optional groups; `input/config.yaml` key inspection. All findings below are anchored to the current tree.

Verified clean (no findings):
- Numbered scripts: all 25 `src/N_*.py` (0–24) tracked and referenced correctly across PIPELINE_SCRIPTS.md, doc/pipeline/README.md, the CLI commands in doc/deployment/, doc/configuration/, README.md (incl. `0_template.py`…`24_intelligent_analysis.py`, `src/main.py`).
- Scripts referenced in doc/pipeline/README.md & doc/execution/FRAMEWORK_AVAILABILITY.md exist: `doc/development/docs_audit.py`, `scripts/check_doc_contracts.py`, `scripts/check_gnn_doc_patterns.py`, `scripts/check_repo_terminology.py`, `scripts/check_maintained_doc_terms.py`, `scripts/check_capability_contracts.py`, `scripts/check_pomdp_gridworld_outputs.py`, `scripts/run_v3_orchestration_acceptance.py`, `scripts/emit_run_manifest.py`, `scripts/run_session_acceptance.py`, `scripts/generate_pipeline_container_plan.py`.
- v3 orchestration modules all tracked: `src/pipeline/{durable_streams,run_session,container_plan,session_acceptance,run_manifest,pipeline_container_plan,mcp}.py`; committed Julia envs `src/execute/rxinfer/Project.toml`, `src/execute/activeinference_jl/{Project,Manifest}.toml`.
- Env vars consumed in `src/`: `OPENAI_API_KEY`, `OLLAMA_MODEL`, `OLLAMA_TEST_MODEL`, `GNN_JAX_PLATFORM`, `GNN_ALLOW_UNSAFE_EXEC`, `GNN_SANDBOX` (all found via grep in processor/sandbox code).
- `pyproject.toml` optional groups `dev, api, ml-ai, audio, gui, graphs, research, scaling` all present (as claimed in SETUP_GUIDE.md / README / CLAUDE.md).
- `justfile` recipes claimed in docs exist: `test, test-full, lint, quality, test-pymdp-focused, pipeline, render-health, audit, bench, steps, setup, validate-stack,…`.
- Framework counts: `src/render/framework_registry.py` contains all 9 target tokens incl. bnlearn → doc/pipeline/README.md & FRAMEWORK_AVAILABILITY.md "9 render targets / 8 executor families" accurate.
- Doc cross-link paths in README.md (gnn_overview, gnn_syntax, gnn_file_structure_doc, quickstart_tutorial, gnn_paper, about_gnn, advanced/gnn_llm_neurosymbolic_active_inference.md, integration/gnn_implementation.md, operations/gnn_tools.md, doc/api/README.md, doc/pymdp/, doc/rxinfer/, doc/mcp/, doc/sympy/, doc/discopy/, doc/cognitive_phenomena/{attention,consciousness,effort,emotion_affect,executive_control}/, doc/templates/, doc/quickstart.md, doc/learning_paths.md, doc/style_guide.md) all resolve to tracked files.

## Findings

doc/development/README.md:307 | ERROR | `from src.gnn.parser import parse_gnn_file` — `src/gnn/parser.py` exists but defines NO `parse_gnn_file` (grep: function lives in `src/gnn/processor.py`); the import raises ImportError. The same broken import/usage also drives the test-writer example at 307–354. Suggested fix: change to `from src.gnn.processor import parse_gnn_file`.

doc/development/README.md:168 | ERROR | code example `from utils.logging_utils import setup_standalone_logging` — the facade `src/utils/logging_utils.py` re-exports `PipelineLogger`, `log_step_*`, `setup_correlation_context`, etc. but does NOT export `setup_standalone_logging` (defined in `src/utils/logging/logging_utils.py`); import fails. Suggested fix: `from utils.logging.logging_utils import setup_standalone_logging`.

doc/development/README.md:362 | ERROR | `uv run --extra dev python -m pytest src/tests/unit/ -v` — `src/tests/unit/` does not exist (tracked layout is `src/tests/<module>/`); command fails. Suggested fix: point at a real test dir, e.g. `src/tests/gnn/`.

doc/development/README.md:369 | ERROR | `uv run --extra dev python -m pytest src/tests/performance/ --benchmark-only` — `src/tests/performance/` does not exist; command fails. Suggested fix: use the performance-marked path that exists (`src/tests/pipeline/` with `-m performance`), or remove.

doc/development/README.md:471 | ERROR | `pytest -vvv --pdb src/tests/unit/test_specific.py` — file does not exist. Suggested fix: sample an actually-tracked test file (e.g. `src/tests/gnn/test_gnn_overall.py`).

doc/development/README.md:120 | WARNING | `└── tests/  # 171 pytest files, mirrored by module` — stale count; current tracked `.py` files under `src/tests/` = 323. Suggested fix: update the count (or drop the hardcoded number and say "mirrored by module").

doc/development/README.md:287–300 | WARNING | test-organization tree shows `tests/unit/`, `tests/integration/`, `tests/fixtures/` — actual layout is `src/tests/<module>/`; `unit/`, `fixtures/`, `performance/` do not exist (only `src/tests/integration/` exists). Suggested fix: rewrite the tree to the real `src/tests/<module>` layout.

doc/configuration/examples.md:1–978 | WARNING | the file presents an aspirational `config/*.yaml` schema (keys `version`, `paths.target_dir`, `pipeline.steps`, `continue_on_error`, `parallel_execution`, `import`-style sections, `config/*.yaml` fixtures) that is not supported and is directly contradicted by `doc/configuration/README.md` (only `input/config.yaml` is auto-loaded; no generic `--config`/profile loader). A reader following it produces a non-functional config. Suggested fix: annotate the file as illustrative/aspirational, or align the examples to the supported `input/config.yaml` schema.

SKILL.md:25 | WARNING | `just render-health  # Check all 8 renderer backends` — authoritative `src/render/framework_registry.py` has 9 targets (incl. bnlearn); CLAUDE.md:38 correctly says "9 renderer backends". Stale count in SKILL.md. Suggested fix: change 8→9 and add bnlearn to the framework list at SKILL.md:14.

SKILL.md:22 | INFO | `just  # List all 21 recipes` — `justfile` currently has ~28 recipes; `just` (default recipe) runs `@just --list --unsorted`, so the pinned "21" is stale/misleading. Suggested fix: drop the count (`just  # List recipes`).

doc/configuration/README.md:65 | INFO | enumerates config sections "testing_matrix, io, logging, validation, performance, and security" but omits `uv`, which is present in `input/config.yaml` (top-level keys: pipeline, setup, uv, io, testing_matrix, llm, logging, validation, performance, security). Suggested fix: add `uv` to the enumeration.

## Notes
- No errors in root README.md, AGENTS.md, ARCHITECTURE.md, DOCS.md, SPEC.md, CONTRIBUTING.md, SETUP_GUIDE.md, CLAUDE.md command/path/import claims. `SETUP_GUIDE.md` optional-group list and `CLAUDE.md` `just steps`/`just bench` recipes verified against `justfile`.
- `README.md:102` "32 module directories" matches the 32 top-level `src/*/AGENTS.md` module dirs; not flagged.
- Dated volatile metrics in README.md:55/58 (310 packages; mypy 758 files; GridWorld output check) are framed with their verification dates and treated as historical, not current-run claims.
