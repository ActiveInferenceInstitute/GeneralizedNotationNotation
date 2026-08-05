# Changelog

All notable changes to the GNN Pipeline are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/) and [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- **Per-iteration VFE trace**: `variational_free_energy` and `vfe_per_iteration` in `rxinfer_simulation_v1` now report the full per-iteration VFE vector from RxInfer's `result.free_energy` (length = INFERENCE_ITERATIONS), replacing the fabricated constant that replicated one scalar across all timesteps.
- **TypedDict contracts**: Added `CanonicalPomdpSpec`, `InitialParameterization`, `RxInferSimulationV1` TypedDict definitions and `ModelKind` enum to `src/render/pomdp_contract.py` for typed renderer contracts.
- **ModelKind enum**: FLAT, FACTORED, HIERARCHICAL, MULTI_AGENT, CONTINUOUS, LEARNING — detected from the GNN spec and carried in `runtime_metadata.model_kind`.
- **Belief entropy validation**: `belief_entropy_ok` field rejects degenerate beliefs (Shannon entropy < 0.1 nats) for non-identity A matrices. Fully observable models (identity A) are exempt.
- **Expanded precompile coverage**: `GnnRxInferModels.jl` now precompiles 6 state-space configurations (2, 3, 4, 8, 9, 16 states) × 7 T values (3–30), covering common GNN exemplar dimensions. Precompile success/failure is logged per config.

### Changed
- **Removed fallback entirely**: The `try/catch` around `infer()` and the Bayesian filter fallback (~30 lines) have been removed. If `infer()` fails, the script crashes with a non-zero exit code and no results JSON is written. `uses_real_rxinfer` is now conditional on actual `infer()` success — set to `true` only when `infer()` returns.
- **Strengthened `all_valid`**: Now includes `inference_converged`, `vfe_present`, and `belief_entropy_ok` in addition to the construction-guaranteed checks. Degenerate beliefs and non-converged inference now fail validation.
- **Accurate pipeline labeling**: All docs and code comments now describe the pipeline as "offline batch inference (Bayesian smoothing) with post-hoc EFE policy evaluation" rather than "active inference". The forward pass is labeled "forward simulation for data collection" not "Bayesian filter".
- **Analyzer VFE handling**: `_normalise_free_energy()` now prefers `vfe_per_iteration` as the authoritative per-iteration VFE source, falling back to `variational_free_energy` and then `expected_free_energy`.

### Fixed (Mahakala C1–C5)
- **C1: VFE trace is a fabricated constant** — FIXED. The per-iteration VFE vector from `result.free_energy` is now reported directly, not replicated across timesteps.
- **C2: `uses_real_rxinfer` hardcoded true even on fallback** — FIXED. Now conditional on actual `infer()` success; the fallback path that could set it incorrectly has been removed entirely.
- **C3: Batch infer() is post-hoc smoothing, not active inference** — FIXED (labeled accurately). Docs and comments now describe the pipeline as offline batch inference with post-hoc policy evaluation.
- **C4: `all_valid` is tautological** — FIXED. `all_valid` now includes `inference_converged`, `vfe_present`, and `belief_entropy_ok`.
- **C5: Beliefs are degenerate** — ADDRESSED. Belief entropy check rejects degenerate beliefs for non-identity A matrices. For fully observable models (identity A), the check is skipped as degeneracy is expected.

### Added (prior)
- **Genuine RxInfer.jl integration**: Replaced the hand-rolled POMDP step simulator with real RxInfer.jl variational message-passing inference. The canonical renderer (`rxinfer_renderer.py`) now emits Julia scripts that define a generative model using `@model` with `Categorical` / `DiscreteTransition` nodes and run `infer()` with `free_energy=true` to obtain posteriors over hidden states and real variational free energy traces. The `variational_free_energy` field in `rxinfer_simulation_v1` is now populated with real VFE values (previously `Float64[]`).
- **Committed Julia environment**: Added `Project.toml` + `Manifest.toml` under `src/execute/rxinfer/` pinning RxInfer 5.5.0 and all dependencies. The runner now passes `--project=<env>` instead of bare `julia`, ensuring reproducible, network-independent execution.
- **Reproducibility tracking**: Generated scripts now include `Random.seed!(seed)` before inference and record the script SHA256, seed, and `uses_real_rxinfer` flag in `runtime_metadata`. Results are byte-identical across runs with the same seed.
- **Inference convergence check**: The `validation` dict now includes `inference_converged` (VFE stabilized) and `vfe_present` fields alongside the existing `all_valid` / `all_beliefs_valid` / `beliefs_sum_to_one` / `actions_in_range`.
- **45/45 GNN exemplar files verified**: All 45 GNN spec files across `discrete/`, `continuous/`, `basics/`, `hierarchical/`, `multiagent/`, `precision/`, `pomdp_gridworld/`, `structured/`, and `pymdp_scaling_study/` render and execute successfully with the real `@model` + `infer()` pipeline.
- **Guarded Julia-native visualization + structured logging in generated RxInfer scripts**: rendered `*_rxinfer.jl` scripts always write `simulation_results.json` (`rxinfer_simulation_v1`) and additionally emit best-effort, guarded artifacts that never cause execution failure — an optional structured runtime log (`simulation.log` / `simulation_log.json`) and optional `Plots.jl` figures (`belief_evolution.png`, `efe_over_time.png`, `policy_posterior.png`) when Plots rendering is available.
- **Comprehensive Step-16 RxInfer analysis**: `src/analysis/rxinfer/` produces the full per-exemplar 10-type visualization set from `rxinfer_simulation_v1` results under `output/16_analysis_output/rxinfer/` — `belief_evolution`, `obs_vs_true`, `belief_heatmap`, `belief_entropy`, `accuracy`, `action_frequencies`, `belief_convergence`, `belief_trace`, `free_energy`, and `observations`. All figures are best-effort and backward-compatible with `rxinfer_simulation_v1`.

### Deprecated
- **`toml_generator.py`**: The legacy TOML-based RxInfer renderer is deprecated. The canonical renderer (`rxinfer_renderer.py`) with genuine `@model` + `infer()` code is the only supported path. `render_gnn_to_rxinfer_toml` now emits a `DeprecationWarning` and is removed from the processor wiring and public exports. The file is retained for git history and reference.

### Fixed
- Fixed A matrix construction in the generated Julia script: the `hcat` approach transposed the observation/likelihood matrix, causing dimension mismatches for non-square A matrices (e.g. HMM baseline with 6 obs × 4 states). Replaced with explicit row-by-row construction matching the original `to_float_matrix` behavior.
- Fixed missing `using Pkg` and `using SHA` imports in the generated Julia script, which caused `UndefVarError` when `package_version()` or `sha256()` were called.
- Hardened MCP execution and LLM entry points against arbitrary local file access and script execution by enforcing repository-local path validation, source-file extension checks, and Step 11 render-summary gating for `process_execute_mcp`.
- Escaped bnlearn generated-code metadata as Python literals, coerced generated timestep literals, and sanitized generated artifact filename stems to prevent code injection and path traversal through model names.
- Resolved default-branch Dependabot alerts by raising patched dependency floors and refreshing `uv.lock` for `msgpack` 1.2.1, `jupyter-server` 2.20.0, `jupyterlab` 4.6.0, and `bleach` 6.4.0.
- Fixed session-scope `test_config` fixture in `conftest.py` to clean up its temporary directory after shutdown (was leaking via `tempfile.mkdtemp()` without finalizer).
- Made render exemplar discovery recursive so all nested GNN spec files under `input/gnn_files/**` are found and rendered to RxInfer.jl. Previously only a shallow subset of the exemplar tree was discovered. All 45 exemplar GNN files now render to RxInfer.jl.
- Fixed an exemplar dimension bug in `input/gnn_files/continuous/stochastic_dynamics.md`: the `A` matrix was declared `A[2,3]` (2 observations × 3 hidden states) but parameterized as a 3×3 matrix, which caused Julia RxInfer dimension validation to fail at execute time. Corrected the `A` matrix rows to a 2-row (2×3) parameterization consistent with the declaration.

### Verified
- **45/45 exemplar GNN files render to RxInfer.jl and execute under RxInfer.jl.** (Recursive discovery + `stochastic_dynamics.md` dimension fix.)

### Changed
- `dag.py`: Added `raise_on_circular` parameter to `resolve_execution_order`; when `True`, circular dependencies raise `ValueError` instead of silently appending unresolved steps as the last execution tier. Added 5 unit tests for DAG resolution behavior.
- `conftest.py`: Added `_auto_seed_rng` autouse function-scope fixture that calls `np.random.seed(0)` before every test, providing a deterministic baseline for unseeded tests.
- `ci.yml`: Set `PYTHONHASHSEED=0` in test job environment for deterministic dict iteration across runs (removes non-reproducibility risk from unsorted JSON keys).
- `ci.yml`: Added `scripts/run_v3_orchestration_acceptance.py --strict` as a CI step (19/19 v3 acceptance checks) — the acceptance gate previously never ran in CI.
- `ci.yml` + `mcp-audit.yml`: Raised MCP tool-count guard threshold from 130 to 140 to match the actual tool count, preventing silent tool-count regressions.
- `pyproject.toml`: Added `fail_under = 50` threshold to `[tool.coverage.run]` — coverage was measured but never enforced.

---

## [3.0.0] — 2026-06-20

> **Long-Running Orchestration & Distributed Ecology Plans.** Safe-by-design durable observation
> streams, resumable run sessions, and auditable container plans. Release gates re-run green for all
> 9 model families (semantic fidelity, cross-framework reliability, model-family acceptance); CI
> matrix restored to green.

### Added
- **v3.0.0 long-running orchestration (safe-by-design, no live mutation)**: three new
  `src/pipeline/` modules — `durable_streams.py` (file/array `StreamManifest` with content checksums,
  `ExecutionTrace` integrity + deterministic replay), `run_session.py` (resumable `RunSession`
  manifests, atomic checkpoint/resume, status inspection, path-safe cancellation cleanup), and
  `container_plan.py` (hardened container plan generation, static security review with
  CRITICAL/HIGH/MEDIUM/LOW findings, rollback descriptors, deterministic plan hashes). Backed by 40
  real-objects-only unit tests with negative controls, a strict end-to-end acceptance gate
  (`scripts/run_v3_orchestration_acceptance.py`), three new MCP tools (`tools_total` 137→140), and a
  doc page at `doc/pipeline/v3_orchestration.md`. No container/cluster is ever executed.
- **v3.0.0 additive live-pipeline integration (safe-by-design)**: `session_acceptance.py` (resumable,
  checkpointed model-family acceptance runs), `run_manifest.py` (emit durable `StreamManifest`s + a
  replayable `ExecutionTrace` from a completed run's `output/`, with re-validation), and
  `pipeline_container_plan.py` (generate a `security_review`-clean container plan from `input/config.yaml`),
  each with a CLI under `scripts/` and real-objects-only tests. Verified on real run artifacts (105 manifests +
  a 25-event trace) and the real config; full `src/tests/pipeline` suite 362 passed. The 25-step
  critical path is unmodified.

### Changed
- **GNN parser extension policy**: `.pkl` is reserved for textual PKL DSL by default; clearly binary pickle payloads in `.pkl` are routed to the pickle parser with a warning, and `.pickle` is the canonical binary pickle extension.
- **API output directories**: API run and job submissions now validate `target_dir` and `output_dir` as repository-local directories and preserve caller-selected output directories during async execution.

### Fixed
- **Pipeline prerequisite checks**: Missing prerequisite output artifacts now fail the prerequisite check instead of only emitting warnings, and readiness checks use the registered GNN extension list.
- **Pipeline completion logging**: Non-warning statuses no longer take the warning branch because of a truthy string condition.

---

## [2.0.0] — 2026-06-12

### Added
- **Semantic fidelity release gate**: `scripts/run_semantic_fidelity_gate.py` writes `gnn_semantic_fidelity_ledger_v1` artifacts for maintained model families.
- **Strict semantic contracts**: representative fixtures now preserve model identity, variables, edges, dimensions, parameter shapes, equations, time, and ontology mappings across JSON parse/serialize/parse checks.
- **Cross-framework reliability release gate**: `scripts/run_cross_framework_reliability.py` writes `gnn_cross_framework_reliability_ledger_v1` artifacts with compatible, required, and unsupported backend statuses.
- **GridWorld three-backend comparison**: GridWorld is profiled for PyMDP, RxInfer, and ActiveInference.jl, including seed, trace length, matrix-shape, and matrix-provenance parity.

### Changed
- GridWorld model-family acceptance now requests PyMDP, RxInfer, and ActiveInference.jl for the v2 comparison fixture instead of a PyMDP-only profile.
- Roadmap next target moves to v3.0.0 for durable streams, long-running sessions, and auditable container plans.

### Fixed
- JSON serialization now emits equation objects instead of lossy stringified dataclasses, preventing silent semantic round-trip drift.
- Cross-framework reliability no longer certifies aggregate Step 12 success without successful non-skipped execution-detail rows and current simulation payloads for required backends.

---

## [1.9.0] — 2026-06-12

### Added
- **Model-family acceptance release gate**: manifest-driven all-family strict acceptance for basics, discrete, continuous, hierarchical, multi-agent, precision, structured, gridworld, and scaling-study fixtures.
- **Cross-step evidence ledger**: release ledger now links Step 3/5/6/11/12/15/16/23 statuses, artifact links, telemetry presence, renderer/execution status, and concrete skip reasons per family.
- **Interpretability summaries**: per-family summaries now include variable/edge inventories, matrix-shape tables, telemetry presence, optional trace previews, renderer/execution status, and artifact links.

### Changed
- Continuous and hierarchical Step 11/12 outcomes are explicit profiled unsupported skips with concrete reasons, not raw render/execute failures accepted by profile math.
- v1.7.0 is retired as a foundation-only track; unfinished runtime-depth ambitions move forward into v2+ reliability and orchestration milestones.
- Current test evidence updated to 2,399 collected tests; final full-suite release evidence is recorded in `TO-DO.md`, `README.md`, and test documentation after the v1.9 release gate rerun.

### Fixed
- Removed the model-family acceptance reason-pattern fallback that could reclassify failed renderer/executor steps as unsupported success.
- Hardened strict acceptance so profiled unsupported steps must be skipped before execution and failed Step 11/12 summaries fail closed.
- Prevented cross-framework analysis from reading stale repo-tracked `output/` artifacts during isolated `/tmp` acceptance runs.
- Relaxed an environment performance smoke threshold to match other slow module smoke tests and avoid full-suite load false negatives.

---

## [1.8.0] — 2026-06-12

### Added
- **Template developer kit**: packaged template index, package-data template assets, `gnn templates list`, `gnn templates show NAME`, and `gnn pull NAME --output-dir ... --dry-run --overwrite`.
- **Template safety contracts**: checksum verification, collision handling, symlink/path traversal rejection, unknown-template failures, and installed-wheel smoke coverage outside the repo checkout.
- **MCP local HTTP orchestration**: bearer-token auth through `GNN_MCP_TOKEN`, localhost default binding, explicit insecure local opt-in with `GNN_MCP_ALLOW_INSECURE_LOCAL=1`, safe-tool filtering, optional rate limiting, and default-denied resource reads unless explicitly allowlisted.
- **Capability-contract verifier**: release-facing checks for template package data, MCP auth/resource safety, acceptance-command isolation, roadmap ordering, count drift, and autonomy non-mutation claims.
- **Roadmap foundations**: contract fixtures for v1.7 multi-agent/rendering/UI/audio/Three.js surfaces and v1.9 model-family acceptance/interpretability ledgers without marking those future release items complete.

### Changed
- v1.8 release evidence moved into the maintained roadmap and verifier surfaces rather than hard-coding historical live counts in this changelog section.
- `TO-DO.md` now treats v1.8.0 as the developer-kit release and v1.9.0 as the next model-family reliability target.
- Developer documentation now advertises verified template and MCP commands only, with `/tmp` output directories in acceptance smokes to avoid tracked `output/` churn.
- Pre-commit/dev tooling remains scoped to Ruff, file hygiene, and `just`/devcontainer ergonomics; dedicated secret scanning is not claimed.

### Fixed
- Removed release-facing false-certification paths around optional framework fallback, stale counts, MCP unauthenticated HTTP, unsafe MCP resource exposure, and template assets that only work from a source checkout.

---

## [1.6.0] — 2026-04-15

### Fixed
- **Testing Constraints**: Entirely removed dependency on internal `hypothesis.internal.conjecture.optimiser` arrays mitigating python 3.13 crashes, transitioning to native parameterized fuzzers.
- **HTML Assertion Accuracy**: Relaxed fixed `test_report_functional` strings to mathematically accept dynamic semantic UI tags (`<html lang="en">` instead of `<html>`).

### Added
- **Global Documentation Guardrails**: Enforced real-implementation documentation mandates. Provisioned `AGENTS.md`, `README.md`, and `SPEC.md` components natively within transient boundaries (`output/`, `.benchmarks/`, `scripts/`) and defined the master environment `SPEC.md` at root.
- **Type checker visual dashboards**: Baseball-card style model summary PNGs (dark neon theme), validity mosaics, issue distribution charts, and type category pie charts generated in `output/5_type_checker_output/visualizations/`
- **Resource estimation integration**: `estimate_file_resources()` now delegates to `estimation_strategies.py` for real FLOPS, memory, and complexity scoring instead of naive heuristics
- **Documentation hub sync**: Automated `doc/gnn/modules/[00-24].md` reconciliation from `src/*/AGENTS.md` source-of-truth

### Changed
- **Real-Implementation Policy Enforcement**: Removed dependency on patch-driven pytest plugins and tightened functional testing constraints.
- **Type checker consolidation**: Deleted redundant `src/type_checker/checker.py`; all logic unified in `processor.py` (`GNNTypeChecker`)
- **Test suite alignment**: `test_type_checker_overall.py` rewired to target production `processor.py` orchestrator
- **Deprecated marker removed**: `safe_to_fail` marker replaced with standard `xfail` in `pyproject.toml` and `pytest.ini`
- **Default local LLM**: Ollama default tag is `smollm2:135m-instruct-q4_K_S` (`llm.defaults.DEFAULT_OLLAMA_MODEL`); override with `OLLAMA_MODEL` or `input/config.yaml` `llm.model`.
- **Core dependencies**: `openai`, `ollama`, `python-dotenv`, and `aiohttp` are installed with the base package (LLM step and OpenRouter/Perplexity providers work from a plain `uv sync`).

---

## [1.3.0] — 2026-03-05

### Added
- **MCP integration**: 131 tools registered across 30 modules
- **LLM step**: Gemma 3 4B default model via Ollama (`--llm-model` configurable)
- **Cross-validation fix**: Dynamic fold logic `min(5, len(X), min_class_count)` eliminates target class sparsity warnings
- **Visual logging**: Real-time progress bars, color-coded output, correlation ID tracking, screen reader support
- **Enhanced audio**: SAPF and Pedalboard dual-backend sonification
- **Security hardening**: Restricted Unpickler (CWE-502), MD5→SHA256, NLTK CVE-2025-14009 and Authlib CVE-2026-28802 remediations

### Fixed
- **MCP deadlock**: Resolved multithreading deadlock in `discover_modules` causing silent timeouts
- **LLM glob**: Fixed recursive path issues during LLM processing logic
- **Orchestrator config**: `skip_steps` in `config.yaml` now properly respected

### Changed
- All 25 pipeline steps follow thin orchestrator pattern (100% compliant)
- Test suite expanded to 1,922+ tests across 108 files

---

## [1.2.0] — 2026-02-15

### Added
- **PyTorch renderer**: Full code generation for PyTorch-based Active Inference
- **NumPyro renderer**: Probabilistic programming code generation
- **Stan renderer**: Statistical modeling code generation
- **DisCoPy renderer**: Categorical diagram generation
- **Distributed execution**: Ray and Dask integration in `execute/distributed.py`
- **GPU utilities**: Auto-detection in `render/jax/gpu_utils.py`
- **Pipeline hasher**: Content-addressable run hashing for reproducibility
- **Preflight checks**: `gnn preflight` for environment validation

### Changed
- Renderer count increased from 4 to 8 (PyMDP, RxInfer, ActInf.jl, JAX, PyTorch, NumPyro, Stan, DisCoPy)
- Execute module expanded with framework-specific runners

---

## [1.1.0] — 2026-01-20

### Added
- **CLI tool**: `gnn` entry point with 12 subcommands (run, validate, parse, render, report, reproduce, preflight, health, serve, lsp, watch, graph)
- **API module**: FastAPI-based Pipeline-as-a-Service with SSE streaming
- **LSP server**: Real-time GNN file diagnostics and hover info for editors
- **GUI module**: 3 interactive editors (form builder, matrix editor, design studio)
- **Website module**: Static site generation with dashboards
- **Research module**: Experimental analysis and benchmarking tools

### Changed
- Pipeline expanded from 20 to 25 steps (0–24)
- Module count increased from 22 to 27

---

## [1.0.0] — 2025-12-01

### Added
- **Core pipeline**: 20-step processing pipeline (0–19)
- **GNN parser**: Markdown-based model file discovery and parsing
- **Type checker**: Static analysis and dimension validation
- **Export module**: JSON, XML, GraphML, Pickle serialization
- **Visualization**: Network graphs and matrix heatmaps
- **Render module**: PyMDP, RxInfer, ActiveInference.jl, JAX code generation
- **Execute module**: Simulation runner with ActiveInferenceAgent
- **LLM analysis**: AI-powered model interpretation
- **Ontology mapping**: Active Inference term annotation
- **Report generation**: Comprehensive pipeline reports

### Infrastructure
- Thin orchestrator pattern established
- UV-based dependency management
- pytest test suite with comprehensive coverage
- MCP tool registration framework

[Unreleased]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/compare/v1.9.0...v2.0.0
[1.9.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/compare/v1.8.0...v1.9.0
[1.8.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/compare/v1.6.0...v1.8.0
[1.6.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/compare/v1.3.0...v1.6.0
[1.3.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/releases
[1.1.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/releases
[1.0.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/releases
