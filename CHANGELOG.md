# Changelog

All notable changes to the GNN Pipeline are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/) and [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

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
- **Core dependencies**: `openai`, `ollama`, `python-dotenv`, and `aiohttp` are installed with the base package (LLM step and OpenRouter/Perplexity providers work without `uv sync --extra llm`).

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
[1.2.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/releases/tag/v1.0.0
