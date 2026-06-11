# Changelog

All notable changes to the GNN Pipeline are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/) and [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- **Developer tooling**: `justfile` (21 recipes), `.pre-commit-config.yaml` (Ruff, file hygiene hooks), `.devcontainer/` (Python 3.11 + uv + just for GitHub Codespaces)
- **NumPyro/Stan E2E tests**: 15-test suite (`test_render_numpyro_stan.py`) validating render success, Python compilation, AST parsing, import correctness, type mapping, and empty-input handling
- **Renderer health verification**: All 8/8 backends confirmed operational (PyMDP, RxInfer, JAX, NumPyro, Stan, PyTorch, ActiveInference.jl, DisCoPy)

### Changed
- Test suite expanded to 2,200 passed, 70 skipped (from 1,906/30)
- Documentation version synchronized to 1.6.0 across 35 files (resolved v1.3.0/v1.5.0 drift)
- TO-DO.md rewritten with verified completed items and restructured roadmap
- Pre-commit detect-secrets hook removed (not in project deps; CI uses GitGuardian)

### Fixed
- 4 broken Mermaid diagram blocks in `doc/gnn/modules/` (00_template, 01_setup, 11_render, 21_mcp)
- Stale line-count claims for orchestrator scripts (11_render.py, 12_execute.py, main.py)
- Root `AGENTS.md` version (1.5.0 → 1.6.0) and `README.md` feature attribution (v1.5.0 → v1.6.0)

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

[Unreleased]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/compare/v1.6.0...HEAD
[1.6.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/compare/v1.3.0...v1.6.0
[1.3.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/releases/tag/v1.0.0
