# REPORT — gnn-render (GNN documentation-vs-code audit)

Repo: /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation
Scope: REPORT-ONLY. No files edited/staged/committed. Only this report written.
Regions audited: doc/gnn/ (language, syntax, integration, implementations, modules, operations, reference, testing, mcp), doc/rxinfer/, doc/activeinference_jl/, doc/pymdp/, doc/execution/FRAMEWORK_*, doc/discopy/, doc/bnlearn/, doc/d2/, doc/pomdp/, doc/templates/

Method: extracted fenced (python/bash/julia/gnn/json) and inline code across all 287 tracked doc files in the region; verified every `python src/N_*.py`/`uv`/`julia` command against `git ls-files`; verified each documented `from X import Y` symbol by importing the real module in `.venv/bin/python` (PYTHONPATH=src) and checking `hasattr`; verified file paths via `git ls-files`; verified env/config tokens by `grep` in src/*.py. Only genuine discrepancies are reported; every command/import/path that resolved was excluded.

Key source anchors (verified present):
- All 26 numbered scripts `src/0_template.py … src/24_intelligent_analysis.py` + `src/main.py` exist and are tracked.
- Renderer files referenced by docs exist: `src/render/rxinfer/rxinfer_renderer.py`, `src/render/pomdp_contract.py`, `src/render/discopy/discopy_renderer.py`, `src/render/stan/stan_renderer.py`, `src/render/processor.py`, `src/render/framework_registry.py` (9 frameworks: PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, Stan, bnlearn).
- Execution envs `src/execute/rxinfer/{Project.toml,Manifest.toml,rxinfer_runner.jl}`, `src/execute/activeinference_jl/{Project.toml,Manifest.toml}` tracked.
- The two `export/discopy/analysis` cross-refs in catcolab.md`(../../../src/render/discopy/discopy_renderer.py`, `../../../src/execute/discopy/discopy_executor.py`, `../../../src/analysis/discopy/analyzer.py`) all resolve. catcolab.md's statements about `src/gnn/catcolab_importer.py` being absent and Step 7 having no `--format` flag are ACCURATE (the file/flag genuinely do not exist; grep confirms 0 uses).
- `scripts/run_pymdp_gnn_scaling_analysis.py` exists; referenced correctly as `../../scripts/...` in doc/pymdp/README.md and gnn_pymdp.md. (doc/pymdp/AGENTS.md / SPEC.md name it only by bare filename — fine.)
- `OLLAMA_MODEL`, `OLLAMA_HOST`, `OLLAMA_TIMEOUT`, `OLLAMA_MAX_TOKENS`, `OLLAMA_DISABLED`, `GNN_JAX_PLATFORM`, `TF_CPP_MIN_LOG_LEVEL` are all consumed in `src/*.py`. `execution_summary_detail` is consumed in `src/execute/processor.py`.

## Findings

### ERROR — broken Python import (module does not exist)

1. doc/gnn/modules/11_render.md:296 (and 309) | ERROR | `from render.renderer import render_gnn_spec` — there is no `src/render/renderer.py` module. Interpreter: `ModuleNotFoundError: No module named 'render.renderer'`. `render_gnn_spec` is real but lives in/under `src/render/processor.py` and is re-exported from the package root.
   Fix: use `from render import render_gnn_spec` (verified importable), or `from render.processor import render_gnn_spec`.

### ERROR — broken Python import (symbol not exported from the documented submodule)

2. doc/gnn/modules/01_setup.md:225, 232, 243, 250, 261 | ERROR | `from setup.setup import setup_uv_environment` / `add_uv_dependency` / `remove_uv_dependency` / `update_uv_dependencies` / `lock_uv_dependencies` — `src/setup/setup.py` does NOT re-export these UV helpers (only `check_system_requirements` is imported there, which is why line 269 works). Interpreter: `setup.setup.setup_uv_environment` FAILS; the same names resolve from the package root `setup`. The helpers actually live in `src/setup/uv_management.py` / `uv_package_ops.py` and are re-exported from `src/setup/__init__.py`.
   Fix: change to `from setup import setup_uv_environment` (and the four companion helpers). This matches the module's own guidance ("All setup helpers are exported from the package root. Prefer `from setup import …`").

3. doc/gnn/operations/REPO_COHERENCE_CHECK.md:221 (and 225, and prose on line 127) | ERROR | `from visualization import process_visualization_main` — `process_visualization_main` does not exist anywhere in `src/` (grep across src finds 0 defs/refs; `hasattr(visualization, 'process_visualization_main')` = False). The actual module-level visualization export is `process_visualization` (defined `src/visualization/core/process.py:91`, re-exported).
   Fix: use `from visualization import process_visualization`.

4. doc/gnn/operations/improvement_analysis.md:159 | ERROR | `from visualization import process_visualization_main` — same nonexistent symbol as finding 3.
   Fix: `from visualization import process_visualization`.

5. doc/gnn/testing/test_patterns.md:44 | ERROR | `from audio import backends` — `backends` is not an importable attribute of the `audio` package. It is a local dict created inside `src/audio/__init__.py:check_audio_backends()` (lines 65–95), never a module-level export. Interpreter: `hasattr(audio, 'backends')` = False.
   Fix: document `from audio import check_audio_backends` (the real public function) and note it returns the `backends` dict.

6. doc/pymdp/pymdp_pomdp/INTEGRATION_SUMMARY.md:194 | ERROR | `from src.execute.pymdp import batch_execute_pymdp` — `batch_execute_pymdp` is defined in `src/execute/pymdp/execute_pymdp.py:63` but is NOT re-exported by `src/execute/pymdp/__init__.py` (its `__all__` covers `execute_pymdp_simulation`, `execute_pymdp_simulation_from_gnn`, `run_pymdp_simulation`, etc. but not `batch_execute_pymdp`). Interpreter: `execute.pymdp.batch_execute_pymdp` FAILS (while `execute_pymdp_simulation` on the same doc lines 55/177/184 resolves OK).
   Fix: either re-export `batch_execute_pymdp` from `src/execute/pymdp/__init__.py`, or change the doc to import from `src.execute.pymdp.execute_pymdp import batch_execute_pymdp`.

### WARNING — file path exists but at a different tracked location

7. doc/gnn/reference/architecture_reference.md:125 | WARNING | "Input: `input/gnn_files/actinf_pomdp_agent.md`" — the tracked exemplar is at `input/gnn_files/discrete/actinf_pomdp_agent.md`; no flat `input/gnn_files/actinf_pomdp_agent.md` exists (`git ls-files` confirms only the `discrete/` path and `src/gnn/gnn_examples/actinf_pomdp_agent.md`).
   Fix: use `input/gnn_files/discrete/actinf_pomdp_agent.md`.

8. doc/gnn/modules/04_model_registry.md:224 | WARNING | JSON example `"file_path": "input/gnn_files/actinf_pomdp_agent.md"` — same flat-path as finding 7 (does not exist as a tracked path).
   Fix: use `input/gnn_files/discrete/actinf_pomdp_agent.md`.

### INFO — illustrative placeholder / cosmetic drift

9. `input/gnn_files/model.md` used as an example input path in doc/gnn/advanced/gnn_ontology.md:90,133; doc/gnn/integration/gnn_export.md:123,145; doc/gnn/integration/gnn_visualization.md:135; doc/gnn/reference/gnn_type_system.md:85,116 | INFO | `model.md` is a generic stand-in name in API examples, not a tracked exemplar (`git ls-files` finds no `model.md`). Utility is fine as an illustrative placeholder, but it is not a real path.
   Fix (optional): annotate as `<your-model>.md` or point to a real exemplar to avoid being read as a factual path.

10. doc/rxinfer/multiagent_trajectory_planning/README.md:56 | INFO | `cd doc/rxinfer/RxInferExamples.jl/scripts/Advanced Examples/Multi-agent Trajectory Planning/` — `RxInferExamples.jl` is not tracked in this repo; it is produced by cloning `doc/rxinfer/clone_rxinfer_examples.sh` (`git clone https://github.com/docxology/RxInferExamples.jl.git`). The README describes the cloned baseline. Not a defect, but the path only resolves after the clone step.
    Fix: add a note that this path requires first running `clone_rxinfer_examples.sh`.

## Clean regions

After verification, these documented items resolved correctly (no findings):
- doc/discopy/, doc/bnlearn/, doc/d2/, doc/pomdp/, doc/templates/: commands, paths, and code snippets verified; no broken paths (their `python src/N_*.py --target-dir input/gnn_files/...` invocations all reference real tracked scripts and real input-dir paths).
- doc/execution/FRAMEWORK_AVAILABILITY.md: framework list and Julia/env references consistent with `src/execute/`; RxInfer committed env paths (`src/execute/rxinfer`, `src/execute/activeinference_jl`) confirmed tracked.
- doc/gnn/mcp/ (client_setup.md, tool_development_guide.md, tool_reference.md): `src/21_mcp.py`, `src/mcp/validate_tools.py`, and `uv run --extra dev python -m pytest src/tests/mcp/test_mcp_audit.py` all resolve. The external `mcp` SDK client imports (`from mcp import ClientSession, StdioServerParameters`) are correct client-side usage of the official Model Context Protocol SDK, not the pipeline's `src/mcp`.
- doc/gnn/implementations/: every renderer path cross-ref resolves (rxinfer, pymdp, activeinference_jl, jax, numpyro, pytorch, stan, discopy). catcolab.md correctly documents the importer's/exporter's absence rather than claiming a broken path.
- doc/gnn/modules/ for 03, 04(except finding 8), 05, 06, 09, 10, 12, 14, 15–24: the `process_*` public functions and their module locations all imported successfully in `.venv/bin/python`.
- All documented `uv run ... python -m pytest src/tests/...` and `python src/main.py --only-steps ...` commands reference real test files, real numbered scripts, and valid steps.

## Summary

- ERROR: 6 (render.renderer nonexistent module; setup.setup helpers not re-exported; process_visualization_main nonexistent in two docs; audio.backends not importable; batch_execute_pymdp not exported)
- WARNING: 2 (actinf_pomdp_agent.md flat path in two docs)
- INFO: 2 (model.md placeholder; RxInferExamples.jl clone-dependent path)

Severity note: the 6 ERROR findings are the actionable ones — each is an import in a documented code example that would currently fail if copied verbatim. Fixes are localized (one import line each). The 4 path/placeholder items are lower impact.
