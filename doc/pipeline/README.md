# Pipeline Documentation

This folder is the canonical home for pipeline-specific documentation.

## Running all execution frameworks

Step 12 (Execute) runs rendered scripts for PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, and NumPyro. JAX, NumPyro, PyTorch, and DisCoPy install with normal `uv sync` (core dependencies). If the environment is incomplete, those scripts are **skipped** (not failed). Julia backends need Julia. Then run the pipeline or `python src/12_execute.py --frameworks all --verbose`.

## Start here

- **Step-by-step script guide**: `../PIPELINE_SCRIPTS.md`
- **Documentation versioning** (language vs doc bundle vs package): [`../SPEC.md`](../SPEC.md)
- **Source specification**: `../../src/SPEC.md`
- **Step index (0–24)**: `../../src/STEP_INDEX.md`
- **Stage hardening review**: [`pipeline_stage_hardening_review.md`](pipeline_stage_hardening_review.md)
- **Orchestrator implementation**: `../../src/main.py`

## Notes

- The pipeline consists of **25 steps (0–24)** implemented as thin orchestrators in `src/`.
- Per-folder routing is controlled by `input/config.yaml` (`testing_matrix`).
- The maintained GridWorld proof path confirms configurable parse, type-check,
  visualization, render, execute, logging, analysis, PNG/GIF generation,
  dashboards, and manifests across PyMDP, RxInfer.jl, and ActiveInference.jl.

## Documentation coverage exclusions

When auditing “every folder under `src/` has `AGENTS.md`/`README.md`/`SPEC.md`”, treat the following as **not numbered pipeline modules**:

- `__pycache__/` (bytecode caches)
- `src/output/` (optional fixture subtree; not step code)
- `src/tests/output/` (generated test artifacts; ignored by git — see `.gitignore`)

Repository-root `output/` is ignored except for `output/.gitkeep`. Do not require full module scaffolding inside `output/` or `src/output/`.
