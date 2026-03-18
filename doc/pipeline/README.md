# Pipeline Documentation

This folder is the canonical home for pipeline-specific documentation.

## Running all execution frameworks

Step 12 (Execute) runs rendered scripts for PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, and NumPyro. JAX, NumPyro, PyTorch, and DisCoPy are optional; if not installed, their scripts are **skipped** (not failed). To run every implementation:

```bash
uv sync --extra execution-frameworks
```

Then run the pipeline or `python src/12_execute.py --frameworks all --verbose`.

## Start here

- **Step-by-step script guide**: `../PIPELINE_SCRIPTS.md`
- **Source specification**: `../../src/SPEC.md`
- **Step index (0–24)**: `../../src/STEP_INDEX.md`
- **Orchestrator implementation**: `../../src/main.py`

## Notes

- The pipeline consists of **25 steps (0–24)** implemented as thin orchestrators in `src/`.
- Per-folder routing is controlled by `input/config.yaml` (`testing_matrix`).

## Documentation coverage exclusions

When auditing “every folder under `src/` has `AGENTS.md`/`README.md`/`SPEC.md`”, treat the following as **generated artifacts**, not source modules:

- `__pycache__/` (bytecode caches)
- `src/output/` (pipeline outputs accidentally placed under `src/`)
- `src/tests/output/` (generated test artifacts, if present)

These are ignored by git (see `.gitignore`) and should not be required to contain module documentation.

