# bnlearn Executor — Specification

## Module boundaries

- `bnlearn_runner.py` — the whole implementation (probe, discovery, execution).
- `__init__.py` — curated re-export surface (`__all__`).

No numbered-script surface: Step 12 (`src/gnn/12_execute.py`) reaches bnlearn
exclusively through the shared script path in `execute.processor` /
`execute.detection` plus this module's probes.

## Result record contract

`execute_bnlearn_script` / `run_bnlearn_scripts` return plain dicts with:

- `script`, `framework` (`"bnlearn"`), `language` (`"python"` | `"r"` |
  `"unknown"`)
- `success` (bool), `skipped` (bool)
- `return_code`, `stdout`, `stderr`, `execution_time_seconds`
- `results_file` (informational pointer to `<out>/simulation_results.json`)
- On skip: `reason` (missing runtime or unknown language)
- On failure: `error`, `error_type` (`"TimeoutExpired"` | envelope error type
  | `"RuntimeError"`)

## Environment

Subprocesses receive `BNLEARN_OUTPUT_DIR=<output_dir>` and run with
`cwd=<output_dir>`, under the shared `run_subprocess_envelope` semantics
(`GNN_SANDBOX` prefix honored; timeout enforced; structured envelope).

## Step 12 wiring points

1. `execute.detection` maps the `bnlearn/` render directory to framework
   `bnlearn` (pre-existing).
2. `execute.processor._build_execution_environment` sets `BNLEARN_OUTPUT_DIR`
   (mirrors `STAN_OUTPUT_DIR`).
3. `execute.planning` classifies bnlearn as a Python-probe framework so
   `plan_execute` reports `skip_dependency` for missing runtimes.
4. `gnn.utils.runtime_safety.framework_availability.FRAMEWORK_IMPORT_CHECK["bnlearn"]` drives the
   pre-flight skip in `execute_single_script` (pre-existing).
