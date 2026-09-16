# bnlearn Executor (`src/gnn/execute/bnlearn/`)

## Purpose

Execute the bnlearn artifacts produced by the generator-backed Step 11 renderer
(`gnn.render.generators.generate_bnlearn_code`). Each rendered model ships a
Python program under `<model>/bnlearn/` (`import bnlearn as bn` +
`bn.make_DAG` + `bn.parameter_learning.fit`). Step 12 discovers it like any
other Python framework script (framework directory `bnlearn/`, output env var
`BNLEARN_OUTPUT_DIR`) and runs it; the shared pre-flight probe
(`utils.framework_availability`, mapping `bnlearn` → `bnlearn`) marks scripts
**skipped** when the module is absent.

## Language handling

The execution lane is derived from each emitted file's suffix, never assumed:

- `.py` — Python interpreter + the `bnlearn` module (the renderer's current
  and only output form).
- `.R` — Rscript + the R `bnlearn` package (`install.packages('bnlearn')`).
  Probed with `Rscript -e 'library(bnlearn)'` (60s timeout).

## Public API

| Symbol | Purpose |
|---|---|
| `is_bnlearn_available(python_executable=None)` | Python `bnlearn` module importable |
| `is_r_bnlearn_available(rscript_executable="Rscript")` | Rscript present and R `bnlearn` loads |
| `script_language(script)` | `"python"` / `"r"` / `"unknown"` from the suffix |
| `find_bnlearn_scripts(render_dir)` | every `.py`/`.R` under a `bnlearn/` render dir |
| `execute_bnlearn_script(script, out_dir)` | run one script with `BNLEARN_OUTPUT_DIR` set; skip with a reason when the lane's runtime is missing |
| `run_bnlearn_scripts(render_dir, out_dir)` | run all drivers; per-script explicit success/failure/skipped records |

## Dependency gating

`utils.framework_availability` maps `bnlearn` → `bnlearn`
(`uv sync --extra bnlearn`). Python-lane scripts skip with that hint; R-lane
scripts skip when Rscript or the R package is absent. Subprocess execution
uses the shared `run_subprocess_envelope` (timeout + `GNN_SANDBOX` semantics,
same as the Stan and PyMDP lanes).

## Tests

`tests/execute/test_execute_bnlearn.py` (offline: monkeypatched probes and
subprocess envelopes; no bnlearn/R runtime required in CI).
