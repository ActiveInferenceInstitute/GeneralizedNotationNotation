# bnlearn Executor

Step 12 execution support for the generator-backed bnlearn render target
(``src/gnn/render/generators.py::generate_bnlearn_code``).

## What it runs

Each rendered model directory contains a ``bnlearn/`` subdirectory with one
Python program per model (``Enhanced<Name>BnlearnAnalyzer``: builds a DAG,
simulates categorical traces, fits CPTs with MLE, and runs exact inference).
The runner executes it with ``BNLEARN_OUTPUT_DIR`` pointing at the model's
``simulation_data`` directory.

## Dependency gates

| Lane | Requirement | Skip behavior |
|---|---|---|
| Python (`.py`) | `bnlearn` module (`uv sync --extra bnlearn`) | script marked `skipped` with the install hint |
| R (`.R`) | `Rscript` on PATH + R `bnlearn` package | script marked `skipped` with the install hint |

Records always carry explicit `success` / `skipped` / `error` /
`error_type` fields; a timeout produces `error_type: "TimeoutExpired"`.

## Usage

```python
from gnn.execute.bnlearn import (
    execute_bnlearn_script,
    find_bnlearn_scripts,
    is_bnlearn_available,
    run_bnlearn_scripts,
)

if is_bnlearn_available():
    results = run_bnlearn_scripts(
        "output/11_render_output", "output/12_execute_output/bnlearn"
    )
    failed = [r for r in results if not r["success"] and not r["skipped"]]
```

Step 12 needs no direct call: rendered scripts are discovered and executed by
the standard script path (`execute.detection` + `execute.processor`), and
`plan_execute` classifies bnlearn scripts as `skip_dependency` when the
module is absent.
