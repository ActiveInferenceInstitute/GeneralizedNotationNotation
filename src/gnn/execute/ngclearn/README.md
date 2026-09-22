# ngc-learn Runner

Discovers and executes ngc-learn-generated POMDP scripts via subprocess.

## Usage

```python
from gnn.execute.ngclearn import run_ngclearn_scripts

results = run_ngclearn_scripts(
    render_dir="output/11_render_output", output_dir="output/12_execute_output"
)
```

## Features

- Dependency validation (checks `ngcsimlib` + `ngclearn` + `jax` availability)
- Syntax pre-validation before execution
- Log persistence (stdout/stderr capture)
- Wall-clock execution timing
- Skip-when-absent: without the `ngclearn` extra the backend reports skipped, never failed (`uv sync --extra ngclearn` installs it)

## See Also

- [Parent: execute/README.md](../README.md)
- [AGENTS.md](AGENTS.md) — Architecture documentation
