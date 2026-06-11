# NumPyro Runner

Discovers and executes NumPyro-generated POMDP scripts via subprocess.

## Usage

```python
from execute.numpyro import run_numpyro_scripts

results = run_numpyro_scripts(
    render_dir="output/11_render_output",
    output_dir="output/12_execute_output"
)
```

## Features

- Dependency validation (checks `numpyro` + `jax` availability)
- Syntax pre-validation before execution
- Log persistence (stdout/stderr capture)
- Wall-clock execution timing

## See Also

- [Parent: execute/README.md](../README.md)
- [AGENTS.md](AGENTS.md) — Architecture documentation
