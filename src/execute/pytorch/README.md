# PyTorch Runner

Discovers and executes PyTorch-generated POMDP scripts via subprocess.

## Usage

```python
from execute.pytorch import run_pytorch_scripts

results = run_pytorch_scripts(
    render_dir="output/11_render_output",
    output_dir="output/12_execute_output"
)
```

## Features

- Dependency validation (checks `torch` availability)
- Syntax pre-validation before execution
- Log persistence (stdout/stderr capture)
- Wall-clock execution timing

## See Also

- [Parent: execute/README.md](../README.md)
- [AGENTS.md](AGENTS.md) — Architecture documentation
