# Core Skill: `cli_dispatch`

**Function**: Unified command-line interface dispatching 12 subcommands to their respective GNN pipeline module APIs.

## Example Flow

```bash
# Full pipeline execution
gnn run --target-dir input/gnn_files --verbose

# Validate a single GNN file
gnn validate input/gnn_files/discrete/actinf_pomdp_agent.md --strict

# Parse to JSON output
gnn parse input/gnn_files/discrete/actinf_pomdp_agent.md --format json

# Generate code for a specific framework
gnn render input/gnn_files/discrete/actinf_pomdp_agent.md --framework pymdp

# Check environment health
gnn preflight
gnn health
```

## Programmatic Usage

```python
from cli import main

# The CLI entry point dispatches to module APIs
# Each subcommand maps to a specific module function:
#   run       → main.main()
#   validate  → gnn.schema.validate_required_sections()
#   parse     → gnn.schema.parse_state_space()
#   render    → render.processor
#   health    → render.health.check_renderers()
#   lsp       → lsp.start_server()
```

## Features

- **Pipeline Execution**: Full 25-step pipeline via `gnn run` with `--skip-steps`, `--only-steps`, `--skip-llm` options
- **File Validation**: Section, state-space, connection, and dimension validation via `gnn validate`
- **Multi-Format Parsing**: JSON, YAML, and summary output via `gnn parse`
- **Code Generation**: PyMDP, RxInfer, JAX, NumPyro, Stan, PyTorch framework output via `gnn render`
- **Environment Checks**: Dependency and configuration validation via `gnn preflight` and `gnn health`
- **Live Development**: File monitoring with 250ms debounce via `gnn watch`
- **Dependency Graphs**: Mermaid/text graph output via `gnn graph`
- **API \& LSP Servers**: FastAPI server (`gnn serve`) and Language Server (`gnn lsp`)


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `cli.health`
- `cli.preflight`
