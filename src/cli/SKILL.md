---
name: gnn-cli-dispatch
description: "GNN command-line interface dispatching 12 subcommands to pipeline module APIs. Use when running GNN pipeline commands, validating GNN files, parsing models, generating code, or checking environment health via the CLI."
---

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

## Recommended Workflow

```bash
# 1. Verify environment is ready
gnn preflight

# 2. Validate GNN files before processing
gnn validate input/gnn_files/discrete/actinf_pomdp_agent.md --strict

# 3. Parse validated files
gnn parse input/gnn_files/discrete/actinf_pomdp_agent.md --format json

# 4. Generate simulation code
gnn render input/gnn_files/discrete/actinf_pomdp_agent.md --framework pymdp

# 5. Run full pipeline (or use --only-steps to target specific phases)
gnn run --target-dir input/gnn_files --verbose --only-steps "3,5,11,12"
```

## Features

- **Pipeline Execution**: Full 25-step pipeline via `gnn run` with `--skip-steps`, `--only-steps`, `--skip-llm` options
- **File Validation**: Section, state-space, connection, and dimension validation via `gnn validate`
- **Multi-Format Parsing**: JSON, YAML, and summary output via `gnn parse`
- **Code Generation**: PyMDP, RxInfer, JAX, NumPyro, Stan, PyTorch framework output via `gnn render`
- **Environment Checks**: Dependency and configuration validation via `gnn preflight` and `gnn health`
- **Live Development**: File monitoring with 250ms debounce via `gnn watch`
- **Dependency Graphs**: Mermaid/text graph output via `gnn graph`
- **API & LSP Servers**: FastAPI server (`gnn serve`) and Language Server (`gnn lsp`)

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Subcommand specification
