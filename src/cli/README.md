# GNN CLI Module

## Overview

Unified command-line interface for the GNN pipeline. Provides subcommands for running, validating, rendering, templating, and managing GNN models.

**Entry point**: `gnn = "src.cli:main"` (defined in `pyproject.toml`)

## Subcommands

| Command | Description |
|---------|-------------|
| `gnn run` | Execute the full 25-step pipeline |
| `gnn validate <file>` | Validate a GNN file (sections, state-space, connections, dimensions) |
| `gnn parse <file>` | Parse a GNN file and output JSON/YAML/summary |
| `gnn render <file>` | Render a GNN file to a specific framework (pymdp, rxinfer, jax, etc.) |
| `gnn report` | Generate pipeline report from existing outputs |
| `gnn reproduce <hash>` | Re-run from a previous run hash (content-addressable) |
| `gnn preflight` | Run environment & config checks |
| `gnn health` | Show renderer generator-module availability and environment preflight status |
| `gnn health --strict` | Exit nonzero when environment preflight reports errors |
| `gnn serve` | Start Pipeline-as-a-Service API (FastAPI) |
| `gnn templates list` | List maintained local GNN templates with checksums |
| `gnn templates show <name>` | Show one maintained template record |
| `gnn pull <name>` | Copy a maintained template into an input directory |
| `gnn lsp` | Launch GNN Language Server (stdio) |
| `gnn watch <dir>` | Monitor directory and live-reparse on file change |
| `gnn graph <file>` | Generate dependency graph from multi-model files |

## Usage

```bash
# Full pipeline
gnn run --target-dir input/gnn_files --verbose

# Validate a model
gnn validate input/gnn_files/discrete/actinf_pomdp_agent.md --strict

# Parse to JSON
gnn parse input/gnn_files/discrete/actinf_pomdp_agent.md

# Check environment
gnn preflight
gnn health
gnn health --strict

# Inspect and dry-run template installation
gnn templates list
gnn templates show pomdp-gridworld-3x3
gnn pull pomdp-gridworld-3x3 --output-dir /tmp/gnn-pull --dry-run
```

## Architecture

The CLI module is a thin dispatcher — each subcommand delegates to the corresponding module's public API:

- `run` → `main.main()`
- `validate` → `gnn.schema.validate_required_sections()` + `parse_state_space()` + `parse_connections()`
- `parse` → `gnn.schema.parse_state_space()` + `gnn.frontmatter.parse_frontmatter()`
- `render` → `render.processor` (planned full integration)
- `report` → `report.pipeline_report.generate_pipeline_report()`
- `reproduce` → `pipeline.hasher.lookup_run()` + `main.main(override_args=...)`
- `preflight` → `pipeline.preflight.run_preflight()`
- `health` → `render.health.check_renderers()` + `pipeline.preflight.check_environment()`
- `serve` → `api.app.start_server()`
- `templates` / `pull` → `cli.templates` maintained template index, checksum, and copy helpers
- `lsp` → `lsp.start_server()`
- `watch` → `gnn.watcher.GNNWatcher()`
- `graph` → `gnn.dep_graph.render_graph_from_file()`

## References

- [SPEC.md](SPEC.md) — Module specification
- [AGENTS.md](AGENTS.md) — Agent documentation
