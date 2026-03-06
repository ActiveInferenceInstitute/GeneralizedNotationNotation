# GNN Documentation Hub — Agent Manifest (`AGENTS.md`)

**Role**: Root-level manifest for the `doc/gnn/` documentation tree.  
**Version**: v2.0.0  
**Context**: Canonical index of all GNN documentation, organized by domain.

## Directory Identity

- **Path**: `doc/gnn/`
- **Purpose**: Complete documentation hub for Generalized Notation Notation — language specification, pipeline modules, framework implementations, testing, operations, and tutorials.
- **Pipeline Steps**: 25 (0–24)
- **Modules**: 38+
- **MCP Tools**: 131
- **Renderers**: 8/8 (pymdp, rxinfer, jax, numpyro, stan, pytorch, activeinference_jl, discopy)
- **Tests**: 1,522+

## Subdirectory Index

| Directory | Purpose | Content Files |
|-----------|---------|---------------|
| `advanced/` | Advanced modeling, ontology, multi-agent, LLM/neurosymbolic | 5 |
| `implementations/` | Framework-specific docs (8 renderers) | 8 |
| `integration/` | Cross-framework integration guides | 4 |
| `language/` | GNN syntax cheatsheet and language-level specifications | 3 |
| `mcp/` | MCP tool reference, client setup, development guide | 3 |
| `modules/` | Per-step documentation for all 25 pipeline steps + init + main | 28 |
| `operations/` | Pipeline tools, troubleshooting, coherence, metrics | 5 |
| `reference/` | Architecture, DSL, schema, standards, type system | 8 |
| `testing/` | Test patterns, MCP audit | 2 |
| `tutorials/` | Quickstart tutorial, GNN examples | 2 |

## Root-Level Files

| File | Purpose |
|------|---------|
| `README.md` | Human-readable documentation index and navigation |
| `AGENTS.md` | Machine-readable manifest (this file) |
| `about_gnn.md` | Detailed GNN specification and motivation |
| `gnn_overview.md` | High-level overview, Triple Play, ecosystem context |
| `gnn_paper.md` | Full text of the GNN Zenodo publication |
| `gnn_syntax.md` | GNN v1.1 syntax specification (living document) |

## New Modules (v1.8.0–v2.0.0)

The following source modules were added since v1.3.0 and should be reflected in documentation:

| Module | Path | Purpose |
|--------|------|---------|
| CLI | `src/cli/__init__.py` | 10 subcommands (run, validate, parse, etc.) |
| API | `src/api/app.py` | FastAPI Pipeline-as-a-Service (6 endpoints) |
| LSP | `src/lsp/__init__.py` | Language Server Protocol diagnostics + hover |
| Stan renderer | `src/render/stan/` | Stan code generation |
| Parse cache | `src/gnn/parse_cache.py` | Section-level incremental cache |
| Multi-model | `src/gnn/multimodel.py` | Multi-model file support |
| Front-matter | `src/gnn/frontmatter.py` | YAML front-matter parser |
| Watcher | `src/gnn/watcher.py` | File watcher with auto-validation |
| Dep graph | `src/gnn/dep_graph.py` | Model dependency visualization |
| Contracts | `src/gnn/contracts.py` | Framework-specific output contracts |
| Hasher | `src/pipeline/hasher.py` | Content-addressable run hashing |
| Preflight | `src/pipeline/preflight.py` | Config + environment validation |
| DAG | `src/pipeline/dag.py` | Topological sort executor |
| Context | `src/pipeline/context.py` | Typed in-memory pipeline state |
| Schemas | `src/pipeline/schemas.py` | Pydantic pipeline data contracts |
| Step registry | `src/pipeline/step_registry.py` | Pluggable step discovery |
| Logging | `src/pipeline/logging_config.py` | Structured JSON logging |
| Health check | `src/render/health.py` | Renderer availability check |
| Remediation | `src/intelligent_analysis/remediation.py` | Auto-fix suggestions |

## Integration

All files in this tree are documentation and specification artifacts. They integrate with:

- `src/main.py` — Pipeline orchestrator (25 steps)
- `src/cli/__init__.py` — CLI entrypoint (`gnn` command)
- `src/api/app.py` — REST API server
- `src/lsp/__init__.py` — Language Server
