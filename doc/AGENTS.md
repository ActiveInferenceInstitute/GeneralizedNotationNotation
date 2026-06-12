# doc/ — agent scaffolding

## Overview

Top-level documentation tree for GNN. **Human onboarding**: [README.md](README.md). **Flat link index**: [INDEX.md](INDEX.md). **Narrative tour**: [START_HERE.md](START_HERE.md). **Curated curricula**: [learning_paths.md](learning_paths.md). **Topic cross-links**: [CROSS_REFERENCE_INDEX.md](CROSS_REFERENCE_INDEX.md).

**Full directory inventory**: [INDEX.md](INDEX.md) (tree) and [expected_dirs.txt](expected_dirs.txt) (canonical name list for scripts and audits).

## Contents

| Path | Role |
|------|------|
| [gnn/](gnn/) | GNN language, syntax, tutorials, pipeline module pages |
| [pipeline/](pipeline/) | Pipeline-oriented documentation |
| [development/](development/) | Dev workflows, `docs_audit.py`, link rewrites |
| [dev/](dev/) | `src/` doc inventory script and notes |
| [testing/](testing/) | Testing strategy guide (see `src/tests/` for layout) |
| [troubleshooting/](troubleshooting/) | Errors, FAQ, pipeline recovery |
| [api/](api/) | REST / API docs |
| [configuration/](configuration/) | Config reference |
| [templates/](templates/) | GNN templates |
| [INDEX.md](INDEX.md), [START_HERE.md](START_HERE.md) | Doc navigation entry points |
| [SPEC.md](SPEC.md) | Versioning policy and doc-tree design requirements |
| Other top-level folders (`pymdp/`, `rxinfer/`, `bnlearn/`, …) | Framework and topic docs; see [INDEX.md](INDEX.md) |

**Last Updated**: 2026-04-14

## Pipeline

Processing steps and modules live under `src/`; see [../src/AGENTS.md](../src/AGENTS.md).

## Related

- [../AGENTS.md](../AGENTS.md) — repository-wide module registry
- [../README.md](../README.md) — project overview
- [development/agents_readme_triple_review.md](development/agents_readme_triple_review.md) — three-pass AGENTS/README audit checklist
