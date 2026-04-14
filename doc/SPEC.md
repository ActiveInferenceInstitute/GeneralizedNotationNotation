# Specification: `doc/` tree

## Versioning policy

These version strings refer to **different things**:

| Name | Where | Meaning |
|------|--------|---------|
| **GNN language / syntax** | [doc/gnn/reference/gnn_syntax.md](gnn/reference/gnn_syntax.md) | Language rules (e.g. v1.1). |
| **GNN documentation bundle** | Front matter on major [doc/gnn/](gnn/) pages | Human-oriented hub revision (e.g. v2.0.0). |
| **Python package** | [pyproject.toml](../pyproject.toml) `version` | Installable distribution (e.g. **1.3.0**). |

Leaf `SPEC.md` files under experimental or research subtrees may omit bundle version; they should still link **up** to a parent SPEC or state scope.

**Dates**: Prefer updating `Last Updated` on hub pages when content changes. Leaf docs may inherit currency from parent README/AGENTS unless materially edited.

## Design requirements

- **`doc/`** is the static documentation tree; it does not execute pipeline steps. Runtime behavior lives under `src/`.
- **Top-level folder list**: [expected_dirs.txt](expected_dirs.txt) is the canonical list of `doc/<name>/` directories for tooling; update it when adding or renaming a top-level doc subtree.
- **Mechanical completeness** is enforced by [development/docs_audit.py](development/docs_audit.py) (`uv run python doc/development/docs_audit.py --strict` from repo root). With `--strict` and failures, the tool prints every issue to stderr by default; use `-q` for summary only. Optional: `--check-anchors` validates `#fragments` against heading slugs (heuristic).
- **GNN authority** for syntax and pipeline narrative: [doc/gnn/README.md](gnn/README.md), [doc/gnn/SPEC.md](gnn/SPEC.md), [CLAUDE.md](../CLAUDE.md), [src/AGENTS.md](../src/AGENTS.md).

## Components

- **Hubs**: [README.md](README.md), [INDEX.md](INDEX.md), [learning_paths.md](learning_paths.md), [SETUP.md](SETUP.md), [AGENTS.md](AGENTS.md).
- **Subtrees**: `gnn/`, `api/`, `deployment/`, `development/`, `cognitive_phenomena/`, framework-specific folders (`pymdp/`, `rxinfer/`, …), each with README + AGENTS where maintained.

## Interfaces

- Cross-links use repo-relative Markdown paths from the linking file.
- For current test commands and pass counts, use [CLAUDE.md](../CLAUDE.md) (measured in CI / local `uv run pytest`); do not treat stale inline numbers as ground truth.
