# gnn.manuscript — Manuscript Token Production

## Overview

**Purpose:** Deterministic production of the manuscript's `{{...}}` token
map: reads the repository snapshot at the current commit, computes every
count the manuscript cites, persists
`output/data/manuscript_variables.json` for audit, and supports config
metadata synchronization and cross-framework family selection.

**Pipeline step:** Manuscript tooling (feeds the render pipeline and the
token checker).

**Category:** Manuscript infrastructure.

**Status:** Maintained.

**Version:** 1.0.0

## Core Functionality

1. `generate_variables()` — compute the full token map from the repository
   snapshot at the current commit.
2. `RepositorySnapshot` — git-snapshot reader backing every count.
3. `load_variables()` / `save_variables()` — read/write the persisted map.
4. `sync_config_metadata()` / `config_metadata_drift()` — keep
   `manuscript/config.yaml` producer-owned fields in step.
5. `select_cross_framework_family()` — family selection for the
   cross-framework coverage note.
6. `token_checksum()` — integrity checksum over the emitted map.
7. `record_render_manifest()` / `custody_issues()` / `verify_fresh_render()`
   — render custody (`render_custody.py`): digest the committed render
   artifacts + hydrated inputs into `output/data/manuscript_render_manifest.json`
   after a render, audit the committed chain against it, and classify a
   fresh render against the committed manifest (`[FAIL]` = artifact missing
   or artifact+input joint drift → chain stale for HEAD; `[WARN]` =
   artifact-only drift with inputs matching → toolchain variance).

## Module Structure

- `__init__.py` — thin re-export surface (`__all__`).
- `variables.py` — the implementation (snapshot counting, token emission).
- `render_custody.py` — render custody manifest: record after a render,
  audit the committed chain, verify a fresh render against the committed
  manifest (SC-22).

## Dependencies

- Stdlib (`ast`, `hashlib`, `json`, `re`, `subprocess`, `fnmatch`,
  `pathlib`, `tomllib`) plus `yaml` (optional). No `gnn.*` imports — the
  producer must stay headless-importable.

## Testing

```bash
uv run --extra dev python -m pytest tests/main/test_manuscript_variables.py tests/main/test_manuscript_variables_api.py -q
uv run --extra dev python -m pytest tests/test_manuscript_latex_log.py -q
```
