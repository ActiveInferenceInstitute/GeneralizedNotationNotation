# JAX Templates - Agent Scaffolding

## Module Overview

**Purpose**: Provide **string templates** for generating JAX-based Python scripts (general model, POMDP solver, combined model).

**Status**: Implemented (templates are present and importable).

**Scope**: This folder contains **data (template strings)**, not a pipeline step and not an orchestrator.

**Current integration status**: The active renderer implementation in `src/render/jax/jax_renderer.py` **does not import or use** these templates; it generates code directly. These templates are kept for future refactors and alternative code layouts.

---

## Public API

Exports are defined in `src/render/jax/templates/__init__.py`:

- `POMDP_TEMPLATE` (from `pomdp_template.py`)
- `GENERAL_TEMPLATE` (from `general_template.py`)
- `COMBINED_TEMPLATE` (from `combined_template.py`)

Each export is a Python string intended to be used with `str.format(...)`.

---

## Template Contracts

### Required `.format(...)` keys

All templates share these required keys:

- `model_name`: display-safe model identifier
- `timestamp`: generation timestamp string
- `source_file`: original GNN source path/name
- `model_parameters`: Python code snippet that defines matrices and scalar parameters used by the script
- `n_states`, `n_observations`, `n_actions`: integer dimensions

Additional keys:

- `discount`: used by `POMDP_TEMPLATE` and `COMBINED_TEMPLATE`
- `use_pomdp`, `use_neural`: used by `COMBINED_TEMPLATE`

### Dependency expectations (generated code)

- `GENERAL_TEMPLATE`: imports `jax`, `optax`, `flax.linen`
- `POMDP_TEMPLATE`: imports `jax`, `optax`, `flax.linen` (even if a future implementation chooses to omit neural pieces)
- `COMBINED_TEMPLATE`: imports `jax`, `optax`, `flax.linen`

If these templates are wired back into `jax_renderer.py`, the parent docs must be updated to reflect that dependency policy.

---

## Testing guidance

These templates are pure strings. The most useful tests are:

- **format-contract tests**: assert that all required keys are documented and formatting does not raise
- **syntax checks**: write formatted templates to a temp file and verify `python -m py_compile` succeeds

End-to-end integration tests remain owned by Step 11 render tests under `src/tests/`.
