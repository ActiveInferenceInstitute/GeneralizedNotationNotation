# JAX Templates

## Overview

`src/render/jax/templates/` contains **Python string templates** for generating JAX-based Python scripts.

**Important**: The current active renderer (`src/render/jax/jax_renderer.py`) **does not use** these templates; it generates code directly. These templates are available for future refactors or alternative rendering paths.

## Exports

From `src/render/jax/templates/__init__.py`:

- `GENERAL_TEMPLATE`
- `POMDP_TEMPLATE`
- `COMBINED_TEMPLATE`

## How to use

Each template is designed to be used with `str.format(...)`.

Example:

```python
from render.jax.templates import GENERAL_TEMPLATE

code = GENERAL_TEMPLATE.format(
    model_name="MyModel",
    timestamp="2026-03-17T00:00:00Z",
    source_file="input/gnn_files/my_model.md",
    model_parameters="A = ...\nB = ...\nC = ...\nD = ...",
    n_states=3,
    n_observations=3,
    n_actions=2,
)
```

## Placeholder keys (contract)

All templates require:

- `model_name`
- `timestamp`
- `source_file`
- `model_parameters`
- `n_states`
- `n_observations`
- `n_actions`

Additional placeholders:

- `discount` (POMDP + combined templates)
- `use_pomdp`, `use_neural` (combined template)

## Dependencies of generated code

These templates import **JAX**, and the general/combined templates also import **Optax** and **Flax**.

If you want a pure-JAX-only output, use the generator path in `src/render/jax/jax_renderer.py` rather than these templates.
