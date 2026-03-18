# Stan Render Backend - Agent Scaffolding

## Module Overview

**Purpose**: Generate **Stan model code** from parsed GNN variables and connections.

**Parent**: `src/render/` (Step 11: Render)

**Primary entrypoint**: `render_stan` in `stan_renderer.py` (re-exported by `__init__.py`).

---

## Public API

From `src/render/stan/__init__.py`:

- `render_stan(variables: List[Dict[str, Any]], connections: List[Dict[str, Any]], model_name: str = "gnn_model") -> str`

**Contract**:
- returns a Stan program as a string containing `data {}`, `parameters {}`, and `model {}` blocks

---

## Implementation notes

`stan_renderer.py`:

- applies simple heuristics to classify variables into `data` vs `parameters`
- emits directed connection comments and a default `normal(source, 1.0)` likelihood for each directed edge

This backend is intentionally conservative: it produces valid Stan syntax from limited structural information. It does not attempt to encode full Active Inference semantics.

---

## Integration points

- **Called by**: `render` module when a Stan target is selected (where supported by the parent processor).

---

## Testing

Preferred tests:

- syntax checks: generated output contains required Stan blocks and is non-empty
- small golden tests: variable/connection inputs → stable Stan output

