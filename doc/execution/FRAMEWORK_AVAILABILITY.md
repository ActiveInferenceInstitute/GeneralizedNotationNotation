# Framework Availability Guide

GNN has two related framework inventories:

- **Render registry** (`src/render/framework_registry.py`): 9 targets, including Stan.
- **Step 12 executor** (`src/execute/processor.py`): 8 executable framework families —
  every render target except bnlearn. Stan executes via the cmdstanpy driver
  (`src/execute/stan/`) and is reported `skipped` when cmdstanpy/CmdStan is absent.

PyTorch is a supported render/execute path and bnlearn is a supported render path;
both are intentionally unavailable in the default lock because their dependency chain
currently carries a known unpatched PyTorch security concern. Julia targets require
their committed project environments.

## Check availability

Use the unified CLI before a run:

```bash
uv run gnn health
uv run gnn preflight
```

For a direct Python status report:

```bash
PYTHONPATH=src uv run python - <<'PY'
from execute import get_execution_health_status

for name, info in get_execution_health_status().items():
    state = "available" if info.get("available") else "unavailable"
    print(f"{name}: {state} — {info.get('reason', '')}")
PY
```

## Runtime checks

### Core Python targets

```bash
uv run python -c "from pymdp import Agent; print('PyMDP available')"
uv run python -c "import jax, numpyro, discopy; print('JAX, NumPyro, and DisCoPy available')"
```

### Julia targets

```bash
julia --startup-file=no --project=src/execute/rxinfer \
  -e 'using RxInfer; println("RxInfer.jl available")'
julia --startup-file=no --project=src/execute/activeinference_jl \
  -e 'using ActiveInference; println("ActiveInference.jl available")'
```

The matching `--project` is required. The executor uses the same project-specific
environments when launching rendered scripts.

## Selection examples

```bash
# Python-only quick preset.
uv run python src/12_execute.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --frameworks lite \
  --verbose

# Explicit requested frameworks. Missing requested frameworks are reported clearly.
uv run python src/12_execute.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --render-output-dir output/11_render_output \
  --frameworks "pymdp,jax" \
  --verbose
```

The executor has no `--dry-run` flag. Use `gnn health` and `gnn preflight` for
non-execution checks.

## Interpret the result

Inspect `output/12_execute_output/` and the pipeline summary. Distinguish:

- **Succeeded**: a rendered script ran and returned a successful result.
- **Skipped/unavailable**: a dependency or runtime was not present.
- **Failed**: an available/requested script ran and returned an error.

Do not report a fixed `N/M` success count in documentation. Counts depend on the
input corpus, selected frameworks, and local runtimes; use the generated execution
summary for a specific run.

## Related references

- [Setup](../SETUP.md)
- [Pipeline](../pipeline/README.md)
- [Render registry](../../src/render/framework_registry.py)
- [Execute module](../../src/execute/AGENTS.md)
- [Troubleshooting](../troubleshooting/README.md)
