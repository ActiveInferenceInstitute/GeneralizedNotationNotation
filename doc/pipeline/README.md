# Pipeline Documentation

This is the canonical home for pipeline-specific guidance. The numbered scripts are
thin orchestrators; the authoritative order and metadata live in
`src/pipeline/step_registry.py`.

## Run a focused pipeline

```bash
uv run python src/main.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --only-steps "3,5,11,12" \
  --verbose
```

Use `--skip-steps "13,15"` or `--skip-llm` when a local runtime is not available.
The main pipeline loads `input/config.yaml` automatically. See the
[configuration guide](../configuration/README.md) for the supported YAML sections.

## Step boundaries

- Steps 0–10 discover, parse, validate, export, visualize, and annotate models.
- Step 11 has 9 render targets: PyMDP, RxInfer.jl, ActiveInference.jl, JAX,
  DisCoPy, PyTorch, NumPyro, Stan, and bnlearn.
- Step 12 has 8 executor families: PyMDP, JAX, DisCoPy, RxInfer.jl,
  ActiveInference.jl, PyTorch, NumPyro, and Stan. bnlearn is render-only: Step 11
  renders bnlearn scripts, Step 12 never executes them. Stan runs the rendered
  cmdstanpy driver (`src/execute/stan/`); it needs `uv sync --extra stan` plus a
  CmdStan toolchain, otherwise it is reported skipped. Continuous (linear-Gaussian)
  exemplars execute on jax, numpyro, pytorch, stan and rxinfer; the categorical
  backends report `unsupported` for them and are not executed.
- Steps 13–24 provide LLM, ML, audio, analysis, integration, security, research,
  website, MCP, GUI, reporting, and intelligent-analysis surfaces.

A missing optional runtime is represented as skipped/unavailable in the execution
summary. It is not equivalent to a successful execution.

## Framework selection

```bash
# Render only selected targets.
uv run python src/11_render.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --frameworks "pymdp,jax" \
  --strict-framework-success

# Execute only the selected framework scripts from a known render directory.
uv run python src/12_execute.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --render-output-dir output/11_render_output \
  --frameworks "pymdp,jax" \
  --timeout 600
```

`all` and `lite` are presets. The exact lists are implemented in
`src/execute/processor.py::parse_frameworks_parameter`; do not infer executor
coverage from the renderer registry.

## Validation and acceptance checks

```bash
# Documentation contracts and link/anchor audit.
uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write
uv run --extra dev python scripts/check_doc_contracts.py --strict

# Pipeline orchestration acceptance.
PYTHONPATH=src uv run --extra dev python scripts/run_v3_orchestration_acceptance.py --strict
```

## Related references

- [Pipeline scripts](../PIPELINE_SCRIPTS.md)
- [Source step index](../../src/STEP_INDEX.md)
- [Source pipeline README](../../src/README.md)
- [Framework integration guide](../gnn/integration/framework_integration_guide.md)
- [Setup](../SETUP.md)
- [Troubleshooting](../troubleshooting/README.md)
