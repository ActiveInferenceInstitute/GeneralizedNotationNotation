# GeneralizedNotationNotation (GNN) Documentation

This is the primary human-facing documentation hub. Other navigation files have
narrow roles:

- [START_HERE.md](START_HERE.md) is a short audience map.
- [INDEX.md](INDEX.md) is a flat link index.
- [learning_paths.md](learning_paths.md) contains the curricula.
- [CROSS_REFERENCE_INDEX.md](CROSS_REFERENCE_INDEX.md) maps topics and frameworks.
- [AGENTS.md](AGENTS.md) describes the documentation tree for maintainers.

The code and the live CLI are authoritative for behavior. Documentation examples are
kept intentionally small and are validated by the repository documentation checks.

## Choose a path

### New to GNN

1. [What is GNN?](gnn/about_gnn.md)
2. [Quickstart tutorial](gnn/tutorials/quickstart_tutorial.md)
3. [Syntax reference](gnn/reference/gnn_syntax.md)
4. [Examples](gnn/tutorials/gnn_examples_doc.md)

### Running the repository

1. [Setup](SETUP.md)
2. [Configuration](configuration/README.md)
3. [Pipeline guide](pipeline/README.md)
4. [Troubleshooting](troubleshooting/README.md)
5. [Framework availability](execution/FRAMEWORK_AVAILABILITY.md)

### Developing or integrating

1. [Pipeline architecture](../src/README.md)
2. [Pipeline scripts](PIPELINE_SCRIPTS.md)
3. [GNN implementation guide](gnn/integration/gnn_implementation.md)
4. [Framework integration](gnn/integration/framework_integration_guide.md)
5. [API documentation](api/README.md)
6. [Development guide](development/README.md)

### Research and theory

- [GNN overview](gnn/gnn_overview.md)
- [GNN paper and formal background](gnn/gnn_paper.md)
- [Active Inference foundations](active_inference/README.md)
- [Cognitive phenomena](cognitive_phenomena/README.md)
- [Advanced modeling patterns](gnn/advanced/advanced_modeling_patterns.md)

## Basic Examples

- [Quickstart tutorial](gnn/tutorials/quickstart_tutorial.md)
- [Example gallery](gnn/tutorials/gnn_examples_doc.md)
- [Basic template](templates/basic_gnn_template.md)

## Framework Integrations

See the [framework integration guide](gnn/integration/framework_integration_guide.md)
and [framework implementations](gnn/implementations/README.md).

## Current implementation map

The main pipeline has 25 numbered steps, 0–24. The canonical order and descriptions
live in `src/pipeline/step_registry.py`; the generated/maintainer-facing tables are
in [src/STEP_INDEX.md](../src/STEP_INDEX.md) and [PIPELINE_SCRIPTS.md](PIPELINE_SCRIPTS.md).

Framework boundaries are deliberately explicit:

- Step 11 exposes 9 render targets: PyMDP, RxInfer.jl, ActiveInference.jl, JAX,
  DisCoPy, PyTorch, NumPyro, Stan, and bnlearn.
- Step 12 executes 8 framework families. Stan is render-only in this pipeline;
  PyTorch and bnlearn are registry-gated and are not installed by the default lock.
- Missing optional runtimes are reported as skipped/unavailable; they are not silently
  represented as successful executions.

Read [Framework Implementations](gnn/implementations/README.md) for per-target
material and [the framework integration guide](gnn/integration/framework_integration_guide.md)
for the render → execute → analyze contract.

## Command reference

All commands below run from the repository root:

```bash
# Install and inspect the environment.
uv sync --extra dev
uv run gnn preflight
uv run gnn health

# Validate one file.
uv run gnn validate input/gnn_files/discrete/actinf_pomdp_agent.md --strict

# Run a focused pipeline path.
uv run python src/main.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --only-steps "3,5,11,12" \
  --verbose
```

The main pipeline uses `input/config.yaml` automatically. Step selection uses
`--only-steps` and `--skip-steps`; setup uses hyphenated flags such as
`--install-optional`, `--optional-groups`, and `--recreate-uv-env`. See the
[configuration guide](configuration/README.md), not a generic `--config` or profile
system.

## Documentation quality checks

```bash
uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write
uv run --extra dev python scripts/check_doc_contracts.py --strict
uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict
uv run --extra dev python scripts/check_maintained_doc_terms.py --strict
```

The audit checks links, anchors, and documentation scaffolding. The contract check
covers the executable quickstart fixture, supported CLI spellings, configuration
location, and framework split. It does not replace running a real pipeline.

## Inbound navigation anchors

These headings preserve inbound links from older topic pages while the navigation
hubs are consolidated above.

### Basic Examples

See [Basic Examples](gnn/tutorials/gnn_examples_doc.md).

### Framework Integrations

See [Framework Integrations](gnn/integration/framework_integration_guide.md).

## Maintainer notes

- Prefer links to source-backed references over duplicated inventories.
- Do not embed test counts, file counts, byte sizes, or “production-ready” claims
  unless the measurement is generated by a checked-in command and scoped to a dated
  run.
- Keep generated pipeline artifacts under `output/`; do not treat them as maintained
  documentation.
- Keep the GNN syntax examples aligned with `src/gnn/schema.py` and validate examples
  with `gnn validate` or Step 5 before publishing them.

For versioning policy, see [SPEC.md](SPEC.md). For contribution conventions, see
[style_guide.md](style_guide.md) and [../CONTRIBUTING.md](../CONTRIBUTING.md).
