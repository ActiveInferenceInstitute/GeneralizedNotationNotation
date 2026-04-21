# Stan Framework Implementation

> **GNN Integration Layer**: Stan probabilistic programming language
> **Framework Base**: Stan (statistical modeling language, MCMC / variational inference)
> **Documentation Version**: v1.6.0 Engine (Bundle v2.0.0)
> **Scope**: **Structural-only.** This backend emits syntactically valid Stan
> from GNN structure (variables, connections, dimensions); it does NOT encode
> full Active Inference semantics (no expected-free-energy policy selection,
> no belief propagation). See "Limitations" below.

## Overview

The Stan renderer translates a parsed GNN specification into a single `.stan`
program file. The generated program has the three canonical Stan blocks
(`data`, `parameters`, `model`) populated from the GNN variables and
connections, with matrix literals emitted directly from
`InitialParameterization`. The output parses cleanly under `stan --check`
and can be sampled with CmdStan or bridged through PyStan / RStan.

Stan is intentionally a **second-class backend**: use it when you want to
expose a GNN-derived graphical model to Stan's mature MCMC tooling rather
than to run a full Active Inference loop. For the latter, use PyMDP,
RxInfer.jl, ActiveInference.jl, or JAX.

## Architecture

| Stage | Module | Description |
|-------|--------|-------------|
| Rendering (Step 11) | `src/render/stan/stan_renderer.py` | GNN → Stan program string |
| Code emission | `_emit_data_block`, `_emit_parameters_block`, `_emit_model_block` | Classify + format variables per block |
| Matrix formatting | `_matrix_to_stan` | Tuple / list literals → Stan array syntax |
| Output | `model.stan` in per-model/per-framework dir | Written by `render/processor.py` |

## GNN Parameter Ingestion

The renderer reads from the parsed GNN dict:

| GNN section | Stan destination |
|-------------|------------------|
| `StateSpaceBlock` | variable declarations, dimensioned into `data` or `parameters` by block classification |
| `Connections` | emitted as `// GNN connection: A > B` comments above the model block |
| `InitialParameterization` | inlined as matrix literals in `data` block (constants) or as Stan priors in `parameters` / `model` |
| `ModelParameters.num_hidden_states` / `num_obs` / `num_actions` | declared as `int` data and used to size arrays |
| `Time.Dynamic` | causes variables subscripted with `_t` to be indexed over `num_timesteps` |

## Block Classification Rules

Stan's strict separation between `data`, `parameters`, and `model` blocks
requires the renderer to classify each GNN variable:

1. **`data` block**: observed variables (those referenced in `## ActInfOntologyAnnotation` as `Observation`) and dimensional constants from `ModelParameters`.
2. **`parameters` block**: unobserved / latent variables (`HiddenState`, `Policy`, `Preference` ontology mappings), and any matrix declared without an initial value.
3. **`model` block**: likelihood assertions derived from directed connections + matrix literals where initial parameters act as priors.

Variables with ambiguous ontology (no `ActInfOntologyAnnotation` entry) default
to `parameters` with a diffuse normal prior.

## Matrix Literal Injection

GNN matrix literals in `InitialParameterization` are converted to Stan's
array syntax. A 2×2 row-major tuple:

```gnn
A={(0.9,0.1),(0.1,0.9)}
```

becomes:

```stan
matrix[2,2] A = [[0.9, 0.1], [0.1, 0.9]];
```

3D tuples (e.g. `B` for per-action transitions) become `array[] matrix[,]`:

```gnn
B={((1,0),(0,1)),((0,1),(1,0))}
```

→

```stan
array[2] matrix[2,2] B = {[[1,0],[0,1]], [[0,1],[1,0]]};
```

## Generated Code Example

For the sample `input/gnn_files/basics/static_perception.md` (2 states, 2
observations, A matrix), the renderer produces approximately:

```stan
// Generated from GNN: static_perception.md
// GNN connection: D > s
// GNN connection: s - A
// GNN connection: A - o

data {
  int<lower=1> num_obs;       // declared in ModelParameters
  int<lower=1> num_hidden_states;
  array[1] int<lower=1,upper=num_obs> o;  // observation
}

parameters {
  simplex[num_hidden_states] D;           // prior over states
  simplex[num_obs] A[num_hidden_states];  // observation model
}

model {
  // Priors
  D ~ dirichlet(rep_vector(1.0, num_hidden_states));
  for (s in 1:num_hidden_states)
    A[s] ~ dirichlet(rep_vector(1.0, num_obs));
  // Likelihood
  int s_latent = categorical_rng(D);
  o[1] ~ categorical(A[s_latent]);
}
```

## Usage

The Stan backend is selected via the standard render step:

```bash
python src/11_render.py --target-dir input/gnn_files --output-dir output --targets stan
```

Direct programmatic use:

```python
from render.stan.stan_renderer import render_gnn_to_stan

with open("my_model.md") as f:
    gnn_dict = parse_gnn_string(f.read())
stan_code = render_gnn_to_stan(gnn_dict)
with open("out/my_model.stan", "w") as f:
    f.write(stan_code)
```

## Compilation & Sampling

The Stan backend only emits `.stan` source. To compile and sample:

```bash
# Install cmdstan
pip install cmdstanpy
# Compile
python -c "from cmdstanpy import CmdStanModel; CmdStanModel(stan_file='out/my_model.stan')"
# Sample
python -c "from cmdstanpy import CmdStanModel; CmdStanModel(stan_file='out/my_model.stan').sample(data={'num_obs':2,'num_hidden_states':2,'o':[1]})"
```

Compilation is NOT part of the GNN pipeline — Step 12 (execute) intentionally
skips `.stan` files because Stan compilation is slow and requires a C++
toolchain. Integrate into your own CI when needed.

## Limitations

- **No Active Inference loop.** The renderer emits a graphical model; it does
  not implement expected-free-energy policy selection, belief updating across
  timesteps, or precision-weighted message passing. Use PyMDP / RxInfer.jl
  / ActiveInference.jl for those.
- **No automatic Julia-style factor graphs.** Stan's model block is a scalar
  program; multi-factor models require manual restructuring.
- **No pymc / numpyro-style random-variable reuse.** Stan's declarative
  syntax makes some patterns verbose.
- **Dimensional mismatch warnings are not auto-fixed.** If GNN declares
  `A[3,2]` but `InitialParameterization` contains a 2×2 literal, the renderer
  emits the literal as-is and Stan rejects at compile time. Run Step 5
  (type-checker) first to catch this.

## Implementation Notes

- Tests: `src/tests/test_render_stan_*.py` (unit) +
  `src/tests/test_render_cli_targets.py::test_every_cli_target_dispatches[stan]`
  (integration).
- Backend status: **Active but structural-only**. Maintained as a bridge to
  Stan ecosystem tools; not a primary Active Inference target.
- For backend-internal details, see
  [src/render/stan/AGENTS.md](../../../src/render/stan/AGENTS.md) and
  [src/render/stan/README.md](../../../src/render/stan/README.md).

## Source References

- Module: [src/render/stan/](../../../src/render/stan)
- Renderer: [src/render/stan/stan_renderer.py](../../../src/render/stan/stan_renderer.py)
- Comparison with other backends: [RxInfer.jl](rxinfer.md) (full AI),
  [PyMDP](pymdp.md) (full AI), [JAX](jax.md) (full AI).

## Navigation

- [← GNN Implementations Index](README.md)
- [← GNN Main Index](../README.md)
