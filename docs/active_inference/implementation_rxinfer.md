# RxInfer.jl Implementation Reference

## Overview

GNN uses RxInfer's native `@model` and `infer()` message-passing APIs for admitted
categorical and linear-Gaussian models. The committed
[Project.toml](../../src/gnn/execute/rxinfer/Project.toml) and
[Manifest.toml](../../src/gnn/execute/rxinfer/Manifest.toml) select RxInfer 6.1.0 /
ReactiveMP 7. The [native workflow](../../.github/workflows/rxinfer.yml) qualifies
Julia 1.13.2, model families, numerical witnesses and custom trajectory rules.

## Source Code Signposting

| Component | Owner |
| --- | --- |
| Model-kind dispatch and rendering | [Renderer](../../src/gnn/render/rxinfer/rxinfer_renderer.py), [strategy guide](../../src/gnn/render/rxinfer/README.md) |
| Native model definitions | [GnnRxInferModels.jl](../../src/gnn/execute/rxinfer/src/GnnRxInferModels.jl) |
| Execution and environment | [RxInfer runner](../../src/gnn/execute/rxinfer/README.md) |
| Native numerical/custom-rule checks | [test_models.jl](../../src/gnn/execute/rxinfer/test_models.jl) |

Generated scripts import the committed model package rather than redefining its
models. They supply source-declared parameters, simulation and action-selection
logic, supervision metadata and result serialization. Execution uses the committed
project environment; it does not install missing packages at runtime.

## Quick Reference

### Installation

From the repository root, instantiate the committed environment during setup:

```bash
julia --startup-file=no --project=src/gnn/execute/rxinfer \
  -e 'using Pkg; Pkg.instantiate()'
```

Run a Julia script with the same `--project` setting. The GNN runner uses this
project-specific environment; an ambient Julia installation alone does not
establish backend readiness.

### Basic Model

This complete categorical example uses the same model and data representation as
[the exhaustive-enumeration native check](../../src/gnn/execute/rxinfer/test_models.jl).
Save it as a Julia script and run it under the project above:

```julia
using RxInfer
using GnnRxInferModels: pomdp_model

A = [0.85 0.2; 0.15 0.8]
B = reshape([0.7 0.25; 0.3 0.75], 2, 2, 1)
D = [0.6, 0.4]
observations = [1, 2, 2]
y = [[Float64(k == o) for k in 1:2] for o in observations]
result = infer(
    model = pomdp_model(A=A, B=B, D=D, u=[1, 1], T=3),
    data = (y=y,), iterations=3, returnvars=KeepLast(), free_energy=true,
)
beliefs = [probvec(q) for q in result.posteriors[:s]]
@assert length(beliefs) == 3
@assert all(isfinite, result.free_energy)
```

### Inference

The batch model returns smoothed posterior states conditioned on the complete
observation sequence. Those results differ from online filtered beliefs.
Generated flat models also support explicit online mode; the
[integration guide](../gnn/integration/framework_integration_guide.md) owns that
selection and action-policy semantics. Free energy is a numerical output of
`infer()`; continuous Gaussian free energy need not be positive.

## Key Concepts

### Model Mapping

- `A[observation, state]` supplies the categorical likelihood.
- `B[next_state, previous_state, action]` supplies action-conditioned transitions.
- `D[state]` supplies the initial categorical prior.
- Preferences and action selection belong to generated policy logic; they are
  separate from the batch posterior model above.
- Continuous models use source-declared `F/H/Q/R` and `prior_mean/prior_cov`;
  the renderer does not invent Gaussian parameters from categorical matrices.

### Advanced Features

Hierarchical, factored and learning strategies use the committed package's model
functions and their required constraints/initialization. Multi-agent composition
and continuous control retain explicit admission limits; see the
[backend guide](../gnn/implementations/rxinfer.md).

Custom factors and rules must use the RxInfer 6 / ReactiveMP 7 APIs. The
[Halfspace node](../rxinfer/multiagent_trajectory_planning/HalfspaceNode.jl) and
[native compatibility checks](../../src/gnn/execute/rxinfer/test_models.jl) provide
working examples. Use their complete definitions and tests when extending a backend.

## Related Resources

- [RxInfer documentation hub](../rxinfer/README.md)
- [GNN integration guide](../rxinfer/gnn_rxinfer.md)
- [Variational inference](variational_inference.md)
- [PyMDP implementation](implementation_pymdp.md)
- [ActiveInference.jl implementation](implementation_activeinference_jl.md)
- [RxInfer source](https://github.com/ReactiveBayes/RxInfer.jl)
- [RxInfer documentation](https://docs.rxinfer.com/stable/)
