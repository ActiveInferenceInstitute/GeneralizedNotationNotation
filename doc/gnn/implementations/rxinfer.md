<!-- markdownlint-disable MD013 -->
# RxInfer.jl Framework Implementation

> **GNN Integration Layer**: Julia
> **Framework Base**: `RxInfer.jl` (Reactive Message Passing)
> **Simulation Architecture**: Online True POMDP Generative Model
> **Documentation Version**: 2.0.0

## Overview

The Generalized Notation Notation (GNN) pipeline translates theoretical model
specifications into executable Julia code natively utilizing the `RxInfer.jl`
ecosystem. RxInfer is a reactive probabilistic programming framework built on
Forney-style factor graphs, where belief propagation is expressed as message
passing over graphical model edges. Within the GNN cross-framework comparison,
RxInfer serves as the primary Bayesian message-passing reference implementation
and is the only framework in the pipeline that performs inference through a
declarative probabilistic programming model (via the `@model` macro).

This document details the full data flow from GNN JSON specification through Julia factor graph construction, reactive belief updating, explicit Expected Free Energy (EFE) computation, and JSON telemetry serialization.

## Architecture

The RxInfer implementation consists of three interconnected layers:

1. **Parameter Parsing**: `pomdp_processor.py` → `rxinfer_renderer.py`
   (Translating GNN variable states into Julia matrix literals)
2. **Generative Loop Generation**: `RxInferRenderer._generate_rxinfer...`
   (Building the Julia script containing the `@model` block, EFE functions,
   and the generative loop)
3. **Execution Context**: `rxinfer_runner.py`
   (Spawning a Julia subprocess to execute the generated script)

### Source File

[rxinfer_renderer.py](../../../src/render/rxinfer/rxinfer_renderer.py)

---

## GNN Parameter Ingestion

### Dimensional Extraction

RxInfer extracts model dimensions from the GNN specification using a multi-source priority chain:

```python
# Priority chain for num_actions
num_actions = (
    model_params.get('num_actions') or      # Explicit GNN model param
    model_params.get('num_controls') or      # Alternative GNN naming
    model_params.get('n_actions') or         # Previous naming convention
    inferred_actions or                      # Inferred from B matrix depth
    3                                        # Hardcoded default
)
```

| GNN Parameter       | Julia Constant     | Extraction Source                              |
| ------------------- | ------------------ | ---------------------------------------------- |
| `num_hidden_states` | `NUM_STATES`       | `model_parameters.num_hidden_states`           |
| `num_obs`           | `NUM_OBSERVATIONS` | `model_parameters.num_obs`                     |
| `num_actions`       | `NUM_ACTIONS`      | Priority chain (see above)                     |
| `num_timesteps`     | `TIME_STEPS`       | `model_parameters.num_timesteps` (default: 20) |

### Matrix Literal Injection

GNN parameters are injected directly as Julia literal expressions into the generated script. Two utility functions handle runtime conversion:

- **`to_matrix(raw)`**: Converts nested Julia `Vector{Vector}` or `Tuple` structures into a proper `Matrix{Float64}` via `hcat()`.
- **`to_tensor(raw)`**: Converts 3-level nested structures into
  `Array{Float64, 3}` tensors for the B transition matrix indexing scheme
  `[next_state, prev_state, action]`.

### Matrix Normalization

| Matrix     | Normalization Rule                 | Purpose                                |
| ---------- | ---------------------------------- | -------------------------------------- | ----- |
| `A_matrix` | Column-sum to 1.0                  | Valid conditional probability `P(o\ \  | s)`   |
| `B_matrix` | Column-sum to 1.0 per action slice | Valid transition prob `P(s'\ \         | s,a)` |
| `D_vector` | Sum to 1.0                         | Valid prior distribution               |

### Preference Vector Transformation

The `C_vector` undergoes a critical transformation unique to RxInfer:

The raw GNN `C` values are treated as **log-preferences**
(unnormalized log-probabilities). The softmax transformation converts
these into a proper probability distribution used in the KL-divergence
risk term of the EFE computation.

---

## Perception-Action Loop (The Generative Process)

RxInfer implements a true POMDP generative environment, fully decoupled from
the agent's internal belief state. The process mirrors PyMDP's architecture.

### Step 1: Initialize Environmental True State

```julia
current_state = rand(Categorical(D_vector))
current_belief = copy(D_vector)
```

### Step 2: Environment Generates Observation

The observation is sampled stochastically from the likelihood column of `A`
corresponding to the true hidden state. The agent never has access to
`current_state`.

This is where RxInfer differs fundamentally from all other frameworks.
Belief updating is performed via a **declarative probabilistic model**:

```julia
@model function belief_update_model(observation, A, prior)
    s ~ Categorical(prior)
    observation ~ DiscreteTransition(s, A)
    return s
end
```

The `@model` macro compiles this into a Forney-style factor graph. RxInfer then runs **5 iterations** of variational message passing to compute the posterior:

````julia
result = infer(
    model = belief_update_model(A=A_matrix, prior=current_belief),
    data = (observation = obs_one_hot,),
    iterations = 5
)
posterior = result.posteriors[:s]
**Recovery Mechanism**: If RxInfer's inference engine fails (e.g., due to
numerical issues), the system falls back to manual Bayesian updating:

```julia
catch e
    likelihood = A_matrix[obs, :]
    unnormalized = current_belief .* likelihood
    current_belief = unnormalized ./ sum(unnormalized)
end
````

### Step 4: Expected Free Energy Computation and Action Selection

RxInfer implements EFE from first principles as `G(a) = Ambiguity + Risk`:

#### Ambiguity (Expected Observation Uncertainty)

````julia
ambiguity = 0.0
for j in 1:length(predicted_state)
    if predicted_state[j] > 1e-16
        col = A[:, j]
        col = max.(col, 1e-16)
        ambiguity -= predicted_state[j] * sum(col .* log.(col))
This computes the expected entropy of `P(o\ | s)` weighted by the predicted
next-state distribution: `H[P(o \ | s')]`.

#### Risk (KL Divergence from Preferences)

```julia
C_safe = max.(C_pref, 1e-16)
risk = sum(predicted_obs .* (log.(predicted_obs) .- log.(C_safe)))
````

This computes `D_KL(P(o') \ | \ | C)`, the divergence between predicted and preferred observations.

#### Action Selection (Softmax Policy)

````julia
neg_efe = -action_precision .* efe_values
action_probs = softmax(neg_efe)
The `action_precision` parameter (configurable via GNN
`ModelParameters.action_precision` or `ModelParameters.gamma`, default: `4.0`)
controls the sharpness of action selection. Higher precision → more
deterministic selection of the lowest-EFE action.

### Step 5: Environment Transition

### Step 6: Predictive Prior Update

After state transition, the agent updates its belief using the transition
model to form a predictive prior for the next timestep:

```julia
B_a = B_matrix[:, :, action]
current_belief = B_a * current_belief
current_belief = max.(current_belief, 1e-16)
current_belief = current_belief ./ sum(current_belief)
````

---

## Expected Free Energy: RxInfer vs PyMDP Convention

| Aspect           | RxInfer                       | PyMDP                          |
| ---------------- | ----------------------------- | ------------------------------ |
| **Sign Conv**    | Positive (Ambiguity + Risk)   | Negative (`neg_efe`)           |
| **Optimal Dir**  | Lower is better               | Higher (closer to 0) is better |
| **Typical Mean** | `~0.72`                       | `~-1.35`                       |
| **Selection**    | Softmax over `-G * precision` | Softmax over `neg_efe`         |

Both are mathematically equivalent Active Inference implementations. The sign difference is purely conventional.

---

## Telemetry & Logging Output

RxInfer exports a comprehensive JSON artifact to `simulation_results.json`:

### Data Schema

| Field               | Shape    | Description                          |
| ------------------- | -------- | ------------------------------------ |
| `framework`         | `string` | Always `"rxinfer"`                   |
| `model_name`        | `string` | From GNN `ModelName`                 |
| `time_steps`        | `int`    | Number of simulation steps           |
| `true_states`       | `[T]`    | True hidden states (1-indexed Julia) |
| `observations`      | `[T]`    | Stochastic emissions from env        |
| `actions`           | `[T]`    | Selected actions (1-indexed Julia)   |
| `beliefs`           | `[T, S]` | Full posterior belief distributions  |
| `efe_history`       | `[T]`    | EFE of the **selected** action       |
| `efe_per_action`    | `[T, A]` | Full EFE vector across all actions   |
| `action_probs`      | `[T, A]` | Softmax policy probabilities         |
| `preferences`       | `[A]`    | Raw C vector                         |
| `val.all_valid`     | `bool`   | All belief entries in `[0, 1]`       |
| `val.sum_to_one`    | `bool`   | All belief vectors sum to 1.0 ± 0.01 |
| `val.action_bounds` | `bool`   | All actions in `[1, NUM_ACTIONS]`    |

> **Critical**: RxInfer uses Julia's **1-indexed** convention.
> All `actions`, `observations`, and `true_states` arrays are 1-indexed.
> The downstream Python analysis pipeline handles this offset automatically.

---

## Dependencies

| Package         | Purpose                     |
| --------------- | --------------------------- |
| `RxInfer`       | Reactive prob programming   |
| `Distributions` | `Categorical` dist sampling |
| `LinearAlgebra` | Matrix operations           |
| `Random`        | PRNG seeding                |
| `StatsBase`     | Action dist counting        |
| `JSON`          | Telemetry serialization     |

---

## Source Code Connections

| Pipeline Stage | Module                                                                 | Key Function                    | Lines  |
| -------------- | ---------------------------------------------------------------------- | ------------------------------- | ------ |
| Rendering      | [rxinfer_renderer.py](../../../src/render/rxinfer/rxinfer_renderer.py) | `_generate_rxinfer_code()`      | —      |
| Entry Point    | [rxinfer_renderer.py](../../../src/render/rxinfer/rxinfer_renderer.py) | `render_to_rxinfer()`           | —      |
| Execution      | [rxinfer_runner.py](../../../src/execute/rxinfer/rxinfer_runner.py)    | `execute_rxinfer_script()`      | 84-175 |
| Julia Check    | [rxinfer_runner.py](../../../src/execute/rxinfer/rxinfer_runner.py)    | `is_julia_available()`          | 18-51  |
| Analysis       | [analyzer.py](../../../src/analysis/rxinfer/analyzer.py)               | `generate_analysis_from_logs()` | —      |
| Visual         | [analyzer.py](../../../src/analysis/rxinfer/analyzer.py)               | `create_rx_visualizations()`    | —      |
| Extraction     | [analyzer.py](../../../src/analysis/rxinfer/analyzer.py)               | `extract_simulation_data()`     | —      |

---

## Improvement Opportunities

| ID   | Area      | Description                          | Impact   |
| ---- | --------- | ------------------------------------ | -------- |
| RX-1 | Execution | Syntx validate pre-check added       | ✅ FIXED |
| RX-2 | Rendering | Errors use `@warn` for Julia logging | ✅ FIXED |
| RX-3 | Rendering | Extract precision from GNN params    | ✅ FIXED |
| RX-4 | Telemetry | Dashboard unused multi-act EFE       | Medium   |
| RX-5 | Execution | Shared `check_julia_availability`    | ✅ FIXED |

## See Also / Next Steps

- **[Cross-Framework Methodology](../integration/cross_framework_methodology.md)**:
  Details on the correlation methodology and benchmarking metrics.
- **[Architecture Reference](../reference/architecture_reference.md)**:
  Deep dive into the pipeline orchestrator and module integration.
- **[GNN Implementations Index](README.md)**: Return to the master framework implementer manifest.
- **[Back to GNN START_HERE](../../START_HERE.md)**
