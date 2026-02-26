# RxInfer.jl Framework Implementation

> **GNN Integration Layer**: Julia
> **Framework Base**: `RxInfer.jl` (Reactive Message Passing)
> **Simulation Architecture**: Online True POMDP Generative Model
> **Documentation Version**: 0.4.1

## Overview

The Generalized Notation Notation (GNN) pipeline translates theoretical model specifications into executable Julia code natively utilizing the `RxInfer.jl` ecosystem. RxInfer is a reactive probabilistic programming framework built on Forney-style factor graphs, where belief propagation is expressed as message passing over graphical model edges. Within the GNN cross-framework comparison, RxInfer serves as the primary Bayesian message-passing reference implementation and is the only framework in the pipeline that performs inference through a declarative probabilistic programming model (via the `@model` macro).

This document details the full data flow from GNN JSON specification through Julia factor graph construction, reactive belief updating, explicit Expected Free Energy (EFE) computation, and JSON telemetry serialization.

## Architecture

The RxInfer implementation consists of three interconnected layers:

1. **Parameter Parsing**: `pomdp_processor.py` → `rxinfer_renderer.py` (Translating GNN variable states into Julia matrix literals)
2. **Generative Loop Generation**: `RxInferRenderer._generate_rxinfer_simulation_code()` (Building the Julia script containing the `@model` block, EFE functions, and the generative loop)
3. **Execution Context**: `rxinfer_runner.py` (Spawning a Julia subprocess to execute the generated script)

### Source File

[rxinfer_renderer.py](#placeholder)

---

## GNN Parameter Ingestion

### Dimensional Extraction

RxInfer extracts model dimensions from the GNN specification using a multi-source priority chain:

```python
# Priority chain for num_actions
num_actions = (
    model_params.get('num_actions') or      # Explicit GNN model param
    model_params.get('num_controls') or      # Alternative GNN naming
    model_params.get('n_actions') or         # Legacy naming
    inferred_actions or                      # Inferred from B matrix depth
    3                                        # Hardcoded default
)
```

| GNN Parameter | Julia Constant | Extraction Source |
|---|---|---|
| `num_hidden_states` | `NUM_STATES` | `model_parameters.num_hidden_states` |
| `num_obs` | `NUM_OBSERVATIONS` | `model_parameters.num_obs` |
| `num_actions` | `NUM_ACTIONS` | Priority chain (see above) |
| `num_timesteps` | `TIME_STEPS` | `model_parameters.num_timesteps` (default: 20) |

### Matrix Literal Injection

GNN parameters are injected directly as Julia literal expressions into the generated script. Two utility functions handle runtime conversion:

- **`to_matrix(raw)`**: Converts nested Julia `Vector{Vector}` or `Tuple` structures into a proper `Matrix{Float64}` via `hcat()`.
- **`to_tensor(raw)`**: Converts 3-level nested structures into `Array{Float64, 3}` tensors for the B transition matrix indexing scheme `[next_state, prev_state, action]`.

### Matrix Normalization

All matrices are normalized after conversion:

| Matrix | Normalization Rule | Purpose |
|---|---|---|
| `A_matrix` | Column-sum to 1.0 | Valid conditional probability `P(o\|s)` |
| `B_matrix` | Column-sum to 1.0 per action slice | Valid transition probability `P(s'\|s,a)` |
| `D_vector` | Sum to 1.0 | Valid prior distribution |

### Preference Vector Transformation

The `C_vector` undergoes a critical transformation unique to RxInfer:

```julia
C_preferred = softmax(C_vector)
```

The raw GNN `C` values are treated as **log-preferences** (unnormalized log-probabilities). The softmax transformation converts these into a proper probability distribution used in the KL-divergence risk term of the EFE computation.

---

## Perception-Action Loop (The Generative Process)

RxInfer implements a true POMDP generative environment, fully decoupled from the agent's internal belief state. The process mirrors PyMDP's architecture exactly.

### Step 1: Initialize Environmental True State

```julia
current_state = rand(Categorical(D_vector))
current_belief = copy(D_vector)
```

### Step 2: Environment Generates Observation

```julia
obs = rand(Categorical(A_matrix[:, current_state]))
observations[t] = obs
```

The observation is sampled stochastically from the likelihood column of `A` corresponding to the true hidden state. The agent never has access to `current_state`.

### Step 3: Belief Inference (RxInfer Factor Graph)

This is where RxInfer differs fundamentally from all other frameworks. Belief updating is performed via a **declarative probabilistic model**:

```julia
@model function belief_update_model(observation, A, prior)
    s ~ Categorical(prior)
    observation ~ DiscreteTransition(s, A)
    return s
end
```

The `@model` macro compiles this into a Forney-style factor graph. RxInfer then runs **5 iterations** of variational message passing to compute the posterior:

```julia
result = infer(
    model = belief_update_model(A=A_matrix, prior=current_belief),
    data = (observation = obs_one_hot,),
    iterations = 5
)
posterior = result.posteriors[:s]
final_posterior = posterior[end]
current_belief = probvec(final_posterior)
```

**Fallback Mechanism**: If RxInfer's inference engine fails (e.g., due to numerical issues), the system falls back to manual Bayesian updating:

```julia
catch e
    likelihood = A_matrix[obs, :]
    unnormalized = current_belief .* likelihood
    current_belief = unnormalized ./ sum(unnormalized)
end
```

### Step 4: Expected Free Energy Computation and Action Selection

RxInfer implements EFE from first principles as `G(a) = Ambiguity + Risk`:

#### Ambiguity (Expected Observation Uncertainty)

```julia
ambiguity = 0.0
for j in 1:length(predicted_state)
    if predicted_state[j] > 1e-16
        col = A[:, j]
        col = max.(col, 1e-16)
        ambiguity -= predicted_state[j] * sum(col .* log.(col))
    end
end
```

This computes the expected entropy of `P(o|s)` weighted by the predicted next-state distribution: `H[P(o|s')]`.

#### Risk (KL Divergence from Preferences)

```julia
C_safe = max.(C_pref, 1e-16)
risk = sum(predicted_obs .* (log.(predicted_obs) .- log.(C_safe)))
```

This computes `D_KL(P(o') || C)`, the divergence between predicted observations and preferred observations.

#### Action Selection (Softmax Policy)

```julia
neg_efe = -action_precision .* efe_values
action_probs = softmax(neg_efe)
action = rand(Categorical(action_probs))
```

The `action_precision` parameter (configurable via GNN `ModelParameters.action_precision` or `ModelParameters.gamma`, default: `4.0`) controls the sharpness of action selection. Higher precision → more deterministic selection of the lowest-EFE action.

### Step 5: Environment Transition

```julia
next_probs = B_matrix[:, current_state, action]
next_probs = max.(next_probs, 1e-16)
next_probs = next_probs ./ sum(next_probs)
current_state = rand(Categorical(next_probs))
```

### Step 6: Predictive Prior Update

After state transition, the agent updates its belief using the transition model to form a predictive prior for the next timestep:

```julia
B_a = B_matrix[:, :, action]
current_belief = B_a * current_belief
current_belief = max.(current_belief, 1e-16)
current_belief = current_belief ./ sum(current_belief)
```

---

## Expected Free Energy: RxInfer vs PyMDP Convention

| Aspect | RxInfer | PyMDP |
|---|---|---|
| **Sign Convention** | Positive (Ambiguity + Risk) | Negative (`neg_efe`) |
| **Optimal Direction** | Lower is better | Higher (closer to 0) is better |
| **Typical Mean** | `~0.72` | `~-1.35` |
| **Selection Method** | Softmax over `-G * precision` | Softmax over `neg_efe` |

Both are mathematically equivalent Active Inference implementations. The sign difference is purely conventional.

---

## Telemetry & Logging Output

RxInfer exports a comprehensive JSON artifact to `simulation_results.json`:

### Data Schema

| Field | Shape | Description |
|---|---|---|
| `framework` | `string` | Always `"rxinfer"` |
| `model_name` | `string` | From GNN `ModelName` |
| `time_steps` | `int` | Number of simulation steps |
| `true_states` | `[T]` | True hidden states (1-indexed Julia) |
| `observations` | `[T]` | Stochastic emissions from environment |
| `actions` | `[T]` | Selected actions (1-indexed Julia) |
| `beliefs` | `[T, S]` | Full posterior belief distributions |
| `efe_history` | `[T]` | EFE of the **selected** action per timestep |
| `efe_per_action` | `[T, A]` | Full EFE vector across all actions |
| `action_probabilities` | `[T, A]` | Softmax policy probabilities |
| `preferences` | `[A]` | Raw C vector |
| `validation.all_beliefs_valid` | `bool` | All belief entries in `[0, 1]` |
| `validation.beliefs_sum_to_one` | `bool` | All belief vectors sum to 1.0 ± 0.01 |
| `validation.actions_in_range` | `bool` | All actions in `[1, NUM_ACTIONS]` |

### Index Convention

> **Critical**: RxInfer uses Julia's **1-indexed** convention. All `actions`, `observations`, and `true_states` arrays are 1-indexed. The downstream Python analysis pipeline handles this offset automatically during ingestion.

---

## Dependencies

| Package | Purpose |
|---|---|
| `RxInfer` | Reactive probabilistic programming and message passing |
| `Distributions` | `Categorical` distribution sampling |
| `LinearAlgebra` | Matrix operations |
| `Random` | PRNG seeding (`Random.seed!(42)`) |
| `StatsBase` | Action distribution counting (`countmap`) |
| `JSON` | Telemetry serialization |

---

## Source Code Connections

| Pipeline Stage | Module | Key Function | Lines |
|---|---|---|---|
| Rendering | [rxinfer_renderer.py](#placeholder) | `_generate_rxinfer_simulation_code()` | L167-531 |
| Entry Point | [rxinfer_renderer.py](#placeholder) | `render_gnn_to_rxinfer()` | L538-598 |
| Execution | [rxinfer_runner.py](#placeholder) | `execute_rxinfer_script()` | L84-168 |
| Julia Check | [rxinfer_runner.py](#placeholder) | `is_julia_available()` | L18-51 (delegates to `julia_setup`) |
| Analysis | [analyzer.py](#placeholder) | `generate_analysis_from_logs()` | L18-62 |
| Visualizations | [analyzer.py](#placeholder) | `create_rxinfer_visualizations()` | L65-240 |
| Data Extraction | [analyzer.py](#placeholder) | `extract_simulation_data()` | L245-280 |

---

## Improvement Opportunities

| ID | Area | Description | Impact |
|---|---|---|---|
| RX-1 | Execution | ~~No `validate_and_clean` pre-check for Julia syntax~~ — now validates file readability and emptiness before execution | ✅ FIXED |
| RX-2 | Rendering | ~~`to_matrix()` and `to_tensor()` rely on silent `try/catch`~~ — now use `@warn` for Julia-level logging | ✅ FIXED |
| RX-3 | Rendering | ~~`action_precision=4.0` was hardcoded~~ — now extracted from GNN `ModelParameters.action_precision` or `gamma` | ✅ FIXED |
| RX-4 | Telemetry | `efe_history` saves only selected-action EFE (`selected_efe`), while `efe_per_action` saves full vectors — the cross-framework comparator uses `efe_history` key, so the full multi-action data is not used in dashboards | Medium |
| RX-5 | Execution | ~~Duplicate `is_julia_available()`~~ — now delegates to shared `julia_setup.check_julia_availability()` | ✅ FIXED |
