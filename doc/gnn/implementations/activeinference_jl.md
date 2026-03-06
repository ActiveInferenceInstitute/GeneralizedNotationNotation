# ActiveInference.jl Framework Implementation

> **GNN Integration Layer**: Julia
> **Framework Base**: `ActiveInference.jl` (Discrete Active Inference Package)
> **Simulation Architecture**: Online True POMDP Generative Model
> **Documentation Version**: 2.0.0

## Overview

The Generalized Notation Notation (GNN) pipeline translates theoretical model specifications into executable Julia code natively utilizing the `ActiveInference.jl` package. ActiveInference.jl is a dedicated Julia implementation of discrete-state Active Inference, providing structured agent initialization (`init_aif`), built-in variational inference (`infer_states!`), policy evaluation (`infer_policies!`), and action sampling (`sample_action!`). Within the GNN cross-framework comparison, ActiveInference.jl serves as the canonical Julia Active Inference reference, directly mirroring the `pymdp` Python API design.

This document details the full data flow from GNN JSON specification through Julia agent struct construction, the unified POMDP generative environment loop, EFE extraction (via the `.G` property), and JSON telemetry serialization.

## Architecture

The ActiveInference.jl implementation consists of three interconnected layers:

1. **Parameter Parsing**: `pomdp_processor.py` → `activeinference_renderer.py` (Translating GNN variable states into Julia matrix literals via `_matrix_to_julia()`)
2. **Generative Loop Generation**: `generate_activeinference_script()` (Building the Julia script containing the AIF agent, the generative environment loop, and telemetry export)
3. **Execution Context**: `activeinference_runner.py` (Spawning a Julia subprocess to execute the generated script)

### Source File

[activeinference_renderer.py](../../../src/render/activeinference_jl/activeinference_renderer.py)

---

## GNN Parameter Ingestion

### Matrix Conversion: `_matrix_to_julia()`

The `_matrix_to_julia()` function is a recursive converter that translates Python lists, tuples, and nested structures into syntactically valid Julia matrix literals:

```python
def _matrix_to_julia(matrix_data: Any) -> str:
    # Handles:
    # - Flat lists → Julia vectors: [1.0, 2.0, 3.0]
    # - 2D lists → Julia matrices: [1.0 2.0; 3.0 4.0]
    # - 3D lists → Julia 3D arrays: cat(slice1, slice2; dims=3)
```

| Input Structure | Output Julia Literal |
|---|---|
| `[0.1, 0.9, 0.0]` | `[0.1, 0.9, 0.0]` |
| `[[0.8, 0.1], [0.2, 0.9]]` | `[0.8 0.1; 0.2 0.9]` |
| `[[[0.9, 0.1], [0.1, 0.9]], ...]` | `cat([0.9 0.1; 0.1 0.9], ...; dims=3)` |

### Model Information Extraction

The `extract_model_info()` function robustly extracts GNN parameters from multiple possible JSON structures:

```python
# Priority chain for parameter extraction
sources = [
    gnn_spec.get('initial_parameterization', {}),    # Standard GNN format
    gnn_spec.get('initialparameterization', {}),      # Alternative casing
    gnn_spec.get('InitialParameterization', {}),      # Pascal case
]
```

| GNN Parameter | Julia Constant | Extraction Logic |
|---|---|---|
| `num_hidden_states` | `N_STATES` | `model_parameters.num_hidden_states` or inferred from `A.shape[1]` |
| `num_obs` | `N_OBSERVATIONS` | `model_parameters.num_obs` or inferred from `A.shape[0]` |
| `num_actions` | `N_CONTROLS` | `model_parameters.num_actions` or inferred from `B.shape[2]` |
| `num_timesteps` | `n_steps` | `model_parameters.num_timesteps` (default: 15) |

### The E Vector (Policy Prior)

ActiveInference.jl uniquely requires an **E vector** — a prior distribution over policies. GNN handles this with automatic dimension adjustment:

```julia
NUM_POLICIES = N_CONTROLS ^ POLICY_LEN

if length(E_vector_raw) != NUM_POLICIES
    E_vector = fill(1.0 / NUM_POLICIES, NUM_POLICIES)  # Uniform prior
end
```

For `POLICY_LEN = 1`, `NUM_POLICIES = N_CONTROLS`, so the E vector has one entry per action.

---

## Matrix Normalization

After injection, all matrices are normalized to ensure valid probability distributions:

```julia
# A matrix: columns sum to 1
for col in 1:size(A_matrix, 2)
    col_sum = sum(A_matrix[:, col])
    if col_sum > 0
        A_matrix[:, col] ./= col_sum
    end
end

# B matrix: columns sum to 1 per action slice
for action in 1:size(B_matrix, 3)
    for col in 1:size(B_matrix, 2)
        col_sum = sum(B_matrix[:, col, action])
        if col_sum > 0
            B_matrix[:, col, action] ./= col_sum
        end
    end
end

# Vectors: sum to 1
D_vector ./= sum(D_vector)
E_vector ./= sum(E_vector)
```

---

## Agent Initialization

The `init_aif` function constructs a full Active Inference agent struct:

```julia
A = [A_matrix]    # Vector of matrices (one per observation modality)
B = [B_matrix]    # Vector of tensors (one per state factor)
C = [C_vector]    # Vector of preference vectors
D = [D_vector]    # Vector of prior vectors
E = ones(n_policies) ./ n_policies   # Uniform policy prior

settings = Dict(
    "policy_len" => POLICY_LEN,
    "n_states" => [N_STATES],
    "n_observations" => [N_OBSERVATIONS],
    "n_controls" => [N_CONTROLS]
)

parameters = Dict(
    "alpha" => 16.0,    # Action precision
    "beta"  => 1.0,     # Policy precision
    "gamma" => 16.0,    # EFE precision
    "eta"   => 0.1,     # Learning rate
    "omega" => 1.0      # Evidence accumulation rate
)

aif_agent = init_aif(A, B; C=C, D=D, E=E, settings=settings, parameters=parameters, verbose=false)
```

### Agent Struct Properties (Key Fields)

| Property | Type | Description |
|---|---|---|
| `aif_agent.qs_current` | `Vector{Vector{Float64}}` | Current posterior beliefs per state factor |
| `aif_agent.action` | `Vector{Int}` | Most recently sampled action per control factor |
| `aif_agent.G` | `Vector{Float64}` | Expected Free Energy per policy |
| `aif_agent.policy` | — | Selected policy index |

> **Critical Note**: The `G` property (not `EFE`) is the correct field for extracting Expected Free Energy values. The `.EFE` field does not exist on the standard `AIF` struct. This mapping was corrected in Phase 6 of the pipeline development.

---

## Perception-Action Loop (The Generative Process)

### Step 1: Initialize Environmental True State

```julia
Random.seed!(42)
true_state = rand(Categorical(D_vector))
```

### Step 2: Environment Generates Observation

```julia
obs_prob = A_matrix[:, true_state]
observation = [rand(Categorical(obs_prob))]
```

The observation is drawn stochastically from the `A` matrix column corresponding to the true environmental state. The agent receives only the integer observation index.

### Step 3: Agent Inference and Action Selection

ActiveInference.jl provides three mutating functions that operate on the agent struct:

```julia
# Update internal beliefs given observation
infer_states!(aif_agent, observation)

# Evaluate policies and compute EFE (populates aif_agent.G)
infer_policies!(aif_agent)

# Sample action from policy posterior
sample_action!(aif_agent)
```

These three calls constitute the full Active Inference perception-action cycle within the agent.

### Step 4: Environment Transitions True State

```julia
next_probs = B_matrix[:, true_state, aif_agent.action[1]]
next_probs = max.(next_probs, 1e-16)
next_probs = next_probs ./ sum(next_probs)
true_state = rand(Categorical(next_probs))
```

The numerical floor (`1e-16`) and re-normalization prevent `Categorical` from receiving zero-probability entries, which would cause a Julia runtime error.

### Step 5: EFE Logging

```julia
try
    if hasfield(typeof(aif_agent), :G) && !isnothing(aif_agent.G)
        push!(efe_log, copy(aif_agent.G))
    else
        push!(efe_log, [NaN])
    end
catch
    push!(efe_log, [NaN])
end
```

The EFE is extracted from `aif_agent.G` per timestep. Each entry is a vector of length `n_policies` containing the Expected Free Energy for each available policy.

---

## JSON Serialization Safety

ActiveInference.jl's Julia arrays can contain `NaN` values (e.g., from numerical edge cases in EFE computation). A safety function sanitizes all floating-point values before JSON export:

```julia
safe_float(x) = isnan(x) ? 0.0 : Float64(x)

json_beliefs_log = [[safe_float(v) for v in b] for b in beliefs_full_log]
json_efe_log = [[safe_float(v) for v in e] for e in efe_log]
```

This prevents JSON serialization failures from `NaN` or `Inf` propagation.

---

## Telemetry & Logging Output

ActiveInference.jl exports **two** output files:

### 1. `simulation_results.csv`

A lightweight tabular format with columns: `step, observation, action, belief_state_1`.

### 2. `simulation_results.json`

The primary telemetry artifact consumed by the cross-framework analysis pipeline:

| Field | Shape | Description |
|---|---|---|
| `framework` | `string` | Always `"activeinference_jl"` |
| `model_name` | `string` | From GNN `ModelName` |
| `time_steps` | `int` | Number of simulation steps |
| `observations` | `[T]` | Stochastic emissions from environment |
| `actions` | `[T]` | Selected actions (1-indexed Julia) |
| `beliefs` | `[T, S]` | Full posterior belief distributions (NaN-safe) |
| `efe_history` | `[T, P]` | Full EFE vector per policy per timestep (NaN-safe) |
| `policy_history` | `[T]` | Selected policy indices |
| `num_states` | `int` | State space dimensionality |
| `num_observations` | `int` | Observation space dimensionality |
| `num_actions` | `int` | Action space dimensionality |
| `validation` | `Dict` | Validation checks: `beliefs_in_range`, `beliefs_sum_to_one`, `actions_in_range`, `all_valid` |

### 3. `model_parameters.json`

Exported separately containing the full A, B, C, D, E matrices used during simulation, enabling exact reproducibility.

---

## Cross-Framework Correlation

In the latest pipeline run, ActiveInference.jl achieved:

| Metric | Value |
|---|---|
| Mean Confidence | 0.9450 |
| EFE Mean | -0.9795 |
| EFE Std | 0.3414 |
| Correlation vs PyMDP | 0.8910 |
| Correlation vs JAX | 0.8910 |
| Correlation vs RxInfer | 0.8910 |

The slightly lower confidence correlation (compared to the 1.0000 achieved by PyMDP/JAX/RxInfer) is expected because ActiveInference.jl uses its own internal variational inference algorithm, which may converge differently from the explicit Bayesian updates in the other frameworks.

---

## Dependencies

| Package | Purpose |
|---|---|
| `ActiveInference` | Core discrete Active Inference agent |
| `Distributions` | `Categorical` distribution sampling |
| `LinearAlgebra` | Matrix operations |
| `Random` | PRNG seeding (`Random.seed!(42)`) |
| `Dates` | Timestamp generation |
| `JSON` | Telemetry serialization |
| `DelimitedFiles` | CSV export (`writedlm`) |

---

## Source Code Connections

| Pipeline Stage | Module | Key Function | Lines |
|---|---|---|---|
| Matrix Conversion | [activeinference_renderer.py](../../../src/render/activeinference_jl/activeinference_renderer.py) | `_matrix_to_julia()` | — |
| Model Extraction | [activeinference_renderer.py](../../../src/render/activeinference_jl/activeinference_renderer.py) | `extract_model_info()` | — |
| Script Generation | [activeinference_renderer.py](../../../src/render/activeinference_jl/activeinference_renderer.py) | `generate_activeinference_script()` | — |
| Env Setup | [activeinference_runner.py](../../../src/execute/activeinference_jl/activeinference_runner.py) | `setup_julia_environment()` | L86-159 |
| Execution | [activeinference_runner.py](../../../src/execute/activeinference_jl/activeinference_runner.py) | `execute_activeinference_script()` | L280-388 |
| Julia Check | [activeinference_runner.py](../../../src/execute/activeinference_jl/activeinference_runner.py) | `is_julia_available()` | L27-84 |
| Analysis | [analyzer.py](../../../src/analysis/activeinference_jl/analyzer.py) | `generate_analysis_from_logs()` | — |
| Trace Reconstruction | [analyzer.py](../../../src/analysis/activeinference_jl/analyzer.py) | `create_trace_reconstruction()` | — |
| Matrix Heatmaps | [analyzer.py](../../../src/analysis/activeinference_jl/analyzer.py) | `create_model_matrix_heatmaps()` | — |

---

## Improvement Opportunities

| ID | Area | Description | Impact |
|---|---|---|---|
| AIF-1 | Telemetry | ~~No `validation` dict in `simulation_results.json`~~ — now includes `beliefs_in_range`, `beliefs_sum_to_one`, `actions_in_range`, `all_valid` | ✅ FIXED |
| AIF-2 | Rendering | ~~4 renderer variants existed~~ — deleted `activeinference_jl_renderer.py`, `_fixed.py`, `_simple.py`; only canonical `activeinference_renderer.py` remains | ✅ FIXED |
| AIF-3 | Execution | Runner is 604 lines with Julia environment setup; `setup_julia_environment()` and `_fallback_environment_setup()` could be shared with `rxinfer_runner.py` — `is_julia_available()` now delegates to shared `julia_setup` module | Medium |
| AIF-4 | Rendering | ~~`POLICY_LENGTH` was referenced but defined as `POLICY_LEN`~~ — removed orphaned `POLICY_LENGTH`; only `POLICY_LEN` is used | ✅ FIXED |
| AIF-5 | Analysis | `parse_julia_matrix()` and `parse_julia_vector()` nested helpers could be extracted to shared Julia parsing utility | Low |

## See Also / Next Steps

- **[Cross-Framework Methodology](../integration/cross_framework_methodology.md)**: Details on the correlation methodology and benchmarking metrics.
- **[Architecture Reference](../reference/architecture_reference.md)**: Deep dive into the pipeline orchestrator and module integration.
- **[GNN Implementations Index](README.md)**: Return to the master framework implementer manifest.
- **[Back to GNN START_HERE](../../START_HERE.md)**
