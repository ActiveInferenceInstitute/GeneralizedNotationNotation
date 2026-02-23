# DisCoPy Framework Implementation

> **GNN Integration Layer**: Python
> **Framework Base**: `discopy` (Categorical Quantum Diagrams)
> **Simulation Architecture**: Structural / Categorical Semantics (No POMDP Simulation)
> **Documentation Version**: 0.4.1

## Overview

The Generalized Notation Notation (GNN) pipeline translates theoretical model specifications into executable Python code utilizing the `discopy` framework for categorical diagram generation. Unlike the other four implementations (PyMDP, JAX, RxInfer, ActiveInference.jl), DisCoPy does **not** run a numerical simulation of the Active Inference perception-action loop. Instead, it constructs a **compositional representation** of the model architecture as string diagrams in a monoidal category, providing structural analysis and formal verification of the model's type-theoretic properties.

Within the GNN cross-framework comparison, DisCoPy serves as the categorical semantics reference — validating that the model's compositional structure (morphisms, types, and their compositions) is well-formed.

## Architecture

The DisCoPy implementation consists of three interconnected layers:

1. **Parameter Parsing**: `pomdp_processor.py` → `discopy_renderer.py` (Extracting GNN variable dimensions and connection topology)
2. **Diagram Generation**: `DisCoPyRenderer._generate_discopy_diagram_code()` (Building the Python script defining categorical types, morphisms, and their compositions)
3. **Execution Context**: `discopy_executor.py` (Executing the generated Python script with syntax validation and log persistence)

### Source File

[discopy_renderer.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/render/discopy/discopy_renderer.py)

---

## GNN Parameter Ingestion

### Dimensional Extraction

DisCoPy extracts dimensions through GNN's model parameters and variable definitions:

```python
model_params = gnn_spec.get('model_parameters', {})
num_states = model_params.get('num_hidden_states', 3)
num_observations = model_params.get('num_obs', 3)
num_actions = model_params.get('num_actions', 3)

# Override from variable definitions if available
for var in gnn_spec.get('variables', []):
    if var.get('name') == 'A' and 'dimensions' in var:
        num_observations = var['dimensions'][0]
        num_states = var['dimensions'][1]
    elif var.get('name') == 'B' and 'dimensions' in var:
        num_actions = var['dimensions'][2]
```

### No Matrix Values

Unlike the numerical frameworks, DisCoPy does not inject actual matrix values. It only uses dimensional information to parameterize the type system.

---

## Categorical Type System

DisCoPy defines four fundamental types representing the Active Inference model's abstract spaces:

```python
S = Ty('S')  # Hidden states
O = Ty('O')  # Observations
A = Ty('A')  # Actions
P = Ty('P')  # Probabilities
```

These types form the objects of the monoidal category in which the Active Inference model is expressed.

---

## Morphisms (Model Components as Boxes)

Each GNN model component is represented as a morphism (Box) in the category, with explicit domain and codomain types:

| Component | Box Name | Domain | Codomain | Interpretation |
|---|---|---|---|---|
| **A matrix** | `'A'` | `S` | `O ⊗ P` | Observation likelihood `P(o\|s)` |
| **B matrix** | `'B'` | `S ⊗ A` | `S ⊗ P` | State transition `P(s'\|s,a)` |
| **C vector** | `'C'` | `I` (unit) | `O ⊗ P` | Preferred observations |
| **D vector** | `'D'` | `I` (unit) | `S ⊗ P` | Prior state beliefs |
| **E vector** | `'E'` | `I` (unit) | `A ⊗ P` | Policy priors |
| **State Inference** | `'StateInf'` | `O` | `S ⊗ P` | Posterior inference |
| **Policy Inference** | `'PolicyInf'` | `S ⊗ P` | `A ⊗ P` | Policy evaluation |
| **Action Selection** | `'ActionSel'` | `A ⊗ P` | `A` | Action sampling |

### Tensor Products (`⊗`)

The `@` operator in DisCoPy represents the tensor product of types. For example, `S @ A` denotes the product type "state combined with action", which is the natural domain of the transition morphism `B`.

---

## Circuit Construction

### Perception-Action Loop

The core Active Inference loop is expressed as a sequential composition of morphisms:

```python
perception_action_loop = (
    state_inf    # O → S ⊗ P
    >> policy_inf  # S ⊗ P → A ⊗ P
    >> action_sel  # A ⊗ P → A
)
```

This composition `>>` represents the functorial mapping:

```
O → S ⊗ P → A ⊗ P → A
```

The loop takes an observation and produces an action through sequential inference stages.

### Generative Model

The full generative model composes the prior vectors (`D`, `C`) as parallel tensor products alongside the perception-action loop. If the type composition succeeds, the result is a properly-typed categorical circuit `(D ⊗ C) >> StateInf ⊗ Id(P) >> PolicyInf >> ActionSel`. If type-dimensional constraints prevent composition, the system falls back gracefully to the perception-action loop alone.

---

## Structural Analysis

The `analyze_circuit_structure()` function performs type-theoretic validation:

```python
analysis_results = {
    'num_components': len(components),      # 8 components total
    'loop_domain': str(loop.dom),            # 'O'
    'loop_codomain': str(loop.cod),          # 'A'
    'model_domain': str(model.dom),          # 'O'
    'model_codomain': str(model.cod)         # 'A'
}
```

Key structural properties verified:

- **Type consistency**: All morphism compositions are well-typed (domain of next = codomain of previous)
- **Composition depth**: Number of sequential morphism stages
- **Component count**: Total number of defined morphisms

---

## Telemetry & Logging Output

DisCoPy exports two JSON files to `discopy_diagrams/`:

### 1. `circuit_analysis.json`

Contains the structural analysis results:

```json
{
  "num_components": 8,
  "loop_domain": "O",
  "loop_codomain": "A",
  "model_domain": "O",
  "model_codomain": "A"
}
```

### 2. `circuit_info.json`

Contains the complete circuit metadata:

```json
{
  "model_name": "Active Inference POMDP Agent",
  "timestamp": "2026-02-22T...",
  "parameters": {
    "num_states": 3,
    "num_observations": 3,
    "num_actions": 3
  },
  "components": [
    "A_matrix", "B_matrix", "C_vector", "D_vector",
    "E_vector", "state_inference", "policy_inference",
    "action_selection"
  ],
  "analysis": { ... }
}
```

### No Simulation Data

DisCoPy does **not** produce `beliefs`, `actions`, `observations`, or `efe_history` arrays. It appears in the cross-framework comparison report with `❌` for all numerical data coverage columns. This is by design — its role is structural, not computational.

---

## Cross-Framework Role

| Aspect | DisCoPy | Numerical Frameworks |
|---|---|---|
| **Purpose** | Structural validation | Computational simulation |
| **Output** | Category diagrams, type analysis | Belief/action/EFE trajectories |
| **GNN Data Used** | Dimensions, variable names, connections | Full matrix values |
| **Numerical Results** | None | beliefs, actions, observations, EFE |
| **Validation** | Type-consistency of compositions | Beliefs sum to 1, actions in range |

---

## Dependencies

| Package | Purpose |
|---|---|
| `discopy` | Categorical diagram construction and composition |
| `discopy.monoidal` | `Ty`, `Box`, `Id` primitives |
| `discopy.drawing` | `Equation` visualization (structural) |
| `numpy` | Numerical utilities |
| `json` | Telemetry serialization |

---

## Source Code Connections

| Pipeline Stage | Module | Key Function | Lines |
|---|---|---|---|
| Rendering | [discopy_renderer.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/render/discopy/discopy_renderer.py) | `_generate_discopy_diagram_code()` | L115-420 |
| Entry Point | [discopy_renderer.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/render/discopy/discopy_renderer.py) | `render_gnn_to_discopy()` | L427-471 |
| GNN Parsing | [discopy_renderer.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/render/discopy/discopy_renderer.py) | `_parse_gnn_content()` | L75-113 |
| Execution | [discopy_executor.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/execute/discopy/discopy_executor.py) | `execute_discopy_script()` | L37-137 |
| Validation | [discopy_executor.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/execute/discopy/discopy_executor.py) | `DisCoPyExecutor.validate_diagram()` | L155-181 |
| Analysis | [analyzer.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/analysis/discopy/analyzer.py) | `generate_analysis_from_logs()` | L18-90 |
| Visualization | [analyzer.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/analysis/discopy/analyzer.py) | `create_discopy_visualizations()` | L93-343 |
| Data Extraction | [analyzer.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/analysis/discopy/analyzer.py) | `extract_circuit_data()` | L349-385 |
| Structure Analysis | [analyzer.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/analysis/discopy/analyzer.py) | `analyze_diagram_structure()` | L388-418 |

---

## Improvement Opportunities

| ID | Area | Description | Impact |
|---|---|---|---|
| D-1 | Execution | ~~No dedicated runner~~ — added `execute_discopy_script()` in `discopy_executor.py` with syntax validation and log persistence | ✅ FIXED |
| D-2 | Rendering | No actual matrix values are injected into the generated script — only dimensions | Low |
| D-3 | Rendering | ~~The `generative_model` was just assigned to `perception_action_loop`~~ — now composes `D_vector @ C_vector` priors with graceful fallback | ✅ FIXED |
| D-4 | Analysis | `create_discopy_visualizations()` at 250 lines generates visualizations from `circuit_info.json` but could also render actual string diagrams using `discopy.drawing` if matplotlib is available | Medium |
| D-5 | Telemetry | No numerical simulation data exported — could add an optional mode to evaluate the circuit with actual tensor values using `discopy.tensor` | Low |
