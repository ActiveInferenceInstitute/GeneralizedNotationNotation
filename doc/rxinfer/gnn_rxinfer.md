---
title: GNN for Program Synthesis and Implementation of Generative Models in RxInfer.jl
type: documentation
status: proposal
created: 2024-07-15
tags:
  - gnn
  - rxinfer
  - program-synthesis
  - generative-models
  - active-inference
  - model-specification
semantic_relations:
  - type: describes_application_of
    source: [[gnn]]
    target: [[rxinfer]]
  - type: related_to
    links:
      - [[program_synthesis]]
      - [[generative_models]]
      - [[active_inference]]
      - [[model_specification]]
      - [[factor_graphs]]
      - [[message_passing]]
---

# GNN for Program Synthesis and Implementation of Generative Models in RxInfer.jl

## 1. Introduction

**Generalized Notation Notation (GNN)** is a text-based language designed to standardize the representation and communication of Active Inference generative models. Its primary goals are to enhance clarity, reproducibility, and interoperability in cognitive modeling. GNN aims to provide a structured, human-readable, and machine-parsable format for describing complex models, bridging the gap between theoretical concepts and practical implementations.

**RxInfer.jl** is a powerful Julia package for Bayesian Inference on Factor Graphs using Message Passing. It offers a flexible and efficient framework for various inference algorithms, supporting both static datasets and dynamic, streaming data. RxInfer's core strength lies in its reactive programming architecture and its expressive `@model` macro for probabilistic model specification.

This document explores the synergistic potential between GNN and RxInfer.jl. Specifically, it investigates how GNN can serve as a high-level specification language for the **program synthesis** of generative models, particularly Active Inference models, executable within the RxInfer.jl ecosystem. The aim is to outline a pathway for translating GNN model descriptions into runnable RxInfer code, thereby leveraging GNN's standardization benefits for RxInfer's advanced inference capabilities.

## 2. Bridging GNN and RxInfer: Conceptual Mapping

The structured nature of GNN lends itself well to a systematic mapping to RxInfer's modeling paradigm.

### 2.1. From GNN Structure to RxInfer `@model`

GNN files are typically organized into distinct sections. These sections can be conceptually mapped to elements within RxInfer's `@model` macro:

*   **`ModelName` (GNN)**: This would directly correspond to the function name in the RxInfer `@model` definition.
    *   Example: `ModelName: MyAgentModel` -> `@model function MyAgentModel(...)`

*   **`StateSpaceBlock` (GNN)**: This section, defining variables, their types, and descriptions, maps to the declaration of random variables and their prior distributions within the `@model` block. GNN parameters can become hyperparameters or fixed values in these declarations.
    *   Example GNN:
        ```gnn
        # StateSpaceBlock
        # Variable: hidden_state, Type: Continuous, Prior: Normal(0, 1)
        # Parameter: observation_precision, Value: 2.0
        ```
    *   Potential RxInfer mapping:
        ```julia
        hidden_state ~ Normal(0.0, 1.0)
        observation_precision = 2.0 # or passed as argument
        ```

*   **`Connections` (GNN)**: This is a crucial section detailing relationships between variables. These connections define the structure of the generative model and can be translated into:
    *   Probabilistic relationships using the `~` operator in RxInfer for stochastic dependencies (e.g., likelihoods, state transitions).
    *   Deterministic relationships using the `:=` operator or standard Julia function calls within the `@model` for transformations.
    *   The overall factor graph structure that RxInfer internally builds.
    *   Example GNN:
        ```gnn
        # Connections
        # Connection: hidden_state -> observation, Function: Normal(mean=hidden_state, precision=observation_precision)
        ```
    *   Potential RxInfer mapping:
        ```julia
        observation ~ Normal(hidden_state, observation_precision)
        ```

*   **`ParameterBlock` / Initial Parameterization (GNN)**: Defines fixed parameters or initial values for model variables. These can be translated into:
    *   Fixed numerical values directly embedded in RxInfer model code.
    *   Arguments passed to the RxInfer `@model` function.
    *   Initial conditions for states in dynamic models.

*   **`Equations` (GNN)**: If GNN specifies explicit mathematical equations for relationships, these would typically map to likelihood functions or deterministic computations within the RxInfer model. For complex equations, they might translate into custom factor node definitions or calls to specialized Julia functions.

*   **`TimeSettings` (GNN)**: For dynamic models, this section would inform the structure of loops (e.g., `for t in 1:T`) and the handling of time-indexed variables in RxInfer.

*   **`ActInfOntologyAnnotation` (GNN)**: While primarily for conceptual clarity and interoperability at a higher level, these annotations could inform variable naming conventions, comments in the generated RxInfer code, or even custom type definitions if a more elaborate mapping is desired.

### 2.2. The GNN "Triple Play" Revisited with RxInfer

GNN emphasizes a "Triple Play" approach to model representation. This aligns well with what a GNN-to-RxInfer pipeline could offer:

1.  **Text-Based Models (GNN) -> RxInfer `@model` Code**: The GNN file itself is the primary text-based representation. The program synthesis process would translate this into Julia code centered around the `@model` macro, which is another, more execution-oriented, text-based form.

2.  **Graphical Models (GNN implies Factor Graph) -> RxInfer's Internal Factor Graph**: The `Connections` section of a GNN file inherently describes a graphical model structure, typically a factor graph. RxInfer, upon parsing an `@model` definition, also constructs an internal factor graph. Thus, GNN's abstract graph translates to RxInfer's concrete computational graph. RxInfer and its underlying packages (like GraphPPL.jl) may offer tools to visualize this internal graph.

3.  **Executable Cognitive Models (GNN as blueprint) -> RxInfer Model is Executable**: A GNN specification serves as a blueprint. The RxInfer `@model` generated from this GNN specification *is* an executable cognitive model, ready for inference using RxInfer's engine.

```mermaid
graph TD
    subgraph "GNN Specification"
        direction LR
        GNN_Text["GNN File (.md)"]
        GNN_Structure["Abstract Model Structure (Variables, Connections)"]
    end

    subgraph "GNN-to-RxInfer Pipeline (Hypothetical)"
        direction TB
        Parser["GNN Parser"]
        Transformer["AST Transformer"]
        CodeGen["RxInfer Code Generator"]
    end

    subgraph "RxInfer Ecosystem"
        direction LR
        RxInfer_Model["`@model` function MyModel(...)"]
        RxInfer_FG["Internal Factor Graph"]
        RxInfer_Engine["Inference Engine (Message Passing)"]
    end

    GNN_Text --> Parser
    Parser --> GNN_Structure
    GNN_Structure --> Transformer
    Transformer --> CodeGen
    CodeGen --> RxInfer_Model
    RxInfer_Model -- RxInfer compiles --> RxInfer_FG
    RxInfer_FG -- Used by --> RxInfer_Engine

    style GNN_Text fill:#cde,stroke:#333
    style GNN_Structure fill:#cde,stroke:#333
    style Parser fill:#ecf,stroke:#333
    style Transformer fill:#ecf,stroke:#333
    style CodeGen fill:#ecf,stroke:#333
    style RxInfer_Model fill:#fce,stroke:#333
    style RxInfer_FG fill:#fce,stroke:#333
    style RxInfer_Engine fill:#fce,stroke:#333
```

## 3. Program Synthesis: GNN to RxInfer Code Generation

The core idea is to use GNN as a high-level specification that can be automatically translated into runnable RxInfer code. This involves a transpilation or code generation process.

### 3.1. A Hypothetical GNN-to-RxInfer Transpiler

A transpiler would perform the following conceptual steps:

1.  **Parsing GNN**: Read the GNN Markdown file and parse its structure into an Abstract Syntax Tree (AST) or an internal model representation. This involves identifying sections like `StateSpaceBlock`, `Connections`, etc., and extracting variables, parameters, and their relationships.
2.  **Semantic Analysis**: Validate the GNN model for consistency, completeness, and adherence to GNN syntax. Resolve references between variables and parameters.
3.  **RxInfer Code Construction**: Traverse the GNN-derived AST/representation and generate corresponding Julia code using RxInfer's `@model` macro.
    *   Map GNN variables to Julia variables within the `@model`.
    *   Translate GNN priors and connections into RxInfer's `~` probabilistic assignments.
    *   Handle GNN parameters by defining them as constants, model arguments, or derived values in Julia.
    *   Structure loops for dynamic models based on `TimeSettings`.
    *   Define necessary helper functions or structures if implied by complex GNN elements.

### 3.2. Example: GNN Snippet and Potential RxInfer Output

Consider a simple GNN for a Hidden Markov Model (HMM) fragment:

**Conceptual GNN Snippet:**

```gnn
# ModelName: SimpleHMM

# StateSpaceBlock
# Variable: s_t, Type: Discrete, States: [1, 2], Description: Latent state at time t
# Variable: o_t, Type: Discrete, States: [A, B, C], Description: Observation at time t
# Parameter: transition_matrix, Value: [[0.7, 0.3], [0.4, 0.6]]
# Parameter: emission_matrix, Value: [[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]]

# InitialConditions
# Variable: s_1, Function: Categorical([0.5, 0.5])

# Connections
# Connection: s_t_minus_1 -> s_t, Function: Categorical(transition_matrix[s_t_minus_1, :])
# Connection: s_t -> o_t, Function: Categorical(emission_matrix[s_t, :])
```

**Potential RxInfer Output:**

```julia
using RxInfer, Distributions

@model function SimpleHMM(observations, T)
    # Assuming transition_matrix and emission_matrix are passed as arguments or defined globally
    # For simplicity, let's assume they are passed as arguments:
    # transition_matrix :: Matrix{Float64}
    # emission_matrix :: Matrix{Float64}

    s = Vector{RandomVariable}(undef, T)
    
    # Initial state (mapping from GNN's s_1)
    s[1] ~ Categorical([0.5, 0.5]) 

    # State transitions
    for t in 2:T
        s[t] ~ Categorical(transition_matrix[s[t-1], :])
    end
    
    # Emissions
    for t in 1:T
        observations[t] ~ Categorical(emission_matrix[s[t], :])
    end
    
    return s, observations
end

# Example usage (matrices would need to be defined)
# transition_matrix_val = [[0.7, 0.3], [0.4, 0.6]]
# emission_matrix_val = [[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]]
# dummy_observations = [1, 2, 1] # Assuming mapped to integers
# result = infer(model = SimpleHMM(transition_matrix = transition_matrix_val, emission_matrix = emission_matrix_val), 
#                data = (observations = dummy_observations, T = 3))
```
*(Note: This example simplifies how states (e.g., `s_t_minus_1`) and indexing would be robustly handled in a real transpiler. State mappings (e.g., `[1,2]` to actual indices) would also be needed.)*

### 3.3. Benefits of GNN as a Front-end for RxInfer

*   **Standardization and Interoperability**: A GNN model definition could potentially target multiple simulation/inference backends (RxInfer being one), promoting model sharing and reuse.
*   **Abstraction**: Researchers can define models at a higher, more conceptual level without immediately diving into the specifics of Julia or RxInfer syntax for the initial model structure. This can lower the barrier to entry for model specification.
*   **Reproducibility**: GNN's clear, plain-text format enhances the reproducibility of model specifications.
*   **Modularity and Reusability**: GNN could encourage defining model components that can be assembled into larger systems, with each component then translatable to an RxInfer sub-model or part of a larger model.
*   **Enhanced Tooling**: A standardized GNN input could foster the development of a broader ecosystem of tools for model validation, visualization, comparison, and management, which could then feed into the RxInfer code generation pipeline.

## 4. GNN for Active Inference Models in RxInfer

RxInfer.jl possesses capabilities relevant to Active Inference, such as variational inference, message passing, and the flexibility to define complex generative models. The RxInfer documentation itself (e.g., `free_energy_message_passing_active_inference.md`, `active_inference_examples.md`) indicates its suitability for such models, detailing aspects like:

*   Expected Free Energy (EFE) computation.
*   Policy selection mechanisms.
*   State estimation within the perception-action loop.

GNN's primary focus is on standardizing Active Inference generative models. This makes it a particularly natural fit for specifying these complex hierarchical and dynamic models before translating them into RxInfer. Key GNN sections relevant to Active Inference might include (or could be extended to include):

*   **`Preferences`**: Defining preferred observations or states (\(C\) matrix).
*   **`Policies` (\(\pi\))**: Specifying the space of possible action sequences.
*   **`ExpectedFreeEnergy` (GCE)**: Describing how EFE is calculated or its components.
*   **`GenerativeProcess` (p)**: The agent's model of how hidden states cause observations.
*   **`GenerativeModel` (q)**: The agent's recognition density used to approximate the posterior over hidden states.

A GNN-to-RxInfer transpiler could map these conceptual blocks into the corresponding RxInfer structures for building active inference agents, such as custom factor nodes for EFE calculation or specific variable structures for policies and beliefs.

For example, a GNN description of an agent's policy selection mechanism based on EFE could be translated into RxInfer code that defines policies as random variables and uses message passing to update beliefs about the utility of these policies.

## 5. Challenges and Future Directions

Several challenges and opportunities exist in realizing a robust GNN-to-RxInfer pipeline:

*   **GNN Standardization and Precision**: The GNN specification itself needs to be sufficiently precise and comprehensive to allow for unambiguous translation into executable code. The current GNN documentation focuses on structure; detailed syntax for functions, distributions, and complex operations would be critical for a transpiler.
*   **Expressivity Mapping**:
    *   Ensuring that the constructs available in GNN can be faithfully represented in RxInfer.
    *   Handling GNN's allowance for natural language descriptions within its structure – how much of this can be formalized for code generation versus serving as comments?
*   **Error Handling and Debugging**: A transpiler would need robust error reporting if a GNN file is malformed or contains ambiguities. Debugging would span two levels: GNN specification errors and RxInfer runtime errors.
*   **Mapping GNN's Flexibility**: GNN aims for broad applicability. RxInfer, while flexible, has its own idioms and constraints (e.g., Julia's type system, specific message passing rules). The mapping must navigate these differences.
*   **Handling Complex GNN Constructs**: Advanced GNN features (e.g., hierarchical models, model composition) would require sophisticated translation logic.
*   **Tooling Development**: The practical realization hinges on the development of robust GNN parsers, semantic analyzers, and RxInfer code generators.
*   **Bidirectional Synchronization**: While GNN-to-RxInfer is the primary goal for program synthesis, exploring tools to (partially) reverse-engineer RxInfer code back to a GNN-like abstract representation could be valuable for documentation and model understanding, though this is a significantly harder problem.
*   **Community Adoption and Evolution**: The success of such an approach depends on community adoption of GNN and collaborative refinement of both GNN and the translation tools.

Future work could focus on:
*   Developing a formal grammar for GNN suitable for automated parsing.
*   Building a prototype GNN-to-RxInfer transpiler for a core subset of GNN features.
*   Creating libraries of GNN components that map to reusable RxInfer model snippets.
*   Integrating GNN validation tools into the transpilation pipeline.

## 6. Conclusion

Generalized Notation Notation (GNN) offers a promising avenue for standardizing the specification of generative models, especially within the Active Inference community. Its structured, text-based format aligns well with the principles of program synthesis. By developing a GNN-to-RxInfer.jl transpilation pipeline, the Active Inference and broader Bayesian modeling communities could benefit from:

*   A higher-level, potentially more accessible language for model definition.
*   Enhanced reproducibility and interoperability of model specifications.
*   The ability to leverage RxInfer's powerful and efficient inference engine for models designed conceptually in GNN.

This synergy would streamline the workflow from theoretical model design to practical implementation and simulation, fostering innovation and collaboration in the field of complex cognitive modeling. While challenges remain in formalizing GNN to the degree required for robust transpilation, the potential benefits warrant further exploration and development.

## 7. References

*   Smékal, J., & Friedman, D. A. (2023). Generalized Notation Notation for Active Inference Models. *Active Inference Journal*. [https://doi.org/10.5281/zenodo.7803328](https://doi.org/10.5281/zenodo.7803328)
*   The GeneralizedNotationNotation Project: [https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation)
*   RxInfer.jl Documentation: [https://reactivebayes.github.io/RxInfer.jl/stable/](https://reactivebayes.github.io/RxInfer.jl/stable/)
*   Bagaev, D., de Vries, B., & van de Laar, T. (2023). RxInfer: A Julia package for reactive real-time Bayesian inference. *Journal of Open Source Software*, 8(92), 5161. [https://doi.org/10.21105/joss.05161](https://doi.org/10.21105/joss.05161)
*   Relevant RxInfer documentation pages:
    *   `model_specification.md`
    *   `model_macro_paradigm.md`
    *   `factor_graphs.md`
    *   `message_passing.md`
    *   `active_inference_examples.md`
    *   `free_energy_message_passing_active_inference.md`

# GNN-RxInfer Integration Guide

## Overview

RxInfer.jl is a powerful Julia package for Bayesian inference on factor graphs using reactive message passing. This guide explains how GNN models are translated to RxInfer.jl code and how to work with the generated implementations for Active Inference applications.

## GNN to RxInfer Translation

### Conceptual Mapping

GNN provides a declarative specification of Active Inference models, while RxInfer implements these models as reactive factor graphs. The translation follows these key mappings:

| GNN Element | RxInfer.jl Equivalent | Description |
|-------------|----------------------|-------------|
| `s_f0`, `s_f1`, ... | Hidden state variables | Random variables in the factor graph |
| `o_m0`, `o_m1`, ... | Observation variables | Observed data nodes |
| `u_c0`, `u_c1`, ... | Control variables | Action/control inputs |
| `A_m0`, `A_m1`, ... | Likelihood matrices | Categorical/Dirichlet distributions |
| `B_f0`, `B_f1`, ... | Transition matrices | Transition model distributions |
| `C_m0`, `C_m1`, ... | Preference vectors | Prior beliefs over observations |
| `D_f0`, `D_f1`, ... | Initial beliefs | Prior distributions over initial states |

### Factor Graph Structure Translation

#### Model Definition Structure
```gnn
# GNN Specification
ModelName: SimpleAgent
```

```julia
# Generated RxInfer Code
@model function SimpleAgent(observations, actions, num_timesteps)
    # Model definition follows
end
```

#### State Space Translation
```gnn
# GNN StateSpaceBlock
s_f0[4,1,type=categorical]      # Hidden state factor 0
o_m0[3,1,type=categorical]      # Observation modality 0
u_c0[2,1,type=categorical]      # Control factor 0
```

```julia
# Generated RxInfer Variables
# Hidden states (latent variables)
s = randomvar(num_timesteps)
# Observations (data variables)  
o = datavar(Vector{Int}, num_timesteps)
# Controls (constrained variables)
u = constvar(Vector{Int}, num_timesteps)
```

#### Likelihood Model (A matrices)
```gnn
# GNN Specification
A_m0 = [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]]
```

```julia
# Generated RxInfer Code
# A matrix as Dirichlet concentration parameters
A_concentration = [
    [0.9, 0.1, 0.0],
    [0.1, 0.8, 0.1], 
    [0.0, 0.1, 0.9]
]

# Likelihood distribution
for t in 1:num_timesteps
    s[t] ~ Categorical(fill(1/4, 4))  # Uniform prior over 4 states
    o[t] ~ Categorical(A_concentration[s[t]])
end
```

#### Transition Model (B matrices)
```gnn
# GNN Specification  
B_f0 = [[[0.8, 0.2], [0.3, 0.7]], [[0.1, 0.9], [0.6, 0.4]]]
```

```julia
# Generated RxInfer Code
# B matrix transitions
B_matrices = [
    [0.8 0.2; 0.3 0.7],  # Action 0 transition matrix
    [0.1 0.9; 0.6 0.4]   # Action 1 transition matrix  
]

# State transitions
for t in 2:num_timesteps
    s[t] ~ Categorical(B_matrices[u[t-1]][s[t-1], :])
end
```

## Working with Generated RxInfer Code

### Basic Agent Structure

Generated RxInfer scripts follow this standard structure:

```julia
using RxInfer, ReactiveMP, Distributions
using GraphPPL

@model function GNNActiveInferenceAgent(observations, actions, num_timesteps, model_params)
    # Extract model parameters
    A_matrices = model_params[:A]
    B_matrices = model_params[:B]  
    C_vectors = model_params[:C]
    D_vectors = model_params[:D]
    
    # Define latent variables
    s = randomvar(num_timesteps)
    
    # Define data variables
    o = datavar(Vector{Int}, num_timesteps)
    u = constvar(Vector{Int}, num_timesteps)
    
    # Initial state distribution
    s[1] ~ Categorical(D_vectors[1])
    
    # State transitions
    for t in 2:num_timesteps
        s[t] ~ Categorical(B_matrices[1][s[t-1], :, u[t-1]])
    end
    
    # Observations  
    for t in 1:num_timesteps
        o[t] ~ Categorical(A_matrices[1][s[t], :])
    end
    
    # Return constrained variables for inference
    return s, o, u
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    # Model parameters from GNN
    model_params = Dict(
        :A => [...],  # From GNN A matrices
        :B => [...],  # From GNN B matrices  
        :C => [...],  # From GNN C vectors
        :D => [...]   # From GNN D vectors
    )
    
    # Observations and actions
    observations = [1, 2, 1, 3, 2]
    actions = [1, 2, 1, 2]
    
    # Run inference
    result = infer(
        model = GNNActiveInferenceAgent(observations, actions, length(observations), model_params),
        data = (o = observations, u = actions)
    )
    
    println("Inferred states: ", result.posteriors[:s])
end
```

### Advanced Features

#### Free Energy Computation
```julia
using RxInfer

# Define model with free energy computation
@model function ActiveInferenceWithFreeEnergy(observations, actions, num_timesteps, params)
    # ... model definition ...
    
    # Meta-node for free energy computation
    fe ~ FreeEnergyMeta()
    
    return s, o, u, fe
end

# Inference with free energy tracking
result = infer(
    model = ActiveInferenceWithFreeEnergy(obs, acts, T, params),
    data = (o = obs, u = acts),
    meta = FreeEnergyMetaRule(),
    iterations = 20
)

free_energy_trace = result.free_energy
println("Free energy evolution: ", free_energy_trace)
```

#### Hierarchical Models
```julia
@model function HierarchicalActiveInference(observations, num_timesteps, hierarchy_params)
    # High-level states (goals, contexts)
    s_high = randomvar(num_timesteps)
    
    # Low-level states (actions, sensorimotor)  
    s_low = randomvar(num_timesteps)
    
    # Hierarchical priors
    for t in 1:num_timesteps
        s_high[t] ~ Categorical(hierarchy_params[:D_high])
        s_low[t] ~ Categorical(hierarchy_params[:B_hierarchical][s_high[t]])
        
        # Observations depend on low-level states
        observations[t] ~ Categorical(hierarchy_params[:A][s_low[t], :])
    end
    
    return s_high, s_low, observations
end
```

#### Planning and Policy Optimization
```julia
@model function PlanningAgent(observations, planning_horizon, params)
    # Current beliefs
    current_state ~ Categorical(params[:current_belief])
    
    # Policy variables (actions to be planned)
    policy = randomvar(planning_horizon)
    
    # Predicted states under policy
    predicted_states = randomvar(planning_horizon + 1)
    predicted_states[1] ~ Categorical(current_state)
    
    # Expected observations under policy
    expected_obs = randomvar(planning_horizon)
    
    # Planning loop
    for t in 1:planning_horizon
        # Sample action from policy
        policy[t] ~ Categorical(params[:policy_prior])
        
        # Predict next state
        predicted_states[t+1] ~ Categorical(params[:B][predicted_states[t], :, policy[t]])
        
        # Predict observation
        expected_obs[t] ~ Categorical(params[:A][predicted_states[t+1], :])
    end
    
    # Expected free energy constraint (preferences)
    for t in 1:planning_horizon
        expected_obs[t] ~ constrain(params[:C])  # Prefer certain observations
    end
    
    return policy, predicted_states, expected_obs
end
```

### Multi-Factor and Multi-Modal Models

#### Multiple State Factors
```julia
@model function MultiFactorAgent(observations, actions, num_timesteps, params)
    # Multiple state factors
    s_f0 = randomvar(num_timesteps)  # Factor 0 (e.g., location)
    s_f1 = randomvar(num_timesteps)  # Factor 1 (e.g., context)
    
    # Initial distributions
    s_f0[1] ~ Categorical(params[:D][1])
    s_f1[1] ~ Categorical(params[:D][2])
    
    # Independent transitions (can be modified for dependencies)
    for t in 2:num_timesteps
        s_f0[t] ~ Categorical(params[:B][1][s_f0[t-1], :, actions[t-1]])
        s_f1[t] ~ Categorical(params[:B][2][s_f1[t-1], :])  # Context evolves independently
    end
    
    # Joint observations from multiple factors
    for t in 1:num_timesteps
        # Observation depends on both factors
        observations[t] ~ Categorical(params[:A][1][s_f0[t], s_f1[t], :])
    end
    
    return s_f0, s_f1, observations
end
```

#### Multiple Observation Modalities
```julia
@model function MultiModalAgent(visual_obs, audio_obs, actions, num_timesteps, params)
    # Single state factor
    s = randomvar(num_timesteps)
    s[1] ~ Categorical(params[:D])
    
    # State transitions
    for t in 2:num_timesteps
        s[t] ~ Categorical(params[:B][s[t-1], :, actions[t-1]])
    end
    
    # Multiple observation modalities
    for t in 1:num_timesteps
        # Visual modality
        visual_obs[t] ~ Categorical(params[:A_visual][s[t], :])
        
        # Audio modality  
        audio_obs[t] ~ Categorical(params[:A_audio][s[t], :])
    end
    
    return s, visual_obs, audio_obs
end
```

## Simulation Examples

### Basic Perception Task
```julia
using RxInfer, Distributions

# Simple perception without actions
@model function PerceptionOnly(observations, num_timesteps)
    # Hidden states
    s = randomvar(num_timesteps)
    
    # Prior beliefs (uniform)
    for t in 1:num_timesteps
        s[t] ~ Categorical([0.5, 0.5])  # Binary state
    end
    
    # Likelihood model
    A_matrix = [0.9 0.1; 0.1 0.9]  # High accuracy perception
    
    for t in 1:num_timesteps
        observations[t] ~ Categorical(A_matrix[s[t], :])
    end
    
    return s, observations
end

# Run perception simulation
observations = [1, 2, 1, 2, 1]
result = infer(
    model = PerceptionOnly(observations, length(observations)),
    data = (observations = observations,)
)

println("Beliefs over time:")
for (t, belief) in enumerate(result.posteriors[:s])
    println("t=$t: $(mean(belief))")
end
```

### Active Inference with Planning
```julia
@model function ActiveInferenceAgent(observations, num_timesteps, planning_horizon, params)
    # Current state beliefs
    current_state = randomvar()
    current_state ~ Categorical(params[:current_belief])
    
    # Planning variables
    planned_actions = randomvar(planning_horizon)
    future_states = randomvar(planning_horizon + 1)
    future_obs = randomvar(planning_horizon)
    
    # Initialize planning
    future_states[1] ~ current_state
    
    # Planning loop
    for h in 1:planning_horizon
        # Sample planned action
        planned_actions[h] ~ Categorical(params[:action_prior])
        
        # Predict future state
        future_states[h+1] ~ Categorical(params[:B][future_states[h], :, planned_actions[h]])
        
        # Predict future observation
        future_obs[h] ~ Categorical(params[:A][future_states[h+1], :])
        
        # Apply preferences (constrain to preferred observations)
        future_obs[h] ~ constrain(params[:C])
    end
    
    return current_state, planned_actions, future_states, future_obs
end

# Execute planning
params = Dict(
    :current_belief => [0.7, 0.3],
    :action_prior => [0.5, 0.5],
    :B => ...,  # Transition matrices
    :A => ...,  # Likelihood matrices  
    :C => [2.0, 0.0]  # Strong preference for observation 1
)

planning_result = infer(
    model = ActiveInferenceAgent(nothing, 1, 5, params),
    iterations = 30
)

optimal_actions = mode.(planning_result.posteriors[:planned_actions])
println("Optimal action sequence: $optimal_actions")
```

### Learning and Adaptation
```julia
@model function LearningAgent(observations, actions, num_timesteps, learning_params)
    # Learnable parameters (Dirichlet priors)
    A_params = randomvar()
    A_params ~ Dirichlet(learning_params[:A_prior])
    
    B_params = randomvar()
    B_params ~ Dirichlet(learning_params[:B_prior])
    
    # States
    s = randomvar(num_timesteps)
    s[1] ~ Categorical(learning_params[:D])
    
    # State transitions with learnable B
    for t in 2:num_timesteps
        s[t] ~ Categorical(B_params[s[t-1], :, actions[t-1]])
    end
    
    # Observations with learnable A
    for t in 1:num_timesteps
        observations[t] ~ Categorical(A_params[s[t], :])
    end
    
    return s, observations, A_params, B_params
end

# Learning simulation
learning_params = Dict(
    :A_prior => ones(2, 2),  # Flat prior
    :B_prior => ones(2, 2, 2),  # Flat prior
    :D => [0.5, 0.5]
)

# Sequential learning
observations_batch = [1, 2, 1, 1, 2]
actions_batch = [1, 2, 1, 2]

for batch in 1:10
    result = infer(
        model = LearningAgent(observations_batch, actions_batch, 5, learning_params),
        data = (observations = observations_batch, actions = actions_batch),
        iterations = 20
    )
    
    # Update priors based on learned posteriors
    learned_A = mean(result.posteriors[:A_params])
    learned_B = mean(result.posteriors[:B_params])
    
    println("Batch $batch - Learned A: $learned_A")
    
    # Update priors for next batch (experience accumulation)
    learning_params[:A_prior] = learned_A .* 10  # Increase confidence
    learning_params[:B_prior] = learned_B .* 10
end
```

## Common Issues and Solutions

### Memory Management
```julia
# Problem: Large factor graphs consume excessive memory
# Solution: Use streaming inference for long sequences

@model function StreamingAgent(observations, window_size, params)
    # Only model a sliding window of states
    s = randomvar(window_size)
    
    # Circular buffer approach for efficient memory usage
    for t in 1:window_size
        if t == 1
            s[t] ~ Categorical(params[:D])
        else
            s[t] ~ Categorical(params[:B][s[t-1], :])
        end
        
        observations[t] ~ Categorical(params[:A][s[t], :])
    end
    
    return s, observations
end
```

### Convergence Issues
```julia
# Problem: Inference doesn't converge
# Solution: Use different message passing schedules

result = infer(
    model = MyModel(...),
    data = data,
    options = (
        iterations = 50,
        tolerance = 1e-6,
        show_progress = true,
        algorithm = BeliefPropagation(
            max_iterations = 100,
            damping = 0.5  # Add damping for stability
        )
    )
)
```

### Numerical Stability
```julia
# Problem: Numerical instabilities with small probabilities
# Solution: Use log-space computations and regularization

@model function StableAgent(observations, num_timesteps, params)
    # Add small regularization to prevent zeros
    epsilon = 1e-8
    
    s = randomvar(num_timesteps)
    
    for t in 1:num_timesteps
        if t == 1
            s[t] ~ Categorical(params[:D] .+ epsilon)
        else
            transition_probs = params[:B][s[t-1], :] .+ epsilon
            s[t] ~ Categorical(transition_probs ./ sum(transition_probs))
        end
        
        likelihood_probs = params[:A][s[t], :] .+ epsilon
        observations[t] ~ Categorical(likelihood_probs ./ sum(likelihood_probs))
    end
    
    return s, observations
end
```

## Performance Optimization

### Efficient Model Structure
```julia
# Use vectorized operations where possible
@model function VectorizedAgent(observations, num_timesteps, params)
    # Batch state variables for efficiency
    s = randomvar(num_timesteps)
    
    # Vectorized initial conditions
    s[1:end] ~ Categorical.(params[:state_priors])
    
    # Efficient transition handling
    for t in 2:num_timesteps
        s[t] ~ Categorical(params[:B_tensor][:, :, s[t-1]])
    end
    
    # Vectorized observations
    observations[1:end] ~ Categorical.(params[:A_tensor][s[1:end], :])
    
    return s, observations
end
```

### Parallel Processing
```julia
using Distributed
@everywhere using RxInfer

# Run multiple inference chains in parallel
function parallel_inference(model_func, data_batches, params)
    results = pmap(data_batches) do batch
        infer(
            model = model_func(batch, params),
            data = batch,
            iterations = 20
        )
    end
    return results
end
```

### Memory-Efficient Streaming
```julia
@model function StreamingInference(observation_stream, params)
    # Process observations one at a time
    current_belief = randomvar()
    current_belief ~ Categorical(params[:initial_belief])
    
    for obs in observation_stream
        # Update belief with new observation
        new_belief = randomvar()
        new_belief ~ Categorical(params[:B][current_belief, :])
        
        # Likelihood of current observation
        obs ~ Categorical(params[:A][new_belief, :])
        
        # Placeholder node for initialization
        current_belief = new_belief
    end
    
    return current_belief
end
```

## Integration with Other Frameworks

### Julia Ecosystem Integration
```julia
# Integration with DifferentialEquations.jl for continuous time
using DifferentialEquations, RxInfer

@model function ContinuousTimeAgent(observations, time_points, params)
    # Continuous-time state evolution
    function state_dynamics!(ds, s, p, t)
        ds[1] = -p[1] * s[1] + p[2] * randn()  # OU process
    end
    
    # Solve differential equation
    prob = SDEProblem(state_dynamics!, [1.0], (0.0, 10.0))
    sol = solve(prob, SRIW1(), dt=0.1)
    
    # Discrete observations of continuous process
    for (i, t) in enumerate(time_points)
        state_at_t = sol(t)[1]
        observations[i] ~ Normal(state_at_t, params[:obs_noise])
    end
end

# Integration with Flux.jl for neural components
using Flux, RxInfer

@model function NeuralActiveInference(observations, neural_params, model_params)
    # Neural network for complex likelihood mapping
    neural_net = Chain(
        Dense(model_params[:state_dim], 64, relu),
        Dense(64, model_params[:obs_dim], softmax)
    )
    
    s = randomvar(length(observations))
    s[1] ~ Categorical(model_params[:D])
    
    for t in 2:length(observations)
        s[t] ~ Categorical(model_params[:B][s[t-1], :])
        
        # Neural likelihood
        likelihood_probs = neural_net([s[t]])[1]
        observations[t] ~ Categorical(likelihood_probs)
    end
    
    return s, observations
end
```

### External Environment Interface
```julia
# Interface with external simulators
mutable struct EnvironmentInterface
    env_state::Any
    step_function::Function
    observe_function::Function
end

function active_inference_loop(agent_model, environment, num_steps)
    observations = []
    actions = []
    
    for step in 1:num_steps
        # Get current observation
        obs = environment.observe_function(environment.env_state)
        push!(observations, obs)
        
        # Run agent inference
        result = infer(
            model = agent_model(observations, length(observations)),
            data = (observations = observations,)
        )
        
        # Select action based on beliefs
        action = sample_action(result.posteriors[:s][end])
        push!(actions, action)
        
        # Update environment
        environment.env_state = environment.step_function(environment.env_state, action)
    end
    
    return observations, actions
end
```

## Best Practices

1. **Model Design**: Start with simple models and gradually add complexity
2. **Numerical Stability**: Always add small regularization terms to avoid numerical issues
3. **Memory Management**: Use streaming inference for long sequences
4. **Convergence**: Monitor free energy and use appropriate stopping criteria
5. **Debugging**: Use RxInfer's visualization tools to inspect factor graphs
6. **Performance**: Profile code and optimize bottlenecks using Julia's profiling tools

## Validation and Testing

### Model Validation
```julia
function validate_gnn_translation(gnn_model, rxinfer_model, test_data)
    # Test that RxInfer model produces sensible results
    result = infer(
        model = rxinfer_model(test_data),
        data = test_data,
        iterations = 50
    )
    
    # Validation checks
    @assert !any(isnan.(result.free_energy)) "Free energy contains NaN values"
    @assert length(result.posteriors) > 0 "No posterior distributions computed"
    
    # Compare with expected behavior
    for (var, posterior) in result.posteriors
        @assert all(0 ≤ p ≤ 1 for p in posterior) "Invalid probability values"
    end
    
    println("Model validation passed")
    return true
end
```

### Performance Benchmarking
```julia
using BenchmarkTools

function benchmark_inference(model_func, data_sizes)
    results = Dict()
    
    for size in data_sizes
        test_data = generate_test_data(size)
        
        benchmark_result = @benchmark infer(
            model = $model_func($test_data),
            data = $test_data,
            iterations = 20
        )
        
        results[size] = benchmark_result
        println("Size $size: $(median(benchmark_result.times) / 1e6) ms")
    end
    
    return results
end
```

## References

- [RxInfer.jl Documentation](https://rxinfer.ml/docs/)
- [ReactiveMP.jl Documentation](https://reactivemp.github.io/ReactiveMP.jl/stable/)
- [GraphPPL.jl Documentation](https://biaslab.github.io/GraphPPL.jl/stable/)
- [Active Inference Tutorial](../gnn/about_gnn.md)
- [GNN Specification](../gnn/gnn_syntax.md)
- [Matrix Algebra in Active Inference](../gnn/gnn_implementation.md)

## Troubleshooting

For RxInfer-specific issues:
1. Check the [RxInfer.jl GitHub Issues](https://github.com/biaslab/RxInfer.jl/issues)
2. Review [ReactiveMP examples](https://reactivemp.github.io/ReactiveMP.jl/stable/examples/)
3. Consult the [GNN troubleshooting guide](../troubleshooting/README.md)
4. Post questions in [GNN Discussions](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/discussions) 