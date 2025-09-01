# actinf_pomdp_agent - Categorical Diagram Analysis

**Generated:** 2025-08-29T11:49:29

## Diagrams Created

### main_pomdp_diagram

- **Domain:** `State @ Action`
- **Codomain:** `Observation`
- **Properties:**
  - type: composite_pomdp_morphism
  - components: transition >> observation
  - input_structure: State @ Action
  - output_structure: Observation
  - category: monoidal

### simple_pomdp_diagram

- **Domain:** `S @ A`
- **Codomain:** `O`
- **Properties:**
  - type: basic_morphism_composition
  - components: f >> g
  - domain_structure: S @ A
  - codomain_structure: O
  - category: monoidal

## Morphisms

### transition_morphism

- **Source:** `S @ A`
- **Target:** `S`
- **Composition Details:**
  - interpretation: State transition function
  - maps: State-Action pairs to next States
  - deterministic: Based on GNN B tensor specification

### observation_morphism

- **Source:** `S`
- **Target:** `O`
- **Composition Details:**
  - interpretation: Observation function
  - maps: States to Observations
  - probabilistic: Based on GNN A matrix specification
