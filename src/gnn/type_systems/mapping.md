# GNN Type System Mapping

This document maps elements from concrete GNN models (like `actinf_pomdp_agent.md`) to their representations in our type system implementations.

## POMDP Agent Model to Type System Mapping

| POMDP Model Element | Scala Implementation | Haskell Implementation | Notes |
|---------------------|----------------------|------------------------|-------|
| A (Likelihood Matrix) | `LikelihoodMatrix` class | `LikelihoodMatrix s o` | Both implementations support this core structure |
| B (Transition Matrix) | `TransitionMatrix` class | `TransitionMatrix s u` | Both implementations support this core structure |
| C (Preference Vector) | `PreferenceVector` class | `PreferenceVector o` | Both implementations support this core structure |
| D (Prior Vector) | `PriorVector` class | `PriorVector s` | Both implementations support this core structure |
| E (Habit Vector) | Not explicitly modeled | Not explicitly modeled | **Gap**: Should be added to both implementations |
| s (Hidden State) | `HiddenState` variable type | `HiddenState` variable type | Both represent hidden states |
| o (Observation) | `Observation` variable type | `Observation` variable type | Both represent observations |
| π (Policy) | Represented in policy inference | Represented in policy inference | Both implementations handle policy distributions |
| u (Action) | `Action` variable type | `Action` variable type | Both represent actions |
| G (Expected Free Energy) | `expectedFreeEnergy()` function | `expectedFreeEnergy()` function | Both implement the calculation |
| t (Time) | `timeHorizon` parameter | `timeHorizon` parameter | Time represented as a horizon, not as an explicit variable |
| Connections (D>s, s-A, etc.) | Implied by model structure | Implied by model structure | **Gap**: Explicit connection syntax not represented |
| Initial Parameterization | Not directly represented | Not directly represented | **Gap**: Values would be instantiations of the structures |

## Implementation Roadmap

To achieve full coverage, the following extensions are needed:

1. Add explicit habit vector (E) representation to both implementations
2. Develop a connection representation system that mirrors the GNN syntax
3. Create mechanisms for specifying initial parameterization values
4. Align ontology annotations with the type systems

These enhancements will ensure the type systems can fully represent and validate all aspects of GNN models. 