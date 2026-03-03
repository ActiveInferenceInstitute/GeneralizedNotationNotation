# EXTRACT_PARAMETERS

Okay, here’s a detailed breakdown of the parameters extracted from the provided GNN specification, organized as requested. This analysis focuses on the Active Inference concepts and mathematical relationships implied by the model structure.

**1. Model Matrices:**

*   **A Matrices (Likelihood Matrices):** These represent the probability of observing an observation given a hidden state.
    *   **A0:** (3x4) – Level 0 likelihood.  `A0[i,j]` represents the probability of observing observation `j` given a hidden state `i`.  This reflects the sensory-motor mapping.
    *   **A1:** (4x3) – Level 1 likelihood. `A1[i,j]` represents the probability of observing a tactical observation given a tactical hidden state `i`.
    *   **A2:** (3x2) – Level 2 likelihood. `A2[i,j]` represents the probability of observing a strategic observation given a strategic hidden state `i`.
*   **B Matrices (Transition Matrices):** These represent the probability of transitioning to a new hidden state given the current hidden state and an action.
    *   **B0:** (4x4x3) – Level 0 transitions. `B0[i,j,k]` represents the probability of transitioning to state `j` from state `i` after taking action `k`.  This is a crucial element for sensorimotor control.
    *   **B1:** (3x3x3) – Level 1 transitions. `B1[i,j,k]` represents the probability of transitioning to state `j` from state `i` after taking action `k`.
    *   **B2:** (2x2x2) – Level 2 transitions. `B2[i,j,k]` represents the probability of transitioning to state `j` from state `i` after taking action `k`.
*   **C Matrices (Preference Vectors):** These represent the preferences or goals at each level, modulating the likelihoods. They are the "top-down" influences.
    *   **C0:** (3x1) – Level 0 preferences. `C0[i]` represents the preference for observation `i` (modulated by Level 1).
    *   **C1:** (4x1) – Level 1 preferences. `C1[i]` represents