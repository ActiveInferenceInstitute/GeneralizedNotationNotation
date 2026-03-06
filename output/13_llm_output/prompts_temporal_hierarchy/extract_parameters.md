# EXTRACT_PARAMETERS

Okay, here’s a detailed breakdown of the GNN specification, organized as requested. This analysis focuses on the Active Inference concepts and mathematical relationships within the model.

**1. Model Matrices:**

*   **A Matrices (Likelihood Matrices):** These represent the probability of observing an observation given a particular state at each level.
    *   **A0:** (3x4) – Level 0 likelihood.  `A0[i,j]` represents the probability of observing observation `j` given that the hidden state is in state `i`.
    *   **A1:** (4x3) – Level 1 likelihood. `A1[i,j]` represents the probability of observing observation `j` given that the hidden state is in state `i`.
    *   **A2:** (3x2) – Level 2 likelihood. `A2[i,j]` represents the probability of observing observation `j` given that the hidden state is in state `i`.
*   **B Matrices (Transition Matrices):** These represent the probability of transitioning to a new state given the current state and action at each level.
    *   **B0:** (4x4) – Level 0 transition. `B0[i,j,k]` represents the probability of transitioning to state `j` from state `i` after taking action `k`.
    *   **B1:** (3x3) – Level 1 transition. `B1[i,j,k]` represents the probability of transitioning to state `j` from state `i` after taking action `k`.
    *   **B2:** (2x2) – Level 2 transition. `B2[i,j,k]` represents the probability of transitioning to state `j` from state `i` after taking action `k`.
*   **C Matrices (Preference Vectors):** These represent the preferences or goals at each level, modulating the likelihoods and transitions. They essentially bias the model towards certain states or actions.
    *   **C0:** (3x1) – Level 0 preference. `C0[i]` represents the preference for state `i`.
    *   **C1:** (4x1) – Level 1 preference. `C1[i]` represents the preference for state `i`.
    *   **C2:** (3x1) – Level 