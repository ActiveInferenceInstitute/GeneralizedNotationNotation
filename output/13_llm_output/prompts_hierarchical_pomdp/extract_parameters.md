# EXTRACT_PARAMETERS

Okay, here’s a detailed breakdown of the parameters extracted from the provided GNN specification, organized according to your requested categories. This analysis focuses on the Active Inference aspects and provides a rigorous interpretation of the parameters within the model’s structure.

**1. Model Matrices:**

*   **A1 (Level 1 Likelihood Matrix):**
    *   Dimensions: 4x4x4 (4 observations, 4 hidden states, type=float)
    *   Structure: A 4x4 matrix representing the likelihood of observing the ‘o1’ (observation) given the hidden state ‘s1’.  This is the core of the bottom-up inference at Level 1.
    *   Interpretation:  Represents the model’s belief about the world given its current hidden state.  The values within the matrix determine the relative probability of observing each observation given a particular hidden state.
*   **B1 (Level 1 Transition Matrix):**
    *   Dimensions: 4x4x3 (next x prev x actions, type=float)
    *   Structure: A 4x4x3 matrix. The first dimension (4) corresponds to the number of hidden states. The second dimension (4) represents the previous hidden state. The third dimension (3) represents the possible actions.
    *   Interpretation:  Defines the transition probabilities between hidden states based on the chosen action.  It’s the core of the forward dynamics at Level 1.
*   **C1 (Level 1 Preference Vector):**
    *   Dimensions: 4x1 (Level 1 preferences over observations, type=float)
    *   Structure: A 4x1 vector.
    *   Interpretation: Represents the model’s preference for observing specific observations.  This vector biases the bottom-up inference towards the preferred observations.
*   **D1 (Level 1 Prior Over Hidden States):**
    *   Dimensions: 4x1 (Level 1 prior over hidden states, type=float)
    *   Structure: A 4x1 vector.
    *   Interpretation:  The initial prior belief distribution over the hidden states at Level 1.  This is the starting point for the Bayesian inference.
*   **A2 (Higher-Level Likelihood Matrix):**
    *   Dimensions: 4x2x4 (maps context to Level 1 hidden