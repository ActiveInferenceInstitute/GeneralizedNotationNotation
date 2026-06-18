# EXTRACT_PARAMETERS

Okay, let's meticulously extract and organize the parameters from this GNN specification, focusing on the Active Inference framework as described. This will be a detailed breakdown based on your requested categories.

**1. Model Matrices:**

*   **A matrices:** These represent transition probabilities between hidden states.  The fixture uses a 3x3 discrete GridWorld, so:
    *   Dimensions: 9 x 9 (representing the 9 possible hidden states).
    *   Structure: A square matrix where `A[i, j]` represents the probability of transitioning from state *i* to state *j*.  Since transitions are boundary-clamped, this matrix will be symmetric.
    *   Interpretation: The values in A define the agent's belief about how likely it is to move between states given its current state and action.

*   **B matrices:** These represent the observation emissions based on hidden states.
    *   Dimensions: 9 x 9 (matching the number of hidden states).
    *   Structure: A square matrix where `B[i, j]` represents the probability of observing a particular observation *j* given that the agent is in state *i*.
    *   Interpretation:  The model specifies noisy identity emissions. This means each observation has an equal probability of being emitted from any hidden state. Therefore, B will be a diagonal matrix with 1s on the diagonal and 0s elsewhere.

*   **C matrices:** These represent the mapping between actions and changes in hidden states.
    *   Dimensions: 9 x 5 (9 hidden states * 5 possible actions).
    *   Structure: A rectangular matrix where `C[i, k]` represents the probability of transitioning to state *i* given that the agent took action *k*.
    *   Interpretation: This matrix defines how each action influences the belief about the current hidden state.

*   **D matrices:** These represent the observation emissions based on actions.
    *   Dimensions: 9 x 5 (9 hidden states * 5 possible actions).
    *   Structure: A rectangular matrix where `D[i, k]` represents the probability of observing a particular observation *k* given that the agent took action *k* and is in state *i*.
    *   Interpretation: Similar to B, but now incorporates the effect of the action on the observation.  Given noisy identity emissions, D will also be a diagonal matrix