# EXTRACT_PARAMETERS

Okay, let’s meticulously extract and organize the parameters from this GNN specification, providing a detailed breakdown as requested. This will focus on the Active Inference aspects and their mathematical implications within the POMDP framework.

**1. Model Matrices:**

*   **A Matrix (Likelihood Matrix):**
    *   Dimensions: 9x9 (representing 9 grid cells in the 3x3 world)
    *   Structure: A dense matrix where each element `A[i, j]` represents the likelihood of observing a particular observation (`o[j]`) given that the hidden state is at cell `s[i]`. This reflects the noisy observation model.  The values within this matrix are crucial for Bayesian inference – they quantify how likely an observation is *given* a specific belief about the world's underlying state.
    *   Interpretation: The diagonal elements (A[i, i]) represent the baseline likelihood of observing the current observation given that you already know your location. Off-diagonal elements capture the increased likelihood due to uncertainty in the hidden state.

*   **B Matrix (Transition Tensor):**
    *   Dimensions: 9x9x5 (representing 9 grid cells, 9 possible previous states, and 5 actions)
    *   Structure: A multi-dimensional tensor. Each element `B[i, j, k]` represents the probability of transitioning to state `s[j]` from state `s[i]` after taking action `u[k]`. This is the core transition dynamics within the POMDP. The 5 actions are: up, down, left, right and stay.
    *   Interpretation:  This matrix encodes the agent's model of how its actions affect its hidden state. It’s a key component in calculating the expected free energy (explained later).

*   **C Matrix (Log-Preferences over Observations):**
    *   Dimensions: 9x1 (representing 9 grid cells and one observation)
    *   Structure: A vector where each element `C[i]` represents the log-preference for observing observation `o[i]` given that the hidden state is at cell `s[i]`. This reflects the agent’s intrinsic preference for certain observations.
    *   Interpretation:  This matrix biases the agent towards preferring certain observations, influencing its belief update process.

*   **D Matrix (Prior over Initial Hidden State):**
    *   Dimensions: