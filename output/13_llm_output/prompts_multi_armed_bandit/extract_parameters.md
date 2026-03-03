# EXTRACT_PARAMETERS

Okay, let’s meticulously extract and organize the parameters from this GNN specification for the Multi-Armed Bandit model. This breakdown will adhere to your requested format, focusing on Active Inference concepts and mathematical relationships.

**1. Model Matrices:**

*   **A Matrix (Reward Likelihood):**
    *   **Dimensions:** 3 x 3 x 3 (3 hidden states, 3 observation types, 3 contexts)
    *   **Structure:** A 3D tensor. Each element A[i, j, k] represents the likelihood of observing reward type *j* (0: no-reward, 1: small-reward, 2: big-reward) given that the current hidden state is *i* (the context representing the best arm).
    *   **Interpretation:** This matrix encodes the agent’s prior belief about the reward landscape.  A[0, 1, 0] would represent the likelihood of observing a small reward (observation type 1) when the agent believes arm 0 is the best (context 0).  The values are normalized to sum to 1 along the observation type dimension.
*   **B Matrix (Context Transitions):**
    *   **Dimensions:** 3 x 3 x 3 x 3 (3 hidden states, 3 actions, 3 next states, 3 contexts)
    *   **Structure:** A 4D tensor. Each element B[i, a, j, k] represents the probability of transitioning from context *k* to next state *j* when the current state is *i* and the agent takes action *a*.
    *   **Interpretation:** This matrix defines the dynamics of the hidden state.  Crucially, it’s designed to be *sticky*, meaning transitions are largely deterministic.  B[0, 0, 0, 0] is very high (0.9), indicating a strong tendency to remain in context 0 if the agent doesn’t take any action. This reflects the core assumption of the degenerate POMDP – the state dynamics are nearly static.
*   **C Matrix (Log Preference Vector):**
    *   **Dimensions:** 3 x 1
    *   **Structure:** A 2D vector. Each element C[i] represents the log-preference for reward type *i* (0: no-reward, 1: small-reward, 2: big