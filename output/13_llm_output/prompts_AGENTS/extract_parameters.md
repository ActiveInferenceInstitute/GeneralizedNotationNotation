# EXTRACT_PARAMETERS

Okay, let's meticulously extract and organize the parameters from this GNN specification document focusing on Active Inference principles. This breakdown will be structured as requested, providing a detailed scientific analysis of each parameter category.

**1. Model Matrices:**

* **A matrices:** The document explicitly states that "explicit `A/B/C/D/E` matrices and matrix provenance comments" are preserved.  The context suggests this refers to the transition model for the POMDP gridworld. We can infer the following:
    * **Dimensions:** Since it’s a 3x3 grid world, we assume a state space of size *n = 9*. The A matrix represents the transition probabilities between these states given an action.  Therefore, `A` is a 9 x 9 matrix.
    * **Structure & Interpretation:**  The A matrix encodes the probabilistic relationships between states. Each entry `A[i, j]` would represent the probability of transitioning from state *i* to state *j* after taking a specific action. The exact values within this matrix are not provided in the document but define the core dynamics of the environment.
    * **Provenance:** The preservation of provenance comments is crucial for understanding how these transition probabilities were derived (e.g., based on simulation, expert knowledge, or learned models).

* **B matrices:**  Again, explicitly stated to be present.
    * **Dimensions:** `B` is a (next_state, previous_state, action) tensor. This implies that the dimensions are related to the state and action spaces. Let’s assume the state space is 9 (as above). The number of actions is 5 (stated in "Runtime metadata uses ‘num_timesteps: 15’, ‘random_seed: 42’, and five actions."). Therefore, `B` would be a tensor with dimensions approximately  (9 x 9 x 5) – representing the probability of transitioning to the *next* state given the *previous* state and the taken *action*.
    * **Structure & Interpretation:** The B matrix represents the conditional transition probabilities. It’s how the environment responds to an action in a particular state.

* **C matrices:** Not explicitly defined, but implied as part of the POMDP framework.  They represent the observation model – the probability of observing an observation given the true underlying state.
    * **Dimensions:** The dimensions would be related to the state and observation spaces. Let