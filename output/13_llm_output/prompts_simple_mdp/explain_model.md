# EXPLAIN_MODEL

Okay, let's dissect this GNN specification for the “Simple MDP Agent.” This model represents a foundational example in Active Inference, specifically designed to illustrate a simplified Markov Decision Process (MDP) under fully observable conditions.

**1. Model Purpose:**

This model simulates a simple agent navigating a 2x2 grid. The core purpose is to demonstrate the fundamental mechanics of Active Inference – how an agent, given its sensory input, updates its beliefs about its environment and then chooses actions to minimize its expected free energy. It’s a crucial starting point for understanding more complex Active Inference models by isolating the key components and their interactions.  It’s essentially a simplified robot navigation problem.

**2. Core Components:**

*   **Hidden States (s_f0, s_f1, s_f2, s_f3):** These represent the agent's belief about its location on the 2x2 grid. Each state corresponds to one of the four corners of the grid.  The agent doesn't *know* its exact location; instead, it maintains a probability distribution across these four possible locations.  The `s[4,1,type=float]` and `s_prime[4,1,type=float]` matrices define these distributions.
*   **Observations (o_m0, o_m1, o_m2, o_m3):** The observations are identical to the hidden states. This is the “fully observable” part – the agent receives perfect information about its location. The `o[4,1,type=int]` matrix stores the current observation.
*   **Actions/Controls (u_c0, u_c1, u_c2, u_c3):** The agent has four discrete actions:
    *   `u_c0` (stay): The agent remains in its current location.
    *   `u_c1`, `u_c2`, `u_c3`: These represent movements to the adjacent grid locations (North, South, East).  The `π[4,type=float]` matrix defines the policy, i.e., the probability distribution over these actions.

**3. Model Dynamics:**

The model operates in discrete time steps (`t`). At each time step, the agent:

1.  **Receives an Observation (o):** Based on its current hidden state (s).
2.  **Updates its Belief