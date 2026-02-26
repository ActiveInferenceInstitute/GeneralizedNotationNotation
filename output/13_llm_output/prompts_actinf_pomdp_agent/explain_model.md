# EXPLAIN_MODEL

Okay, let’s dissect this GNN specification for the Active Inference POMDP Agent. This model represents a classic scenario in embodied robotics and cognitive science – an agent navigating an environment where it doesn’t have complete knowledge and must actively explore to reduce uncertainty.

**1. Model Purpose:**

This model simulates a simple agent learning to navigate a 3D environment (represented by the hidden states) by observing its surroundings (observations). The agent’s goal is to minimize its uncertainty about its location, driven by a preference for observing certain states and a tendency to take actions that lead to those states.  It’s a foundational example for demonstrating Active Inference principles in a discrete POMDP setting.

**2. Core Components:**

*   **Hidden States (s_f0, s_f1, s_f2):** These represent the agent's belief about its location within the 3D environment.  Each state (indexed 0, 1, and 2) represents a possible location. The model assumes a fully controllable hidden state, meaning the agent can directly influence its location through its actions.
*   **Observations (o_m0, o_m1, o_m2):** These are the sensory inputs the agent receives from the environment. The model defines three possible observation outcomes, each corresponding to a specific hidden state.  The agent’s goal is to observe states that are likely to be associated with its true location.
*   **Actions/Controls (u_c0, π_c0, etc.):** The agent has three discrete actions (indexed 0, 1, and 2) that it can take. These actions are used to transition between the hidden states. The policy vector (π) defines the probability distribution over these actions. Importantly, this model *doesn’t* implement planning; the agent simply selects actions based on its current policy (habit).

**3. Model Dynamics:**

The model’s dynamics are governed by the following key relationships, as defined by the GNN specification:

*   **Transition (B):** The `B` matrix dictates how the agent’s hidden state changes based on the action it takes.  Each row of `B` represents a specific action, and the values within that row indicate the probability of transitioning to a particular hidden state after taking that action.  The deterministic nature of this matrix (e.g., action 0 always moves to state 0) simplifies the dynamics.