# EXPLAIN_MODEL

Okay, let’s dissect this GNN specification for the Multi-Armed Bandit (MAB) model. This is a carefully constructed example designed to illustrate key concepts within Active Inference and Bayesian modeling, specifically targeting the exploration-exploitation dilemma.

**1. Model Purpose:**

This GNN specification represents a classic multi-armed bandit problem.  The goal is for an agent to learn which of three slot machines (arms) yields the highest average reward over time. It’s a degenerate POMDP – meaning it’s simplified to the point where the state dynamics are extremely regular, allowing us to focus on the core Active Inference mechanisms of belief updating and action selection.  It’s a valuable test case for understanding how an agent can learn to exploit a known good arm while still exploring to potentially discover even better options.

**2. Core Components:**

*   **Hidden States (s):** The model employs three hidden states, denoted as `s_f0`, `s_f1`, and `s_f2`. These represent the agent’s *belief* about which arm is currently the “best” or most rewarding.  Crucially, these aren’t absolute reward values; they’re *context* beliefs.  The agent doesn’t know the true reward distribution of each arm, but it believes one of the arms is currently more likely to yield a high reward.  Think of it as the agent’s internal representation of the environment’s “reward context.”

*   **Observations (o):** The agent receives three types of observations:
    *   `o_m0`: Represents a “no-reward” signal.
    *   `o_m1`: Represents a “small-reward” signal.
    *   `o_m2`: Represents a “big-reward” signal.
    These observations are the agent’s sensory input, reflecting the outcome of pulling an arm.

*   **Actions/Controls (u):** The agent has three possible actions:
    *   `u_c0`: Pull arm 0.
    *   `u_c1`: Pull arm 1.
    *   `u_c2`: Pull arm 2.
    The agent selects an action based on a policy, represented by the vector `π`.

**3. Model Dynamics:**

The model’s dynamics are governed by the following relationships:

*   **Transition (B):** The `