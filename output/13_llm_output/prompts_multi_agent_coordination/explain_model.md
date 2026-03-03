# EXPLAIN_MODEL

Okay, let's dissect this GNN specification for a Multi-Agent Cooperative Active Inference model. This is a fascinating example demonstrating how Active Inference can be used to model social coordination without explicit communication.

**1. Model Purpose:**

This model aims to represent and simulate the emergence of cooperation between two agents in a shared environment. Specifically, it’s designed to capture how agents can coordinate their actions to achieve a common goal – in this case, reaching a specific state – simply by observing each other’s actions and sharing a common observation space. It’s a simplified representation of social cognition and coordination, focusing on the underlying Bayesian inference processes driving agent behavior.  It's essentially a testbed for exploring how agents can learn to cooperate through observation and shared goals.

**2. Core Components:**

* **Hidden States (s_f0, s_f1, etc.):** These represent the agents’ internal beliefs about their own position in the shared environment.  Each agent maintains a hidden state vector of length 4, representing their estimate of their own position (likely x, y coordinates in a 2D space).  The `s1` and `s2` variables are the key hidden states.  These states are *latent* – the agents don't directly perceive their own position, but infer it based on their observations and prior beliefs.

* **Observations (o_m0, o_m1, etc.):** The observations are the information the agents receive about the environment.  Crucially, each agent receives observations from *both* itself and the other agent.
    * `o1` and `o2` represent the agent's observation of the *other* agent's last action (or, more accurately, the last state of the other agent).  The `o_joint` observation represents the overall goal achievement.
    * The observation space is 4x1, meaning each agent receives a single value representing the state of the shared environment.

* **Actions/Controls (u_c0, π_c0, etc.):**  Each agent has a policy (π) that dictates its action (u). The policy is a vector of 3 probabilities, representing the likelihood of taking each of the 3 available actions. The action itself (u) is a discrete choice (1, 2, or 3).  The model assumes the agents choose actions based on their policies, effectively implementing a stochastic control mechanism.


**3