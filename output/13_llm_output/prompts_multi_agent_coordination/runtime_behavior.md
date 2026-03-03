# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

Okay, let's break down how this GNN model would run and its potential behavior across different settings. This is a fascinating example of using GNNs to model social cognition and cooperation, and it’s a good illustration of Active Inference principles.

**1. Core Mechanics of the Model Run**

This model operates within the framework of Active Inference, specifically using a POMDP (Partially Observable Markov Decision Process) structure. Here’s a step-by-step breakdown of how it would run:

* **Initialization:** The model starts with initial beliefs (D1, D2), prior states (s1, s2), and policies (π1, π2) for each agent. The shared preference vector (C1, C2) is set to a strong preference for state 4 (index 3) – representing the goal.
* **Time Step (t):**  At each timestep ‘t’ (up to 20 in this case):
    * **Agent Inference:** Each agent (Agent 1 and Agent 2) independently performs the following:
        * **Observe:** Receives its observation (o1, o2) – this includes information about the other agent’s *last* action.
        * **Calculate Likelihood:**  Uses the likelihood matrix (A1, A2) to assess the probability of its current hidden state (s1, s2) given the observation.
        * **Update Belief:**  Combines the likelihood with its prior belief (D1, D2) to update its belief about its hidden state (s1, s2).
        * **Calculate Expected Free Energy (EFE):**  This is the core of Active Inference.  The agent calculates the expected free energy (G1, G2) – a measure of how good its current state is, considering its uncertainty and preferences. This involves using the transition matrix (B1, B2) and the preference vector (C1, C2).
        * **Generate Policy:** Based on the EFE, the agent generates a policy (π1, π2) – a probability distribution over its possible actions.
        * **Select Action:**  The agent selects an action (u1, u2) according to its policy.
    * **Environment Update:** The agent’s action (u1, u2) is applied to the shared environment, updating the joint state (s_joint).
    * **Shared Observation