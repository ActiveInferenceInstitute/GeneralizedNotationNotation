# EXPLAIN_MODEL

Here is a summary of the key components:

1. **Model Purpose**: This document provides an overview of how the Multi-Agent Cooperative Active Inference (MCAF) algorithm works and its core components. It covers what happens when agents interact with each other, what actions are available to them, and how they evolve over time.

2. **Core Components**:
   - **Hidden states**: The model represents a set of hidden states that encode the agent's beliefs about their own state and the shared environment space. These hidden states can be thought of as "actions" or "prior probabilities".
   - **Observations**: The network learns to update its beliefs based on new observations, which are represented by actions (represented by vectors). Actions represent a sequence of actions that are available for each agent.
   - **Actions**: The network updates its belief in the shared environment space using a policy vector and action vector. These actions can be thought of as "actions" or "prior probabilities".

3. **Model Dynamics**: This document provides an overview of how the model evolves over time, including key relationships between hidden states (s_f0, s_f1, etc.), actions (u_c0, π_c0), and observations (o_m0, o_m1). It also discusses what happens when agents interact with each other.

4. **Active Inference Context**: This document provides a concise overview of how the model implements Active Inference principles in practice by learning to update its beliefs based on new observations. It describes key relationships between hidden states (s_f0, s_f1), actions (u_c0, π_c0), and observable actions (o_m0, o_m1).

5. **Practical Implications**: This document provides a concise overview of how the model can inform decisions by providing accurate predictions based on current beliefs and actions. It also discusses what happens when agents interact with each other in practice to improve their performance over time.