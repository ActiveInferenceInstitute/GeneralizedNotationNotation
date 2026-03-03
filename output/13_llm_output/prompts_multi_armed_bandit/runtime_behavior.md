# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

Okay, let's break down what happens when this GNN Multi-Armed Bandit model runs and how its behavior would shift across different settings. This is a fantastic example for illustrating Active Inference and GNN specifications.

**Core Mechanics – How it Runs (Active Inference in Action)**

This model embodies Active Inference at its core. Here’s a step-by-step breakdown of what happens during each time step:

1. **Observation:** The agent receives an observation (`o`). This observation represents the reward signal – it could be ‘no reward’, ‘small reward’, or ‘big reward’ (indexed 0, 1, or 2).

2. **Inference (qs = infer_states(reward_obs)):**  This is the crucial Active Inference step. The GNN takes the observation (`o`) as input and *infers* the most likely hidden state (`s`) representing the reward context.  The GNN learns a mapping between observations and the underlying reward context.  Essentially, it’s asking, "Given this reward signal, what’s the most likely state of the world (which arm is best)?"  The `infer_states` function is the GNN’s internal mechanism for this inference.

3. **Predict Next State (s_prime):** Based on the inferred hidden state (`s`), the model predicts the next hidden state (`s_prime`) using the transition matrix (`B`). Because the transitions are *sticky*, meaning the context doesn’t change based on the action, the next state is almost always the same as the current state. This is a key simplification for testing exploration vs. exploitation.

4. **Action Selection (u ~ softmax(-G)):** The agent then chooses an action (`u`) – pulling one of the three arms – based on the expected free energy (`G`). The `softmax` function converts the expected free energy into a probability distribution over the actions.  The agent is trying to minimize its free energy, which represents the trade-off between:
   * **Pragmatic Value (Reward):**  The expected reward from taking a particular action in the current context.
   * **Epistemic Value (Information Gain):** The uncertainty about the hidden state.  The agent is incentivized to explore and reduce this uncertainty.

5. **Execute Action & Update Belief (u>s_prime):** The agent pulls the chosen arm (`u`), and this action *immediately* updates the belief