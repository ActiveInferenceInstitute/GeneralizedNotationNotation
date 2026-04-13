# EXPLAIN_MODEL

Here's a concise overview of the GNN representation:

**Model Purpose:**
This is a degenerate POMDP (Multi-Armed Bandit) that represents a simple decision-making problem with two actions and three hidden states. The model learns to optimize between exploration and exploitation based on the available rewards, while also updating beliefs about the agents' intentions and their relative positions in the reward space.

**Core Components:**

1. **Likelihood Matrix**: A probability distribution over all possible outcomes (actions) that represent the uncertainty of future actions. The likelihood matrix captures the probabilities of different actions given a specific state, allowing for exploration and exploitation based on rewards.

2. **Transition Matrix**: A probability distribution over all possible transitions between states (arms). This represents the probability of observing arms in one arm at another arm. It also allows for exploring the reward space by choosing which arm is best.

3. **Probable Actions**: A set of actions that represent the available actions, capturing the uncertainty about future actions and their relative positions in the reward space. These actions are represented as a probability distribution over all possible outcomes (actions).
4. **Action Vector**: A vector representing the rewards received by each arm based on its current state. This represents the uncertainty around which arms will be best for exploration or exploitation, while also allowing for exploring the reward space using different actions.

5. **Probable Actions**: A set of actions that represent the available actions and their corresponding probabilities (probabilities). These actions are represented as a probability distribution over all possible outcomes (actions), capturing the uncertainty around which arms will be best for exploration or exploitation, while also allowing for exploring the reward space using different actions.
6. **Action Vector**: A vector representing the rewards received by each arm based on its current state and corresponding probabilities of receiving rewards from other arms in that arm. This represents the uncertainty around which arms will be best for exploration or exploitation, while also allowing for exploring the reward space using different actions.
7. **Probable Actions**: A set of actions that represent the available actions and their corresponding probabilities (probabilities). These actions are represented as a probability distribution over all possible outcomes (actions), capturing the uncertainty around which arms will be best for exploration or exploitation, while also allowing for exploring the reward space using different actions.
8. **Action Vector**: A vector representing the rewards received by each arm based on its current state and corresponding probabilities of receiving rewards from other arms in that arm. This