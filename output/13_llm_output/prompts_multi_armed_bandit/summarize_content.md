# SUMMARIZE_CONTENT

Here's a concise summary of the GNN specification:

**Model Overview**
This is a degenerate POMDP that models multi-armed bandit decision-making under sticky context. The model consists of three hidden states, five observation types (arms), and 3 actions. It performs well in testing when the reward dynamics are trivial but becomes less effective with more complex rewards or when there's an asymmetry between arms.

**Key Variables**

1. **Hidden States**: A list of categories representing different arm configurations. Each category has a corresponding reward value, which is tracked across actions and actions' outcomes. The reward values represent the reward received from each arm based on its action.

2. **Observations**: A list of rewards for each arm in the POMDP. Each observation represents an arm's reward.

3. **Actions/Controls**: A list of actions, which are actions that affect the reward distribution across arms and their outcomes. Actions can be either identity (pulling one arm) or identity-exchange (pulling two arms).

**Critical Parameters**

1. **Most Important Matrices**: The Likelihood Matrix for each hidden state category represents the probability of observing a reward in each action, which is tracked across actions and actions' outcomes. This matrix captures the uncertainty about rewards based on the available actions.

2. **Key Variables**: A list of matrices representing the probabilities of observing different types of rewards (identity vs identity-exchange) for each hidden state category. These matrices represent the probability distribution over reward distributions across arms, with each row representing a single action and its corresponding reward value.

**Notable Features**

1. **Unique Constraints**: The model's key features are:
   - **Random Walks**: Each arm has an equal chance of receiving different rewards (identity vs identity-exchange).
   - **Symmetry**: The reward distributions across arms are symmetric, with the same probability distribution for each arm.
   - **Uniformity**: The reward distributions across arms are uniform, with the same probability distribution for each arm.

2. **Unique Constraints**: The model's key features are:
   - **Random Walks**: Each arm has an equal chance of receiving different rewards (identity vs identity-exchange).
   - **Symmetry**: The reward distributions across arms are symmetric, with the same probability distribution for each arm.

**Use Cases**

1. **Testing**: This model performs well in testing when the