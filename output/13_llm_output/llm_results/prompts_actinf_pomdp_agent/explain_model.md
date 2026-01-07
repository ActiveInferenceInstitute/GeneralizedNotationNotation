# EXPLAIN_MODEL

Based on the doc's description, I'll provide a comprehensive explanation of GNNs for Active Inference models:

1. **Definition**: This model represents a classic Active Inference agent used to learn from probability distributions with two observation modalities (`observation_outcomes` and `hidden_states`) with an initial policy (`habit`) based on action choices (`actions`). It learns through the following steps:
- The agent maps each observed state and actions onto new hidden states. Each time, it updates its prior over these shared observations and biases in probability distributions over actions to predict future ones.

2. **Model Purpose**: This model represents a class of Active Inference POMDP agents that learn to infer the probability distribution of each observation based on their associated state transitions (`states`) and action choices (`actions`). It uses Variational Free Energy (VFE) as its goal, with initial policies acting as actions.

3. **Core Components**: The models represent:
   - `hidden_state`: The probability distribution over states for each of the observed states in the agent's set-up; it encodes both belief and action biases during inference.
   - `observations`: A 2D numpy array representing the observations, which capture a sequence of random actions taken by the agent. These actions are mapped onto the hidden states for later learning based on the probability distribution over actions across observed states.
   - `actions**: A list of sequences that represent an action chosen from each of the available actions (action choices) to be learned from by the agent, with a corresponding reward and return state whenever they're chosen. These are initialized in sequence order according to their action choice probabilities (`pi_c0`), allowing for inference based on initial beliefs without prior knowledge or predictions about future outcomes given actions choices.
   - `states`: A 2D numpy array representing the sequence of observed states, with each step being a state transition followed by subsequent actions chosen from available actions (action choices). These are initialized in sequence order according to their action choices (`actions`), allowing for inference based on current beliefs before they're updated towards new learned probability distributions.

4. **Active Inference Context**: The POMDP agent learns by sequentially updating its beliefs over the entire history of observed states and actions, until there's no longer an incentive to continue learning. During this process:
   - The `belief_updates` function is used to update a belief from prior probabilities across all observed states (`π[states]`) based on posterior probability distributions within each observation (actions) for that state (`σ[state][action]) and then backout towards the beginning of the history, choosing actions after initial beliefs are updated.
   - The `belief_update` function is used to update an action's belief into a new observable under the assumption it will be later chosen from subsequent observations (`u`) as the agent learns through action choices across observed states (policy posterior).

5. **Active Inference**: When predictions converge towards uncertain actions, the hypothesis of future actions and beliefs are updated using Bayesian inference to update our beliefs about future actions and biases based on probabilities within each observable space (`σ[observation][action])`), allowing for active inference through prediction updates after initial belief updates back out once more observations are observed in a given observation.

Please provide more details or context specific to your question, such as the types of actions used (actions taken by the agent) and where action choices were made across available actions when using Bayesian inference methods, if applicable.