# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

Here's the GNN representation for the Active Inference POMDP agent:
```python
G = Gnn(
    num_hidden_states=3,
    num_actions=10,
    num_timesteps=256,
    num_observations=48,
    action_prior="habit",
    policy_prior="policy"
)
```
This representation is based on the GNN implementation from [https://github.com/david-springer/gnn](https://github.com/david-springer/gnn). The `G` symbol represents a GNN instance, and the `Varname` variable defines the name of the model annotation.
The `num_hidden_states`, `num_actions`, and `num_timesteps` are all integers representing the number of hidden states, actions, and timesteps in the POMDP agent, respectively. The `action_prior` is a dictionary mapping each action to its probability distribution over the history of observations (represented as an integer).
The `policy_prior`, `belief_prior`, and `hidden_state_prior` are dictionaries representing the prior distributions for each hidden state and actions in the POMDP agent. The `num_actions` is a list containing the number of actions, which can be used to initialize the policy vector.
The `num_observations` represents the total number of observations (represented as an integer) that are available for inference. This allows us to compute the total number of predictions and beliefs in the POMDP agent.
Finally, the `action_prior`, `policy_prior`, and `belief_prior` represent the prior distributions over actions, policies, and habit probabilities, respectively. These variables can be used to initialize the policy vector, belief vector, and action vector for inference.