# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

Here's the signature for the GNN model:
```python
G = gnn_signature(
    "Active Inference POMDP Agent",
    num_hidden_states=3,
    num_actions=10,
    num_timesteps=256,
    action_prior="Habit" # Uniform prior over actions.
)
```