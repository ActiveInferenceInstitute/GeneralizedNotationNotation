# Nontechnical Description

Prompt:

> Describe this GNN model comprehensively, in non-technical language suitable for a broad audience.

Response:

You can also use a `GNNModel` with an optional `action_bias` parameter to specify the direction of inference:
```python
model = CryptoSequential(
    GNNModel("cryptography", "actions"),
    action_bias=0.1,
    action_bias="forward" if action == "backward" else 0)
```