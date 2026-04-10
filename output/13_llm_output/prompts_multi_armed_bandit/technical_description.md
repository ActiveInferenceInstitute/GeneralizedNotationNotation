# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

You can use the following code to create a GNN representation of the multi-armed bandit:

1. Create an instance of the GNN model using `GnnModel`:
   ```python
  gnn = GNN(num_hidden_states=3, num_actions=4)
  gnn.encode([
    (0.2567989699999991, 0.2567989699999991),
    (0.2567989699999991, 0.2567989699999991),
    (0.2567989699999991, 0.2567989699999991)
  ])

  gnn.encode([
    (0.3418271630000001, 0.3418271630000001),
    (0.3418271630000001, 0.3418271630000001)
  ])

  gnn.encode([
    (0.5916268808000001, 0.5916268808000001),
    (0.5916268808000001, 0.5916268808000001)
  ])

  gnn.encode([
    (0.3476169213000001, 0.3476169213000001),
    (0.3476169213000001, 0.3476169213000001)
  ])

  gnn.encode([
    (0.5830790123000001, 0.5830790123000001),
    (0