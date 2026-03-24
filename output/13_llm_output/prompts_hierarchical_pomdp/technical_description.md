# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

Here is the complete implementation of the GNN model:
```python
import numpy as np
from scipy import stats
from scipy import linalg

def gnn_model(num_hidden_states, num_actions):
    """GNN Model Representation.

    Args:
        num_hidden_states (int): Number of hidden states in the POMDP.
        num_actions (int): Number of actions to be inferred from.

    Returns:
        A numpy array representing the GNN model representation.
    """
    # Initialize state space block
    d1 = np.array([[0, 0], [num_hidden_states - 2 * num_actions + 3 * num_hidden_states // 4]
              for _ in range(num_actions)])

    # Initialize action-level probabilities matrix
    d2 = np.array([
        [[1., 0.], [num_actions - 2 * num_actions + 3 * num_actions // 4]]
          for _ in range(num_hidden_states)]
      )

    # Initialize state transition matrices
    d1a = np.array([[[0, 0], [num_actions - 2 * num_actions + 3 * num_actions // 4]
              for _ in range(num_actions)])
    d1b = np.array([
        [[1., 0.], [num_actions - 2 * num_actions + 3 * num_actions // 4]]
          for _ in range(num_hidden_states)]
      )

    # Initialize action-level probabilities matrix
    d2a = np.array([[[0, 0], [1., 0.]]
              for _ in range(num_actions)])
    d2b = np.array([
        [[1.], [num_actions - 2 * num_actions + 3 * num_actions // 4]]
          for _ in range(num_hidden_states)]
      )

    # Initialize state-level probabilities matrix
    d1a_prime = np.array([[[0, 0], [1., 0.]]
              for _ in range(num_actions)])
    d2a_prime = np.array([
        [[1.], [num_actions - 2 * num_actions + 3 * num_actions // 4]]
          for _ in range(num_hidden_states)]
      )

    # Initialize