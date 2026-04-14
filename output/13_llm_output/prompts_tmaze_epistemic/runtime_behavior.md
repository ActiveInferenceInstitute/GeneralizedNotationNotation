# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

You can use the following code for your GNN implementation:
```python
import numpy as np

def gnn_agent(x):
    x = np.array([x])

    # Initialize state and reward matrices
    A = np.zeros((4, 3))
    B = np.ones((2, 1))
    C = np.ones((2, 1))
    D = np.ones((2, 1))

    # Initialize action vectors
    A[0] = (x[:, 0], x[:, 1])
    A[1] = (x[:, 0], x[:, 1])
    A[2][:, :, :] = (np.array([x[:3]])).T

    B[0] = (x[:, 0], x[:, 1])
    B[1] = (x[:, 0], x[:, 1])
    B[2][:, :, :] = (np.array([x[:4]]).T)

    C[0] = (x, x)
    C[1] = (x,)

    D[0] = (x, x)
    D[1] = (x,)

    return A
```
This implementation uses a state-value matrix (`A`) and reward vector (`B`), as well as the transition matrices (`C` and `D`, respectively). The agent starts at the center of the maze. It then explores one arm, visits another arm, explores an adjacent arm, and so on until it reaches the cue location or ends up in a corner (which is not explored yet).
The GNN representation for this agent can be obtained using the following code:
```python
G = np.array([[x[0] + x[:, 1]], [x[2]]])
```