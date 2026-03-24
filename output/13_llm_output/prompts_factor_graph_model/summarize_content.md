# SUMMARIZE_CONTENT

Here is a concise summary of the GNN specification:

**GNN Example:** Factor Graph Active Inference Model (Version 1)

```python
import numpy as np
from scipy import stats

def factor_graph(x):
    """
    A factor graph decomposition for tractable inference in structured models.

    Parameters
    ----------
    x : array-like of shape (n, n)
        Input data to be decomposed into observation and hidden state variables.

    Returns
    -------
    array: array of size (n,) with the same shape as input data.
    """
    # Define initial parameters
    A = np.array([x])  # Observation matrix
    B = np.array([[(0, -1), (-1, 0)])  # Variable matrix
    C = np.array([[(2, 0), (4, 0)], [[(-3, 5), (6, 7)]])

    # Define the hidden state matrices
    A_vis = np.array([x] * n)
    A_prop = np.array([x] * n)
    B_vis = np.array([[(2, -1), (-0.8, 0)], [[(-3, 5), (6, 7)]])

    # Define the action matrices
    C_vis=np.array([])
    C_prop=np.array([x] * n)
    D_vis = np.array([[(2, -1), (-4)])
    D_prop = np.array([x] * n)
    G_vis = np.array([x] * n)

    # Define the transition matrices
    D_vel=np.array([])
    D_pos=np.array([x] * n)
    F_vis = np.array([[(1, 0), (2, -3)], [[(-4, 5), (6, 7)]])

    # Define the belief updates
    π_vis = np.array([x] * n)
    π_prop=np.array([])
    G_vis = np.array([x] * n)
    B_pos = np.array([[(2, -1), (-4)], [[(-3, 5), (6, 7)]])

    # Define the action matrices
    C_vel = np.array([x] * n)
    C_prop=np.array([])
    D_