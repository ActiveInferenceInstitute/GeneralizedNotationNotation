# Nontechnical Description

Prompt:

> Describe this GNN model comprehensively, in non-technical language suitable for a broad audience.

Response:

```python
import json
from typing import List

def gnn(input: List[List[int]]) -> List[Dict[str, float]]:
    """GNN implementation for hierarchical active inference (HAI) on a multi-armed bandit problem.

    Parameters
    ----------
    input : list of lists
        A list of 2D arrays representing the data points in the HAI problem. Each array represents an action and its probability distribution over the reward space, with each element being a scalar value between 0 (no action) to 1 (action).

    Returns
    -------
    list[Dict[str, float]]
        A list of dictionaries containing the following keys:
            "actions": [
                {
                    "type": str,
                    "probabilities": [
                        {
                            "value": int(input[i]),
                            "probability": int(input[j])
                        }
                    ]
                },
                ...
            ]
    """

    # TODO: Implement GNN implementation for HAI on multi-armed bandit problem.
```