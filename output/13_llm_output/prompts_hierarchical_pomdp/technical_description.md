# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

```python
import numpy as np
from typing import List, Dict, Tuple

def gnn(
    level: List[Tuple[List[int], int]],
    num_hidden_states: int = 4,
    num_actions: int = 3,
    timesteps: int = 20,
    timescale_ratio=5,
    signature="GNN",
) -> Dict[str, List[float]]:
    """
    GNN Representation of a hierarchical POMDP.

    Args:
        level (List[Tuple[int], int]): A list of Level objects representing the hierarchy.
        num_hidden_states (int): The number of hidden states in the hierarchy.
        num_actions (int): The number of actions per level.
        timesteps (int): The time step size for each level.
        signature: A string indicating a GNN signature, which is used to encode the POMDP representation.

    Returns:
        Dict[str, List[float]]: A dictionary representing the hierarchical POMDP.
    """
    # Create a list of Level objects representing the hierarchy
    level_objects = [level] + level_objects for level in level
    
    # Create a list of timesteps to compute the GNN updates
    timestep_list = []

    # Initialize an empty list to store the aggregated values
    aggregated_values = []

    # Iterate over all Level objects and update their probabilities
    for level, level_obj in enumerate(level):
        aggregated_value = level.get("prob", {}).sum()

        if level_obj["type"] == "float":
            aggregated_value += level_obj["probability"].mean().astype(np.float32) * timestep_list[0]
        
        elif level_obj["type"] == "int":
            aggregated_value = level_obj["prob"][timestep_list[1]]
            
            # Update the aggregated value by applying a weighted average of the probabilities
            aggregated_values += aggregated_value

        else:
            raise ValueError("Unknown type for Level object")

    return aggregated_values
```