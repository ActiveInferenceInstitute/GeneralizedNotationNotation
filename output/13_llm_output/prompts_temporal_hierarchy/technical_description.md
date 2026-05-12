# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

```python
import json
from typing import List

def gNN(input: List[str], output: List[List[int]]) -> List[Dict[str, int]]:
    """GNN implementation of the GNN algorithm.

    Args:
        input (list): A list of strings representing actions and goals for a single agent.
        output (list): A list of lists containing the probabilities of each action based on its level.

    Returns:
        list(dict) : A dictionary with two keys, 'action' and 'probabilities', where
            'actions' are dictionaries that contain the actions to be performed in the current layer
            and 'probs' is a dictionary containing the probability of performing an action for each state.
    """
    # Initialize the output lists with empty dictionaries
    outputs = []

    # Loop through all levels of the hierarchy
    for level_idx, (actions, probabilities) in enumerate(output):
        # Perform actions on the current layer and update the corresponding probabilities
        if isinstance(level_idx + 1, int):
            actions.append((action=input[level_idx], probability=0))

        # Update the probabilities of each action based on its level
        for state_index in range(len(actions)):
            next_state = actions[(level_idx + 1) % len(outputs)]

            if isinstance(next_state, int):
                next_action = actions[next_state]

                # Perform a single action and update the probabilities of all states
                if actions.index(next_action) == level_idx:
                    outputs.append((actions[(level_idx + 1) % len(outputs)]))

    return outputs
```