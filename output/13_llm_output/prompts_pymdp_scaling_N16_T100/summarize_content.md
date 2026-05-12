# SUMMARIZE_CONTENT

```python
import json
from typing import List, Dict, Tuple

def generate_ontology(
    ontology: List[Dict[str, str]],
    num_hidden_states: int = 16,
    num_obs: int = 16,
    num_actions: int = 16,
    num_timesteps: int = 100,
) -> Dict[str, str]:
    """Generate a structured summary of the model.

    This function generates a summary of the model using the given parameters.

    Parameters
    ----------
    ontology : List[Dict[str, str]]
        A list containing the models and their corresponding hyperparameters.
    num_hidden_states: int
        The number of hidden states in the model.
    num_obs: int
        The number of observations in the model.
    num_actions: int
        The number of actions in the model.
    num_timesteps: int
        The number of timesteps in the model.

    Returns
    -------
    Summary : Dict[str, str]
        A dictionary containing the summary of the model and its parameters.
    """
    summary = {}
    for node in ontology:
        if isinstance(node, dict):
            summary[node["name"]] = node['value']

        elif isinstance(node, list) or isinstance(node, tuple):
            summary[node] = [
                {
                    "key": key_to_idx(node),
                    "label": label
                } for idx in range(num_hidden_states)
            ]

    return summary
```