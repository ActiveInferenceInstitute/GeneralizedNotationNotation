# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

```python
import json
from typing import List, Dict

def gnn(model: str) -> Dict[str, int]:
    """GNN implementation for multi-agent cooperative active inference.

    Args:
        model (str): The name of the GNN model to use.

    Returns:
        Dict[str, int]: A dictionary containing the predicted actions and their corresponding probabilities.
    """
    # Load the data from JSON files
    with open("data/input_10_ontology_output/simple_mdp_ontology_report.json", "r") as f:
        json_data = json.load(f)

    # Load the data into a list of dictionaries
    predicted_actions, actions_probabilities = [], []

    for action in json_data["predictedActions"]:
        if isinstance(action, str):
            predicted_actions.append(json_data[action])

        elif isinstance(action, int):
            actions_probabilities.append(eval(json_data[action]))
    return {"predictions": predicted_actions}, actions_probabilities
```