# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

```python
import numpy as np
from scipy import stats

# Load the data from the database file
data = load_dataset("input/10_ontology_output/simple_mdp_ontology_report.json")

# Load the data from a JSON file
with open('input/10_ontology_output/multi_armed_bandit_ontology_report.json', 'r') as f:
    json_data = json.load(f)

# Load the data from a CSV file
with open('input/10_ontology_output/actinf_pomdp_agent_ontology_report.csv', 'r') as f:
    csv_data = csv.reader(f, delimiter=', ')

    # Read and parse the data into a list of dictionaries
    for row in csv_data:
        # Check if there are any missing values
        if len(row) != 10:
            raise ValueError("Missing value at index {}.".format(len(row)))

        # Initialize an empty dictionary to store the action probabilities
        actions = []
        # Iterate through all actions and add them to a list of actions
        for action in row[2]:
            if len(action) != 10:
                raise ValueError("Missing value at index {}.".format(len(action)))

            actions.append({
                "id": int(action["id"]),
                "type_": str(action["type"]),
                "probability": stats.pdist(actions[action['id']])
            })

        # Add the action to the list of actions
        actions.append({"id": row[0], "type_": str(row[1]), "probability": int(stats.pdist(actions))})

    # Add all actions to a dictionary
    for action in actions:
        if len(action) != 2:
            raise ValueError("Missing value at index {}.".format(len(action)))

        actions["id"] = action['id']
        actions["type_"] = str(action['type'])
        actions.append({
            "id": int(action["id"]),
            "probability": int(stats.pdist(actions))
        })

    # Add the actions to a dictionary
    for action in actions:
        if len(action) != 2:
            raise ValueError("Missing value at index {}.".format(len(action)))

        actions[action['id']] = {"type