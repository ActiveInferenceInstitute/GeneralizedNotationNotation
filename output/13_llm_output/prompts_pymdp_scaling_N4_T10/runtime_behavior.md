# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

```python
import numpy as np
from pymdp import *

# Load the data from JSON file
data = json.load(open("input/10_ontology_output/README_ontology_report.json"))['data']

# Load the data into a list of dictionaries
obj_list = [
    {"name": "A", "type": "Feature"},
    {"name": "B", "type": "Feature"},
    {"name": "C", "type": "Feature"}
]

# Load the data from JSON file and create an object for each feature type
for obj in obj_list:
    obj = {
        "name": obj["name"],
        "type": obj["type"]
    }

    # Create a list of dictionaries with the same name but different types
    for key, value in obj.items():
        if isinstance(value, np.ndarray):
            obj[key] = PyMDPScalingDataset(
                data=data,
                **obj
            )
```