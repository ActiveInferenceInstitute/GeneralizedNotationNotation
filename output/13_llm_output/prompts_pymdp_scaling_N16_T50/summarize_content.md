# SUMMARIZE_CONTENT

```python
import numpy as np
from scipy import stats

# Load data from JSON file (optional)
data = json.load(open("input/10_ontology_output/README_ontology_report.json"))

# Create a structured summary including key variables, critical parameters, and use cases
summary = {
    "model": {
        "name": "Active Inference on to-rule ontology",
        "type": "active inference",
        "description": "This model is used for generating new active inferences based on the given data.",
        "outputs": [
            {"id": 1, "label": "A"},
            {"id": 2, "label": "B"}
        ],
    },
    "summary_structure": {
        "name": "Summary Summary",
        "description": "This summary is generated from a structured JSON file.",
        "outputs": [
            {"id": 1, "label": "A"},
            {"id": 2, "label": "B"}
        ],
    },
    "errors": []
}


# Load data from JSON file (optional)
data = json.load(open("input/10_ontology_output/README_ontology_report.json"))
```