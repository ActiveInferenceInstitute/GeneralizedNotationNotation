# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

```python
import numpy as np
from pymDP import *

# Load the data from JSON file
data = json.load(open("input/10_ontology_output/simple_mdp_ontology_report.json"))['data']

# Load the data from CSV file
data = pd.read_csv('input/10_ontology_output/multi_armed_bandit_ontology_report.csv')
```