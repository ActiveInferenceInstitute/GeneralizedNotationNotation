# SUMMARIZE_CONTENT

Here is a concise summary of the GNN specification:

**GNN Example:**
```python
import numpy as np
from bnlearn import gnn_syntax

# Create a Bayesian Network (BN) from a neural network (BN), with hidden state and action variables.
A = np.array([[('s', 'S'), ('a', 'S')], dtype=[(0.9, 0.1)], shape=[2])
bnlearn_network=gnn_syntax([
    {'hidden': A}, 
    {'action': B}], name='bnlearnCausalModel'
  )
```
**Key Variables:**
- **A**: Hidden state (represented as a list of lists) with each element representing an observation.
- **B**: Action variable, represented as a list of lists containing the action assignments for each observation.
- **C**: Observation mapping matrix, which maps actions to states and transitions between states.
**Notable Features:**
- **A**: Hidden state (represented as a list with elements representing an observation)
- **B**: Action variable (represented as a list of lists containing the action assignments for each observation)
- **C**: Observation mapping matrix, which maps actions to states and transitions between states.