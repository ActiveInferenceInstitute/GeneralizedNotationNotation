# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

```python
import json
from typing import List, Dict, Tuple

def gnn(
    inputs: List[Dict[str, str]],
    outputs: List[Tuple[str, int]] = [],
    num_hidden_states: int = 4,
    num_actions: int = 3,
    timesteps: int = 20,
    timescale_ratio: int = 5
) -> Dict[str, int]:
  """GNN implementation on to-go.

  This implementation uses a hierarchical active inference model (HAM). It is based on the
  GNN algorithm from https://github.com/davidbjorn/gnn and is implemented in
  Python using the `pytorch` library. The HAM consists of two layers:
  - A higher-level layer with 4 hidden states, each corresponding to a different action
  - A lower-level layer with 2 contextual states that modulate the actions
  - A sequence of "actions" are computed by combining the input and output from the previous
  layers.

  The HAM is implemented in Python using PyTorch's `pytorch` library. It uses the
  GNN algorithm from https://github.com/davidbjorn/gnn to compute the
  "action-level inference".

  The model outputs are stored as JSON files with timesteps and actions separated by
  commas. Each action is represented in a dictionary, where the key is an integer
  representing the action (e.g., "A1"), and the value is a list of dictionaries containing
  the input data from the previous layer.

  The output file contains all the actions that are computed for each action. For example,
  if there are 3 actions A1-B2 with timesteps 0 to 5, then the output file will contain
  "A1: B2".

Parameters:
    inputs (List[Dict[str, str]]) - List of input data from the previous layer.
    outputs (List[Tuple[str, int]] = []): A list of actions computed by combining the input and output from the previous layers.
    num_hidden_states (int): Number of hidden states in the HAM.
    num_actions (int): Number of actions in the HAM.
    timesteps (int): Number of timesteps to compute action-level inference.
    timescale_ratio (int): Ratio between the number of time steps and the number of timest