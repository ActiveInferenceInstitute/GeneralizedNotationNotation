# EXPLAIN_MODEL

Here is a detailed explanation of the GNN example:

**GNN Example:**

```python
import numpy as np
from pyMDPScaling import PyMDPScalingScores

# Define the model parameters and annotations
num_hidden_states = 2
num_obs = 2
num_actions = 2
num_timesteps = 10

model = PyMDPScalingScores(
    num_hidden_states=num_hidden_states,
    num_observations=num_observations,
    num_actions=num_actions,
    num_timesteps=num_timesteps,
    annotations=[
        {"type": "log", "value": 0.9},
        {"type": "prob", "value": 1.2}
    ],
)
```

**Model Purpose:**

1. **Initialization**: The model starts with a set of hidden states (s_f0, s_f1), observations (o_m0, o_m1), and actions (u_c0, π_c0). These are represented as a dictionary `s` for hidden state, `o`, `m`, and `c`.

2. **Initialization**: The model starts with the initial parameters (`num_hidden_states`, `num_obs`) and sets up the data structure to represent the input data (actions) and predictions of actions based on the observed states.
3. **Model Dynamics**: The model implements Active Inference principles by updating the hidden state based on the observed observations. It updates the hidden state using a Markov Chain, which is represented as a dictionary `s` for hidden state, `o`, `m`, and `c`. This process continues until all states have been updated or there are no more actions available to update the model parameters.
4. **Active Inference Context**: The model uses Active Inference principles by updating its beliefs based on the observed actions. It updates its belief for each action using a Markov Chain, which is represented as a dictionary `s` for hidden state, `o`, and `m`. This process continues until all states have been updated or there are no more actions available to update the model parameters.
5. **Practical Implications**: The GNN example provides accurate predictions of future actions based on past observations. It also predicts that there will be a transition from one state to another, which is modeled as an