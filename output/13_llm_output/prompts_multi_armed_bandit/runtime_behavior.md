# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

You can use the following code to implement a multi-arm bandit:
```python
import numpy as np

# Define the action space and reward distribution for each arm
actions = np.array([
    (0, 1), 
    (1, 2)
])
reward_obs = np.array([
    0.957643818,
    0.057643818
])

# Initialize the action space and reward distribution for each arm
actions[action] = actions[:,:num_actions].T  # Apply action to action pairs
reward_obs[action] = rewards[:num_actions][:,:]   # Apply reward to action pair
```
This code defines a list of actions, then applies the action to each action pair in the action space. The reward distribution is initialized as a numpy array with shape (3,) and initializes it with the first two elements of the action space: `(0.957643818, 0.057643818)`.
The code then defines the reward distributions for each arm using NumPy arrays (`reward_obs`). The actions are initialized as a numpy array with shape (num_actions,), and their corresponding rewards are initialized with `(0.957643818, 0.057643818)`.
Finally, the code initializes the signature of the multi-armed bandit by using the action space as a numpy array (`actions`). The actions and reward distributions are initialized with `(0.957643818, 0.057643818)`.
The signature is then used to encode the action sequences into a dictionary of dictionaries (each representing an arm), where each dictionary represents one arm's action sequence and its reward distribution. The code uses `collections` module to create a dictionary from the actions, which allows for easy serialization of actions as dictionaries with keys "action", "reward".