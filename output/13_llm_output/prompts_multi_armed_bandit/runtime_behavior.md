# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

You can use the following code to implement a multi-armed bandit:


```python
import numpy as np
from scipy import stats

def infer_states(reward_obs):
    """
    Implies inference based on reward observations.

    Args:
        reward_obs (numpy array): A sequence of rewards, each with the same length and type.

    Returns:
        numpy array: The inferred reward distribution for arm 0 given a reward observation.
    """
    # Initialize state space
    states = np.array([reward_obs])
    
    # Perform inference based on reward observations
    actions = [np.random.randint(low=1, high=num_actions)] + rewards

    # Perform inference using the action-observation mapping (A)
    for i in range(num_states):
        next_state = np.random.choice([reward_obs[i]])
        
        # Perform inference based on reward observations
        if actions[next_state] == 1:
            # If arm is active, choose arm with reward 0 and action 2
            if actions[actions[next_state]] != 2:
                next_action = np.random.choice([actions[next_state]])
            
            # Perform inference based on reward observations
            if actions[next_state] == 1:
                # If arm is active, choose arm with reward 0 and action 3
                if actions[actions[next_state]] != 2:
                    next_action = np.random.choice([actions[next_state]])
            
            # Perform inference based on reward observations
            if actions[next_state] == 1:
                # If arm is active, choose arm with reward 0 and action 4
                if actions[actions[next_state]] != 2:
                    next_action = np.random.choice([actions[next_state]])
            
            # Perform inference based on reward observations
            if actions[next_state] == 1:
                # If arm is active, choose arm with reward 0 and action 5
                if actions[actions[next_state]] != 2:
                    next_action = np.random.choice([actions[next_state]])
            
            # Perform inference based on reward observations
            if actions[next_state] == 1:
                # If arm is active, choose arm with reward 0 and action 6
                if actions[actions[next_state]] != 2:
                    next_action =