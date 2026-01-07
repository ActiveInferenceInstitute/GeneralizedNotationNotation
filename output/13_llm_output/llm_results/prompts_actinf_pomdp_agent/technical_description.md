# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

A:  
```python
import numpy as np
from scipy import stats
def gNN(state_observation):
    states = state_observation[:,:-1]
    actions = state_observation[:num_hidden_states], state_observation[num_hidden_states:]
    
    # Actions and their probabilities
    probability_actions=stats.pois("Probabilities over actions:")
    
 
    hidden_probabilities=[
        [
            (state_observation) 
            .shape([n,1]) 
        ]
      , 
 
       ([
         {
           "probs":
             [ 
                states[0:num_hidden_states],
                state_observation[:num_hidden_states]
                
              ],
              probes={"prop:", probabilities}
     ])
    )
 
    
    # Hidden and observed actions
    hidden_actions=[
        {
            "probabilities":
                   [{
                     "idx":
                         :,
                  ","
               },
                       {"state":
                             [
                                 states[0:n],
                             state_observation[:num]
                 ]
              ],
                probes={"prop:", probabilities}
     ])
 
    
    # Probability of a new observation and its hidden actions
    prev_action = states[:, 1:-2, :] 
    action=" " + ", ".join(states[:-1])

    # Random choice on the previous state (in case you want to choose one more time)
    next_state=[]
    for i in range(num_hidden_states):
        if np.random.randint(low=0).astype('int'):
            next_state.append((i, states[1:2**i]))

        # Set the previous state to be a random index (in case you want to choose one more time)
    hidden_actions[:] = [[", ".join([states[next_index]]) for i in range(num_hidden_states)]],
     """{"prop:", probabilities}"""
 
    
    return [], [action]
```