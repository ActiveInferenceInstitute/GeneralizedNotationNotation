# IDENTIFY_COMPONENTS

Here is a systematic breakdown of the elements in the GNN specification:

1. **Action Variables** (represented as `C` matrices): 
   - What each action represents conceptually
    - Action type, actions/actions

    - For example:

      - Action 1 to state 2
      - Action 2 from state A
 
- **Observation Variables**:
   - Observation modalities and their meanings
   - Sensor/measurement interpretations

   - Example:

      - Observation 1 is the prior belief.
      - Observation 2 may be in view of other states or actions

2. **State Variable** (represented as `A` matrices): 
   - What each state represents conceptually
    - History/history-relevant variables

    - Example:

      - State A to previous states
    
    - Example:

      - Observation A is the prior probability for the last observation 
      
 
3. **Action Variables**:
   - Available actions and their effects
    - Example:

    - Action 1 (in action state) to state 2
    
      - Action(action_history)(observer,state) -> observation

     - Example:

      - Actions(observation_states)[observation] is the probability of the first observation
      
 
4. **Model Variables** (represented as `A` matrices): 
   - What each model variable represents
    - Variational-free nature

    - Example:

    - Model variables 2 (policy, history) to a policy vector
    
      - Value function (initial),

      - Policy vector for an action
  
 
5. **Parameter** (represented as `V` matrices): 
   - What each parameter is 
    - Decision models
    - Initialization of initial belief
    - Learning rate/adaptation parameters

   - Example:

    - Model variables 1 to states, histories

6. **Hyperparameter** (represented as `A=B=C=D=E`):
    
    - P(state_observations)
    - P(history|action) 
    - P(policy | initial beliefs, history-relevant) 
    - P(initial belief for policy | history/beliefs)
```