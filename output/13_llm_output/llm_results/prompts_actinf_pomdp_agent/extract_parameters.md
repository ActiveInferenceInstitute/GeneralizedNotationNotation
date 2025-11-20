# EXTRACT_PARAMETERS

Based on the information provided in the doc, here are the specifications and parameters for the Active Inference POMDP Agent:
1. **Model Matrices**:
   - A matrices representing the model structure and the parameter values are shown. The matrix is initialized to initial states. There are two types of matrices: Likelihood Matrix (low dimensional) and Probability Matrix (high dimensional). The list provides information about how these two matrices represent the state-level models, as well as the type of parameters that can be associated with each model variable.
   - A description of the likelihood matrix is also provided in the doc which outlines its structure and interpretation for each parameter value.

2. **Precision Parameters**:
   - α (alpha) represents a learning rate parameter and determines how much to perturb actions in order to increase accuracy. There are different values for α; default values are 0.1, 0.05, 0.03, etc., each with varying effects on the model's performance.
   - γ is set at 0.

3. **Dimensional Parameters**:
   - C represents the choice of classification or prediction types (decision trees vs. random forests), while D points to which parameter values are associated with a particular type of dimensionality, respectively.

4. **Initial Conditions**:
   - Initial conditions provide information about how the model changes after each action selection. There is a description for each initial condition and its parameters.
   - This provides an overview of what happens before each actions choice.
 
To calculate these specifications:

1. `alphabeticalOrder`: A numerical ordering (e.g., list) to represent a parameter value's position in the alphabet, where subsequent values are placed after the last one found.

2. `state_shape` represents a shape of state space dimensions for each parameter variable and its corresponding parameter value's dimensionality: 3 if all variables have the same type as probability matrix (low dimensional), 1 if all variables have the same number but different types, etc..
For example, the list representing the parameters states A = Lambda[state_shape] B=[states_shape for state in range(num_hidden_states)], B['observation']B[:,0]=True and so on.

3. `probability_matrix` represents a matrix of probabilities across all dimensionality (number of states plus one).

4. `action` represent the type of action selected from the list of actions, while `policy`. 

5. **initialization**: Each initial condition has an associated parameter value based on their position in the alphabet and its size. The parameters are represented as lists representing the variables in the alphabet.
The choice of classification is based on the last occurrence of each type and its corresponding dimensionality (number of states plus one).

6. `parameter_file`: A file that provides information about how parameter values change after each action selection, with their position in the alphabet (a list for the number of parameters) representing each value's index within the dictionary.
For example:
```python
    'actions': ['randomize', 'remove'] 
    'policy' : ['./options/action_policy.', './options/action_selection0','.'./options/action_selection1'],
    'state_observation'  = ['new state', '_update_beliefs_slowly' .replace('mean([{'}][[']{}) is a dictionary that stores the parameter values at each position, along with their corresponding order in alphabetical order (from last to first). The value for next observation will be accessed directly by index number of current observation. For example', 'next_observation'+ []+'are'][0][1:] denotes the next observation.']))
```