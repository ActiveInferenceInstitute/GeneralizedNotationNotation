# ANALYZE_STRUCTURE

Here is a detailed analysis of the GNN Representation:

**1. Graph Structure:**

1.1. **Number of Variables and Types**:
    - Number of variables: 3 (one observation modality, one hidden state factor)
   - Type: `{(0.9,), (0.25,)` is used to represent the agent's actions as an action vector, while `{()}` represents the policy prior.

1.2. **Variable Analysis**:
    - Number of variables for each variable: 3
       - `num_observations`: 3
       - `num_actions`: 3
       - `numberOfStepCounts:**
           - `nsteps=1`: for every step, there are 4 steps
       - `shape(0):** "([a.b..])" represents the structure of each observation (state) with one observation at a time and two actions in it
   - `connections`, `actions` :
        - `num_states = num_observations + nsteps/2`: for every step, there are 4 states
          - `current_history=[0.,1.]]` represents the history of each action (state)
```python
    # state space dimensionality for each variable:
 
    # Network topology (hierarchical vs hierarchical):
     ```python
   - Network structure:
      - Hierarchical, with "levels" at which actions are taken
       - Number of nodes in layers = number_of_steps/2
         - Each node has 3 paths to reach the next level
         - Each path is connected to a path within that layer (see below)
```
    ```python
   # Policy topology:
      `actions` -> `current_history[0..nsteps]]`
       - Each step, there are N steps left
        - Forward propagation across previous states
          - $p_{step1}$ = current_state(x+a.b), where a and b 
              are actions
          
  ```
   **Type**
    - `type=float`: Likelihood map (each observation is equally probable)
    """
    
    # Variable types:
        type('A', 'LikelihoodMatrix[3]'),
            type('B', 'TransitionVector[1]'),
                type('C', 'ProbabilityVec[<numbers>]')
```
   **Connection patterns**:
     `connections` 
       - `current_history = [0..nsteps]]`. Each step is taken with a path
        - The current state(x) can move to any of the states in the layer
          - We are doing forward propagation across each path
              - `next**path`, which has an (a.b..)` at its end, we then update the next_state
            for each subsequent step:
                - "next_states" now represents a new state
                      $x+a$ -> x+a + b
                  and so on

    **Actions**:
    *   `actions` can have multiple actions (action is taken).
        - If action = 0.5 then there are two possible actions
          - For step 1:
            - "previous_states" now represents the state before it
              $x-a$. Now we move to the previous node for action 'b'.
                We will then have another path forward from this current node in next steps
```python
    # Activation pattern 
    `action = [0.5]*[1..nsteps]`

    **Type**
        - "actions" now represents the sequence of actions
        - These sequences are connected to each other
        - Each action has two possible outcomes (success or failure)
            - Each subsequent sequence can then be propagated across previous steps
    """
```