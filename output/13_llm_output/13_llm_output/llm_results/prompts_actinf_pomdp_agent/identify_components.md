# IDENTIFY_COMPONENTS

You've already covered the key concepts in the response for understanding the structure of GNN models with active inference, Bayesian inference, and generalized notational notation (GANN). Here's a more detailed breakdown:

1. **State Variables**:
    - Variable names and dimensions: 
    - Hidden states by number of observations and actions
    - Input parameters are denoted as `A`, `B` for action/policy types, e.g., "x" for "action x", etc.

2. **Observation Variables**:
    - Observation modalities include the following
   - 3 variables: `o[n]`:
      - Initial position (0 to n-1) of observation

      - Observation direction and orientation (e.g., up/down, right/left)

    - `p_i` or `u_i`:
      - Current state in index i

      - `S_{i}^{k}` or `B_{i}^{k}**: A transition matrix with states indexed by indices of observation pairs
      - `s_{i}^{j}** = S_{i}'*(A(x,o[j])+u_(y))$
   
   - A and B represent input/output variables for each action
3. **Action Variables**:
    - Action types are denoted as `action`

    - 3 actions: "S_1", "B_1"
      - "T_2": "P(x^*,u^*)" is defined to be P([p],[s])

      A and B denote action variables, where
        A represents the input parameter for a single action
        B denotes the output variable in the same action

   
4. **Model Matrices**:
    - A matrices:
      - A set of matrices representing observable transitions over actions (actions).

      For each observation pair

  - B sets of matrices representing observed observables, including their distribution over actions and actions-permitting/unallowing observations
  - C represents the probability distributions over states
   - D represents a vector or column array containing the action probabilities
5. **Parameters**:
    - Precision parameters (γ)
      - Initialize the `gnn_p` parameter to 0, meaning that you can't learn anything from an unbounded input horizon

  - Learning rates and adaptation:
     - Initialize learning rate `l` for each time step based on observation probabilities

    - Adjusting parameters in future steps

6. **Temporal Structure**:
    - Time horizons (fixed)
   - Dynamic vs. static components
7. **Signature**:
    - The signature provides a clear description of the model's structure:
      - A(x,o[n]) represents actions taken

      Plugated inputs with state-wise variables for each observation pair

  - The "A" matrix is used to represent observable transitions over action pairs,
      while the "B" and "D" matrices are used to describe the corresponding observed observables.