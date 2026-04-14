# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN implementation:

1. **Model Matrices**:
   - A matrices representing the state space dimensions and dimensionality of each agent's environment (e.g., `A`)
   - B matrices representing the action spaces and dimensionality of each agent's actionspace (`B`)
   - C matrices representing the observation space dimensions for each agent (`C`)

2. **Precision Parameters**:
   - γ: precision parameters, which are used to determine how well each agent cooperates with others (e.g., `γ=0.9`)
   - α: learning rates and adaptation parameters, which control the rate at which agents learn from their actions
3. **Dimensional Parameters**:
   - State space dimensions for each agent (`A`):
    - `num_agents`: number of agents
    - `grid_size`: size of grid in time (9)
    - `num_obs`: number of observations per agent (`4`)
    - `num_actions`: number of actions per agent (`1`)
   - `state_space_dimensions` for each agent:
      - `A`: dimensions of state space, which are used to compute the action probabilities and reward distributions (e.g., `A=0.9`, `B=0.3`)
    - `B`: dimensionality of observation space, which is used to compute the policy distribution and reward distributions (e.g., `B=1`)
   - `state_space_dimensions` for each agent:
      - `C`: dimensions of state space, which are used to compute the action probabilities and reward distributions (e.g., `C=0`)
    - `observation_dimensionality` for each agent:
      - `A`: dimensionality of observation space, which is used to compute the policy distribution and reward distributions (e.g., `A=1`, `B=2`)
   - `action_space_dimensions` for each agent:
      - `C`: dimensions of action space, which are used to compute the policy distribution and reward distributions (e.g., `C=0`)
    - `observation_dimensionality` for each agent:
      - `A`: dimensionality of observation space, which is used to compute the policy distribution and reward distributions (e.g., `A=1`, `B=2`)
   - `action_space_dimensions` for each agent:
      - `C`: dimensions