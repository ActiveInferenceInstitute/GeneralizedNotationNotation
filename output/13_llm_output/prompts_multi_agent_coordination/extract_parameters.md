# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN implementation:

1. **Model Matrices**:
   - A matrices representing the model structure and interpretation of each agent's actions and beliefs.
   - B matrices representing the action-belief matrix representation of each agent.
   - C matrices representing the action-observation matrix representation of each agent, with the goal state as a hidden state.
   - D matrices representing the action-observation matrix representation of each agent, with the goal state as an observable.

2. **Precision Parameters**:
   - γ (gamma): precision parameters and their roles.
   - α (alpha): learning rates and adaptation parameters.
   - Other precision/confidence parameters for each dimension.

3. **Dimensional Parameters**:
   - State space dimensions for each agent:
    - A matrices representing the state-action matrix representation of each agent, with the goal state as a hidden state.
    - B matrices representing the action-observation matrix representation of each agent, with the goal state as an observable.
    - C matrices representing the action-observation matrix representation of each agent, with the goal state as an observable.

4. **Temporal Parameters**:
   - Time horizons (T) for each dimension:
    - Initial time horizon: 20 timesteps
    - Temporal dependencies and windows:
      - Time Horizon: 16 timesteps
      - Temporal dependencies and windows:
        - Time Horizon: 3 timesteps
        - Temporal dependencies and windows:
            - Time Horizon: 5 timesteps
            - Temporal dependencies and windows:
                - Time Horizon: 7 timesteps
                - Temporal dependencies and windows:
                    - Time Horizon: 9 timesteps
                    - Temporal dependencies and windows:
                        - Time Horizon: 10 timesteps
                      - Temporal dependencies and windows:
                            - Time Horizon: 12 timesteps
                          - Temporal dependencies and windows:
                                - Time Horizon: 14 timesteps

These parameters are used to compute the following metrics for each agent, including:

1. **Initial Conditions**:
   - Initial time horizon (T): 20 timesteps
   - Initial observation window (OUW): 3 timesteps
   
   - Temporal dependencies and windows (TPD)
   - Temporal dependencies and windows (DTD):
      - Time Horizon: 5 timesteps
      - Temporal dependencies and windows:
            - Time Horizon: 7 timesteps
            - Temporal dependencies and windows: