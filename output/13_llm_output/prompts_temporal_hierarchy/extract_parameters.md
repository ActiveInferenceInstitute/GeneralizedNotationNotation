# EXTRACT_PARAMETERS

Based on the document, here are the key information and parameters for the GNN implementation:

1. **Model Matrices**:
   - A matrices: dimensions, structure, interpretation
   - B matrices: dimensions, structure, interpretation
   - C matrices: dimensions, structure, interpretation

   The model matrix represents the set of all possible actions that can be taken given a sequence of actions and their corresponding states. It is used to represent the inference process in the GNN framework.

2. **Precision Parameters**:
   - γ (gamma): precision parameters and their roles
   - α (alpha): learning rates and adaptation parameters
   - Other precision/confidence parameters

   These are related to the accuracy of predictions made by the agent based on its actions, which is represented as a matrix in the model matrices.

3. **Dimensional Parameters**:
   - State space dimensions for each modality:
    - A matrices representing the set of all possible states and their corresponding actions (represented as vectors)
    - B matrices representing the set of all possible actions that can be taken given a sequence of actions and their corresponding states (represented as vectors)

4. **Temporal Parameters**:
   - Time horizons for each modality:
    - T matrices representing the time horizon over which predictions are made based on the model matrix
    - Temporal dependencies and windows for each control factor

    The temporal parameters represent how well the agent can predict its own actions given a sequence of actions, with different values indicating that the agent is more or less likely to make a specific action.

5. **Initial Conditions**:
   - Initial conditions:
    - Prior beliefs over initial states (represented as vectors)
    - Initial parameter values for each modality

    These parameters represent how well the agent can predict its own actions given a sequence of actions and their corresponding states, with different values indicating that the agent is more or less likely to make specific actions.

6. **Configuration Summary**:
   - Parameter file format recommendations:
    - Cryptographic signature goes here
