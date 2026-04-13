# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN implementation:

1. **Model Matrices**:
   - A matrices representing the model structure and its relationships with other models (e.g., Multi-Agent Cooperative Active Inference).
   - B matrices representing the action probabilities and their roles in the model structure.
   - C matrices representing the policy distributions and their roles in the model structure.

2. **Precision Parameters**:
   - γ: precision parameters, which are used to evaluate the performance of each agent independently (e.g., for each agent).
   - α: learning rates and adaptation parameters, which can be adjusted based on user input or other factors (e.g., network architecture) to improve model accuracy.

3. **Dimensional Parameters**:
   - State space dimensions for each modality (e.g., Multi-Agent Cooperative Active Inference):
   - Number of states per agent: 4
   - Number of actions per agent: 3
   - Number of timesteps per agent: 20
   - Number of action probabilities over initial states and actions: 16

4. **Temporal Parameters**:
   - Time horizons (t)
   - Temporal dependencies and windows for each modality (e.g., time horizon):
   - Update frequencies and timescales for each modality (e.g., temporal frequency, update frequency).

5. **Initial Conditions**:
   - Initial parameters:
   - Initial state space dimensions: 4
   - Initial action probabilities over initial states: 16
   - Initialization strategies:
   - Simple initialization with a fixed parameter value and time horizon for each modality (e.g., simple initialization, fixed parameters).

6. **Configuration Summary**:
   - Parameter file format recommendations:
   - For each agent independently, provide the following information:
     - Model structure: A matrix representing the model structure and its relationships with other models (e.g., Multi-Agent Cooperative Active Inference).
     - Action probabilities over initial states: A matrix representing the action probabilities for each agent based on their actions.
     - Policy distributions over initial states: A matrix representing the policy distributions for each modality, including the policy distribution of each agent's state and actions.
     - Observation space dimensions: A matrix representing the observation space dimensions for each modality (e.g., Multi-Agent Cooperative Active Inference).

These parameters provide a systematic way to evaluate performance of each agent independently based on their action probabilities over initial states, as well as