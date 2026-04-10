# EXTRACT_PARAMETERS

Based on the document, here are the key components of the model:

1. **Model Matrices**:
   - A matrices representing the location and reward distributions for each action/observation pair (A)
   - A matrix representing the probability distribution over actions (B), which is used to update the agent's belief in its current state space based on the observed rewards
   - A matrix representing the probability distribution of the reward at a particular time step (C)

2. **Precision Parameters**:
   - γ: precision parameters, which are used for training and evaluation purposes
   - α: learning rate parameter, which is set to 0.1 during training
   - Other precision/confidence parameters

3. **Dimensional Parameters**:
   - State space dimensions for each modality (D)
   - Observation space dimensions for each action/observation pair (A), with the goal of reducing uncertainty and exploring different actions based on their reward distributions
   - Action space dimensions for each control factor (C)
4. **Temporal Parameters**:
   - Time horizons (T)
   - Temporal dependencies and windows (W)
   - Update frequencies and timescales (F)

5. **Initial Conditions**:
   - Initial parameters:
      - γ

      - α

      - Other precision/confidence parameters

   6. **Configuration Summary**:
    - Parameter file format recommendations
    - Tunable vs. fixed parameters
    - Sensitivity analysis priorities