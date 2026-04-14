# EXTRACT_PARAMETERS

Based on the information provided, here are the key parameters for the GNN example:

1. **Model Matrices**:
   - A matrices representing the hidden states and their corresponding actions (represented as matrices)
   - B matrices representing the transition matrices and their corresponding action probabilities (represented as matrices)
   - D matrices representing the initial state distribution and its probability of being updated over time
2. **Precision Parameters**:
   - γ: precision parameters for each factor
   - α: learning rates and adaptation parameters
   - Other precision/confidence parameters

3. **Dimensional Parameters**:
   - State space dimensions for each modality (represented as matrices)
   - Observation space dimensions for each modality (represented as matrices)
   - Action space dimensions for each control factor (represented as matrices)
4. **Temporal Parameters**:
   - Time horizons (represented as timestamps in the parameter file format)
   - Temporal dependencies and windows

5. **Initial Conditions**:
   - Prior beliefs over initial states
   - Initial parameter values
   - Initialization strategies

6. **Configuration Summary**:
   - Parameter file formats recommendations for each parameter:
    - `cryptographic-signature` (represented as a JSON file): Use as baseline comparison for POMDP Active Inference variants.
    - `parametricity-summary`: Use as baseline comparison for POMDP Active Inference variants.

Here's the complete list of parameters and their corresponding values in the parameter file format:
```json
{
  "model": {
    "type": "GNN",
    "num_hidden_states": 4,
    "num_observations": 6,
    "num_timesteps": 50,
    "precision_parameters": [
      {"name": "gamma"},
      {"name": "alpha"}],
    },
    "initial_state_distribution": {
      "shape": [[1]],
      "type": "ndarray",
      "dtype": "float64"
    }
  },
  "observation_space": [
    {
      "shape": [[1]],
      "type": "ndarray",
      "dtype": "float32"
    },
    {
      "shape": [[1]]},
    ],
    "action_space": [
      {
        "shape": [[1]],
        "type": "ndarray",
        "dtype": "float64"
      }
    ]
  },
  "forward_algorithm": {
    "num_hidden_states": 4,
    "