# EXTRACT_PARAMETERS

Here is a systematic parameter breakdown of the GNN model:

1. **Model Matrices**:
   - A matrices: dimensions, structure, interpretation
   - B matrices: dimensions, structure, interpretation
   - C matrices: dimensions, structure, interpretation
   - D matrices: dimensions, structure, interpretation

   These matrices represent the state space and observation spaces for each factor in the model.

2. **Precision Parameters**:
   - γ (gamma): precision parameters and their roles
   - α (alpha): learning rates and adaptation parameters
   - Other precision/confidence parameters

3. **Dimensional Parameters**:
   - State space dimensions for each modality:
      - A matrices: dimensions, structure, interpretation

      These represent the state spaces of each modality in the model. For example, for a particular action, we can have two states (A) and one observation (B). Each dimension represents an element of the state space.

4. **Temporal Parameters**:
   - Time horizons (T):
      - Initialization strategies:

      - Random initialization with random initializations

      - Random initialization with fixed parameters

      - Random initialization with fixed parameter values

      - Random initialization with fixed parameter values and time-varying parameters

      - Random initialization with fixed parameter values, temporal dependencies, and window sizes

5. **Initial Conditions**:
   - Initializing the model:
      - Initialization strategies:

        - Random initialization with random initializations
        - Random initialization with fixed parameters
        - Random initialization with fixed parameter values
        - Random initialization with fixed parameter values and time-varying parameters

      - Random initialization with fixed parameter values (fixed parameter value)

6. **Configuration Summary**:
   - Parameter file format recommendations
      - For each modality, we have a list of dictionaries representing the initial conditions for that modality:

        - Initialization strategies
        - Initializations based on the state space dimensions and observation spaces

        - Random initialization with random initializations

      - Initialization based on the state space dimensions and observation spaces

    - Tunable vs. fixed parameters
      - For each modality, we have a list of dictionaries representing the initial conditions for that modality:

        - Initialization strategies
        - Initializations based on the state space dimensions and observation spaces

        - Random initialization with random initializations

      - Initialization based on the state space dimensions and observation spaces

    - Sensitivity analysis priorities
    - For each modality, we have a list of dictionaries representing the initial conditions for that modality:

        - Initialization strategies
        - Initializations based on the state space dimensions and observation spaces

       