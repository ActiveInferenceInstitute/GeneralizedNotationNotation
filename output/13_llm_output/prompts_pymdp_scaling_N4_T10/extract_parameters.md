# EXTRACT_PARAMETERS

Based on the provided specifications, here are the key parameters that will be used to generate a GNN model:

1. **Model Matrices**:
   - A matrices: dimensions, structure, interpretation
   - B matrices: dimensions, structure, interpretation
   - C matrices: dimensions, structure, interpretation

   These matrices represent the input data and can be composed of multiple layers or different types of models (e.g., neural networks).

2. **Initial Parameters**:
   - γ (gamma): precision parameters and their roles
   - α (alpha): learning rates and adaptation parameters
   - Other precision/confidence parameters

The parameter file format recommendations are:

  1. **Model Matrices**:
    - A matrices: dimensions, structure, interpretation

    This matrix represents the input data for each layer of the model. It can be composed of multiple layers or different types of models (e.g., neural networks).

2. **Initial Parameters**:
   - γ (gamma): precision parameters and their roles

   This parameter is used to initialize the initial learning rate, which will guide the training process. The default value for γ is 0.1.

3. **Dimensional Parameters**:
    - State space dimensions for each modality:
      - Initial state space dimensions for each modality

      This represents the input data and can be composed of multiple layers or different types of models (e.g., neural networks).

4. **Observation Space Dimensions for Each Control Factor**:
   - Initialization strategies:
      - Random initialization strategy: Use a random initialization method to initialize the observation space dimensions, which is suitable when there are many control factors and no prior knowledge about them.

      This approach can be useful if you have a large number of control factors but don't know their specific properties or relationships with each other.

5. **Temporal Parameters**:
    - Time horizons (T)

    This parameter represents the time horizon over which the model is trained, which will guide the training process. The default value for T is 10 seconds.

6. **Initial Conditions**:
   - Initialization strategies:
      - Random initialization strategy: Use a random initialization method to initialize the initial state space dimensions and observation space dimensions at the beginning of each iteration.

      This approach can be useful if you have a large number of control factors but don't know their specific properties or relationships with each other.

7. **Configuration Summary**:
    - Parameter file format recommendations:
      - The parameter file format recommendation is based on the provided specification and provides guidance