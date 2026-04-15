# EXTRACT_PARAMETERS

Based on the provided specifications, here are the key parameters for the GNN implementation:

1. **Model Matrices**:
   - A matrices: dimensions, structure, interpretation
   - B matrices: dimensions, structure, interpretation
   - C matrices: dimensions, structure, interpretation

   The matrix representation of each parameter is as follows:
   
   - A matrices with 5 rows and 4 columns (representing the number of hidden states)
   - B matrices with 3 rows and 2 columns (representing the number of actions in a navigation context)
   - C matrices with 1 row and 0 columns (representing the number of observations)
   - D matrices with 5 rows and 4 columns (representing the number of hidden states)

2. **Precision Parameters**:
   - α: learning rates and adaptation parameters
   - γ: precision parameters and their roles

   The parameter file format recommendations are as follows:
   
	- Initialization strategies:
	   - Random initialization with a fixed initial value for each parameter (default).
	  - Random initialization with a random initial value for each parameter.
	  - Random initialization with a random initial value for each parameter, but only if the parameter is not already initialized.

3. **Dimensional Parameters**:
   - State space dimensions:
   	- A matrices representing the number of hidden states (represented by rows)
   	- B matrices representing the number of actions in a navigation context (represented by columns)
   	- C matrices representing the number of observations (represented by rows)
   	- D matrices representing the number of hidden states and actions (represented by columns)

4. **Temporal Parameters**:
   - Time horizons:
   	- A matrix representing the number of timesteps for each modality (represented by rows)
   	- B matrices representing the number of observations in a navigation context (represented by columns)
   	- C matrices representing the number of actions in a navigation context (represented by rows)

5. **Initial Conditions**:
   - Initialization strategies:
	   - Random initialization with a fixed initial value for each parameter (default).
	  - Random initialization with a random initial value for each parameter, but only if the parameter is not already initialized.

6. **Configuration Summary**:
   	- Tunable parameters:
   							
    1. **Initialization strategies**:
     	   - Random initialization with a fixed initial value for each parameter (default).
     	  - Random initialization with a random initial value for each parameter