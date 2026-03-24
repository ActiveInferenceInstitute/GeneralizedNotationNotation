# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN model:

1. **Model Parameters**:
   - `A`: Likelihood matrix
   - `B` (Transition Matrix): Transition matrix with 5 hidden states and 4 actions in a navigation context
   - `C`: Log Preference Vector
   - `D`: Prior Over Hidden States
   - `γ`: Precision parameter
   - `F`: Variational Free Energy

2. **Initial Parameters**:
   - `A` (initial parameters): Likelihood matrix with initial state and action values
   - `B` (initial parameters): Transition Matrix with 5 hidden states and 4 actions in a navigation context
   - `C` (initial parameters): Log Preference Vector
   - `D`: Prior Over Hidden States

3. **Initial Parameters**:
   - `A`: Likelihood matrix
   - `B`: Transition Matrix
   - `C`: Log Preference Vector
   - `D`: Prior Over Hidden States

4. **Initial Parameter File Format Recommendations**:
   - `GNNVersionAndFlags` (optional): list of GNN version and flags for the model
   - `ModelName` (optional): name of the model to be used in parameter file format