# EXTRACT_PARAMETERS

Based on the specifications provided, here is a summary of the Active Inference POMDP agent:

1. **Model Matrices**:
   - A matrices structure with dimensions `(num_hidden_states,)` and length 3 (representing each hidden state). This represents the network for each type of belief estimation method applied to each observation, while allowing control over the behavior of subsequent actions based on prior knowledge or inference rules. The size of the matrix is not specified in this document but can be inferred from its shape if you provide a list of matrices.

2. **Precision Parameters**:
   - Parameters for each modality are `(num_actions,)` (representing action selection) and length 3 (representing actions). This represents information available to make decisions based on actions, while controlling the behavior of subsequent actions based on prior knowledge or inference rules. Each parameter corresponds to a decision made by an individual agent using a specific probability distribution over a particular action chosen during prediction time step.

3. **Dimensional Parameters**:
   - Parameters for each modality are `(num_states,)` and length 1 (representing the number of states used in inference). This represents information available at each observable level, including their possible values based on prior beliefs or inference rules. Each parameter represents a probability distribution over actions assigned to each state. Each choice made during prediction time step corresponds to an action chosen by agent.

4. **Temporal Parameters**:
   - Initial Conditions are `(time=10)` (representing the number of simulations running before final initialization process begins). This specifies the start point for predictions, and will be used as a baseline when specifying new parameters or updating prediction sequences based on input data points during subsequent iterations.

5. **Initialization Strategies**:
   - Initializing the parameter files with a list of matrices is specified as `(num_hidden_states,)` and length 3 (representing each hidden state). This allows initializations to be done using different distributions for prediction time step, but will not affect subsequent predictions based on prior beliefs. 
   - Initialization strategies are specified in the document as a list of arrays with dimensions of `(num_actions,),` which is nested lists representing matrices containing probabilities over actions selected during prediction iteration timesteps, and values corresponding to action choices made during observation time step (specified by the parameter number 12). This allows initialization process to be performed using different distributions for predictions.