# EXTRACT_PARAMETERS

You've already provided a comprehensive list of parameters for the Active Inference model, including:
1. **Model Matrices**:
	* A matrices representing the input and output data sets (A) and (D).
2. **Initial Parameters**
    * Initial belief values over hidden states (`B`)
    * Initialization strategy to avoid vanishing gradient issues

3. **Prediction Parameters**
    * Initial prediction parameters for each modality
    * Initial parameter value range based on the training dataset

4. **Model Parameters**:
    * Number of hidden states and prior beliefs (num_hidden_states)
    * Number of observation and action spaces (`num_obs`)
    * Number of actions and actions learned from the data set
5. **Initialization Strategies**
    * Random initialization for each modality, with a fixed number of initial parameters

6. **Parameter File Format Recommendations**:
    * Use JSON format to store parameter file information (e.g., `input/10_ontology_output/simple_mdp_ontology_report.json`)
	* Use `model_file` for storing model-specific metadata and configuration details
	* Use `parameter_file` for storing parameter file information, including initial parameters, initialization strategies, etc.