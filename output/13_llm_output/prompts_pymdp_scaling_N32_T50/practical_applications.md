# PRACTICAL_APPLICATIONS

Based on the provided code, here are some key points about the proposed implementation:

1. **Input Data**: The input data is a JSON file containing a list of `ontology` objects with `label`, `action`, and `timestep`. Each object has an optional `time_varying` attribute that can be used to track changes in time or state over different timesteps.

2. **Model Parameters**: There are 32 hidden states, each represented by a dictionary (`hiddenState`) containing the corresponding label values for each action and timesteps. The `num_actions`, `num_states`, and `num_timesteps` parameters specify how to represent each object in the model.

3. **Model Parameters**: There are 10 input files, each representing a single ontology object with labels (`label`) and actions (`action`) for different timesteps (represented by integers). Each input file contains a list of `ontology` objects along with their corresponding label values and action values.

4. **Output Files**: There is one JSON file containing the output data from the model, which includes the following fields:
   - `output/10_ontology_output/simple_mdp_ontology_report`: A JSON file containing a single ontology object representing a single ontology instance with label and action values for each timestep.
   - `output/10_ontology_output/multi_armed_bandit_ontology_report`: A JSON file containing multiple instances of the same ontology, which represent different actions across different timesteps.
   - `output/10_ontology_output/deep_planning_horizon_ontology_report`: A JSON file containing multiple instances of the same ontology, representing different actions across different timesteps and levels (represented by integers).

5. **Output Files**: There is one JSON file containing the output data from the model, which includes the following fields:
   - `output/10_ontology_output/actinf_pomdp_agent_ontology_report`: A JSON file containing a single ontology object representing an agent instance with label and action values for each timestep.
   - `output/10_ontology_output/two_state_bistable_ontology_report`: A JSON file containing multiple instances of the same ontology, representing different actions across different timesteps and levels (represented by integers).

6. **Output Files**: There is one JSON file containing the output data from the model, which includes the following