# EXTRACT_PARAMETERS

Here's a comprehensive summary of the key components in your GNN specification:

**Variance and Variability Parameters:**
- **Variance** parameter set up for the GNN variable with prior values that determine which actions are taken based on their likelihood. This allows you to specify parameters aligned with previous state inference models (in this case, GNNs) or different distributions used during analysis of future observations (`state_observation` variables).
- **Variability** parameter set up for the GNN variable with prior values that determine which actions are taken based on their likelihood. This allows you to specify parameters aligned with previous states and actions (e.g., policy updates, action selection) or different distributions used during analysis of future observations (`state_observation` variables).
- **Variability** parameter set up for the GNN variable with prior values that determine which actions are taken based on their likelihoods (not mentioned explicitly in your specification). This allows you to specify parameters aligned with previous states and actions while enabling flexibility for different action selection strategies.