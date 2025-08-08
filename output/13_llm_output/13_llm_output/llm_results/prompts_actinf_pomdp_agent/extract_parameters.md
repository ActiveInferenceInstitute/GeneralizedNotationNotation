# EXTRACT_PARAMETERS

You can provide the following steps to calculate the model parameters for your GNN, POMDP agent:

1.   For each observation modality "state_observation", apply a set of probability distributions to predict the state and then update these probabilities with action selection from policy. This provides 3-step predictions based on prior beliefs over observable states (policy), actions selected via habit/prior belief distribution, and initial parameters for learning.

2.   Apply similar logic to all observation modalities in order to infer posterior knowledge about next observed states and subsequent action selections. 

3.   Iterate through all actions using a decision tree-based algorithm or a greedy mechanism until convergence of the model is reached (see GNN example below).