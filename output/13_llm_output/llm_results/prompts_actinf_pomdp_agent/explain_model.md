# EXPLAIN_MODEL

Here is a concise description of the GNN Representation:
ModelPurpose=GNN 
What are the hidden states (s_f0, s_f1, etc.) and what do they represent?
"The model represents a POMDP that describes an agent acting independently on its observation space with one observation modality ("state-observation"), one hidden state factor ('location'), a policy prior (habit), actions as transitions between states. Each action takes a single observation ('observation') from the policy, and each action is uniformly distributed over all observed observations in the system"