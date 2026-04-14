# EXPLAIN_MODEL

Here is a concise summary of the GNN Section:

**GNNSection:**
A standard discrete Hidden Markov Model (HMM) with 4 hidden states and 6 observation symbols has been implemented for comparison to Active Inference POMDP variants, including the HMM Baseline. The model consists of four hidden states, two action-based states, and a single observable state. The input data are represented as sparse matrices representing the observed and predicted outcomes from the model.

**GNNVersionAndFlags:**
The GNN version v1 is implemented for comparison to Active Inference POMDP variants with HMM Baseline. The model consists of four hidden states, two action-based states, and a single observable state. The input data are represented as sparse matrices representing the observed outcomes from the model.

**ModelAnnotation:**
A standard discrete Hidden Markov Model (HMM) with 4 hidden states and 6 observation symbols has been implemented for comparison to Active Inference POMDP variants with HMM Baseline. The model consists of four hidden states, two action-based states, and a single observable state. The input data are represented as sparse matrices representing the observed outcomes from the model.

**StateSpaceBlock:**
The GNN specification has 4 hidden states, 6 observation symbols, and fixed transition matrix (no action dependence). The initial state distribution is specified to be a random state with probability of 1/256 for each hidden state. The next hidden state is specified to have the same probability as the previous one but now it represents the current state instead of the previous state. The last hidden state has no observable and is fixed in its value at time t = 0, which corresponds to the initial state.

**InferenceQuantities:**
The GNN inference quantities are specified for each hidden state and action-based states:

1. **Forward Algorithm:**
   - Alpha_t(s) = sum_{s'} P(o_t|s) * P(s|s')
   - Beta_t(s) = sum_{s'} P(o_{t+1}|s') * B(s'|s)

2. **Backward Algorithm:**
   - Alpha_b(s) = sum_{s'} P(o_b|s') * B(s'|s)
   - Beta_b(s) = sum_{s'} P(o_{t