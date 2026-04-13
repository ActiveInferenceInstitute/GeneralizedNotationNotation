# SUMMARIZE_CONTENT

Here's a concise summary of the GNN specification:

**Model Overview**
This is a simple active inference model that demonstrates pure perception without temporal dynamics or action components. It uses two hidden states (A) and one observation (D), with each state mapping to a probability distribution over future observations, and each observation representing an action. The model's key variables are:

1. **hidden_states**: A 2-dimensional array of arrays containing the probabilities of observing a particular state at different times.
2. **observation**: A 1-dimensional array with the same shape as `A`, mapping to one-hot encoded observations.
3. **actions/controls** (optional): A list of matrices representing actions and control variables for each observation, where each matrix represents an action in the model's inference pipeline.
4. **hidden_states**: A 2-dimensional array containing arrays with probabilities over hidden states at different times.
5. **observations**: A 1-dimensional array with the same shape as `A`, mapping to one-hot encoded observations for each observation.
6. **actions/controls** (optional): A list of matrices representing actions and control variables for each observation, where each matrix represents an action in the model's inference pipeline.
7. **hidden_states**: A 2-dimensional array containing arrays with probabilities over hidden states at different times.
8. **observations** (optional): A list of arrays mapping to one-hot encoded observations for each observation.