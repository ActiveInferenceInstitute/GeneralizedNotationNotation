# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

SmolLM has included the following sections:

1. GNN Example: Active Inference POMDP Agent v1
   - In this example we have an instance of a traditional GNN model and it can be represented as:
    - Model name: Classic Active Inference POMDP agent v1
   - Type: GNN example with one observation modality, one hidden state factor.
   - Parameters: A = Likelihood map to the observed observations (observations) and H(x).
   - Initialization of state space: There are no initial states here but we assume it is a binary vector representing one observable with 3 elements for each action selection.
    - Example: For a single observation, there can be 2 possible outcomes based on actions and the value from the set {0-9} to support hypothesis testing (each element represents an outcome).
    - Input matrix:
      A = LikelihoodMatrix(observations,actions)
    1
      0       0     0       ...    0       1
       0        0         1   0   ...     1   0
         2           14.875-0 -1-36.995 -0.333  4.00e+2
            0       0     0       ...    0       1
           0          0                0        0
        -                                               
          
  Example: For a single observation, there are two possible outcomes that support hypothesis testing (each element is either "on" or "off").
   - Input matrix
    A = LikelihoodMatrix(observations)
    1
      0       0     0       ...    0       1
        2         4.598-0  1        4.597  3.631          32
        -                                               
          
Note: This GNN model is not limited to a specific parameterization (choices for states, actions). If you want your model with more parameters to fit different data types and distributions, you can provide it with parameters. For example, if your case has 5 observations per observation (observations), then in this case we need only one input matrix A = LikelihoodMatrix(observations)
GNN Example: Active Inference POMDP Agent v1 - GNN Representation.

2. GNN Version and Flags
   - Current version of GNN is V1, which is suitable for rendering into various simulation or inference backends.
  - Flags are given to specify how GNN represents the model:
    - "G(x)", A = Likelihood map to observed observation (observations) -> observed observation
    - "V(a)", B = Transition matrix from observed observations to observed actions

    - Flags 2 and 3 are used for specifying action selections, i.e., there is no prior over a given observation in this case
  - Flags 1-4 can be used to specify action selection from state estimation algorithm (HGSE), i.e.:
    - "B", A = Hidden State
    - "F", B = Habit
    - "E", A = Observation
# GNNVersionAndFlags
   - Current version of GNN is V1, which is suitable for rendering into various simulation or inference backends.
  - Flags are given to specify action selections, i.e., there is no prior over a given observation in this case
```