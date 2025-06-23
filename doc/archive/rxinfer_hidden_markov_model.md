# GNN Example: RxInfer Hidden Markov Model
# Format: Markdown representation of a Hidden Markov Model for RxInfer.jl
# Version: 1.0
# This file is machine-readable and represents a 3-state Hidden Markov Model with 3 observation categories.

## GNNSection
RxInferHiddenMarkovModel

## GNNVersionAndFlags
GNN v1

## ModelName
RxInfer Hidden Markov Model v1

## ModelAnnotation
This model represents a Hidden Markov Model with 3 hidden states and 3 observation categories for RxInfer.jl.
- Hidden states: "Bedroom" (state 1), "Living room" (state 2), "Bathroom" (state 3)
- Observations: 3 categorical outcomes corresponding to noisy observations of the true state
- Transition matrix A: Controls state-to-state transitions over time
- Observation matrix B: Controls emission probabilities from hidden states to observations
- The model uses Dirichlet priors on both A and B matrices for Bayesian learning
- Initial state distribution is uniform over the 3 states

## StateSpaceBlock
# Model dimensions
T[1,type=int]                    # Time horizon / number of time steps
n_states[1,type=int]             # Number of hidden states (3)
n_obs[1,type=int]                # Number of observation categories (3)

# Transition and observation matrices
A[3,3,type=float]                # State transition matrix P(s_t|s_{t-1})
B[3,3,type=float]                # Observation/emission matrix P(x_t|s_t)

# Dirichlet hyperparameters for priors
A_prior[3,3,type=float]          # Dirichlet hyperparameters for transition matrix
B_prior[3,3,type=float]          # Dirichlet hyperparameters for observation matrix

# State and observation sequences
s_0[3,type=float]                # Initial state distribution
s[3,T,type=float]                # Hidden state sequence (categorical distributions)
x[3,T,type=float]                # Observation sequence (categorical distributions)

# Posterior marginals (inference results)
q_A[3,3,type=float]              # Posterior marginal for transition matrix
q_B[3,3,type=float]              # Posterior marginal for observation matrix
q_s[3,T,type=float]              # Posterior marginals for hidden states

# Inference parameters
n_iterations[1,type=int]         # Number of variational inference iterations
free_energy[n_iterations,type=float] # Free energy trace during inference

# Data generation parameters (for simulation)
seed[1,type=int]                 # Random seed for reproducibility
n_samples[1,type=int]            # Number of data samples to generate

## Connections
# Prior specifications
A_prior > A
B_prior > B
s_0 > s

# Generative model structure
s_0 > s[1]                       # Initial state influences first hidden state
A > s                            # Transition matrix influences state sequence  
B > x                            # Observation matrix influences observations
s > x                            # Hidden states generate observations

# Temporal dependencies
s[t-1] > s[t]                    # Previous state influences current state (for t > 1)
s[t] > x[t]                      # Current state generates current observation

# Inference connections
(A, B, s_0, x) > (q_A, q_B, q_s) # Inference from observations to posteriors
(q_A, q_B, q_s) > free_energy    # Posteriors contribute to free energy calculation

## InitialParameterization
# Model dimensions
T=100
n_states=3
n_obs=3
n_iterations=20
n_samples=100
seed=42

# Dirichlet hyperparameters for transition matrix A
# Encouraging diagonal structure (agents tend to stay in same room)
A_prior={
  (10.0, 1.0, 1.0),  # From state 0 (Bedroom): strong preference to stay
  (1.0, 10.0, 1.0),  # From state 1 (Living room): strong preference to stay  
  (1.0, 1.0, 10.0)   # From state 2 (Bathroom): strong preference to stay
}

# Dirichlet hyperparameters for observation matrix B  
# Diagonal structure with some noise (observations mostly match true state)
B_prior={
  (1.0, 1.0, 1.0),   # Uniform prior for observations from each state
  (1.0, 1.0, 1.0),   
  (1.0, 1.0, 1.0)    
}

# True data generation parameters (from the Julia example)
# Ground truth transition matrix for data generation
A_true={
  (0.9, 0.05, 0.0),   # From Bedroom: 90% stay, 5% to Living room, 0% to Bathroom
  (0.1, 0.9, 0.1),    # From Living room: 10% to Bedroom, 90% stay, 10% to Bathroom
  (0.0, 0.05, 0.9)    # From Bathroom: 0% to Bedroom, 5% to Living room, 90% stay
}

# Ground truth observation matrix for data generation  
B_true={
  (0.9, 0.05, 0.05),  # From Bedroom: 90% correct obs, 5% each wrong obs
  (0.05, 0.9, 0.05),  # From Living room: 90% correct obs, 5% each wrong obs
  (0.05, 0.05, 0.9)   # From Bathroom: 90% correct obs, 5% each wrong obs
}

# Initial state distribution (starts in Bedroom with certainty)
s_0={(1.0, 0.0, 0.0)}

# Expected posterior results (approximate, from Julia example output)
# These would be learned through inference
q_A_expected={
  (0.9, 0.05, 0.0),   # Learned transition probabilities
  (0.1, 0.9, 0.1),    
  (0.0, 0.05, 0.9)    
}

q_B_expected={
  (0.9, 0.05, 0.05),  # Learned observation probabilities
  (0.05, 0.9, 0.05),  
  (0.05, 0.05, 0.9)   
}

## Equations
# Hidden Markov Model generative equations:
# s_0 ~ Categorical([1/3, 1/3, 1/3])  # Initial state (uniform in model, deterministic in data)
# A ~ DirichletCollection(A_prior)     # Prior on transition matrix
# B ~ DirichletCollection(B_prior)     # Prior on observation matrix
# 
# For t = 1, ..., T:
#   s[t] ~ DiscreteTransition(s[t-1], A)  # State transition
#   x[t] ~ DiscreteTransition(s[t], B)    # Observation emission
#
# Inference objective:
# Minimize: F = E_q[log q(s,A,B) - log p(x,s,A,B)]
# where q(s,A,B) is the variational posterior approximation

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=T

## ActInfOntologyAnnotation
A=StateTransitionMatrix
B=ObservationMatrix
A_prior=TransitionMatrixPrior
B_prior=ObservationMatrixPrior
s_0=InitialStateDistribution
s=HiddenStateSequence
x=ObservationSequence
q_A=PosteriorTransitionMatrix
q_B=PosteriorObservationMatrix  
q_s=PosteriorHiddenStates
free_energy=VariationalFreeEnergy
T=TimeHorizon
n_states=NumberOfHiddenStates
n_obs=NumberOfObservationCategories
n_iterations=InferenceIterations

## ModelParameters
n_states=3              # Hidden states: Bedroom, Living room, Bathroom
n_obs=3                 # Observation categories: 3 discrete outcomes
n_iterations=20         # Variational inference iterations
model_type="HMM"        # Hidden Markov Model
inference_method="variational_message_passing"
backend="RxInfer.jl"

## Footer
RxInfer Hidden Markov Model v1 - GNN Representation

## Signature
Creator: AI Assistant for GNN
Date: 2024-12-19
Status: Example for RxInfer.jl Hidden Markov Model
Source: RxInferExamples.jl/Basic Examples/Hidden Markov Model 