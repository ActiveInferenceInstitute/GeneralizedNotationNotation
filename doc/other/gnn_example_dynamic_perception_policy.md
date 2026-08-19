# GNN Example: Dynamic Perception Model with Policy Selection
# Format: Markdown representation of a Dynamic Perception model with Policy Selection in Active Inference format
# Version: 1.0
# This file is machine-readable

## GNNSection
DynamicPerceptionWithPolicySelection

## ImageFromPaper
image.png

## GNNVersionAndFlags
GNN v1

## ModelName
Dynamic perception with Policy Selection v1

## ModelAnnotation
This model relates a single hidden state to a single observable modality. It is a dynamic model because it tracks changes in the hidden state through time. There is Action applied via policy selection (π).

## StateSpaceBlock
A[2,2,type=float]
D[2,1,type=float]
B[2,len(π),2,type=float]
π=[2]
C=[2,1]
G[len(π),type=float]
s[2,1,type=float]
s_prime[2,1,type=float] # Next state (s at t+1)
o[2,1,type=float]
t[1,type=int]

## Connections
D-s
s-A
A-o
s-B
B-s_prime
C>G
G>π

## InitialParameterization
A={(0.7,0.3),(0.4,0.6)}
D={(0.5),(0.5)}
C={(0.8),(0.2)}
B={( ( (0.9,0.1),(0.1,0.9) ), ( (0.1,0.9),(0.9,0.1) ) )} # B[policy_index][state_from][state_to]

## Equations
s=sigma((1/2)(lnD+ln(B^dagger_{pi,tau}s_{pi,tau+1}))+lnA^T*o_tau) # for tau=1, policy π
s=sigma((1/2)(ln(B_{pi,tau-1}s_{pi,tau-1})+ln(B^dagger_{pi,tau}s_{pi,tau+1}))+lnA^T*o_tau) # for tau>1, policy π
G[π]=sum_tau(As_{pi,tau}(ln(A*s_{pi,tau})-lnC_tau)-diag(A^TlnA)*s_{pi,tau})
π=sigma(-G)

## Time
Dynamic
DiscreteTime=s
ModelTimeHorizon=Unbounded

## ActInfOntologyAnnotation
A=RecognitionMatrix
B=TransitionMatrix
C=Preference
D=Prior
G=ExpectedFreeEnergy
s=HiddenState
s_prime=NextHiddenState
o=Observation
π=PolicyVector
t=Time

## ModelParameters
num_hidden_states_factors: [2]
num_obs_modalities: [2]
num_control_action_dims: [2] # From len(π) which is 2, used in B matrix

## Footer
Dynamic perception with Policy Selection v1

## Signature
NA 