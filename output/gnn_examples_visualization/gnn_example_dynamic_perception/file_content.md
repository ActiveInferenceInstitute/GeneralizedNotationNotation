# GNN File: src/gnn/examples/gnn_example_dynamic_perception.md\n\n## Raw File Content\n\n```\n# GNN Example: Dynamic Perception Model
# Format: Markdown representation of a Dynamic Perception model in Active Inference format
# Version: 1.0
# This file is machine-readable

## GNNSection
DynamicPerception

## ImageFromPaper
image.png

## GNNVersionAndFlags
GNN v1

## ModelName
Dynamic perception v1

## ModelAnnotation
This model relates a single hidden state to a single observable modality. It is a dynamic model because it tracks changes in the hidden state through time.

## StateSpaceBlock
D[2,1,type=float]
B[2,2,type=float]
s[2,1,type=float]
s_prime[2,1,type=float] # Next state (s at t+1)
A[2,2,type=float]
o[2,1,type=float]
t[1,type=int]
C[2,1,type=float]

## Connections
D-s
s-A
A-o
s-B
B-s_prime

## InitialParameterization
A={(0.7,0.3),(0.4,0.6)}
B={(0.9,0.1),(0.1,0.9)}
D={(0.5),(0.5)}
C={(0.0),(0.0)} # Neutral preferences

## Equations
s=softmax((1/2)(ln(D)+ln(B^dagger_tau*s_{tau+1})+ln(trans(A)o_tau))) # for tau=1
s=softmax((1/2)(ln(D)+ln(B^dagger_tau*s_{tau+1})+ln(trans(A)o_tau))) # for tau>1

## Time
Dynamic
DiscreteTime=s
ModelTimeHorizon=Unbounded

## ActInfOntologyAnnotation
A=RecognitionMatrix
B=TransitionMatrix
D=Prior
s=HiddenState
s_prime=NextHiddenState
o=Observation
t=Time

## ModelParameters
num_hidden_states_factors: [2]
num_obs_modalities: [2]
num_control_action_dims: [1] # Implicit, as no explicit control factors defined

## Footer
Dynamic perception v1

## Signature
NA \n```\n\n## Parsed Sections

### _HeaderComments

```
# GNN Example: Dynamic Perception Model
# Format: Markdown representation of a Dynamic Perception model in Active Inference format
# Version: 1.0
# This file is machine-readable
```

### ModelName

```
Dynamic perception v1
```

### GNNSection

```
DynamicPerception
```

### ImageFromPaper

```
image.png
```

### GNNVersionAndFlags

```
GNN v1
```

### ModelAnnotation

```
This model relates a single hidden state to a single observable modality. It is a dynamic model because it tracks changes in the hidden state through time.
```

### StateSpaceBlock

```
D[2,1,type=float]
B[2,2,type=float]
s[2,1,type=float]
s_prime[2,1,type=float] # Next state (s at t+1)
A[2,2,type=float]
o[2,1,type=float]
t[1,type=int]
C[2,1,type=float]
```

### Connections

```
D-s
s-A
A-o
s-B
B-s_prime
```

### InitialParameterization

```
A={(0.7,0.3),(0.4,0.6)}
B={(0.9,0.1),(0.1,0.9)}
D={(0.5),(0.5)}
C={(0.0),(0.0)} # Neutral preferences
```

### Equations

```
s=softmax((1/2)(ln(D)+ln(B^dagger_tau*s_{tau+1})+ln(trans(A)o_tau))) # for tau=1
s=softmax((1/2)(ln(D)+ln(B^dagger_tau*s_{tau+1})+ln(trans(A)o_tau))) # for tau>1
```

### Time

```
Dynamic
DiscreteTime=s
ModelTimeHorizon=Unbounded
```

### ActInfOntologyAnnotation

```
A=RecognitionMatrix
B=TransitionMatrix
D=Prior
s=HiddenState
s_prime=NextHiddenState
o=Observation
t=Time
```

### ModelParameters

```
num_hidden_states_factors: [2]
num_obs_modalities: [2]
num_control_action_dims: [1] # Implicit, as no explicit control factors defined
```

### Footer

```
Dynamic perception v1
```

### Signature

```
NA
```

