# GNN Syntax Reference

Quick reference for GNN syntax with working examples.

## Variable Declaration

```gnn
## StateSpaceBlock
s[2,1,type=int]      # 2D state vector, integer type
o[3,1,type=float]    # 3D observation vector, float type
A[3,2,type=float]    # 3×2 matrix, float type
t[1,type=int]        # Time scalar, integer type
```

## Subscripts and Superscripts

```gnn
s_t[2,1,type=float]      # s with subscript t
s_t+1[2,1,type=float]    # s with subscript t+1
X^observed[3,1,type=int] # X with superscript observed
π_f1[3,type=float]       # π with subscripts f1
```

## Connections

```gnn
## Connections
s>o          # Directed: s causes o
s-A          # Undirected: s relates to A
s_t>s_t+1    # Temporal: current state to next state
(s,u)>B      # Multiple inputs to B
```

## Dimensions and Types

```gnn
X[2]           # Vector of length 2
X[2,3]         # 2×3 matrix
X[2,3,4]       # 3D tensor: 2×3×4
X[len(π)]      # Dynamic size based on policy length
X[1,type=int]  # Explicit type declaration
```

## Initial Values

```gnn
## InitialParameterization
D={0.5,0.5}                    # Vector
A={(0.9,0.1),(0.2,0.8)}       # Matrix rows
B={((1,0),(0,1)),((0,1),(1,0))} # 3D tensor
```

## Mathematical Operations

```gnn
P(X|Y)    # Conditional probability
X+Y       # Addition
X*Y       # Multiplication  
X/Y       # Division
X^2       # Power
```

## Comments

```gnn
s[2,1,type=float]  # Hidden state vector
### This is a full-line comment
A[2,2,type=float]  ### Recognition matrix
```

## Time Specifications

```gnn
## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=10
```

## Complete Minimal Example

```gnn
## GNN v1

# Simple Static Model

## StateSpaceBlock
s[2,1,type=float]
o[2,1,type=float]
A[2,2,type=float]

## Connections
s-A-o

## InitialParameterization
A={(0.9,0.1),(0.1,0.9)}

## Time
Static

# Simple Static Model
```

## Ontology Mapping

```gnn
## ActInfOntologyAnnotation
s=HiddenState
o=Observation
A=RecognitionMatrix
```

This syntax produces models that parse cleanly and execute correctly in the GNN pipeline.