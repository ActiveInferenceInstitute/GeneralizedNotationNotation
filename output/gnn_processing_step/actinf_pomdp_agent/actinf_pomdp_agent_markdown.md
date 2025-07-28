## GNNVersionAndFlags
Version: 1.0

## ModelName
Classic Active Inference POMDP Agent v1

## StateSpaceBlock
A[3,3],float
B[3,3,3],float
C[3],float
D[3],float
E[3],float

## Connections
A>s
B>s_prime
C>o
D>s
E>π
s>o
π>u
u>s_prime

## InitialParameterization
A = []
B = []
C = []
D = []
E = []

## Footer
Generated: 2025-07-28T07:42:28.759726

## Signature
