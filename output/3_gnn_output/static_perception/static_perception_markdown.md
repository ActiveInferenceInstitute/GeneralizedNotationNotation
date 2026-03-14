## GNNVersionAndFlags
Version: 1.0

## ModelName
Static Perception Model

## ModelAnnotation
The simplest Active Inference model demonstrating pure perception:

- 2 hidden states mapped to 2 observations via a recognition matrix A
- Prior D encodes initial beliefs over hidden states
- No temporal dynamics — single-shot inference
- Demonstrates the core observation model: P(o|s) = A
- Suitable as a minimal baseline and for testing perception-only inference

## StateSpaceBlock
A[2,2],float
D[2,1],float
s[2,1],float
o[2,1],integer

## Connections
D>s
s-A
A-o

## InitialParameterization
A = [[0.9, 0.1], [0.2, 0.8]]
D = [[0.5, 0.5]]

## Time
Static

## ActInfOntologyAnnotation
A = RecognitionMatrix
D = Prior
s = HiddenState
o = Observation

## Footer
Generated: 2026-03-13T18:17:53.172644

## Signature
