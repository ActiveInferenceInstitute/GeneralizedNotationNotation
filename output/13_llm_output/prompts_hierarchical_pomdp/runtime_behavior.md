# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

Here's a concise summary of the key points:

1. The GNN model is an unsupervised active inference system that uses hierarchical Bayesian networks to learn and update beliefs from data. It consists of two main components:
   - Level 1 (fast dynamics)
   - Level 2 (slow dynamics)
   - Higher-level beliefs are updated at a slower timescale

The model's structure includes:
   - Level 1 (fast dynamics):
     - A1 = LikelihoodMatrix
     - B1 = TransitionMatrix
     - C1 = LogPreferenceVector
     - D1 = PriorOverHiddenStates
     - s1=LikelihoodMatrix
     - o1=LogPreferenceVector
     - G1 = HigherLevelLikelihoodMatrix
   - Level 2 (slow dynamics):
     - A2 = TransitionMatrix
     - B2 = ContextTransitionMatrix
     - C2 = LogPreferenceVector
     - D2 = PriorOverHiddenStates
     - s2=HierarchicalMessagePassingMatrix
     - G2 = HigherLevelExpectedFreeEnergy
   - Hierarchical belief propagation: Top-down updates based on observed data

The model's behavior is characterized by fast and slow dynamics, as well as hierarchical message passing. The rate of slower dynamics corresponds to the timescale ratio (5). This allows for a more efficient learning process compared to other active inference systems like GNNPOMDP or GNNV2017.