
# Processed by GNN Pipeline Template
# Original file: /home/trim/Documents/GitHub/GeneralizedNotationNotation/input/gnn_files/actinf_pomdp_agent.md
# Processed on: 2025-08-06T18:43:39.049008
# Options: {'verbose': True, 'recursive': True, 'example_param': 'default_value'}

## ModelName
Classic Active Inference POMDP Agent v1

## StateSpaceBlock
A[3, 3] # likelihood_matrix
B[3, 3, 3] # transition_matrix
C[3] # preference_vector
D[3] # prior_vector
E[3] # policy

## InitialParameterization
A={{0.8, 0.1, 0.1}, {0.1, 0.8, 0.1}, {0.1, 0.1, 0.8}}
B={{{{0.9, 0.05, 0.05}, {0.05, 0.9, 0.05}, {0.05, 0.05, 0.9}}, {{0.8, 0.1, 0.1}, {0.1, 0.8, 0.1}, {0.1, 0.1, 0.8}}, {{0.7, 0.15, 0.15}, {0.15, 0.7, 0.15}, {0.15, 0.15, 0.7}}}}
C={0.1, 0.1, 0.8}
D={0.33, 0.33, 0.34}
E={0.33, 0.33, 0.34}

## Connections
A > s
B > s_prime
C > o
D > s
E > π
s > o
π > u
u > s_prime

