#!/usr/bin/env python3
# PyMDP Active Inference Simulation
# Generated from GNN Model: Unknown
# Generated: 2025-07-24 10:58:26

import numpy as np
from pymdp import utils
from pymdp.agent import Agent
from pymdp.envs import Env

# Model parameters
num_states = 3
num_obs = 3
num_controls = 3

# Initialize likelihood matrix (A matrix)
A = np.eye(num_obs, num_states)  # Identity mapping for now

# Initialize transition matrix (B matrix) 
B = np.zeros((num_states, num_states, num_controls))
for i in range(num_controls):
    B[:, :, i] = np.eye(num_states)  # Identity transitions for now

# Initialize preference vector (C vector)
C = np.zeros(num_obs)

# Initialize prior over states (D vector)
D = np.ones(num_states) / num_states  # Uniform prior

# Create agent
agent = Agent(A=A, B=B, C=C, D=D)

# Create environment (simple identity mapping)
env = Env(A=A, B=B)

# Simulation parameters
T = 10  # Number of time steps

# Run simulation
for t in range(T):
    # Get observation from environment
    obs = env.step()
    
    # Agent inference and action selection
    qs = agent.infer_states(obs)
    q_pi, _ = agent.infer_policies()
    action = agent.sample_action()
    
    print(f"Step {t}: Observation={obs}, Action={action}")
    print(f"  State beliefs: {qs}")
    print(f"  Policy beliefs: {q_pi}")

print("Simulation completed!")
