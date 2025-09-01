#!/usr/bin/env python3
"""
PyMDP simulation code for actinf_pomdp_agent
Generated from GNN specification
"""

import numpy as np
from pymdp import utils
from pymdp.agent import Agent
from pymdp.envs import Env

def create_actinf_pomdp_agent_agent():
    """Create a PyMDP agent for actinf_pomdp_agent."""
    
    # Define observation space
    num_obs = 4
    num_obs_modalities = 1
    
    # Define action space  
    num_controls = 2
    
    # Define state space
    num_states = 3
    
    # Create likelihood matrix (A matrix)
    A = utils.random_A_matrix(num_obs_modalities, num_obs, num_states)
    
    # Create transition matrix (B matrix)
    B = utils.random_B_matrix(num_states, num_controls)
    
    # Create preference matrix (C matrix)
    C = utils.obj_array_zeros([num_obs])
    
    # Create prior over states (D matrix)
    D = utils.obj_array_uniform(num_states)
    
    # Create agent
    agent = Agent(A=A, B=B, C=C, D=D)
    
    return agent

def create_actinf_pomdp_agent_environment():
    """Create a PyMDP environment for actinf_pomdp_agent."""
    
    # Define environment parameters
    num_states = 3
    num_obs = 4
    num_controls = 2
    
    # Create likelihood matrix
    A = utils.random_A_matrix(1, num_obs, num_states)
    
    # Create transition matrix
    B = utils.random_B_matrix(num_states, num_controls)
    
    # Create environment
    env = Env(A=A, B=B)
    
    return env

def run_actinf_pomdp_agent_simulation(num_steps=100):
    """Run a simulation of actinf_pomdp_agent."""
    
    # Create agent and environment
    agent = create_actinf_pomdp_agent_agent()
    env = create_actinf_pomdp_agent_environment()
    
    # Initialize
    obs = env.reset()
    total_reward = 0
    
    # Run simulation
    for step in range(num_steps):
        # Agent action
        q_pi, _ = agent.infer_states(obs)
        q_u = agent.infer_policies()
        action = agent.sample_action()
        
        # Environment step
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    return total_reward

if __name__ == "__main__":
    # Run simulation
    reward = run_actinf_pomdp_agent_simulation()
    print(f"Simulation completed with total reward: {reward}")
