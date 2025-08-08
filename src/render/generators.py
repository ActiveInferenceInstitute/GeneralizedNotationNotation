#!/usr/bin/env python3
"""
Render generators module for GNN code generation.
"""

from typing import Dict, Any
import re

def _sanitize_identifier(base: str, *, lowercase: bool = True, allow_empty_fallback: str = "model") -> str:
    """Sanitize an arbitrary string into a safe Python/Julia identifier (snake_case).
    - Replace non-alphanumeric chars with underscores
    - Collapse repeated underscores
    - Lowercase if requested
    - Prefix with fallback if it would start with a digit or be empty
    """
    s = base.lower() if lowercase else base
    s = re.sub(r"\W+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = allow_empty_fallback
    if s[0].isdigit():
        s = f"{allow_empty_fallback}_{s}"
    return s

def _to_pascal_case(base: str, *, allow_empty_fallback: str = "Model") -> str:
    """Convert arbitrary string to PascalCase for Julia struct/type names.
    Non-alphanumeric separators are treated as word boundaries.
    """
    parts = re.split(r"\W+", base)
    parts = [p for p in parts if p]
    if not parts:
        parts = [allow_empty_fallback]
    name = "".join(p.capitalize() for p in parts)
    if name[0].isdigit():
        name = f"{allow_empty_fallback}{name}"
    return name

def generate_pymdp_code(model_data: Dict) -> str:
    """Generate PyMDP simulation code using the modular PyMDP renderer."""
    try:
        # Get model name for file paths and sanitize for identifiers
        model_name = model_data.get('model_name', 'GNN Model')
        model_snake = _sanitize_identifier(model_name, lowercase=True, allow_empty_fallback="model")
        
        # Generate PyMDP code
        code = f'''#!/usr/bin/env python3
"""
PyMDP simulation code for {model_name}
Generated from GNN specification
"""

import numpy as np
from pymdp import utils
from pymdp.agent import Agent
from pymdp.envs import Env

def create_{model_snake}_agent():
    """Create a PyMDP agent for {model_name}."""
    
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

def create_{model_snake}_environment():
    """Create a PyMDP environment for {model_name}."""
    
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

def run_{model_snake}_simulation(num_steps=100):
    """Run a simulation of {model_name}."""
    
    # Create agent and environment
    agent = create_{model_snake}_agent()
    env = create_{model_snake}_environment()
    
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
    reward = run_{model_snake}_simulation()
    print(f"Simulation completed with total reward: {{reward}}")
'''
        
        return code
        
    except Exception as e:
        return f"# Error generating PyMDP code: {e}"

def generate_rxinfer_code(model_data: Dict) -> str:
    """Generate RxInfer.jl simulation code."""
    try:
        model_name = model_data.get('model_name', 'GNN Model')
        model_snake = _sanitize_identifier(model_name, lowercase=True, allow_empty_fallback="model")
        model_pascal = _to_pascal_case(model_name, allow_empty_fallback="Model")
        
        code = f'''#!/usr/bin/env julia
"""
RxInfer.jl simulation code for {model_name}
Generated from GNN specification
"""

using RxInfer
using Distributions
using LinearAlgebra

@model function {model_snake}_model(n)
    # Define variables
    x = randomvar(n)
    y = datavar(Float64, n)
    
    # Define priors
    x_prior ~ NormalMeanVariance(0.0, 1.0)
    
    # Define likelihood
    for i in 1:n
        x[i] ~ NormalMeanVariance(x_prior, 1.0)
        y[i] ~ NormalMeanVariance(x[i], 0.1)
    end
end

function run_{model_snake}_inference(data, n)
    """Run inference for {model_name}."""
    
    # Create model
    model = {model_snake}_model(n)
    
    # Set up constraints
    constraints = @constraints begin
        q(x) :: NormalMeanVariance
        q(x_prior) :: NormalMeanVariance
    end
    
    # Run inference
    result = inference(
        model = model,
        data = (y = data,),
        constraints = constraints,
        initmarginals = (x_prior = NormalMeanVariance(0.0, 1.0),),
        iterations = 10
    )
    
    return result
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    # Generate sample data
    n = 100
    true_x = randn(n)
    data = true_x .+ 0.1 .* randn(n)
    
    # Run inference
    result = run_{model_snake}_inference(data, n)
    
    println("Inference completed successfully")
    println("Posterior mean of x_prior: ", mean(result.posteriors[:x_prior]))
end
'''
        
        return code
        
    except Exception as e:
        return f"# Error generating RxInfer code: {e}"

def generate_rxinfer_fallback_code(model_data: Dict) -> str:
    """Generate fallback RxInfer.jl code when main generator fails."""
    try:
        model_name = model_data.get('model_name', 'GNN Model')
        model_snake = _sanitize_identifier(model_name, lowercase=True, allow_empty_fallback="model")
        
        code = f'''#!/usr/bin/env julia
"""
Fallback RxInfer.jl code for {model_name}
"""

using RxInfer
using Distributions

@model function {model_name.lower().replace('-', '_')}_fallback_model(n)
    # Simple fallback model
    x = randomvar(n)
    y = datavar(Float64, n)
    
    # Prior
    x_prior ~ NormalMeanVariance(0.0, 1.0)
    
    # Likelihood
    for i in 1:n
        x[i] ~ NormalMeanVariance(x_prior, 1.0)
        y[i] ~ NormalMeanVariance(x[i], 0.1)
    end
end

function run_{model_name.lower().replace('-', '_')}_fallback_inference(data, n)
    """Run fallback inference."""
    
    model = {model_name.lower().replace('-', '_')}_fallback_model(n)
    
    constraints = @constraints begin
        q(x) :: NormalMeanVariance
        q(x_prior) :: NormalMeanVariance
    end
    
    result = inference(
        model = model,
        data = (y = data,),
        constraints = constraints,
        initmarginals = (x_prior = NormalMeanVariance(0.0, 1.0),),
        iterations = 5
    )
    
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    n = 50
    data = randn(n)
    result = run_{model_name.lower().replace('-', '_')}_fallback_inference(data, n)
    println("Fallback inference completed")
end
'''
        
        return code
        
    except Exception as e:
        return f"# Error generating RxInfer fallback code: {e}"

def generate_activeinference_jl_code(model_data: Dict) -> str:
    """Generate ActiveInference.jl simulation code."""
    try:
        model_name = model_data.get('model_name', 'GNN_Model')
        
        code = f'''#!/usr/bin/env julia
"""
ActiveInference.jl simulation code for {model_name}
Generated from GNN specification
"""

using ActiveInference
using Distributions
using LinearAlgebra

struct {model_pascal}Agent
    A::Matrix{Float64}  # Likelihood matrix
    B::Array{Float64, 3}  # Transition matrices
    C::Vector{Float64}  # Preferences
    D::Vector{Float64}  # Prior over states
end

function create_{model_snake}_agent()
    """Create an ActiveInference agent for {model_name}."""
    
    # Define dimensions
    num_states = 3
    num_obs = 4
    num_controls = 2
    
    # Create likelihood matrix A
    A = rand(num_obs, num_states)
    A ./= sum(A, dims=1)  # Normalize
    
    # Create transition matrices B
    B = zeros(num_states, num_states, num_controls)
    for u in 1:num_controls
        B[:, :, u] = rand(num_states, num_states)
        B[:, :, u] ./= sum(B[:, :, u], dims=1)  # Normalize
    end
    
    # Create preferences C
    C = zeros(num_obs)
    
    # Create prior over states D
    D = ones(num_states) / num_states
    
    return {model_pascal}Agent(A, B, C, D)
end

function run_{model_snake}_simulation(agent, num_steps=100)
    """Run ActiveInference simulation for {model_name}."""
    
    # Initialize
    s = agent.D  # Initial state
    total_free_energy = 0.0
    
    for step in 1:num_steps
        # Generate observation
        o = agent.A * s
        o ./= sum(o)  # Normalize
        
        # Infer states (variational message passing)
        qs = agent.D  # Initialize with prior
        
        # Update beliefs
        for i in 1:10  # Iterative inference
            qs = softmax(log.(agent.D) + agent.A' * log.(o))
        end
        
        # Infer policies
        # (Simplified - in practice would use more sophisticated policy inference)
        
        # Update state
        s = qs
        
        # Calculate free energy
        F = sum(o .* log.(o ./ (agent.A * s)))
        total_free_energy += F
    end
    
    return total_free_energy
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Create agent and run simulation
    agent = create_{model_snake}_agent()
    free_energy = run_{model_snake}_simulation(agent)
    
    println("ActiveInference simulation completed")
    println("Total free energy: ", free_energy)
end
'''
        
        return code
        
    except Exception as e:
        return f"# Error generating ActiveInference.jl code: {e}"

def generate_activeinference_jl_fallback_code(model_data: Dict) -> str:
    """Generate fallback ActiveInference.jl code."""
    try:
        model_name = model_data.get('model_name', 'GNN_Model')
        
        code = f'''#!/usr/bin/env julia
"""
Fallback ActiveInference.jl code for {model_name}
"""

using Distributions
using LinearAlgebra

function create_{model_name.lower().replace('-', '_')}_fallback_agent()
    """Create a simple fallback agent."""
    
    num_states = 2
    num_obs = 2
    
    # Simple likelihood matrix
    A = [0.8 0.2; 0.2 0.8]
    
    # Simple preferences
    C = [1.0, -1.0]
    
    # Uniform prior
    D = [0.5, 0.5]
    
    return (A=A, C=C, D=D)
end

function run_{model_name.lower().replace('-', '_')}_fallback_simulation(agent, num_steps=50)
    """Run fallback simulation."""
    
    s = agent.D
    total_energy = 0.0
    
    for step in 1:num_steps
        # Generate observation
        o = agent.A * s
        
        # Simple belief update
        qs = softmax(log.(agent.D) + agent.A' * log.(o))
        
        # Update state
        s = qs
        
        # Calculate energy
        energy = -sum(o .* log.(o ./ (agent.A * s)))
        total_energy += energy
    end
    
    return total_energy
end

if abspath(PROGRAM_FILE) == @__FILE__
    agent = create_{model_name.lower().replace('-', '_')}_fallback_agent()
    energy = run_{model_name.lower().replace('-', '_')}_fallback_simulation(agent)
    println("Fallback simulation completed. Total energy: ", energy)
end
'''
        
        return code
        
    except Exception as e:
        return f"# Error generating ActiveInference.jl fallback code: {e}"

def generate_discopy_code(model_data: Dict) -> str:
    """Generate DisCoPy categorical diagram code."""
    try:
        model_name = model_data.get('model_name', 'GNN_Model')
        
        code = f'''#!/usr/bin/env python3
"""
DisCoPy categorical diagram code for {model_name}
Generated from GNN specification
"""

from discopy import *
from discopy.quantum import *
import numpy as np

def create_{model_snake}_diagram():
    """Create a DisCoPy categorical diagram for {model_name}."""
    
    # Define types
    State = Ty('State')
    Observation = Ty('Observation')
    Action = Ty('Action')
    
    # Create basic boxes
    transition = Box('transition', State @ Action, State)
    observation = Box('observation', State, Observation)
    
    # Create diagram
    diagram = (State @ Action >> transition >> State >> observation >> Observation)
    
    return diagram

def create_{model_snake}_quantum_diagram():
    """Create a quantum-inspired diagram for {model_name}."""
    
    # Define quantum types
    Qubit = Ty('Qubit')
    
    # Create quantum gates
    hadamard = Box('H', Qubit, Qubit)
    cnot = Box('CNOT', Qubit @ Qubit, Qubit @ Qubit)
    
    # Create quantum circuit
    circuit = (Qubit @ Qubit >> cnot >> Qubit @ Qubit)
    
    return circuit

def run_{model_snake}_simulation():
    """Run DisCoPy simulation for {model_name}."""
    
    # Create diagrams
    classical_diagram = create_{model_snake}_diagram()
    quantum_diagram = create_{model_snake}_quantum_diagram()
    
    print("Classical diagram:")
    print(classical_diagram)
    print("\\nQuantum diagram:")
    print(quantum_diagram)
    
    return classical_diagram, quantum_diagram

if __name__ == "__main__":
    classical, quantum = run_{model_snake}_simulation()
    print("\\nDisCoPy simulation completed successfully")
'''
        
        return code
        
    except Exception as e:
        return f"# Error generating DisCoPy code: {e}"

def generate_discopy_fallback_code(model_data: Dict) -> str:
    """Generate fallback DisCoPy code."""
    try:
        model_name = model_data.get('model_name', 'GNN_Model')
        
        code = f'''#!/usr/bin/env python3
"""
Fallback DisCoPy code for {model_name}
"""

from discopy import *

def create_{model_name.lower().replace('-', '_')}_fallback_diagram():
    """Create a simple fallback diagram."""
    
    # Define basic types
    A = Ty('A')
    B = Ty('B')
    
    # Create simple box
    f = Box('f', A, B)
    
    # Create simple diagram
    diagram = f
    
    return diagram

if __name__ == "__main__":
    diagram = create_{model_name.lower().replace('-', '_')}_fallback_diagram()
    print("Fallback diagram created:")
    print(diagram)
'''
        
        return code
        
    except Exception as e:
        return f"# Error generating DisCoPy fallback code: {e}"

def create_active_inference_diagram():
    """Create a DisCoPy diagram representing Active Inference."""
    
    code = '''#!/usr/bin/env python3
"""
Active Inference DisCoPy Diagram
"""

from discopy import *
import numpy as np

def create_active_inference_diagram():
    """Create a DisCoPy diagram representing Active Inference."""
    
    # Define types
    State = Ty('State')
    Observation = Ty('Observation')
    Action = Ty('Action')
    Belief = Ty('Belief')
    
    # Create boxes for Active Inference components
    generative_model = Box('generative_model', State, Observation)
    inference = Box('inference', Observation, Belief)
    policy_selection = Box('policy_selection', Belief, Action)
    action_execution = Box('action_execution', Action, State)
    
    # Create the Active Inference loop
    diagram = (
        State >> generative_model >> Observation >> 
        inference >> Belief >> policy_selection >> 
        Action >> action_execution >> State
    )
    
    return diagram

def create_free_energy_diagram():
    """Create a diagram representing Free Energy minimization."""
    
    # Define types
    Internal_State = Ty('Internal_State')
    External_State = Ty('External_State')
    Sensory_Input = Ty('Sensory_Input')
    
    # Create boxes
    sensory_encoding = Box('sensory_encoding', External_State, Sensory_Input)
    internal_model = Box('internal_model', Internal_State, Sensory_Input)
    free_energy = Box('free_energy', Sensory_Input @ Sensory_Input, Ty())
    
    # Create diagram
    diagram = (
        External_State >> sensory_encoding >> Sensory_Input >>
        free_energy << Sensory_Input << internal_model << Internal_State
    )
    
    return diagram

if __name__ == "__main__":
    ai_diagram = create_active_inference_diagram()
    fe_diagram = create_free_energy_diagram()
    
    print("Active Inference Diagram:")
    print(ai_diagram)
    print("\\nFree Energy Diagram:")
    print(fe_diagram)
'''
    
    return code
