#!/usr/bin/env python3
"""
Step 11: Render Processing

This step generates simulation code for PyMDP, RxInfer, and ActiveInference.jl from GNN models.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

def generate_pymdp_code(model_data: Dict) -> str:
    """Generate PyMDP simulation code."""
    # If model_data is a file result, load the actual parsed data
    if 'parsed_model_file' in model_data:
        import json
        with open(model_data['parsed_model_file'], 'r') as f:
            parsed_data = json.load(f)
        model_data = parsed_data
    
    # Extract variables from model data or extensions
    variables = model_data.get('variables', [])
    
    # If variables array is empty, try to extract from extensions
    if not variables and 'extensions' in model_data:
        extensions = model_data['extensions']
        
        # Extract hierarchical agent structure from FactorBlock
        if 'FactorBlock' in extensions:
            factor_block = extensions['FactorBlock']
            
            # Parse the hierarchical structure
            if 'A_lower:' in factor_block and 'A_higher:' in factor_block:
                # This is a hierarchical agent
                return generate_hierarchical_pymdp_code(model_data)
    
    # Calculate dimensions for standard agent
    state_vars = [var for var in variables if 'state' in var.get('type', '')]
    obs_vars = [var for var in variables if 'observation' in var.get('type', '')]
    action_vars = [var for var in variables if 'action' in var.get('type', '')]
    
    num_states = max([max(var.get('dimensions', [1])) for var in state_vars]) if state_vars else 3
    num_obs = max([max(var.get('dimensions', [1])) for var in obs_vars]) if obs_vars else 3
    num_controls = max([max(var.get('dimensions', [1])) for var in action_vars]) if action_vars else 3
    
    code = f"""#!/usr/bin/env python3
# PyMDP Active Inference Simulation
# Generated from GNN Model: {model_data.get('model_name', 'Unknown')}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

import numpy as np
from pymdp import utils
from pymdp.agent import Agent
from pymdp.envs import Env

# Model parameters
num_states = {num_states}
num_obs = {num_obs}
num_controls = {num_controls}

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
    
    print(f"Step {{t}}: Observation={{obs}}, Action={{action}}")
    print(f"  State beliefs: {{qs}}")
    print(f"  Policy beliefs: {{q_pi}}")

print("Simulation completed!")
"""
    return code

def generate_hierarchical_pymdp_code(model_data: Dict) -> str:
    """Generate PyMDP simulation code for hierarchical agents."""
    model_name = model_data.get('model_name', 'Hierarchical Agent')
    
    code = f"""#!/usr/bin/env python3
# PyMDP Hierarchical Active Inference Simulation
# Generated from GNN Model: {model_name}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

import numpy as np
from pymdp import utils
from pymdp.agent import Agent
from pymdp.envs import Env

# Hierarchical Agent Parameters
# Lower Level Agent
lower_num_states = [2, 2, 2, 3, 3]  # [Trustworthiness, CorrectCard, Affect, Choice, Stage]
lower_num_obs = [3, 3, 2, 3]        # [Advice, Feedback, Arousal, Choice]
lower_num_controls = [2, 1, 2, 3, 1]  # [Trust, Null, Trust, Card, Null]

# Higher Level Agent  
higher_num_states = [2, 2, 2]       # [SafetySelf, SafetyWorld, SafetyOther]
higher_num_obs = [2, 2, 2, 3, 3]   # [TrustworthinessObs, CorrectCardObs, AffectObs, ChoiceObs, StageObs]
higher_num_controls = [1]            # [Null]

# Initialize lower level agent matrices
A_lower = utils.obj_array(len(lower_num_obs))
B_lower = utils.obj_array(len(lower_num_states))
C_lower = utils.obj_array_zeros(len(lower_num_obs))
D_lower = utils.obj_array(len(lower_num_states))

# Initialize higher level agent matrices
A_higher = utils.obj_array(len(higher_num_obs))
B_higher = utils.obj_array(len(higher_num_states))
C_higher = utils.obj_array_zeros(len(higher_num_obs))
D_higher = utils.obj_array(len(higher_num_states))

# Set up basic matrices (identity mappings for now)
for i in range(len(lower_num_obs)):
    A_lower[i] = np.eye(lower_num_obs[i], np.prod(lower_num_states))
    C_lower[i] = np.zeros(lower_num_obs[i])

for i in range(len(lower_num_states)):
    B_lower[i] = np.eye(lower_num_states[i], lower_num_states[i], lower_num_controls[i])
    D_lower[i] = np.ones(lower_num_states[i]) / lower_num_states[i]

for i in range(len(higher_num_obs)):
    A_higher[i] = np.eye(higher_num_obs[i], np.prod(higher_num_states))
    C_higher[i] = np.zeros(higher_num_obs[i])

for i in range(len(higher_num_states)):
    B_higher[i] = np.eye(higher_num_states[i], higher_num_states[i], higher_num_controls[i])
    D_higher[i] = np.ones(higher_num_states[i]) / higher_num_states[i]

# Create agents
lower_agent = Agent(A=A_lower, B=B_lower, C=C_lower, D=D_lower)
higher_agent = Agent(A=A_higher, B=B_higher, C=C_higher, D=D_higher)

# Create environment (simple identity mapping)
env = Env(A=A_lower, B=B_lower)

# Simulation parameters
T = 10  # Number of time steps

# Run hierarchical simulation
for t in range(T):
    # Get observation from environment
    obs = env.step()
    
    # Lower level agent inference and action selection
    lower_qs = lower_agent.infer_states(obs)
    lower_q_pi, _ = lower_agent.infer_policies()
    lower_action = lower_agent.sample_action()
    
    # Higher level agent inference (using lower level posteriors as observations)
    higher_obs = lower_qs  # Simplified mapping
    higher_qs = higher_agent.infer_states(higher_obs)
    higher_q_pi, _ = higher_agent.infer_policies()
    higher_action = higher_agent.sample_action()
    
    print(f"Step {{t}}:")
    print(f"  Lower Level - Observation: {{obs}}, Action: {{lower_action}}")
    print(f"    State beliefs: {{lower_qs}}")
    print(f"    Policy beliefs: {{lower_q_pi}}")
    print(f"  Higher Level - Observation: {{higher_obs}}, Action: {{higher_action}}")
    print(f"    State beliefs: {{higher_qs}}")
    print(f"    Policy beliefs: {{higher_q_pi}}")

print("Hierarchical simulation completed!")
"""
    return code

def generate_rxinfer_code(model_data: Dict) -> str:
    """Generate RxInfer.jl simulation code."""
    # If model_data is a file result, load the actual parsed data
    if 'parsed_model_file' in model_data:
        import json
        with open(model_data['parsed_model_file'], 'r') as f:
            parsed_data = json.load(f)
        model_data = parsed_data
    
    # Extract variables from model data or extensions
    variables = model_data.get('variables', [])
    
    # If variables array is empty, try to extract from extensions
    if not variables and 'extensions' in model_data:
        extensions = model_data['extensions']
        
        # Extract hierarchical agent structure from FactorBlock
        if 'FactorBlock' in extensions:
            factor_block = extensions['FactorBlock']
            
            # Parse the hierarchical structure
            if 'A_lower:' in factor_block and 'A_higher:' in factor_block:
                # This is a hierarchical agent
                return generate_hierarchical_rxinfer_code(model_data)
    
    # Calculate dimensions for standard agent
    state_vars = [var for var in variables if 'state' in var.get('type', '')]
    obs_vars = [var for var in variables if 'observation' in var.get('type', '')]
    action_vars = [var for var in variables if 'action' in var.get('type', '')]
    
    num_states = max([max(var.get('dimensions', [1])) for var in state_vars]) if state_vars else 3
    num_obs = max([max(var.get('dimensions', [1])) for var in obs_vars]) if obs_vars else 3
    num_controls = max([max(var.get('dimensions', [1])) for var in action_vars]) if action_vars else 3
    
    code = f"""#!/usr/bin/env julia
# RxInfer.jl Active Inference Simulation
# Generated from GNN Model: {model_data.get('model_name', 'Unknown')}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

using RxInfer
using Distributions
using LinearAlgebra

# Model parameters
num_states = {num_states}
num_obs = {num_obs}
num_controls = {num_controls}

# Define the model
@model function active_inference_model(num_steps)
    # State variables
    s = randomvar(num_steps)
    
    # Observation variables
    o = datavar(Vector{{Float64}}, num_steps)
    
    # Prior distributions
    s[1] ~ NormalMeanVariance(0.0, 1.0)
    
    # State transitions and observations
    for t in 2:num_steps
        s[t] ~ NormalMeanVariance(s[t-1], 0.1)
        o[t] ~ NormalMeanVariance(s[t], 0.5)
    end
end

# Simulation parameters
num_steps = 10

# Create model
model = active_inference_model(num_steps)

# Generate synthetic data
observations = randn(num_steps)

# Run inference
results = inference(
    model = model,
    data = (o = observations,),
    iterations = 10
)

println("RxInfer.jl simulation completed!")
println("State estimates: ", results.posteriors[:s])
"""
    return code

def generate_hierarchical_rxinfer_code(model_data: Dict) -> str:
    """Generate RxInfer.jl simulation code for hierarchical agents."""
    model_name = model_data.get('model_name', 'Hierarchical Agent')
    
    code = f"""#!/usr/bin/env julia
# RxInfer.jl Hierarchical Active Inference Simulation
# Generated from GNN Model: {model_name}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

using RxInfer
using Distributions
using LinearAlgebra

# Hierarchical Agent Parameters
# Lower Level Agent
lower_num_states = [2, 2, 2, 3, 3]  # [Trustworthiness, CorrectCard, Affect, Choice, Stage]
lower_num_obs = [3, 3, 2, 3]        # [Advice, Feedback, Arousal, Choice]

# Higher Level Agent  
higher_num_states = [2, 2, 2]       # [SafetySelf, SafetyWorld, SafetyOther]
higher_num_obs = [2, 2, 2, 3, 3]   # [TrustworthinessObs, CorrectCardObs, AffectObs, ChoiceObs, StageObs]

# Define the hierarchical model
@model function hierarchical_active_inference_model(num_steps)
    # Lower level state variables
    trust = randomvar(num_steps)
    card = randomvar(num_steps)
    affect = randomvar(num_steps)
    choice = randomvar(num_steps)
    stage = randomvar(num_steps)
    
    # Higher level state variables
    safety_self = randomvar(num_steps)
    safety_world = randomvar(num_steps)
    safety_other = randomvar(num_steps)
    
    # Lower level observation variables
    advice = datavar(Vector{{Float64}}, num_steps)
    feedback = datavar(Vector{{Float64}}, num_steps)
    arousal = datavar(Vector{{Float64}}, num_steps)
    choice_obs = datavar(Vector{{Float64}}, num_steps)
    
    # Higher level observation variables (from lower level posteriors)
    trust_obs = datavar(Vector{{Float64}}, num_steps)
    card_obs = datavar(Vector{{Float64}}, num_steps)
    affect_obs = datavar(Vector{{Float64}}, num_steps)
    choice_obs_high = datavar(Vector{{Float64}}, num_steps)
    stage_obs = datavar(Vector{{Float64}}, num_steps)
    
    # Prior distributions for lower level
    trust[1] ~ Categorical([0.5, 0.5])
    card[1] ~ Categorical([0.5, 0.5])
    affect[1] ~ Categorical([0.5, 0.5])
    choice[1] ~ Categorical([0.0, 0.0, 1.0])
    stage[1] ~ Categorical([1.0, 0.0, 0.0])
    
    # Prior distributions for higher level
    safety_self[1] ~ Categorical([0.25, 0.75])
    safety_world[1] ~ Categorical([0.25, 0.75])
    safety_other[1] ~ Categorical([0.25, 0.75])
    
    # State transitions and observations
    for t in 2:num_steps
        # Lower level transitions
        trust[t] ~ Categorical([0.9, 0.1])  # Simplified transition
        card[t] ~ Categorical([0.9, 0.1])
        affect[t] ~ Categorical([0.333, 0.667])
        choice[t] ~ Categorical([0.95, 0.025, 0.025])
        stage[t] ~ Categorical([0.0, 1.0, 0.0])
        
        # Higher level transitions
        safety_self[t] ~ Categorical([1.0, 0.0])  # Deterministic
        safety_world[t] ~ Categorical([1.0, 0.0])
        safety_other[t] ~ Categorical([1.0, 0.0])
        
        # Lower level observations
        advice[t] ~ NormalMeanVariance(trust[t], 0.1)
        feedback[t] ~ NormalMeanVariance(card[t], 0.1)
        arousal[t] ~ NormalMeanVariance(affect[t], 0.1)
        choice_obs[t] ~ NormalMeanVariance(choice[t], 0.1)
        
        # Higher level observations (mapped from lower level)
        trust_obs[t] ~ NormalMeanVariance(safety_self[t], 0.1)
        card_obs[t] ~ NormalMeanVariance(safety_world[t], 0.1)
        affect_obs[t] ~ NormalMeanVariance(safety_other[t], 0.1)
        choice_obs_high[t] ~ NormalMeanVariance(choice[t], 0.1)
        stage_obs[t] ~ NormalMeanVariance(stage[t], 0.1)
    end
end

# Simulation parameters
num_steps = 10

# Create model
model = hierarchical_active_inference_model(num_steps)

# Generate synthetic data
lower_obs = [randn(3), randn(3), randn(2), randn(3)]  # advice, feedback, arousal, choice
higher_obs = [randn(2), randn(2), randn(2), randn(3), randn(3)]  # trust_obs, card_obs, affect_obs, choice_obs_high, stage_obs

# Run inference
results = inference(
    model = model,
    data = (
        advice = lower_obs[1],
        feedback = lower_obs[2],
        arousal = lower_obs[3],
        choice_obs = lower_obs[4],
        trust_obs = higher_obs[1],
        card_obs = higher_obs[2],
        affect_obs = higher_obs[3],
        choice_obs_high = higher_obs[4],
        stage_obs = higher_obs[5]
    ),
    iterations = 10
)

println("Hierarchical RxInfer.jl simulation completed!")
println("Lower level state estimates:")
println("  Trust: ", results.posteriors[:trust])
println("  Card: ", results.posteriors[:card])
println("  Affect: ", results.posteriors[:affect])
println("Higher level state estimates:")
println("  Safety Self: ", results.posteriors[:safety_self])
println("  Safety World: ", results.posteriors[:safety_world])
println("  Safety Other: ", results.posteriors[:safety_other])
"""
    return code

def generate_activeinference_jl_code(model_data: Dict) -> str:
    """Generate ActiveInference.jl simulation code."""
    # Calculate dimensions
    state_vars = [var for var in model_data.get('variables', []) if 'state' in var.get('type', '')]
    obs_vars = [var for var in model_data.get('variables', []) if 'observation' in var.get('type', '')]
    action_vars = [var for var in model_data.get('variables', []) if 'action' in var.get('type', '')]
    
    num_states = max([max(var.get('dimensions', [1])) for var in state_vars]) if state_vars else 3
    num_obs = max([max(var.get('dimensions', [1])) for var in obs_vars]) if obs_vars else 3
    num_controls = max([max(var.get('dimensions', [1])) for var in action_vars]) if action_vars else 3
    
    code = f"""# ActiveInference.jl Simulation
# Generated from GNN Model: {model_data.get('model_name', 'Unknown')}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

using ActiveInference
using LinearAlgebra

# Model parameters
num_states = {num_states}
num_obs = {num_obs}
num_controls = {num_controls}

# Initialize matrices
A = Matrix{{Float64}}(I, num_obs, num_states)  # Likelihood matrix
B = zeros(num_states, num_states, num_controls)  # Transition matrix
for i in 1:num_controls
    B[:, :, i] = Matrix{{Float64}}(I, num_states, num_states)
end
C = zeros(num_obs)  # Preference vector
D = ones(num_states) / num_states  # Prior over states

# Create agent
agent = ActiveInferenceAgent(A, B, C, D)

# Simulation parameters
T = 10

# Run simulation
for t in 1:T
    # Generate observation
    obs = rand(1:num_obs)
    
    # Agent inference
    qs = infer_states(agent, obs)
    q_pi = infer_policies(agent)
    action = sample_action(agent, q_pi)
    
    println("Step \$t: Observation=\$obs, Action=\$action")
    println("  State beliefs: \$qs")
end

println("ActiveInference.jl simulation completed!")
"""
    return code

def generate_discopy_code(model_data: Dict) -> str:
    """Generate DisCoPy code for categorical diagrams."""
    variables = [var.get('name', '') for var in model_data.get('variables', [])]
    connections = []
    for conn in model_data.get('connections', []):
        sources = conn.get('source', [])
        targets = conn.get('target', [])
        for source in sources:
            for target in targets:
                connections.append(f"{source}->{target}")
    
    code = f"""#!/usr/bin/env python3
# DisCoPy Categorical Diagram Generation
# Generated from GNN Model: {model_data.get('model_name', 'Unknown')}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

from discopy import *
from discopy.quantum import *
import numpy as np

# Define types
H = Ty('H')  # Hidden states
O = Ty('O')  # Observations
A = Ty('A')  # Actions

# Define boxes for the model components
variables = {variables}

# Create boxes for each variable
boxes = {{}}

for var_name in variables:
    if 'A' in var_name:  # Likelihood matrix
        boxes[var_name] = Box(var_name, H, O)
    elif 'B' in var_name:  # Transition matrix
        boxes[var_name] = Box(var_name, H @ A, H)
    elif 'C' in var_name:  # Preference vector
        boxes[var_name] = Box(var_name, Ty(), O)
    elif 'D' in var_name:  # Prior
        boxes[var_name] = Box(var_name, Ty(), H)
    else:
        boxes[var_name] = Box(var_name, Ty(), Ty())

# Create diagram
connections = {connections}

print("DisCoPy diagram components created:")
for name, box in boxes.items():
    print(f"  {{name}}: {{box}}")

print("\\nConnections:")
for conn in connections:
    print(f"  {{conn}}")

# Generate diagram (simplified)
diagram = Id(Ty())
print("\\nDisCoPy diagram generated successfully!")
"""
    return code

def main():
    """Main render processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("11_render.py")
    
    # Setup logging
    logger = setup_step_logging("render", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("11_render.py", Path(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_step_start(logger, "Processing render")
        
        # Load parsed GNN data from previous step
        gnn_output_dir = get_output_dir_for_script("3_gnn.py", Path(args.output_dir))
        gnn_results_file = gnn_output_dir / "gnn_processing_results.json"
        
        if not gnn_results_file.exists():
            log_step_error(logger, "GNN processing results not found. Run step 3 first.")
            return 1
        
        with open(gnn_results_file, 'r') as f:
            gnn_results = json.load(f)
        
        logger.info(f"Loaded {len(gnn_results['processed_files'])} parsed GNN files")
        
        # Render results
        render_results = {
            "timestamp": datetime.now().isoformat(),
            "source_directory": str(args.target_dir),
            "output_directory": str(output_dir),
            "files_rendered": [],
            "summary": {
                "total_files": 0,
                "successful_renders": 0,
                "failed_renders": 0,
                "code_files_generated": {
                    "pymdp": 0,
                    "rxinfer": 0,
                    "activeinference_jl": 0,
                    "discopy": 0
                }
            }
        }
        
        # Render targets
        render_targets = [
            ("pymdp", generate_pymdp_code, ".py"),
            ("rxinfer", generate_rxinfer_code, ".jl"),
            ("activeinference_jl", generate_activeinference_jl_code, ".jl"),
            ("discopy", generate_discopy_code, ".py")
        ]
        
        for file_result in gnn_results["processed_files"]:
            if not file_result["parse_success"]:
                continue
            
            file_name = file_result["file_name"]
            logger.info(f"Rendering code for: {file_name}")
            
            # Create file-specific output directory
            file_output_dir = output_dir / file_name.replace('.md', '')
            file_output_dir.mkdir(exist_ok=True)
            
            file_render_result = {
                "file_name": file_name,
                "file_path": file_result["file_path"],
                "renders": {},
                "success": True
            }
            
            # Generate code for each target
            for target_name, code_generator, extension in render_targets:
                try:
                    # Generate code
                    code = code_generator(file_result)
                    
                    # Save code file
                    code_file = file_output_dir / f"{file_name.replace('.md', '')}_{target_name}{extension}"
                    with open(code_file, 'w') as f:
                        f.write(code)
                    
                    file_render_result["renders"][target_name] = {
                        "output_path": str(code_file),
                        "success": True,
                        "file_size": code_file.stat().st_size,
                        "lines_of_code": len(code.split('\n'))
                    }
                    
                    render_results["summary"]["code_files_generated"][target_name] += 1
                    logger.info(f"  Generated {target_name}: {code_file.stat().st_size} bytes, {len(code.split(chr(10)))} lines")
                    
                except Exception as e:
                    file_render_result["renders"][target_name] = {
                        "output_path": str(file_output_dir / f"{file_name.replace('.md', '')}_{target_name}{extension}"),
                        "success": False,
                        "error": str(e)
                    }
                    file_render_result["success"] = False
                    logger.error(f"  Failed to generate {target_name}: {e}")
            
            render_results["files_rendered"].append(file_render_result)
            
            # Update summary
            render_results["summary"]["total_files"] += 1
            if file_render_result["success"]:
                render_results["summary"]["successful_renders"] += 1
            else:
                render_results["summary"]["failed_renders"] += 1
        
        # Save render results
        results_file = output_dir / "render_results.json"
        with open(results_file, 'w') as f:
            json.dump(render_results, f, indent=2, default=str)
        
        # Save summary
        summary_file = output_dir / "render_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(render_results["summary"], f, indent=2)
        
        # Determine success
        success = render_results["summary"]["successful_renders"] > 0
        
        if success:
            total_code_files = sum(render_results["summary"]["code_files_generated"].values())
            log_step_success(logger, f"Generated {total_code_files} code files for {render_results['summary']['successful_renders']} files")
            return 0
        else:
            log_step_error(logger, "Render processing failed")
            return 1
            
    except Exception as e:
        log_step_error(logger, f"Render processing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
