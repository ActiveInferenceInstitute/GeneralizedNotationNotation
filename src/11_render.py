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
    """Generate PyMDP simulation code using the pipeline's PyMDP renderer."""
    try:
        from render import render_gnn_spec
        
        # Use the pipeline's PyMDP renderer to generate proper simulation code
        success, message, warnings = render_gnn_spec(
            gnn_spec=model_data,
            target='pymdp',
            output_directory='temp_render',
            options={'embedded_mode': True}
        )
        
        if success:
            # Read the generated code
            from pathlib import Path
            model_name = model_data.get('model_name', 'GNN_Model')
            temp_file = Path('temp_render') / f"{model_name}_pymdp_simulation.py"
            
            if temp_file.exists():
                with open(temp_file, 'r') as f:
                    code = f.read()
                # Clean up temp file
                temp_file.unlink()
                temp_file.parent.rmdir()
                return code
            
    except Exception as e:
        logging.warning(f"Failed to use pipeline PyMDP renderer: {e}, falling back to simple generation")
    
    # Fallback to simple code generation
    model_name = model_data.get('model_name', 'Unknown')
    num_states = model_data.get('model_parameters', {}).get('num_hidden_states', 3)
    num_obs = model_data.get('model_parameters', {}).get('num_obs', 3) 
    num_actions = model_data.get('model_parameters', {}).get('num_actions', 3)
    
    code = f"""#!/usr/bin/env python3
# PyMDP Active Inference Simulation
# Generated from GNN Model: {model_name}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 
# This is a simplified fallback implementation.
# For full pipeline integration, use the execute/pymdp module.

import sys
from pathlib import Path

# Add src to path for pipeline integration
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.execute.pymdp import execute_pymdp_simulation
    
    def main():
        \"\"\"Run PyMDP simulation using pipeline integration.\"\"\"
        
        # GNN specification (embedded)
        gnn_spec = {model_data}
        
        # Output directory
        output_dir = Path("output") / "pymdp_simulations" / "{model_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run simulation using pipeline
        success, results = execute_pymdp_simulation(
            gnn_spec=gnn_spec,
            output_dir=output_dir
        )
        
        if success:
            print(f"✓ Simulation completed successfully!")
            print(f"Output: {{results.get('output_directory')}}")
        else:
            print(f"✗ Simulation failed: {{results.get('error')}}")
        
        return 0 if success else 1

except ImportError:
    # Fallback for basic PyMDP usage
    import numpy as np
    
    def main():
        \"\"\"Fallback PyMDP simulation.\"\"\"
        try:
            from pymdp import utils
            from pymdp.agent import Agent
            
            print("Running basic PyMDP simulation...")
            
            # Model parameters from GNN
            num_states = {num_states}
            num_obs = {num_obs}
            num_actions = {num_actions}
            
            # Create simple POMDP matrices
            A = utils.obj_array(1)
            A[0] = np.eye(num_obs, num_states)
            A[0] = utils.norm_dist(A[0])
            
            B = utils.obj_array(1) 
            B[0] = np.random.uniform(0.1, 1.0, (num_states, num_states, num_actions))
            B[0] = utils.norm_dist(B[0])
            
            C = utils.obj_array(1)
            C[0] = np.random.uniform(-1, 1, num_obs)
            
            D = utils.obj_array(1)
            D[0] = np.ones(num_states) / num_states
            
            # Create agent
            agent = Agent(A=A, B=B, C=C, D=D)
            
            # Simple simulation
            for t in range(10):
                obs = np.random.choice(num_obs)
                qs = agent.infer_states([obs])
                action = agent.sample_action()
                print(f"Step {{t}}: obs={{obs}}, action={{action[0]}}")
            
            print("Basic simulation completed!")
            return 0
            
        except ImportError:
            print("PyMDP not available. Please install: pip install inferactively-pymdp")
            return 1

if __name__ == "__main__":
    sys.exit(main())
"""
    return code

def generate_rxinfer_code(model_data: Dict) -> str:
    """Generate RxInfer.jl simulation code."""
    # Calculate dimensions
    state_vars = [var for var in model_data.get('variables', []) if 'state' in var.get('type', '')]
    obs_vars = [var for var in model_data.get('variables', []) if 'observation' in var.get('type', '')]
    action_vars = [var for var in model_data.get('variables', []) if 'action' in var.get('type', '')]
    
    num_states = max([max(var.get('dimensions', [1])) for var in state_vars]) if state_vars else 3
    num_obs = max([max(var.get('dimensions', [1])) for var in obs_vars]) if obs_vars else 3
    num_controls = max([max(var.get('dimensions', [1])) for var in action_vars]) if action_vars else 3
    
    code = f"""# RxInfer.jl Active Inference Simulation
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
@model function active_inference_model(n_steps)
    # Priors
    s_prior ~ Dirichlet(ones(num_states))
    A ~ MatrixDirichlet(ones(num_obs, num_states))
    B ~ ArrayDirichlet(ones(num_states, num_states, num_controls))
    C ~ Normal(0, 1)
    
    # State sequence
    s = randomvar(n_steps)
    o = datavar(Vector{{Float64}}, n_steps)
    
    # Initial state
    s[1] ~ Categorical(s_prior)
    
    # State transitions and observations
    for t in 2:n_steps
        s[t] ~ Categorical(B[:, :, 1])  # Assuming single action for now
        o[t] ~ Normal(A * s[t], 0.1)
    end
end

# Inference
n_steps = 10
results = inference(
    model = active_inference_model(n_steps),
    data = (o = [randn(num_obs) for _ in 1:n_steps],),
    initmarginals = (s = Categorical(ones(num_states) / num_states),),
    returnvars = (s = KeepLast(),)
)

println("RxInfer.jl simulation completed!")
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
                    logger.info(f"  Generated {target_name}: {code_file.stat().st_size} bytes, {len(code.split('\n'))} lines")
                    
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
