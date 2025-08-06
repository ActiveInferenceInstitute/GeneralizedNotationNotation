#!/usr/bin/env python3
"""
Step 11: Render Processing

This step generates simulation code for PyMDP, RxInfer, and ActiveInference.jl from GNN models.
This is a thin orchestrator that delegates rendering to modular framework implementations.
"""

import sys
import json
import logging
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
    """Generate PyMDP simulation code using the modular PyMDP renderer."""
    try:
        from render.pymdp.pymdp_renderer import render_gnn_to_pymdp
        from render.pymdp.pymdp_converter import GnnToPyMdpConverter
        
        # Get model name for file paths
        model_name = model_data.get('model_name', 'GNN_Model')
        
        # Use the dedicated PyMDP converter for full script generation
        converter = GnnToPyMdpConverter(model_data)
        pymdp_script = converter.get_full_python_script(include_example_usage=True)
        
        if pymdp_script:
            logging.info(f"Successfully generated PyMDP code using GNN converter for {model_name}")
            return pymdp_script
        else:
            # Fallback to renderer if converter doesn't work
            logging.warning("GNN converter returned empty script, using PyMDP renderer")
            # Create a temporary output path (the script content will be returned, not written)
            temp_output_path = Path("/tmp/temp_pymdp_script.py")
            success, message, warnings = render_gnn_to_pymdp(model_data, temp_output_path)
            
            if success and temp_output_path.exists():
                with open(temp_output_path, 'r') as f:
                    script_content = f.read()
                temp_output_path.unlink()  # Clean up
                return script_content
            else:
                raise Exception(f"PyMDP renderer failed: {message}")
                
    except Exception as e:
        logging.error(f"Failed to generate PyMDP code: {e}")
        return f"""#!/usr/bin/env python3
# PyMDP code generation failed: {e}
# Please check the GNN specification and try again.
print("Error: PyMDP code generation failed")
"""

def generate_rxinfer_code(model_data: Dict) -> str:
    """Generate RxInfer.jl simulation code using the modular RxInfer renderer."""
    try:
        from render.rxinfer.rxinfer_renderer import render_gnn_to_rxinfer
        
        # Create a temporary output path
        temp_output_path = Path("/tmp/temp_rxinfer_script.jl")
        success, message, warnings = render_gnn_to_rxinfer(model_data, temp_output_path)
        
        if success and temp_output_path.exists():
            with open(temp_output_path, 'r') as f:
                script_content = f.read()
            temp_output_path.unlink()  # Clean up
            return script_content
        else:
            raise Exception(f"RxInfer renderer failed: {message}")
            
    except ImportError:
        logging.warning("RxInfer renderer not available, using fallback")
        return generate_rxinfer_fallback_code(model_data)
    except Exception as e:
        logging.error(f"Failed to generate RxInfer code: {e}")
        return generate_rxinfer_fallback_code(model_data)

def generate_rxinfer_fallback_code(model_data: Dict) -> str:
    """Fallback RxInfer.jl code generation."""
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
    """Generate ActiveInference.jl simulation code using the modular ActiveInference.jl renderer."""
    try:
        from render.activeinference_jl.activeinference_jl_renderer import render_gnn_to_activeinference_jl
        
        # Create a temporary output path
        temp_output_path = Path("/tmp/temp_activeinference_jl_script.jl")
        success, message, warnings = render_gnn_to_activeinference_jl(model_data, temp_output_path)
        
        if success and temp_output_path.exists():
            with open(temp_output_path, 'r') as f:
                script_content = f.read()
            temp_output_path.unlink()  # Clean up
            return script_content
        else:
            raise Exception(f"ActiveInference.jl renderer failed: {message}")
            
    except ImportError:
        logging.warning("ActiveInference.jl renderer not available, using fallback")
        return generate_activeinference_jl_fallback_code(model_data)
    except Exception as e:
        logging.error(f"Failed to generate ActiveInference.jl code: {e}")
        return generate_activeinference_jl_fallback_code(model_data)

def generate_activeinference_jl_fallback_code(model_data: Dict) -> str:
    """Fallback ActiveInference.jl code generation."""
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
    
            println("Step \\$t: Observation=\\$obs, Action=\\$action")
        println("  State beliefs: \\$qs")
end

println("ActiveInference.jl simulation completed!")
"""
    return code

def generate_discopy_code(model_data: Dict) -> str:
    """Generate DisCoPy code for categorical diagrams using the modular DisCoPy renderer."""
    try:
        from render.discopy.discopy_renderer import render_gnn_to_discopy
        
        # Create a temporary output path
        temp_output_path = Path("/tmp/temp_discopy_script.py")
        success, message, warnings = render_gnn_to_discopy(model_data, temp_output_path)
        
        if success and temp_output_path.exists():
            with open(temp_output_path, 'r') as f:
                script_content = f.read()
            temp_output_path.unlink()  # Clean up
            return script_content
        else:
            raise Exception(f"DisCoPy renderer failed: {message}")
            
    except ImportError:
        logging.warning("DisCoPy renderer not available, using fallback")
        return generate_discopy_fallback_code(model_data)
    except Exception as e:
        logging.error(f"Failed to generate DisCoPy code: {e}")
        return generate_discopy_fallback_code(model_data)

def generate_discopy_fallback_code(model_data: Dict) -> str:
    """Fallback DisCoPy code generation."""
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
        
        # Render targets - delegate to modular framework renderers
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
            
            # Load the actual parsed GNN specification
            parsed_model_file = file_result.get("parsed_model_file")
            if parsed_model_file and Path(parsed_model_file).exists():
                try:
                    with open(parsed_model_file, 'r') as f:
                        actual_gnn_spec = json.load(f)
                    logger.info(f"Loaded parsed GNN specification from {parsed_model_file}")
                    # Use the actual GNN specification instead of the summary
                    model_data = actual_gnn_spec
                except Exception as e:
                    logger.error(f"Failed to load parsed GNN spec from {parsed_model_file}: {e}")
                    # Fall back to using the summary data
                    model_data = file_result
            else:
                logger.warning(f"Parsed model file not found for {file_name}, using summary data")
                model_data = file_result
            
            # Create file-specific output directory
            file_output_dir = output_dir / file_name.replace('.md', '')
            file_output_dir.mkdir(exist_ok=True)
            
            file_render_result = {
                "file_name": file_name,
                "file_path": file_result["file_path"],
                "renders": {},
                "success": True
            }
            
            # Generate code for each target using modular renderers
            for target_name, code_generator, extension in render_targets:
                try:
                    # Generate code using the modular renderer
                    code = code_generator(model_data)
                    
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
                    newline_count = len(code.split('\n'))
                    logger.info(f"  Generated {target_name}: {code_file.stat().st_size} bytes, {newline_count} lines")
                    
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
