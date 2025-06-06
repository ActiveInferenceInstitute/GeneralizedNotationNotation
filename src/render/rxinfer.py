"""
Module for rendering GNN specifications to RxInfer.jl configuration files.

This module acts as a top-level caller for the rxinfer package, which processes
GNN (Generalized Notation Notation) files and generates RxInfer.jl configuration files.
"""

import logging
import os
import sys
import toml  # Import toml for TOML file generation
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Ensure that the src directory is in the Python path for importing rxinfer
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from rxinfer.gnn_parser import parse_gnn_file
from rxinfer.config_generator import generate_rxinfer_config
from rxinfer.toml_generator import generate_rxinfer_toml_config  # New import for TOML generation

logger = logging.getLogger(__name__)

def render_gnn_to_rxinfer_jl(
    gnn_spec: Dict[str, Any],
    output_script_path: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Renders a GNN specification to a RxInfer.jl configuration file.
    
    Args:
        gnn_spec: Dictionary containing the GNN specification
        output_script_path: Path to the output configuration file
        options: Optional additional options for rendering
        
    Returns:
        Tuple containing (success, message, list of artifact URIs)
    """
    options = options or {}
    logger.info(f"Rendering GNN spec to RxInfer.jl configuration '{output_script_path}'.")
    
    try:
        # Ensure the output directory exists
        output_script_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate the configuration file
        success = generate_rxinfer_config(gnn_spec, output_script_path)
        
        if success:
            logger.info(f"Successfully wrote RxInfer.jl configuration to {output_script_path}")
            # Convert relative path to absolute for URI
            abs_path = output_script_path.absolute()
            return True, f"Successfully rendered to RxInfer.jl: {output_script_path.name}", [abs_path.as_uri()]
        else:
            logger.error(f"Failed to generate RxInfer.jl configuration")
            return False, f"Failed to generate RxInfer.jl configuration", []
    
    except Exception as e:
        logger.error(f"Error rendering GNN to RxInfer.jl: {e}", exc_info=True)
        return False, f"Error rendering to RxInfer.jl: {str(e)}", []

def render_gnn_to_rxinfer_toml(
    gnn_spec: Dict[str, Any],
    output_config_path: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Renders a GNN specification to a RxInfer TOML configuration file.
    
    Args:
        gnn_spec: Dictionary containing the GNN specification
        output_config_path: Path to the output TOML configuration file
        options: Optional additional options for rendering
        
    Returns:
        Tuple containing (success, message, list of artifact URIs)
    """
    options = options or {}
    logger.info(f"Rendering GNN spec to RxInfer TOML configuration '{output_config_path}'.")
    
    try:
        # Ensure the output directory exists
        output_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Make sure we're using .toml extension for TOML files
        if not output_config_path.suffix.lower() == '.toml':
            original_path = output_config_path
            output_config_path = output_config_path.with_suffix('.toml')
            logger.warning(f"Changed output extension from '{original_path.suffix}' to '.toml' for TOML config: {output_config_path}")
        
        # Generate the TOML configuration
        try:
            # Try to use the dedicated TOML generator function
            success = generate_rxinfer_toml_config(gnn_spec, output_config_path)
        except (ImportError, AttributeError):
            # Fallback to manual TOML generation if the module doesn't exist
            logger.warning("Could not import generate_rxinfer_toml_config, using fallback TOML generation")
            success = _fallback_generate_toml(gnn_spec, output_config_path, options)
        
        if success:
            logger.info(f"Successfully wrote RxInfer TOML configuration to {output_config_path}")
            # Convert relative path to absolute for URI
            abs_path = output_config_path.absolute()
            return True, f"Successfully rendered to RxInfer TOML: {output_config_path.name}", [abs_path.as_uri()]
        else:
            logger.error(f"Failed to generate RxInfer TOML configuration")
            return False, f"Failed to generate RxInfer TOML configuration", []
    
    except Exception as e:
        logger.error(f"Error rendering GNN to RxInfer TOML: {e}", exc_info=True)
        return False, f"Error rendering to RxInfer TOML: {str(e)}", []

def _fallback_generate_toml(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Dict[str, Any]
) -> bool:
    """
    Fallback function to generate a TOML configuration file from a GNN specification.
    This is used when the dedicated toml_generator module is not available.
    
    Args:
        gnn_spec: Dictionary containing the GNN specification
        output_path: Path to the output TOML file
        options: Additional options for rendering
        
    Returns:
        Boolean indicating success
    """
    try:
        # Extract model name and description
        model_name = gnn_spec.get("name", "GNN_Model")
        model_description = gnn_spec.get("description", f"TOML configuration for {model_name}")
        
        # Create a comprehensive TOML structure based on the GNN spec and gold standard structure
        toml_config = {
            "model": {
                "dt": gnn_spec.get("dt", 1.0),
                "gamma": gnn_spec.get("gamma", 1.0),
                "nr_steps": gnn_spec.get("time_steps", 40),
                "nr_iterations": gnn_spec.get("nr_iterations", 350),
                "nr_agents": gnn_spec.get("nr_agents", 4),
                "softmin_temperature": gnn_spec.get("softmin_temperature", 10.0),
                "intermediate_steps": gnn_spec.get("intermediate_steps", 10),
                "save_intermediates": gnn_spec.get("save_intermediates", False)
            },
            "priors": {
                "initial_state_variance": gnn_spec.get("initial_state_variance", 100.0),
                "control_variance": gnn_spec.get("control_variance", 0.1),
                "goal_constraint_variance": gnn_spec.get("goal_constraint_variance", 1e-5),
                "gamma_shape": gnn_spec.get("gamma_shape", 1.5),
                "gamma_scale_factor": gnn_spec.get("gamma_scale_factor", 0.5)
            },
            "visualization": {
                "x_limits": gnn_spec.get("x_limits", [-20, 20]),
                "y_limits": gnn_spec.get("y_limits", [-20, 20]),
                "fps": gnn_spec.get("fps", 15),
                "heatmap_resolution": gnn_spec.get("heatmap_resolution", 100),
                "plot_width": gnn_spec.get("plot_width", 800),
                "plot_height": gnn_spec.get("plot_height", 400),
                "agent_alpha": gnn_spec.get("agent_alpha", 1.0),
                "target_alpha": gnn_spec.get("target_alpha", 0.2),
                "color_palette": gnn_spec.get("color_palette", "tab10")
            }
        }
        
        # Add state space matrices if available
        if "matrices" in gnn_spec:
            matrices = gnn_spec["matrices"]
            toml_config["model"]["matrices"] = {
                "A": matrices.get("A", [
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0]
                ]),
                "B": matrices.get("B", [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 1.0]
                ]),
                "C": matrices.get("C", [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]
                ])
            }
        else:
            # Add default matrices if not present
            toml_config["model"]["matrices"] = {
                "A": [
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                "B": [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 1.0]
                ],
                "C": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]
                ]
            }
        
        # Extract environment definitions
        toml_config["environments"] = {}
        
        # Add standard environment definitions if they aren't in the GNN spec
        standard_environments = {
            "door": {
                "description": "Two parallel walls with a gap between them",
                "obstacles": [
                    {"center": [-40.0, 0.0], "size": [70.0, 5.0]},
                    {"center": [40.0, 0.0], "size": [70.0, 5.0]}
                ]
            },
            "wall": {
                "description": "A single wall obstacle in the center",
                "obstacles": [
                    {"center": [0.0, 0.0], "size": [10.0, 5.0]}
                ]
            },
            "combined": {
                "description": "A combination of walls and obstacles",
                "obstacles": [
                    {"center": [-50.0, 0.0], "size": [70.0, 2.0]},
                    {"center": [50.0, 0.0], "size": [70.0, 2.0]},
                    {"center": [5.0, -1.0], "size": [3.0, 10.0]}
                ]
            }
        }
        
        # Use environments from GNN spec if available, otherwise use standard ones
        if "environments" in gnn_spec:
            for env_name, env_info in gnn_spec["environments"].items():
                toml_config["environments"][env_name] = {
                    "description": env_info.get("description", f"Environment {env_name}"),
                    "obstacles": []
                }
                
                if "obstacles" in env_info:
                    for obstacle in env_info["obstacles"]:
                        toml_config["environments"][env_name]["obstacles"].append({
                            "center": obstacle.get("center", [0.0, 0.0]),
                            "size": obstacle.get("size", [1.0, 1.0])
                        })
        else:
            # Use standard environments if none provided
            toml_config["environments"] = standard_environments
        
        # Add agent configurations
        toml_config["agents"] = []
        
        # Extract agents if available
        if "agents" in gnn_spec:
            for i, agent_info in enumerate(gnn_spec["agents"]):
                agent_config = {
                    "id": agent_info.get("id", i + 1),
                    "radius": agent_info.get("radius", 1.0),
                    "initial_position": agent_info.get("initial_position", [0.0, 0.0]),
                    "target_position": agent_info.get("target_position", [0.0, 0.0])
                }
                toml_config["agents"].append(agent_config)
        else:
            # Add default agents if none provided
            default_agents = [
                {
                    "id": 1,
                    "radius": 2.5,
                    "initial_position": [-4.0, 10.0],
                    "target_position": [-10.0, -10.0]
                },
                {
                    "id": 2,
                    "radius": 1.5,
                    "initial_position": [-10.0, 5.0],
                    "target_position": [10.0, -15.0]
                },
                {
                    "id": 3,
                    "radius": 1.0,
                    "initial_position": [-15.0, -10.0],
                    "target_position": [10.0, 10.0]
                },
                {
                    "id": 4,
                    "radius": 2.5,
                    "initial_position": [0.0, -10.0],
                    "target_position": [-10.0, 15.0]
                }
            ]
            toml_config["agents"] = default_agents
        
        # Add experiment configurations
        toml_config["experiments"] = {
            "seeds": gnn_spec.get("seeds", [42, 123]),
            "results_dir": gnn_spec.get("results_dir", "results"),
            "animation_template": gnn_spec.get("animation_template", "{environment}_{seed}.gif"),
            "control_vis_filename": gnn_spec.get("control_vis_filename", "control_signals.gif"),
            "obstacle_distance_filename": gnn_spec.get("obstacle_distance_filename", "obstacle_distance.png"),
            "path_uncertainty_filename": gnn_spec.get("path_uncertainty_filename", "path_uncertainty.png"),
            "convergence_filename": gnn_spec.get("convergence_filename", "convergence.png")
        }
        
        # Add a header comment to the file
        header_comment = f"# {model_name} Configuration\n\n"
        
        # Write the TOML configuration to file with header
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header_comment)
            toml.dump(toml_config, f)
        
        return True
    
    except Exception as e:
        logger.error(f"Error in fallback TOML generation: {e}", exc_info=True)
        return False

def process_gnn_file(gnn_file_path: Path, output_path: Path, output_format: str = "julia") -> Tuple[bool, str, List[str]]:
    """
    Process a GNN file to generate a RxInfer configuration file.
    
    Args:
        gnn_file_path: Path to the GNN file
        output_path: Path to the output configuration file
        output_format: Format of the output ("julia" or "toml")
        
    Returns:
        Tuple containing (success, message, list of artifact URIs)
    """
    try:
        # Parse the GNN file
        logger.info(f"Parsing GNN file: {gnn_file_path}")
        if not gnn_file_path.exists():
            logger.error(f"GNN file not found: {gnn_file_path}")
            return False, f"GNN file not found: {gnn_file_path}", []
        
        parsed_gnn = parse_gnn_file(gnn_file_path)
        
        # Render the configuration file
        if output_format.lower() == "toml":
            return render_gnn_to_rxinfer_toml(parsed_gnn, output_path)
        else:
            return render_gnn_to_rxinfer_jl(parsed_gnn, output_path)
    
    except Exception as e:
        logger.error(f"Error processing GNN file: {e}", exc_info=True)
        return False, f"Error processing GNN file: {str(e)}", []

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Process command line arguments
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <gnn_file_path> <output_path> [format: julia|toml]")
        sys.exit(1)
    
    gnn_file_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    output_format = sys.argv[3] if len(sys.argv) > 3 else "julia"
    
    success, message, artifacts = process_gnn_file(gnn_file_path, output_path, output_format)
    print(message)
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    test_output_dir = Path("temp_rxinfer_render_test")
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Original linear regression test (can be kept or removed)
    dummy_gnn_data_linear_regression = {
        "name": "LinearRegressionGNN",
        "arguments": ["y_obs", "x_matrix", "sigma_sq_val_arg"], # sigma_sq_val_arg to avoid clash with node
        "nodes": [
            {"id": "beta", "type": "random_variable", "distribution": "Normal", "params": {"mean": 0.0, "variance": 1.0}, "report_posterior": True},
            {"id": "intercept", "type": "random_variable", "distribution": "Normal", "params": {"mean": 0.0, "variance": 10.0}, "report_posterior": True},
            {"id": "sigma_sq_val_node", "type": "constant", "initial_value": "0.1"}, # Node for internal constant
            {"id": "y_obs", "type": "observed_data", "distribution": "Normal", 
             "params": {"mean": "x_matrix * beta + intercept", "variance": "sigma_sq_val_arg"}, # use arg here
             "is_vectorized": True, "dependencies": ["x_matrix", "beta", "intercept", "sigma_sq_val_arg"]
            }
        ],
        # No model_logic, so it will use node-based processing.
        "constraints": [{"type": "mean_field", "factors": [["beta", "intercept"]]}],
        "meta": [{"factor_ref": "Normal", "settings": {"damped": "true"}}], # Ensure boolean is string for _format_params
        "julia_imports": ["using Distributions"] # Example of specific imports
    }
    
    output_script_lr = test_output_dir / "generated_linear_regression_script.jl"
    render_options_lr = {
        "data_bindings": {
            "y_obs": "actual_y_data",      # This is data
            "x_matrix": "actual_x_data",    # This is data
            "sigma_sq_val_arg": "0.05"      # This is a parameter passed to model
        },
        "inference_iterations": 75,
        "calculate_free_energy": True
    }

    success_lr, msg_lr, artifacts_lr = render_gnn_to_rxinfer_jl(
        dummy_gnn_data_linear_regression,
        output_script_lr, 
        options=render_options_lr
    )

    if success_lr:
        logger.info(f"RxInfer.jl LR rendering test successful: {msg_lr}")
        if output_script_lr.exists():
            logger.info(f"--- Generated LR Julia Script ({output_script_lr.name}) ---")
            print(f"\n{output_script_lr.read_text(encoding='utf-8')}\n")
            logger.info(f"--- End of LR Script ---")
    else:
        logger.error(f"RxInfer.jl LR rendering test failed: {msg_lr}")

    # Also test TOML rendering
    output_toml_lr = test_output_dir / "generated_linear_regression_config.toml"
    success_toml_lr, msg_toml_lr, artifacts_toml_lr = render_gnn_to_rxinfer_toml(
        dummy_gnn_data_linear_regression,
        output_toml_lr,
        options=render_options_lr
    )

    if success_toml_lr:
        logger.info(f"RxInfer TOML rendering test successful: {msg_toml_lr}")
        if output_toml_lr.exists():
            logger.info(f"--- Generated TOML Config ({output_toml_lr.name}) ---")
            print(f"\n{output_toml_lr.read_text(encoding='utf-8')}\n")
            logger.info(f"--- End of TOML Config ---")
    else:
        logger.error(f"RxInfer TOML rendering test failed: {msg_toml_lr}")

    # --- HMM Test using model_logic ---
    dummy_gnn_data_hmm_ml = {
        "name": "SimpleHMM_from_Logic",
        "arguments": ["observations", "T", "A", "B", "initial_dist_p"], # A=transition, B=emission
        "nodes": [ 
            # Nodes can still define types or be informative, but model structure comes from model_logic
            {"id": "observations", "type": "observed_data", "description": "Vector of observed states"},
            {"id": "T", "type": "constant", "description": "Time horizon / number of observations"},
            {"id": "A", "type": "constant", "description": "Transition matrix"},
            {"id": "B", "type": "constant", "description": "Emission matrix"},
            {"id": "initial_dist_p", "type": "constant", "description": "Initial state distribution parameters (vector p)"},
            {"id": "s", "type": "random_variable_vector", "description": "Latent state sequence"}
        ],
        "model_logic": [
            {"item_type": "raw_julia", "code": "# Hidden Markov Model implementation from GNN model_logic"},
            {"item_type": "rv_vector_declaration", "name": "s", "size_var": "T", "element_type": "RandomVariable"},
            {"item_type": "assignment", "variable": "s[1]", "distribution": "Categorical", "params": {"p": "initial_dist_p"}},
            {
                "item_type": "loop", "variable": "t", "range_start": 2, "range_end": "T",
                "body": [
                    {"item_type": "assignment", "variable": "s[t]", "distribution": "Categorical", 
                     "params": {"p": "A[s[t-1], :]"}} # Assuming A is passed as matrix
                ]
            },
            {
                "item_type": "loop", "variable": "t", "range_start": 1, "range_end": "T",
                "body": [
                    # observations[t] is the data placeholder from model arguments
                    {"item_type": "assignment", "variable": "observations[t]", "is_observed_data": True, 
                     "distribution": "Categorical", "params": {"p": "B[s[t], :]"}} # Assuming B is passed
                ]
            },
            {"item_type": "return_statement", "values": ["s", "observations"]}
        ],
        "julia_imports": ["using Distributions"], # Explicitly list required packages for the model
        "constraints": { # Example of named constraints
            "name": "MyHMMConstraints", # Optional custom name
            "is_anonymous": False, # Explicitly not anonymous
             "raw_lines": ["q(s) :: MeanField()"] # Raw lines for constraints body
        },
        "meta": { # Example of anonymous meta
            "is_anonymous": True,
            "raw_lines": ["Categorical(p) -> ((p = p ./ sum(p)),)"] # Example meta line
        }
    }

    output_script_hmm_ml = test_output_dir / "generated_hmm_model_logic_script.jl"
    render_options_hmm_ml = {
        "data_bindings": {
            "observations": "my_observed_sequence", # Name of Julia variable holding observations
            "T": "length(my_observed_sequence)",   # Julia expression for T
            "A": "transition_matrix_data",         # Name of Julia variable for transition matrix
            "B": "emission_matrix_data",           # Name of Julia variable for emission matrix
            "initial_dist_p": "initial_probabilities_vector" # Julia variable for initial dist
        },
        "inference_iterations": 100,
        "calculate_free_energy": True
    }

    success_hmm_ml, msg_hmm_ml, artifacts_hmm_ml = render_gnn_to_rxinfer_jl(
        dummy_gnn_data_hmm_ml,
        output_script_hmm_ml,
        options=render_options_hmm_ml
    )

    if success_hmm_ml:
        logger.info(f"RxInfer.jl HMM (model_logic) rendering test successful: {msg_hmm_ml}")
        if output_script_hmm_ml.exists():
            logger.info(f"--- Generated HMM (model_logic) Julia Script ({output_script_hmm_ml.name}) ---")
            print(f"\n{output_script_hmm_ml.read_text(encoding='utf-8')}\n")
            logger.info(f"--- End of HMM (model_logic) Script ---")
    else:
        logger.error(f"RxInfer.jl HMM (model_logic) rendering test failed: {msg_hmm_ml}")

    # Test TOML rendering for HMM model
    output_toml_hmm_ml = test_output_dir / "generated_hmm_model_logic_config.toml"
    success_toml_hmm_ml, msg_toml_hmm_ml, artifacts_toml_hmm_ml = render_gnn_to_rxinfer_toml(
        dummy_gnn_data_hmm_ml,
        output_toml_hmm_ml,
        options=render_options_hmm_ml
    )

    if success_toml_hmm_ml:
        logger.info(f"RxInfer TOML HMM rendering test successful: {msg_toml_hmm_ml}")
        if output_toml_hmm_ml.exists():
            logger.info(f"--- Generated HMM TOML Config ({output_toml_hmm_ml.name}) ---")
            print(f"\n{output_toml_hmm_ml.read_text(encoding='utf-8')}\n")
            logger.info(f"--- End of HMM TOML Config ---")
    else:
        logger.error(f"RxInfer TOML HMM rendering test failed: {msg_toml_hmm_ml}")

    # Cleanup (optional)
    # import shutil
    # if test_output_dir.exists():
    #     shutil.rmtree(test_output_dir)
    #     logger.info(f"Cleaned up test directory: {test_output_dir}") 