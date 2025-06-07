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
import traceback

# Ensure that the src directory is in the Python path for importing rxinfer
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from rxinfer.gnn_parser import parse_gnn_file
from rxinfer.config_generator import generate_rxinfer_config, generate_rxinfer_config_from_spec
from rxinfer.toml_generator import generate_rxinfer_toml_config  # New import for TOML generation

logger = logging.getLogger(__name__)

def render_gnn_to_rxinfer_toml(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Renders a GNN specification dictionary to an RxInfer TOML configuration file.

    Args:
        gnn_spec (Dict[str, Any]): The GNN specification as a Python dictionary.
        output_path (Path): The path where the output TOML file should be saved.
        options (Dict[str, Any], optional): Rendering options. Defaults to None.

    Returns:
        Tuple[bool, str, List[str]]: A tuple containing success status, a message, and a list of artifact URIs.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Starting rendering of GNN spec to RxInfer TOML config at {output_path}")
    
    try:
        config_content = generate_rxinfer_config_from_spec(gnn_spec, logger)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(config_content)
        
        success_msg = f"Successfully wrote RxInfer TOML configuration to {output_path}"
        logger.info(success_msg)
        return True, success_msg, [output_path.as_uri()]

    except Exception as e:
        error_msg = f"An error occurred during RxInfer TOML generation: {e}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg, []

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
            return render_gnn_to_rxinfer_toml(parsed_gnn, output_path)
    
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

    success_lr, msg_lr, artifacts_lr = render_gnn_to_rxinfer_toml(
        dummy_gnn_data_linear_regression,
        test_output_dir / "generated_linear_regression_config.toml",
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

    success_hmm_ml, msg_hmm_ml, artifacts_hmm_ml = render_gnn_to_rxinfer_toml(
        dummy_gnn_data_hmm_ml,
        test_output_dir / "generated_hmm_model_logic_config.toml",
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