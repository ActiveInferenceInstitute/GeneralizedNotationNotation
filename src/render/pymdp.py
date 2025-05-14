"""
Module for rendering GNN specifications to PyMDP compatible formats.

This module translates a parsed GNN (Generalized Notation Notation) specification
into Python code and data structures suitable for use with the PyMDP library,
which is designed for active inference agents in discrete state spaces.
The conversion is based on the technical documentation provided for PyMDP.
"""

import logging
import numpy as np # For matrix/array generation if needed directly
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import json # For placeholder_gnn_parser_pymdp
import re # for _numpy_array_to_string refinement

logger = logging.getLogger(__name__)

# Removed: logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Global Configuration & Constants ---

# Default directory for rendered PyMDP model outputs, relative to the main output directory

def _numpy_array_to_string(arr: np.ndarray, indent=8) -> str:
    """Converts a NumPy array to a string representation for Python script, with proper indentation."""
    if arr.ndim == 0: # Scalar
        return str(arr.item())
    if arr.ndim == 1:
        # Single list for 1D array
        list_str = np.array2string(arr, separator=', ', prefix=' ' * indent)
    else:
        # List of lists for 2D+ arrays
        list_str = np.array2string(arr, separator=', ', prefix=' ' * indent)
        list_str = list_str.replace('\n', '\n' + ' ' * indent)
    # Remove extra spaces around brackets if any from np.array2string
    list_str = re.sub(r'\[\s+', '[', list_str)
    list_str = re.sub(r'\s+\]', ']', list_str)
    list_str = re.sub(r'\s+,', ',', list_str)
    return f"np.array({list_str})"

def format_list_recursive(data_list: list, current_indent: int, item_formatter: callable) -> str:
    """Formats a potentially nested list of items (like NumPy arrays) into a string for Python script."""
    lines = []
    base_indent_str = ' ' * current_indent
    item_indent_str = ' ' * (current_indent + 4)
    
    if not any(isinstance(item, list) for item in data_list): # Flat list of items (e.g., np.arrays)
        formatted_items = [item_formatter(item, current_indent + 4) for item in data_list]
        if len(formatted_items) > 3: # Heuristic for multiline formatting
            lines.append("[")
            for fi in formatted_items:
                lines.append(item_indent_str + fi + ",")
            lines.append(base_indent_str + "]")
        else:
            lines.append("[" + ", ".join(formatted_items) + "]")
    else: # Nested list - currently not expected for pymdp A, B, C, D which are lists of arrays
        # This part would need more sophisticated handling if truly nested lists of lists of arrays are needed.
        # For now, assumes a flat list of arrays.
        lines.append("[")
        for item in data_list:
            if isinstance(item, np.ndarray):
                lines.append(item_indent_str + item_formatter(item, current_indent + 4) + ",")
            # Add handling for other types if necessary
        lines.append(base_indent_str + "]")
    return '\n'.join(lines)

def generate_pymdp_matrix_definition(
    matrix_name: str,
    data: Any, 
    is_object_array: bool = False, # True if data is a list of np.arrays for different modalities/factors
    num_modalities_or_factors: Optional[int] = None, # Used if is_object_array is True
    is_vector: bool = False
) -> str:
    """
    Generates Python code for a PyMDP matrix (A, B, C, D, etc.).
    Handles single matrices, lists of matrices (object arrays), and vectors.
    """
    lines = []
    indent_str = "    " # 4 spaces for base indent within script

    if data is None:
        lines.append(f"{matrix_name} = None")
        return '\n'.join(lines)

    if is_object_array and isinstance(data, list):
        # This is for lists of np.arrays (e.g., A for multiple modalities)
        # Each item in the list `data` should be a np.ndarray
        array_strs = []
        for i, arr_item in enumerate(data):
            if not isinstance(arr_item, np.ndarray):
                # Fallback: if an item isn't a numpy array, try to convert it.
                # This might happen if parsing yielded lists of lists instead of ndarrays directly.
                try:
                    arr_item = np.array(arr_item)
                except Exception as e:
                    lines.append(f"# Warning: Could not convert item {i} of {matrix_name} to np.array: {e}")
                    array_strs.append("None") # Placeholder for problematic item
                    continue
            array_strs.append(_numpy_array_to_string(arr_item, indent=8))
        
        lines.append(f"{matrix_name} = [ # Object array for {num_modalities_or_factors} modalities/factors")
        for arr_s in array_strs:
            lines.append(indent_str + indent_str + arr_s + ",")
        lines.append(indent_str + "]")
        lines.append(f"{matrix_name} = np.array({matrix_name}, dtype=object)")

    elif isinstance(data, (list, tuple)) and not is_object_array:
        # Single matrix provided as list/tuple of lists/tuples from GNN spec
        try:
            np_array = np.array(data)
            lines.append(f"{matrix_name} = {_numpy_array_to_string(np_array, indent=4)}")
        except ValueError as e:
            lines.append(f"# ERROR: Could not convert {matrix_name} data to numpy array: {e}")
            lines.append(f"# Raw data: {data}")
            lines.append(f"{matrix_name} = None")
            
    elif isinstance(data, np.ndarray) and not is_object_array:
        # Already a numpy array (e.g. if converter pre-processed it)
        lines.append(f"{matrix_name} = {_numpy_array_to_string(data, indent=4)}")
    else:
        lines.append(f"# Note: Data for {matrix_name} is of unexpected type: {type(data)}. Assigning as is or None.")
        lines.append(f"{matrix_name} = {data if data is not None else 'None'}")

    return '\n'.join(lines)


def generate_pymdp_agent_instantiation(
    agent_name: str,
    model_params: Dict[str, str], # Matrix names as strings, assuming they are defined in the global scope
    control_params: Optional[Dict[str, Any]] = None, 
    learning_params: Optional[Dict[str, Any]] = None, 
    algorithm_params: Optional[Dict[str, Any]] = None
) -> str:
    lines = [f"{agent_name} = Agent("]
    indent = "    "

    # Model parameters (A, B, C, D, etc.)
    for key, matrix_name_str in model_params.items():
        lines.append(f"{indent}{key}={matrix_name_str},")

    # Control parameters (e.g., E, F, policy_len, etc.)
    if control_params:
        for key, value in control_params.items():
            if isinstance(value, str): # If it's a variable name
                lines.append(f"{indent}{key}={value},")
            else: # If it's a literal value
                lines.append(f"{indent}{key}={repr(value)},") # Use repr for correct literal representation
    
    # Learning parameters (e.g. use_B_learning, etc.)
    if learning_params:
         for key, value in learning_params.items():
            lines.append(f"{indent}{key}={repr(value)},")

    # Algorithm parameters (e.g. G_learn_method, etc.)
    if algorithm_params:
        for key, value in algorithm_params.items():
            lines.append(f"{indent}{key}={repr(value)},")

    # Remove trailing comma from the last parameter if any
    if lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]
    
    lines.append(")")
    return "\n".join(lines)


class GnnToPyMdpConverter:
    """Converts a GNN specification dictionary to a PyMDP compatible Python script."""

    def __init__(self, gnn_spec: Dict[str, Any]):
        self.gnn_spec = gnn_spec
        self.model_name = self.gnn_spec.get("model_name", "pymdp_agent_model").replace(" ", "_").replace("-", "_")
        self.script_parts: Dict[str, List[str]] = {
            "imports": ["import numpy as np", "from pymdp.agent import Agent"],
            "comments": ["# Generated PyMDP script from GNN specification"],
            "matrix_definitions": [],
            "agent_instantiation": [],
            "example_usage": []
        }
        self.conversion_log: List[str] = [] # Log messages for summary

        # Directly use dimension info from gnn_spec
        self.num_hidden_states_factors = self.gnn_spec.get("num_hidden_states_factors", [])
        self.num_obs_modalities = self.gnn_spec.get("num_obs_modalities", [])
        self.num_control_factors = self.gnn_spec.get("control_states_factors", [])
        
        self.matrix_params = self.gnn_spec.get("matrix_parameters", {})

    def _add_log(self, message: str, level: str = "INFO"): 
        self.conversion_log.append(f"{level}: {message}")
        # print(f"[{level}] {message}") # Optional: print to console during conversion

    def _get_matrix_data(self, base_name: str, factor_idx: Optional[int] = None, modality_idx: Optional[int] = None) -> Any:
        """Helper to retrieve matrix data from gnn_spec['matrix_parameters'] based on naming convention."""
        if factor_idx is not None:
            matrix_key = f"{base_name}_f{factor_idx}"
        elif modality_idx is not None:
            matrix_key = f"{base_name}_m{modality_idx}"
        else:
            matrix_key = base_name
        
        data = self.matrix_params.get(matrix_key)
        if data is None:
            self._add_log(f"Matrix {matrix_key} not found in GNN InitialParameterization.", "WARNING")
        return data

    def convert_A_matrix(self) -> str:
        """Converts GNN's A matrix (likelihood) to PyMDP format."""
        if not self.num_obs_modalities: # No observation modalities defined
            self._add_log("A_matrix: No observation modalities (num_obs_modalities) defined in GNN spec. 'A' will be None.", "WARNING")
            return generate_pymdp_matrix_definition("A", None)

        if len(self.num_obs_modalities) == 1:
            # Single modality, look for plain "A" or "A_m0"
            a_data = self._get_matrix_data("A")
            if a_data is None: # Try A_m0 as fallback for single modality
                 a_data = self._get_matrix_data("A", modality_idx=0)
            
            if a_data is None:
                self._add_log("A_matrix: 'A' or 'A_m0' not found for single observation modality.", "ERROR")
                return generate_pymdp_matrix_definition("A", None)
            self._add_log(f"A_matrix: Using single matrix for 1 modality.")
            return generate_pymdp_matrix_definition("A", a_data)
        else:
            # Multiple modalities
            A_list = []
            all_found = True
            for i in range(len(self.num_obs_modalities)):
                a_mod_data = self._get_matrix_data("A", modality_idx=i)
                if a_mod_data is None:
                    self._add_log(f"A_matrix: A_m{i} not found for modality {i}.", "ERROR")
                    A_list.append(None) # Or handle error more strictly
                    all_found = False
                else:
                    A_list.append(np.array(a_mod_data)) # Convert to ndarray here
            
            if not all_found:
                self._add_log("A_matrix: Not all modality-specific A matrices found. 'A' might be incomplete.", "ERROR")
            self._add_log(f"A_matrix: Using list of matrices for {len(self.num_obs_modalities)} modalities.")
            return generate_pymdp_matrix_definition("A", A_list, is_object_array=True, num_modalities_or_factors=len(self.num_obs_modalities))

    def convert_B_matrix(self) -> str:
        """Converts GNN's B matrix (transition) to PyMDP format."""
        if not self.num_hidden_states_factors:
            self._add_log("B_matrix: No hidden state factors (num_hidden_states_factors) defined. 'B' will be None.", "WARNING")
            return generate_pymdp_matrix_definition("B", None)

        if len(self.num_hidden_states_factors) == 1:
            b_data = self._get_matrix_data("B")
            if b_data is None: # Try B_f0 as fallback
                b_data = self._get_matrix_data("B", factor_idx=0)
            
            if b_data is None:
                # It's okay for B to be None if it's a static model (no transitions)
                # Check GNN time spec
                if self.gnn_spec.get("time_specification", "Static").lower() == "dynamic":
                    self._add_log("B_matrix: 'B' or 'B_f0' not found for single dynamic hidden state factor.", "ERROR")
                else:
                    self._add_log("B_matrix: Not specified for single factor static model. 'B' will be None (acceptable).", "INFO")
                return generate_pymdp_matrix_definition("B", None)
            self._add_log(f"B_matrix: Using single matrix for 1 hidden state factor.")
            return generate_pymdp_matrix_definition("B", b_data)
        else:
            # Multiple hidden state factors
            B_list = []
            all_found = True
            for i in range(len(self.num_hidden_states_factors)):
                b_factor_data = self._get_matrix_data("B", factor_idx=i)
                if b_factor_data is None:
                    self._add_log(f"B_matrix: B_f{i} not found for factor {i}.", "ERROR")
                    B_list.append(None)
                    all_found = False
                else:
                    B_list.append(np.array(b_factor_data))
            
            if not all_found:
                self._add_log("B_matrix: Not all factor-specific B matrices found. 'B' might be incomplete.", "ERROR")
            self._add_log(f"B_matrix: Using list of matrices for {len(self.num_hidden_states_factors)} hidden state factors.")
            return generate_pymdp_matrix_definition("B", B_list, is_object_array=True, num_modalities_or_factors=len(self.num_hidden_states_factors))

    def convert_C_vector(self) -> str:
        """Converts GNN's C vector (preferences over outcomes) to PyMDP format."""
        # C can also be named G in some contexts (goal preferences)
        if not self.num_obs_modalities:
             self._add_log("C_vector: No observation modalities defined. 'C' will be None.", "INFO") # C is optional
             return generate_pymdp_matrix_definition("C", None)

        if len(self.num_obs_modalities) == 1:
            c_data = self._get_matrix_data("C")
            if c_data is None: c_data = self._get_matrix_data("C", modality_idx=0)
            if c_data is None: c_data = self._get_matrix_data("G") # Try G as an alias for C
            if c_data is None: c_data = self._get_matrix_data("G", modality_idx=0)
            
            if c_data is None:
                self._add_log("C_vector: 'C', 'C_m0', 'G', or 'G_m0' not found for single modality. 'C' will be None.", "INFO")
                return generate_pymdp_matrix_definition("C", None)
            self._add_log(f"C_vector: Using single vector for 1 modality.")
            return generate_pymdp_matrix_definition("C", c_data, is_vector=True)
        else:
            C_list = []
            all_found = True # Or some found is okay for C
            any_found = False
            for i in range(len(self.num_obs_modalities)):
                c_mod_data = self._get_matrix_data("C", modality_idx=i)
                if c_mod_data is None: c_mod_data = self._get_matrix_data("G", modality_idx=i)
                
                if c_mod_data is None:
                    self._add_log(f"C_vector: C_m{i} or G_m{i} not found for modality {i}. Will use zeros or None.", "INFO")
                    # PyMDP expects a C vector for each modality, even if it's neutral (zeros)
                    # Infer dimension from num_obs_modalities[i]
                    num_outcomes_mod = self.num_obs_modalities[i]
                    C_list.append(np.zeros(num_outcomes_mod))
                    # all_found = False # Don't mark as error if one C is missing, just use zeros
                else:
                    C_list.append(np.array(c_mod_data))
                    any_found = True
            
            if not any_found:
                self._add_log("C_vector: No modality-specific C or G vectors found. 'C' will be None.", "INFO")
                return generate_pymdp_matrix_definition("C", None)

            self._add_log(f"C_vector: Using list of vectors for {len(self.num_obs_modalities)} modalities.")
            return generate_pymdp_matrix_definition("C", C_list, is_object_array=True, num_modalities_or_factors=len(self.num_obs_modalities), is_vector=True)

    def convert_D_vector(self) -> str:
        """Converts GNN's D vector (initial hidden state priors) to PyMDP format."""
        if not self.num_hidden_states_factors:
            self._add_log("D_vector: No hidden state factors defined. 'D' will be None.", "ERROR")
            return generate_pymdp_matrix_definition("D", None)

        if len(self.num_hidden_states_factors) == 1:
            d_data = self._get_matrix_data("D")
            if d_data is None: d_data = self._get_matrix_data("D", factor_idx=0)

            if d_data is None:
                self._add_log("D_vector: 'D' or 'D_f0' not found for single hidden state factor.", "ERROR")
                return generate_pymdp_matrix_definition("D", None)
            self._add_log(f"D_vector: Using single vector for 1 hidden state factor.")
            return generate_pymdp_matrix_definition("D", d_data, is_vector=True)
        else:
            D_list = []
            all_found = True
            for i in range(len(self.num_hidden_states_factors)):
                d_factor_data = self._get_matrix_data("D", factor_idx=i)
                if d_factor_data is None:
                    self._add_log(f"D_vector: D_f{i} not found for factor {i}.", "ERROR")
                    D_list.append(None)
                    all_found = False
                else:
                    D_list.append(np.array(d_factor_data))
            
            if not all_found:
                 self._add_log("D_vector: Not all factor-specific D vectors found. 'D' might be incomplete.", "ERROR")
            self._add_log(f"D_vector: Using list of vectors for {len(self.num_hidden_states_factors)} hidden state factors.")
            return generate_pymdp_matrix_definition("D", D_list, is_object_array=True, num_modalities_or_factors=len(self.num_hidden_states_factors), is_vector=True)

    def extract_agent_hyperparameters(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str,Any]]]:
        """Extracts control, learning, and algorithm parameters from GNN spec if available."""
        # This is a placeholder. GNN spec needs a dedicated section for these.
        # For now, assume they are not in the current GNN spec format.
        control_params = None # e.g. self.gnn_spec.get("control_parameters") 
        learning_params = None # e.g. self.gnn_spec.get("learning_parameters")
        algorithm_params = None # e.g. self.gnn_spec.get("algorithm_parameters")

        # Example: if policy_len is in GNN's time_specification or a specific hyperparameter section
        # if self.num_control_factors and self.num_control_factors[0] > 0: # Assuming first control factor is policy length
        #     control_params = control_params or {}
        #     control_params["policy_len"] = self.num_control_factors[0] # This is a guess, needs proper GNN spec support

        # Check for E matrix (prior over policies)
        e_data = self._get_matrix_data("E")
        if e_data is not None:
            self.script_parts["matrix_definitions"].append(generate_pymdp_matrix_definition("E", e_data, is_vector=True))
            control_params = control_params or {}
            control_params["E"] = "E" # Pass the name of the E matrix
        
        self._add_log("AgentHyperparameters: Extracted control, learning, algorithm params (mostly placeholders).", "INFO")
        return control_params, learning_params, algorithm_params

    def generate_agent_instantiation_code(self) -> str:
        model_params_dict = {
            "A": "A", "B": "B", "C": "C", "D": "D"
        }
        control_params, learning_params, algorithm_params = self.extract_agent_hyperparameters()
        return generate_pymdp_agent_instantiation(
            self.model_name,
            model_params_dict,
            control_params=control_params,
            learning_params=learning_params,
            algorithm_params=algorithm_params
        )

    def get_full_python_script(self, include_example_usage: bool = True) -> str:
        """Generates the full Python script string for the PyMDP agent."""
        self._add_log("Starting PyMDP script generation.", "INFO")

        # Process core matrices
        self.script_parts["matrix_definitions"].append(self.convert_A_matrix())
        self.script_parts["matrix_definitions"].append(self.convert_B_matrix())
        self.script_parts["matrix_definitions"].append(self.convert_C_vector())
        self.script_parts["matrix_definitions"].append(self.convert_D_vector())

        # Agent instantiation (also handles E matrix if present via extract_agent_hyperparameters)
        self.script_parts["agent_instantiation"].append(self.generate_agent_instantiation_code())

        # Example Usage (placeholder)
        if include_example_usage:
            self.script_parts["example_usage"].extend([
                "# --- Example Usage ---",
                "if __name__ == '__main__':",
                "    # Example: Initialize agent",
                f"    # agent = {self.model_name}",
                "    # print(\"Agent initialized\")",
                "    # obs = [0] # Example observation (modality 0, outcome 0)",
                "    # if hasattr(agent, 'num_obs_modalities') and agent.num_obs_modalities > 1:",
                "    #    obs = [[0],[1]] # Example for multi-modal observations", 
                "    # Qs = agent.infer_states(obs)",
                "    # print(f\"Posterior states (Qs): {Qs}\")",
                "    # Q_pi, EFE = agent.infer_policies()",
                "    # print(f\"Posterior policies (Q_pi): {Q_pi}\")",
                "    # print(f\"Expected Free Energy (EFE): {EFE}\")",
                "    # action = agent.sample_action()",
                "    # print(f\"Sampled action: {action}\")"
            ])
        
        # Assemble the script
        script_lines = []
        script_lines.extend(self.script_parts["imports"])
        script_lines.append("\n# --- Model Comments ---")
        for log_msg in self.conversion_log:
            script_lines.append(f"# {log_msg}")
        script_lines.append("\n# --- Matrix Definitions ---")
        script_lines.extend(self.script_parts["matrix_definitions"])
        script_lines.append("\n# --- Agent Instantiation ---")
        script_lines.extend(self.script_parts["agent_instantiation"])
        if include_example_usage:
            script_lines.append("\n")
            script_lines.extend(self.script_parts["example_usage"])
        
        self._add_log("PyMDP script generation complete.", "INFO")
        return "\n".join(script_lines)


def render_gnn_to_pymdp(
    gnn_spec: Dict[str, Any],
    output_script_path: Path,
    options: Optional[Dict[str, Any]] = None # e.g. {"include_example_usage": True}
) -> Tuple[bool, str, List[str]]:
    """Renders a GNN specification to a PyMDP Python script."""
    
    if options is None: options = {}
    include_example = options.get("include_example_usage", True)

    converter = GnnToPyMdpConverter(gnn_spec)
    script_content = converter.get_full_python_script(include_example_usage=include_example)
    
    summary_message = f"GNN to PyMDP conversion for model: {converter.model_name}"
    
    # Add conversion log to the top of the script as comments
    log_header = ["# --- GNN to PyMDP Conversion Summary ---"]
    log_header.extend([f"# {log}" for log in converter.conversion_log])
    log_header.append("# --- End of GNN to PyMDP Conversion Summary ---")
    
    # Prepend log summary to the script_content, after imports
    # Find where imports end
    num_imports = len(converter.script_parts["imports"])
    script_lines = script_content.split('\n')
    final_script_content = "\n".join(
        script_lines[:num_imports] + 
        ["\n"] + log_header + ["\n"] + 
        script_lines[num_imports:]
    )

    try:
        output_script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_script_path, "w", encoding='utf-8') as f:
            f.write(final_script_content)
        # print(f"Successfully wrote PyMDP script to {output_script_path}")
        return True, f"Successfully wrote PyMDP script: {output_script_path.name}", converter.conversion_log
    except IOError as e:
        # print(f"Error writing PyMDP script to {output_script_path}: {e}")
        error_msg = f"Error writing PyMDP script: {e}"
        converter.conversion_log.append(f"FATAL_ERROR: {error_msg}")
        return False, error_msg, converter.conversion_log

# Placeholder for GNN parser if this script is run standalone or for testing
# In the full pipeline, gnn_spec would come from the export step (5_export.py)
def placeholder_gnn_parser_pymdp(gnn_file_path: Path) -> Optional[Dict[str, Any]]:
    """Placeholder GNN parser, loads from a JSON file if it exists (simulating export step output)."""
    # This assumes the export step has already run and created a .json for the .md GNN file.
    # In the main pipeline, this function would not be used; gnn_spec comes from previous steps.
    json_gnn_path = gnn_file_path.with_suffix(".json") # Assuming export creates this.
    
    if not json_gnn_path.exists():
        print(f"Error: JSON GNN spec file not found: {json_gnn_path}. Run export step first.")
        # Fallback: Try to simulate a very basic parse from the .md itself if needed for standalone testing
        # This would be a very simplified version of what format_exporters.py does.
        # For now, just error out if JSON isn't there.
        return None
        
    try:
        with open(json_gnn_path, 'r', encoding='utf-8') as f:
            gnn_data = json.load(f)
        return gnn_data
    except Exception as e:
        print(f"Error loading or parsing GNN JSON file {json_gnn_path}: {e}")
        return None

# Example main for standalone testing:
if __name__ == '__main__':
    print("Running GnnToPyMdpConverter standalone example...")
    # This example assumes you have a GNN file exported to JSON by the pipeline
    # For instance, from: output/gnn_exports/gnn_example_pymdp_agent/gnn_example_pymdp_agent.json
    
    # Replace with the actual path to an exported GNN JSON file
    example_gnn_json_path_str = "output/gnn_exports/gnn_example_pymdp_agent/gnn_example_pymdp_agent.json"
    # Or provide a path to a .md file if you want to test placeholder_gnn_parser_pymdp
    # example_gnn_md_path_str = "src/gnn/examples/gnn_example_pymdp_agent.md"

    gnn_spec_data = None
    gnn_path = Path(example_gnn_json_path_str)

    if gnn_path.suffix == '.json':
        try:
            with open(gnn_path, 'r') as f:
                gnn_spec_data = json.load(f)
            print(f"Successfully loaded GNN spec from: {gnn_path}")
        except Exception as e:
            print(f"Failed to load GNN spec from JSON {gnn_path}: {e}")
    elif gnn_path.suffix == '.md': # Using the placeholder parser
        print(f"Attempting to use placeholder parser for .md file (expects corresponding .json): {gnn_path.name}")
        gnn_spec_data = placeholder_gnn_parser_pymdp(gnn_path)
    else:
        print(f"Unsupported file type: {gnn_path.suffix}")

    if gnn_spec_data:
        output_dir = Path("output/gnn_rendered_simulators/pymdp/") / gnn_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        output_script = output_dir / (gnn_path.stem + "_rendered_test.py")
        
        print(f"Rendering GNN spec to PyMDP script: {output_script}")
        success, message, log = render_gnn_to_pymdp(gnn_spec_data, output_script)
        
        print("\n--- Conversion Log ---")
        for log_entry in log:
            print(log_entry)
        print("--- End Log ---")
        
        if success:
            print(f"\nSuccessfully generated PyMDP script: {output_script}")
            print(f"Message: {message}")
        else:
            print(f"\nFailed to generate PyMDP script.")
            print(f"Error Message: {message}")
    else:
        print("Could not load GNN specification. Aborting example.")

# Note: The GNN spec for PyMDP needs to be carefully designed.
# For A, B, C, D, it can provide:
# 1. "type": "random_A_matrix", "num_obs": ..., "num_states": ... (similar for B, C_uniform, D_uniform)
# 2. "type": "obj_array_from_list", "arrays": [ <list_of_numpy_array_like_lists_or_actual_arrays> ]
# 3. A direct list of lists (for simple, non-object arrays).
# The converter tries to handle these. More complex scenarios might need GNN spec refinement
# or more sophisticated parsing/conversion logic.
# Normalization of matrices (summing to 1 over appropriate axes) is assumed to be
# handled by PyMDP's utils functions (like random_A_matrix) or needs to be ensured
# by the GNN spec if providing raw arrays. This renderer currently doesn't add explicit normalization steps.
# </rewritten_file> is not part of the python code. It is a marker for the system. 