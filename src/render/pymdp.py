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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union # Added Union
import json # For placeholder_gnn_parser_pymdp
import re # for _numpy_array_to_string refinement
import sys # for sys.modules
import inspect # for inspect.signature
import traceback # for traceback.format_exc()

# Attempt to import pymdp utilities, maths, and agent.
# These are for the *generated* script, not for the renderer itself,
# but having them available for type hints or constants can be useful if shared.
_PMDP_AVAILABLE = False
try:
    from pymdp import utils as pymdp_utils
    from pymdp import maths as pymdp_maths
    from pymdp.agent import Agent as PymdpAgent # Alias to avoid confusion with internal Agent concepts
    _PMDP_AVAILABLE = True
except ImportError:
    # These are not strictly needed for the renderer to generate code strings,
    # but if we were to, e.g., validate matrix shapes, they would be.
    pymdp_utils = None
    pymdp_maths = None
    PymdpAgent = None


logger = logging.getLogger(__name__)

# Removed: logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Global Configuration & Constants ---

# Default directory for rendered PyMDP model outputs, relative to the main output directory

def _numpy_array_to_string(arr: np.ndarray, indent=8) -> str:
    """Converts a NumPy array to a string representation for Python script, with proper indentation."""
    if arr is None: # Handle cases where a None might be passed
        return "None"
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
    If data is already a string (e.g. "pymdp.utils.get_A_likelihood_identity(...)""), use it directly.
    """
    lines = []
    indent_str = "    " # 4 spaces for base indent within script

    if data is None:
        if is_object_array: # For A, B, D which are lists of np.arrays (object arrays)
            logger.debug(f"Data for object array {matrix_name} is None, defaulting to [].")
            lines.append(f"{matrix_name} = []")
        else: # For C or other single arrays/vectors where PyMDP might handle None internally
            logger.debug(f"Data for non-object array {matrix_name} is None, setting to None.")
            lines.append(f"{matrix_name} = None")
        return '\n'.join(lines)
    
    if isinstance(data, str) and ("pymdp." in data or "np." in data or "utils." in data or "maths." in data): # Heuristic for pre-formatted code string
        lines.append(f"{matrix_name} = {data}")
        return '\n'.join(lines)

    if is_object_array and isinstance(data, list):
        # This is for lists of np.arrays (e.g., A for multiple modalities)
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
    algorithm_params: Optional[Dict[str, Any]] = None,
    # Added new parameters
    policy_len: Optional[int] = None,
    control_fac_idx_var_name: Optional[str] = None, # Changed from List[int] to var_name
    use_utility: Optional[bool] = None, # Changed to Optional
    use_states_info_gain: Optional[bool] = None, # Changed to Optional
    use_param_info_gain: Optional[bool] = None, # Changed to Optional
    action_selection: Optional[str] = None, # Changed to Optional
    # num_obs_var_name: Optional[str] = None, # REMOVED
    # num_states_var_name: Optional[str] = None, # REMOVED
    # num_controls_var_name: Optional[str] = None # REMOVED
) -> str:
    lines = [f"{agent_name} = Agent("]
    indent = "    "

    # Dimensionality parameters - PyMDP infers these from A and B
    # if num_obs_var_name: lines.append(f"{indent}num_obs={num_obs_var_name},") # REMOVED
    # if num_states_var_name: lines.append(f"{indent}num_states={num_states_var_name},") # REMOVED
    # if num_controls_var_name: lines.append(f"{indent}num_controls={num_controls_var_name},") # REMOVED

    # Model parameters (A, B, C, D, E, etc.)
    for key, matrix_name_str in model_params.items():
        lines.append(f"{indent}{key}={matrix_name_str},")

    # Control parameters (e.g., E, F, policy_len, etc.)
    if control_params:
        for key, value in control_params.items():
            if isinstance(value, str): # If it's a variable name
                lines.append(f"{indent}{key}={value},")
            else: # If it's a literal value
                lines.append(f"{indent}{key}={repr(value)},") # Use repr for correct literal representation
    
    # Specific agent constructor parameters (handled separately now)
    if control_fac_idx_var_name: lines.append(f"{indent}control_fac_idx={control_fac_idx_var_name},")
    if policy_len is not None: lines.append(f"{indent}policy_len={policy_len},")
    if use_utility is not None: lines.append(f"{indent}use_utility={use_utility},")
    if use_states_info_gain is not None: lines.append(f"{indent}use_states_info_gain={use_states_info_gain},")
    if use_param_info_gain is not None: lines.append(f"{indent}use_param_info_gain={use_param_info_gain},")
    if action_selection is not None: lines.append(f"{indent}action_selection='{action_selection}',")

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
        self.model_name = self.gnn_spec.get("ModelName", "pymdp_agent_model").replace(" ", "_").replace("-", "_")
        self.script_parts: Dict[str, List[str]] = {
            "imports": [
                "import numpy as np",
                "from pymdp.agent import Agent",
                "from pymdp import utils", # Added
                "from pymdp import maths", # Added, for softmax etc.
                "import copy", # Added for A_gp, B_gp
                "import sys", # for sys.modules
                "import inspect", # for inspect.signature
                "import traceback" # for traceback.format_exc()
            ],
            "preamble_vars": [], # For obs_names, state_names etc.
            "comments": [f"# --- GNN Model: {self.model_name} ---"],
            "matrix_definitions": [],
            "agent_instantiation": [],
            "example_usage": []
        }
        self.conversion_log: List[str] = [] # Log messages for summary

        # Extracted and processed GNN data
        self.obs_names: List[str] = []
        self.state_names: List[str] = []
        self.action_names_per_control_factor: Dict[int, List[str]] = {} # factor_idx -> list of action names

        self.num_obs: List[int] = [] # num_outcomes per modality
        self.num_states: List[int] = [] # num_states per factor
        self.num_actions_per_control_factor: Dict[int, int] = {} # factor_idx -> num_actions

        self.num_modalities: int = 0
        self.num_factors: int = 0
        self.control_factor_indices: List[int] = [] # List of hidden state factor indices that are controllable

        self.A_spec: Optional[Union[Dict, List[Dict]]] = None # Can be complex structure
        self.B_spec: Optional[Union[Dict, List[Dict]]] = None
        self.C_spec: Optional[Union[Dict, List[Dict]]] = None
        self.D_spec: Optional[Union[Dict, List[Dict]]] = None
        self.E_spec: Optional[Dict] = None # For expected future utilities / policies

        self.agent_hyperparams: Dict[str, Any] = {}
        self.simulation_params: Dict[str, Any] = {}
        
        self._extract_gnn_data() # Populate the above fields


    def _add_log(self, message: str, level: str = "INFO"): 
        self.conversion_log.append(f"{level}: {message}")
        # print(f"[{level}] {message}") # Optional: print to console during conversion

    def _extract_gnn_data(self):
        """Parses the raw gnn_spec and populates structured attributes."""
        self._add_log("Starting GNN data extraction.")

        # StateSpaceBlock parsing
        ss_block = self.gnn_spec.get("StateSpaceBlock", {})

        # ObservationModalities
        obs_modalities_spec = ss_block.get("ObservationModalities", [])
        if not obs_modalities_spec:
            self._add_log("No 'ObservationModalities' defined in StateSpaceBlock.", "WARNING")
        for i, mod_spec in enumerate(obs_modalities_spec):
            name = mod_spec.get("modality_name", f"modality_{i}")
            num_outcomes = mod_spec.get("num_outcomes")
            if num_outcomes is None:
                self._add_log(f"Modality '{name}' missing 'num_outcomes'. Skipping.", "ERROR")
                continue
            self.obs_names.append(name)
            self.num_obs.append(int(num_outcomes))
        self.num_modalities = len(self.num_obs)
        # Always define these, even if empty
        self.script_parts["preamble_vars"].append(f"obs_names = {self.obs_names if self.num_modalities > 0 else []}")
        self.script_parts["preamble_vars"].append(f"num_obs = {self.num_obs if self.num_modalities > 0 else []}")
        self.script_parts["preamble_vars"].append(f"num_modalities = {self.num_modalities}")


        # HiddenStateFactors
        hidden_factors_spec = ss_block.get("HiddenStateFactors", [])
        if not hidden_factors_spec:
            self._add_log("No 'HiddenStateFactors' defined in StateSpaceBlock.", "WARNING")
        for i, fac_spec in enumerate(hidden_factors_spec):
            name = fac_spec.get("factor_name", f"factor_{i}")
            num_states_val = fac_spec.get("num_states")
            if num_states_val is None:
                self._add_log(f"Factor '{name}' missing 'num_states'. Skipping.", "ERROR")
                continue
            self.state_names.append(name)
            self.num_states.append(int(num_states_val))
            if fac_spec.get("controllable", False):
                self.control_factor_indices.append(i)
                num_actions = fac_spec.get("num_actions")
                if num_actions is None:
                    self._add_log(f"Controllable factor '{name}' missing 'num_actions'. Defaulting to num_states ({num_states_val}).", "WARNING")
                    num_actions = int(num_states_val) # Default for pymdp if not specified
                self.num_actions_per_control_factor[i] = int(num_actions)
                action_names = fac_spec.get("action_names")
                if action_names and len(action_names) == num_actions:
                    self.action_names_per_control_factor[i] = action_names
                else:
                    self.action_names_per_control_factor[i] = [f"{name}_action_{j}" for j in range(num_actions)]


        self.num_factors = len(self.num_states)
        # Always define these, even if empty
        self.script_parts["preamble_vars"].append(f"state_names = {self.state_names if self.num_factors > 0 else []}")
        self.script_parts["preamble_vars"].append(f"num_states = {self.num_states if self.num_factors > 0 else []}")
        self.script_parts["preamble_vars"].append(f"num_factors = {self.num_factors}")
        self.script_parts["preamble_vars"].append(f"control_fac_idx = {self.control_factor_indices if self.control_factor_indices else []}")

        # Derive and add num_controls
        num_controls_list = []
        if self.num_factors > 0:
            for f_idx in range(self.num_factors):
                if f_idx in self.control_factor_indices:
                    # Default to num_states for that factor if num_actions not specified
                    num_actions_for_factor = self.num_actions_per_control_factor.get(f_idx, self.num_states[f_idx])
                    num_controls_list.append(num_actions_for_factor)
                else:
                    num_controls_list.append(1) # PyMDP convention for uncontrollable factors
        self.script_parts["preamble_vars"].append(f"num_controls = {num_controls_list}")
        
        # InitialParameterization (or specific matrix blocks)
        # This needs to be adapted based on how the GNN spec is structured for matrices.
        # Assuming a structure like: gnn_spec.get("LikelihoodMatrixA", {})
        param_block = self.gnn_spec.get("InitialParameterization", self.gnn_spec.get("MatrixParameters", {})) # Backward compatibility
        
        self.A_spec = param_block.get("A_Matrix", param_block.get("A")) # Support "A" or "A_Matrix"
        self.B_spec = param_block.get("B_Matrix", param_block.get("B"))
        self.C_spec = param_block.get("C_Vector", param_block.get("C")) # C is a vector of preferences over outcomes
        self.D_spec = param_block.get("D_Vector", param_block.get("D")) # D is initial hidden states
        self.E_spec = param_block.get("E_Vector", param_block.get("E")) # E is prior preferences over policies

        self.agent_hyperparams = self.gnn_spec.get("AgentHyperparameters", {})
        self.simulation_params = self.gnn_spec.get("SimulationParameters", {})

        self._add_log("Finished GNN data extraction.")


    def _get_matrix_data(self, base_name: str, factor_idx: Optional[int] = None, modality_idx: Optional[int] = None) -> Any:
        """DEPRECATED: Use direct spec attributes like self.A_spec instead."""
        # This method is kept for now if old parts of convert_X_matrix still use it,
        # but the goal is to use the structured self.A_spec, self.B_spec etc.
        param_block = self.gnn_spec.get("InitialParameterization", self.gnn_spec.get("MatrixParameters", {}))

        if factor_idx is not None:
            matrix_key = f"{base_name}_f{factor_idx}"
        elif modality_idx is not None:
            matrix_key = f"{base_name}_m{modality_idx}"
        else:
            matrix_key = base_name
        
        data = param_block.get(matrix_key)
        if data is None:
            self._add_log(f"Matrix {matrix_key} not found in GNN InitialParameterization/MatrixParameters.", "WARNING")
        return data

    def convert_A_matrix(self) -> str:
        """Converts GNN's A matrix (likelihood) to PyMDP format."""
        if not self.num_modalities:
            self._add_log("A_matrix: No observation modalities defined. 'A' will be [].", "WARNING")
            self.script_parts["matrix_definitions"].append("A = []")
            return "# A matrix set to [] due to no observation modalities."

        init_code = f"A = utils.obj_array_zeros([[o_dim] + num_states for o_dim in num_obs])"
        self.script_parts["matrix_definitions"].append(init_code)
        
        if not self.A_spec:
            self._add_log("A_matrix: No A_spec provided in GNN. 'A' will be default (zeros from obj_array_zeros).", "INFO")
            return "# A matrix remains default initialized (zeros)."

        # Helper to generate assignment string for A[mod_idx]
        def get_assignment_string(spec_value, indent_level=4) -> Optional[str]:
            if isinstance(spec_value, np.ndarray):
                # Ensure normalization for arrays that are likely distributions
                # Check if it sums to 1.0 along the first axis (axis of outcomes)
                if not np.allclose(np.sum(spec_value, axis=0), 1.0):
                    self._add_log(f"A_matrix (array spec): Array for modality does not sum to 1.0 over outcomes. Will be wrapped with utils.normdist(). Original sum: {np.sum(spec_value, axis=0)}", "DEBUG")
                    return f"utils.normdist({_numpy_array_to_string(spec_value, indent=indent_level+4)})"
                return _numpy_array_to_string(spec_value, indent=indent_level)
            elif isinstance(spec_value, list): # list of lists from JSON, convert to array then format
                try:
                    np_arr = np.array(spec_value)
                    if not np.allclose(np.sum(np_arr, axis=0), 1.0):
                        self._add_log(f"A_matrix (list spec): Array for modality does not sum to 1.0 over outcomes. Will be wrapped with utils.normdist(). Original sum: {np.sum(np_arr, axis=0)}", "DEBUG")
                        return f"utils.normdist({_numpy_array_to_string(np_arr, indent=indent_level+4)})"
                    return _numpy_array_to_string(np_arr, indent=indent_level)
                except Exception as e:
                    self._add_log(f"A_matrix: Error converting list to numpy array: {e}", "ERROR")
                    return None
            return None # Should not happen if spec is validated

        if isinstance(self.A_spec, list): # List of specs per modality
            if len(self.A_spec) != self.num_modalities:
                self._add_log(f"A_matrix: Length of A_spec list ({len(self.A_spec)}) does not match num_modalities ({self.num_modalities}). Processing up to shorter length.", "ERROR")
            
            for mod_idx, mod_spec in enumerate(self.A_spec):
                if mod_idx >= self.num_modalities:
                    break # Avoid index out of bounds if A_spec is longer
                
                if not isinstance(mod_spec, dict):
                    self._add_log(f"A_matrix (modality {mod_idx}): Spec item is not a dictionary. Skipping.", "ERROR")
                    continue

                array_data = mod_spec.get("array")
                rule_string = mod_spec.get("rule_string") # General rule string
                rule_type = mod_spec.get("rule") # Specific predefined rule like "uniform"

                assignment_val_str = None
                log_msg_prefix = f"A_matrix (modality {self.obs_names[mod_idx] if mod_idx < len(self.obs_names) else mod_idx})"

                if array_data is not None:
                    try:
                        np_array_val = np.array(array_data) # Convert list from JSON to numpy array
                        expected_shape = tuple([self.num_obs[mod_idx]] + self.num_states)
                        if np_array_val.shape == expected_shape:
                            assignment_val_str = get_assignment_string(np_array_val)
                            self._add_log(f"{log_msg_prefix}: Defined from 'array' spec.", "INFO")
                        else:
                            self._add_log(f"{log_msg_prefix}: Shape mismatch for 'array'. Expected {expected_shape}, got {np_array_val.shape}. Skipping.", "ERROR")
                    except Exception as e:
                        self._add_log(f"{log_msg_prefix}: Error processing 'array' data: {e}. Skipping.", "ERROR")
                
                elif rule_string is not None:
                    assignment_val_str = rule_string # Use the rule string directly as code
                    self._add_log(f"{log_msg_prefix}: Defined from 'rule_string': {rule_string}", "INFO")
                
                elif rule_type == "uniform":
                    shape_tuple_str = f"({self.num_obs[mod_idx]}, {', '.join(map(str, self.num_states))})"
                    assignment_val_str = f"utils.normdist(np.ones({shape_tuple_str}))"
                    self._add_log(f"{log_msg_prefix}: Defined using 'uniform' rule.", "INFO")
                
                # Add other specific rules here e.g. "identity_mapping" if meaningful for A
                # elif rule_type == "identity_mapping":
                #    if self.num_factors == 1 and self.num_obs[mod_idx] == self.num_states[0]:
                #        assignment_val_str = f"utils.onehot(np.arange({self.num_obs[mod_idx]}), num_values={self.num_obs[mod_idx]})" 
                #    # This is a pymdp specific structure for identity A matrices, needs careful construction
                #    # pymdp.utils.get_A_likelihood_identity might be better if it fits the multi-factor case.
                #    # For now, let user specify via rule_string for complex identity A.
                #    else:
                #        self._add_log(f"{log_msg_prefix}: 'identity_mapping' rule for A is complex for multi-factor or mismatched dims. Use 'rule_string' for precise control.", "WARNING")

                if assignment_val_str:
                    self.script_parts["matrix_definitions"].append(f"A[{mod_idx}] = {assignment_val_str}")
                else:
                    self._add_log(f"{log_msg_prefix}: No valid definition found (array, rule_string, or known rule). A[{mod_idx}] will remain default (zeros).", "WARNING")

        elif isinstance(self.A_spec, dict) and self.num_modalities == 1: # Single spec dict for single modality
            mod_idx = 0
            log_msg_prefix = f"A_matrix (modality {self.obs_names[0] if self.obs_names else 0})"
            array_data = self.A_spec.get("array")
            rule_string = self.A_spec.get("rule_string")
            rule_type = self.A_spec.get("rule")
            assignment_val_str = None

            if array_data is not None:
                try:
                    np_array_val = np.array(array_data)
                    expected_shape = tuple([self.num_obs[mod_idx]] + self.num_states)
                    if np_array_val.shape == expected_shape:
                        assignment_val_str = get_assignment_string(np_array_val)
                        self._add_log(f"{log_msg_prefix}: Defined from 'array' spec.", "INFO")
                    else:
                        self._add_log(f"{log_msg_prefix}: Shape mismatch for 'array'. Expected {expected_shape}, got {np_array_val.shape}. Skipping.", "ERROR")
                except Exception as e:
                    self._add_log(f"{log_msg_prefix}: Error processing 'array' data: {e}. Skipping.", "ERROR")
            
            elif rule_string is not None:
                assignment_val_str = rule_string
                self._add_log(f"{log_msg_prefix}: Defined from 'rule_string': {rule_string}", "INFO")
            
            elif rule_type == "uniform":
                shape_tuple_str = f"({self.num_obs[mod_idx]}, {', '.join(map(str, self.num_states))})"
                assignment_val_str = f"utils.normdist(np.ones({shape_tuple_str}))"
                self._add_log(f"{log_msg_prefix}: Defined using 'uniform' rule.", "INFO")

            if assignment_val_str:
                self.script_parts["matrix_definitions"].append(f"A[{mod_idx}] = {assignment_val_str}")
            else:
                self._add_log(f"{log_msg_prefix}: No valid definition for single modality. A[0] will remain default (zeros).", "WARNING")
        else:
             self._add_log("A_matrix: A_spec format not recognized or incompatible with num_modalities. 'A' will be default (zeros).", "WARNING")

        return "# A matrix processing complete."


    def convert_B_matrix(self) -> str:
        """Converts GNN's B matrix (transition) to PyMDP format."""
        if not self.num_factors:
            self._add_log("B_matrix: No hidden state factors defined. 'B' will be [].", "WARNING")
            self.script_parts["matrix_definitions"].append("B = []")
            return "# B matrix set to [] due to no hidden state factors."

        init_code = f"B = utils.obj_array(num_factors)"
        self.script_parts["matrix_definitions"].append(init_code)

        # Default initialization for all B slices first
        for f_idx in range(self.num_factors):
            num_states_f = self.num_states[f_idx]
            log_msg_prefix_f = f"B_matrix (factor {self.state_names[f_idx] if f_idx < len(self.state_names) else f_idx})"
            if f_idx in self.control_factor_indices:
                num_actions_f = self.num_actions_per_control_factor.get(f_idx, num_states_f) # Default actions = num_states for that factor
                # PyMDP default for controllable B: tile identity matrices for each action
                self.script_parts["matrix_definitions"].append(f"B[{f_idx}] = np.tile(np.eye({num_states_f}), ({num_actions_f}, 1, 1)).transpose(1, 2, 0) # Default for controllable")
                self._add_log(f"{log_msg_prefix_f}: Initialized with default structure for controllable factor (repeated identity matrices).", "DEBUG")
            else: # Uncontrolled
                self.script_parts["matrix_definitions"].append(f"B[{f_idx}] = np.eye({num_states_f})[:, :, np.newaxis] # Default for uncontrolled (identity)")
                self._add_log(f"{log_msg_prefix_f}: Initialized with default identity matrix for uncontrolled factor.", "DEBUG")

        if not self.B_spec:
            self._add_log("B_matrix: No B_spec provided in GNN. B slices will use default initializations (identities).", "INFO")
            return "# B matrix slices remain default initialized."

        # Helper to generate assignment string for B[f_idx]
        def get_b_assignment_string(spec_value, num_states_val, is_controlled_val, num_actions_val, indent_level=4) -> Optional[str]:
            if isinstance(spec_value, (np.ndarray, list)):
                try:
                    np_arr = np.array(spec_value) # Convert if it's a list
                    # Check normalization (sum over first axis (next_state) should be 1 for each current_state, action combination)
                    # For controlled: sum over axis 0. For uncontrolled: sum over axis 0 of the (Ns,Ns) part.
                    if is_controlled_val:
                        if not np.allclose(np.sum(np_arr, axis=0), 1.0):
                            self._add_log(f"B_matrix (array spec, factor): Array for controlled factor does not sum to 1.0 over next_states. Will be wrapped with utils.normdist(). Sums: {np.sum(np_arr, axis=0)}", "DEBUG")
                            return f"utils.normdist({_numpy_array_to_string(np_arr, indent=indent_level+4)})"
                    else: # uncontrolled, expect (Ns, Ns) or (Ns, Ns, 1)
                        arr_to_check = np_arr[:,:,0] if np_arr.ndim == 3 else np_arr
                        if not np.allclose(np.sum(arr_to_check, axis=0), 1.0):
                             self._add_log(f"B_matrix (array spec, factor): Array for uncontrolled factor does not sum to 1.0 over next_states. Will be wrapped with utils.normdist(). Sums: {np.sum(arr_to_check, axis=0)}", "DEBUG")
                             return f"utils.normdist({_numpy_array_to_string(np_arr, indent=indent_level+4)})" # normdist handles 2D or 3D

                    return _numpy_array_to_string(np_arr, indent=indent_level)
                except Exception as e:
                    self._add_log(f"B_matrix: Error converting/processing array/list for B factor: {e}", "ERROR")
                    return None
            return None # Should not happen

        if isinstance(self.B_spec, list): # List of specs per factor
            if len(self.B_spec) != self.num_factors:
                 self._add_log(f"B_matrix: Length of B_spec list ({len(self.B_spec)}) does not match num_factors ({self.num_factors}). Processing up to shorter length.", "ERROR")

            for f_idx, fac_spec in enumerate(self.B_spec):
                if f_idx >= self.num_factors: break

                if not isinstance(fac_spec, dict):
                    self._add_log(f"B_matrix (factor {f_idx}): Spec item is not a dictionary. Using default.", "ERROR")
                    continue
                
                num_states_f = self.num_states[f_idx]
                is_controlled = f_idx in self.control_factor_indices
                num_actions_f = self.num_actions_per_control_factor.get(f_idx, 1)
                log_msg_prefix_f = f"B_matrix (factor {self.state_names[f_idx] if f_idx < len(self.state_names) else f_idx})"

                array_data = fac_spec.get("array")
                rule_string = fac_spec.get("rule_string")
                rule_type = fac_spec.get("rule")
                assignment_val_str = None

                if array_data is not None:
                    try:
                        np_array_val = np.array(array_data)
                        expected_shape = (num_states_f, num_states_f, num_actions_f) if is_controlled else (num_states_f, num_states_f) # or (Ns,Ns,1)
                        
                        if is_controlled and np_array_val.shape == expected_shape:
                            assignment_val_str = get_b_assignment_string(np_array_val, num_states_f, is_controlled, num_actions_f)
                            self._add_log(f"{log_msg_prefix_f}: Defined from 'array' spec (controlled).", "INFO")
                        elif not is_controlled and (np_array_val.shape == expected_shape or np_array_val.shape == (num_states_f, num_states_f, 1)):
                            arr_to_assign = np_array_val if np_array_val.ndim == 3 else np_array_val[:, :, np.newaxis]
                            assignment_val_str = get_b_assignment_string(arr_to_assign, num_states_f, is_controlled, 1)
                            self._add_log(f"{log_msg_prefix_f}: Defined from 'array' spec (uncontrolled).", "INFO")
                        else:
                            self._add_log(f"{log_msg_prefix_f}: Shape mismatch for 'array'. Expected {expected_shape}, got {np_array_val.shape}. Using default.", "ERROR")
                    except Exception as e:
                        self._add_log(f"{log_msg_prefix_f}: Error processing 'array' data: {e}. Using default.", "ERROR")
                
                elif rule_string is not None:
                    assignment_val_str = rule_string
                    self._add_log(f"{log_msg_prefix_f}: Defined from 'rule_string': {rule_string}", "INFO")

                elif rule_type == "identity":
                    # Default initialization already handles this, but explicit GNN spec confirms it.
                    self._add_log(f"{log_msg_prefix_f}: Confirmed 'identity' rule (already default).", "INFO")
                    # No need to change assignment_val_str, default is already set up.
                    continue # Continue to next factor, current B[f_idx] is already identity.
                
                if assignment_val_str:
                    self.script_parts["matrix_definitions"].append(f"B[{f_idx}] = {assignment_val_str}")
                else:
                    if rule_type != "identity": # If not identity and no other rule matched, log warning
                        self._add_log(f"{log_msg_prefix_f}: No valid definition found. Using default identity B[{f_idx}].", "WARNING")
        
        # Handling for B_spec as a single dict for a single factor model (less common for B but possible)
        elif isinstance(self.B_spec, dict) and self.num_factors == 1:
            f_idx = 0
            log_msg_prefix_f = f"B_matrix (factor {self.state_names[0] if self.state_names else 0})"
            num_states_f = self.num_states[f_idx]
            is_controlled = f_idx in self.control_factor_indices
            num_actions_f = self.num_actions_per_control_factor.get(f_idx, 1)
            
            array_data = self.B_spec.get("array")
            rule_string = self.B_spec.get("rule_string")
            rule_type = self.B_spec.get("rule")
            assignment_val_str = None

            if array_data is not None:
                try:
                    np_array_val = np.array(array_data)
                    expected_shape = (num_states_f, num_states_f, num_actions_f) if is_controlled else (num_states_f, num_states_f)
                    if is_controlled and np_array_val.shape == expected_shape:
                        assignment_val_str = get_b_assignment_string(np_array_val, num_states_f, is_controlled, num_actions_f)
                    elif not is_controlled and (np_array_val.shape == expected_shape or np_array_val.shape == (num_states_f, num_states_f, 1)):
                        arr_to_assign = np_array_val if np_array_val.ndim == 3 else np_array_val[:, :, np.newaxis]
                        assignment_val_str = get_b_assignment_string(arr_to_assign, num_states_f, is_controlled, 1)
                    else:
                        self._add_log(f"{log_msg_prefix_f}: Shape mismatch for 'array'. Expected {expected_shape}, got {np_array_val.shape}. Using default.", "ERROR")
                except Exception as e:
                    self._add_log(f"{log_msg_prefix_f}: Error processing 'array' data for single factor: {e}. Using default.", "ERROR")
            
            elif rule_string is not None:
                assignment_val_str = rule_string
            
            elif rule_type == "identity":
                 self._add_log(f"{log_msg_prefix_f}: Confirmed 'identity' rule (already default).", "INFO")
                 assignment_val_str = None # Will keep default
            
            if assignment_val_str:
                self.script_parts["matrix_definitions"].append(f"B[{f_idx}] = {assignment_val_str}")
            else:
                if rule_type != "identity":
                    self._add_log(f"{log_msg_prefix_f}: No valid definition for single factor. Using default identity B[0].", "WARNING")
        else:
            self._add_log("B_matrix: B_spec format not recognized or incompatible with num_factors. B slices will use defaults.", "WARNING")

        return "# B matrix processing complete."

    def convert_C_vector(self) -> str:
        """Converts GNN's C vector (preferences) to PyMDP format."""
        if not self.num_modalities:
            self._add_log("C_vector: No observation modalities defined. 'C' will be None.", "INFO") # C is optional
            # Directly append the definition to script_parts
            self.script_parts["matrix_definitions"].append(generate_pymdp_matrix_definition("C", None)) 
            return "# C matrix set to None due to no observation modalities." # Return value is for logging/info

        # Initialize C = utils.obj_array_zeros(num_obs)
        init_code = f"C = utils.obj_array_zeros(num_obs)" # num_obs is list of outcome counts per modality
        self.script_parts["matrix_definitions"].append(init_code)

        if self.C_spec:
            if isinstance(self.C_spec, list): # List of specs per modality
                for mod_idx, mod_c_spec in enumerate(self.C_spec):
                    if mod_c_spec is None: continue # Allow sparse C definition
                    
                    array_data = mod_c_spec.get("array") # Expects 1D array of length num_obs[mod_idx]
                    if array_data:
                        try:
                            np_array = np.array(array_data)
                            expected_shape = (self.num_obs[mod_idx],)
                            if np_array.shape == expected_shape:
                                self.script_parts["matrix_definitions"].append(f"C[{mod_idx}] = {_numpy_array_to_string(np_array, indent=4)}")
                            else:
                                self._add_log(f"C_vector (modality {mod_idx}): Shape mismatch. Expected {expected_shape}, got {np_array.shape}.", "ERROR")
                        except Exception as e:
                             self._add_log(f"C_vector (modality {mod_idx}): Error processing array data: {e}.", "ERROR")
                    else:
                         self._add_log(f"C_vector (modality {mod_idx}): No 'array' found in spec. C[{mod_idx}] will be zeros.", "INFO")
            # Add handling for C_spec as single dict if num_modalities == 1
            else:
                self._add_log("C_vector: C_spec format not recognized for detailed construction. C will be zeros.", "WARNING")
        else:
            self._add_log("C_vector: No C_spec. C will be initialized to zeros by obj_array_zeros.", "INFO")
        
        return "# C vector definition appended to script parts"
        
    def convert_D_vector(self) -> str:
        """Converts GNN's D vector (initial hidden states) to PyMDP format."""
        if not self.num_factors:
            self._add_log("D_vector: No hidden state factors defined. 'D' will be [].", "ERROR") # D is usually required
            self.script_parts["matrix_definitions"].append(
                generate_pymdp_matrix_definition("D", None, is_object_array=True)
            )
            return "# D matrix set to [] due to no hidden state factors."

        # Initialize D = utils.obj_array(num_factors)
        init_code = f"D = utils.obj_array(num_factors)"
        self.script_parts["matrix_definitions"].append(init_code)
        
        # Default to uniform if no D_spec
        for f_idx in range(self.num_factors):
            self.script_parts["matrix_definitions"].append(f"D[{f_idx}] = utils.normdist(np.ones({self.num_states[f_idx]})) # Default: uniform D for factor {f_idx}")

        if self.D_spec:
            if isinstance(self.D_spec, list): # List of specs per factor
                for f_idx, fac_d_spec in enumerate(self.D_spec):
                    if fac_d_spec is None: continue

                    array_data = fac_d_spec.get("array") # Expects 1D array of length num_states[f_idx]
                    if array_data:
                        try:
                            np_array = np.array(array_data)
                            expected_shape = (self.num_states[f_idx],)
                            if np_array.shape == expected_shape:
                                # Ensure it's normalized if it's a distribution
                                if np.isclose(np.sum(np_array), 1.0):
                                     assign_str = _numpy_array_to_string(np_array, indent=4)
                                else: # Normalize
                                     assign_str = f"utils.normdist({_numpy_array_to_string(np_array, indent=4)})"
                                self.script_parts["matrix_definitions"].append(f"D[{f_idx}] = {assign_str}")
                            else:
                                self._add_log(f"D_vector (factor {f_idx}): Shape mismatch. Expected {expected_shape}, got {np_array.shape}. Using default uniform.", "ERROR")
                        except Exception as e:
                             self._add_log(f"D_vector (factor {f_idx}): Error processing array data: {e}. Using default uniform.", "ERROR")
                    else:
                        self._add_log(f"D_vector (factor {f_idx}): No 'array' found. Using default uniform D[{f_idx}].", "INFO")
            # Add handling for D_spec as single dict if num_factors == 1
            else:
                 self._add_log("D_vector: D_spec format not recognized. D factors will be default uniform.", "WARNING")
        else:
            self._add_log("D_vector: No D_spec. D factors will be default uniform.", "INFO")

        return "# D vector definition appended to script parts"

    def convert_E_vector(self) -> str:
        """Converts GNN's E vector (prior preferences for policies) to PyMDP format."""
        # E is optional. It's a flat vector (num_policies) or not used if policy_len=1
        if self.E_spec:
            array_data = self.E_spec.get("array")
            if array_data:
                try:
                    np_array = np.array(array_data)
                    # Shape depends on policy_len and num_control_states, complex to validate here without policy_len
                    # For now, just generate the assignment
                    self.script_parts["matrix_definitions"].append(generate_pymdp_matrix_definition("E", np_array))
                    self._add_log("E_vector: Defined from GNN spec.", "INFO")
                except Exception as e:
                    self._add_log(f"E_vector: Error processing array data: {e}. 'E' will be None.", "ERROR")
                    self.script_parts["matrix_definitions"].append("E = None")
            else:
                self.script_parts["matrix_definitions"].append("E = None") # No array in E_spec
        else:
            self.script_parts["matrix_definitions"].append("E = None") # E is often None by default
        return "# E vector definition appended to script parts"


    def extract_agent_hyperparameters(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str,Any]]]:
        """Extracts control, learning, and algorithm parameters from GNN spec for Agent."""
        # This method needs to be updated if GNN spec for these changes
        self._add_log("AgentHyperparameters: Extracting learning and algorithm parameter dicts.")
        
        # These are generic dictionaries for other parameters not explicitly handled by Agent constructor args
        # control_params_dict = self.agent_hyperparams.get("control_parameters", {}) # Potentially unused or for other control aspects
        learning_params_dict = self.agent_hyperparams.get("learning_parameters", {})
        algorithm_params_dict = self.agent_hyperparams.get("algorithm_parameters", {})
        
        # Return None for control_params_dict if it's not meant for direct Agent args here.
        # Or ensure it only contains parameters that pymdp.Agent would accept via **kwargs if that's supported.
        # For now, assume it's not directly used for fixed Agent constructor args.
        return None, learning_params_dict, algorithm_params_dict

    def generate_agent_instantiation_code(self) -> str:
        model_matrix_params = {"A": "A", "B": "B", "C": "C"}
        if self.num_factors > 0:
            model_matrix_params["D"] = "D"
        if self.E_spec and self.E_spec.get("array") is not None:
             model_matrix_params["E"] = "E"

        _unused_control_dict, learning_params, algorithm_params = self.extract_agent_hyperparameters()
        
        # Agent-specific constructor arguments from self.agent_hyperparams
        policy_len = self.agent_hyperparams.get("policy_len") 
        # Pass the variable name "control_fac_idx" which is defined in preamble
        control_fac_idx_var_name_to_pass = "control_fac_idx" 
        
        use_utility = self.agent_hyperparams.get("use_utility") 
        use_states_info_gain = self.agent_hyperparams.get("use_states_info_gain")
        use_param_info_gain = self.agent_hyperparams.get("use_param_info_gain")
        action_selection = self.agent_hyperparams.get("action_selection")

        return generate_pymdp_agent_instantiation(
            self.model_name, 
            model_params=model_matrix_params,
            # Pass names of script variables for dimensions
            # Pass specific agent constructor args
            control_fac_idx_var_name=control_fac_idx_var_name_to_pass,
            policy_len=policy_len,
            use_utility=use_utility,
            use_states_info_gain=use_states_info_gain,
            use_param_info_gain=use_param_info_gain,
            action_selection=action_selection,
            # Pass through other parameter dicts
            learning_params=learning_params,
            algorithm_params=algorithm_params
        )

    def generate_example_usage_code(self) -> List[str]:
        """Generates a runnable example usage block based on the GNN spec and user's example."""
        usage_lines = ["", "# --- Example Usage ---", "if __name__ == '__main__':"]
        indent = "    "

        sim_T = self.simulation_params.get("timesteps", 5)
        init_o_raw = self.simulation_params.get("initial_observations")
        init_s_raw = self.simulation_params.get("initial_true_states")
        use_gp_copy = self.simulation_params.get("use_generative_process_copy", True)

        print_obs = self.simulation_params.get("print_observations", True)
        print_beliefs = self.simulation_params.get("print_belief_states", True)
        print_actions = self.simulation_params.get("print_actions", True)
        print_states = self.simulation_params.get("print_true_states", True)
        
        usage_lines.append(f"{indent}# Initialize agent (already done above)")
        usage_lines.append(f"{indent}agent = {self.model_name}")
        usage_lines.append(f"{indent}print(f\"Agent '{self.model_name}' initialized with {self.num_factors} factors and {self.num_modalities} modalities.\")")

        # Initial observation
        if init_o_raw and isinstance(init_o_raw, list) and len(init_o_raw) == self.num_modalities:
            usage_lines.append(f"{indent}o_current = {init_o_raw} # Initial observation from GNN spec")
        else: # Default initial observation (e.g., first outcome for each modality or a placeholder)
            default_o = [0] * self.num_modalities if self.num_modalities > 0 else "None"
            usage_lines.append(f"{indent}o_current = {default_o} # Example initial observation (e.g. first outcome for each modality)")
            if not init_o_raw and self.num_modalities > 0 : self._add_log("Simulation: No 'initial_observations' in GNN, using default.", "INFO")
        
        # Initial true state (for simulation purposes, not agent's belief D)
        if init_s_raw and isinstance(init_s_raw, list) and len(init_s_raw) == self.num_factors:
             usage_lines.append(f"{indent}s_current = {init_s_raw} # Initial true states from GNN spec")
        else:
            default_s = [0] * self.num_factors if self.num_factors > 0 else "None" # Example: first state for each factor
            usage_lines.append(f"{indent}s_current = {default_s} # Example initial true states for simulation")
            if not init_s_raw and self.num_factors > 0: self._add_log("Simulation: No 'initial_true_states' in GNN, using default.", "INFO")

        usage_lines.append(f"{indent}T = {sim_T} # Number of timesteps")

        if use_gp_copy:
            usage_lines.append(f"{indent}A_gen_process = copy.deepcopy(A)")
            usage_lines.append(f"{indent}B_gen_process = copy.deepcopy(B)")
        else:
            usage_lines.append(f"{indent}A_gen_process = A")
            usage_lines.append(f"{indent}B_gen_process = B")
        
        usage_lines.append("")
        usage_lines.append(f"{indent}for t_step in range(T):")
        inner_indent = indent * 2

        if print_obs:
            usage_lines.append(f"{inner_indent}print(f\"\\n--- Timestep {{t_step + 1}} ---\")")
            usage_lines.append(f"{inner_indent}if o_current is not None:")
            usage_lines.append(f"{inner_indent}{indent}for g_idx, o_val in enumerate(o_current):")
            usage_lines.append(f"{inner_indent}{indent}{indent}print(f\"Observation ({{obs_names[g_idx] if obs_names else f'Modality {{g_idx}}'}}): {{o_val}}\")")
        
        usage_lines.append(f"{inner_indent}# Infer states")
        usage_lines.append(f"{inner_indent}qs_current = agent.infer_states(o_current)")
        if print_beliefs:
            usage_lines.append(f"{inner_indent}if qs_current is not None:")
            usage_lines.append(f"{inner_indent}{indent}for f_idx, q_val in enumerate(qs_current):")
            usage_lines.append(f"{inner_indent}{indent}{indent}print(f\"Beliefs about {{state_names[f_idx] if state_names else f'Factor {{f_idx}}'}}: {{q_val}}\")")

        usage_lines.append("")
        usage_lines.append(f"{inner_indent}# Infer policies and sample action")
        usage_lines.append(f"{inner_indent}q_pi_current, efe_current = agent.infer_policies()")
        usage_lines.append(f"{inner_indent}action_agent = agent.sample_action()")
        
        # Map agent action (control_factor list) to environment action (all factors list)
        usage_lines.append(f"{inner_indent}# Map agent's action (on control factors) to full environment action vector")
        usage_lines.append(f"{inner_indent}action_env = np.zeros(num_factors, dtype=int)")
        usage_lines.append(f"{inner_indent}if control_fac_idx and action_agent is not None:")
        usage_lines.append(f"{inner_indent}{indent}for i, cf_idx in enumerate(control_fac_idx):")
        usage_lines.append(f"{inner_indent}{indent}{indent}action_env[cf_idx] = int(action_agent[i])")


        if print_actions:
            usage_lines.append(f"{inner_indent}# Construct action names for printing")
            usage_lines.append(f"{inner_indent}action_names_str_list = []")
            usage_lines.append(f"{inner_indent}if control_fac_idx and action_agent is not None:")
            usage_lines.append(f"{inner_indent}{indent}for i, cf_idx in enumerate(control_fac_idx):")
            usage_lines.append(f"{inner_indent}{indent}{indent}factor_action_name_list = agent.action_names.get(cf_idx, []) if hasattr(agent, 'action_names') and isinstance(agent.action_names, dict) else []")
            usage_lines.append(f"{inner_indent}{indent}{indent}action_idx_on_factor = int(action_agent[i])")
            usage_lines.append(f"{inner_indent}{indent}{indent}if factor_action_name_list and action_idx_on_factor < len(factor_action_name_list):")
            usage_lines.append(f"{inner_indent}{indent}{indent}{indent}action_names_str_list.append(f\"{{state_names[cf_idx] if state_names else f'Factor {{cf_idx}}'}}: {{factor_action_name_list[action_idx_on_factor]}} (idx {{action_idx_on_factor}})\")")
            usage_lines.append(f"{inner_indent}{indent}{indent}else:")
            usage_lines.append(f"{inner_indent}{indent}{indent}{indent}action_names_str_list.append(f\"{{state_names[cf_idx] if state_names else f'Factor {{cf_idx}}'}}: Action idx {{action_idx_on_factor}}\")")
            usage_lines.append(f"{inner_indent}print(f\"Action taken: {{', '.join(action_names_str_list) if action_names_str_list else 'No controllable actions or names not found'}}\")")
            usage_lines.append(f"{inner_indent}# Raw sampled action_agent: {{action_agent}}")
            usage_lines.append(f"{inner_indent}# Mapped action_env for B matrix: {{action_env}}")

        usage_lines.append("")
        usage_lines.append(f"{inner_indent}# Update true states of the environment based on action")
        usage_lines.append(f"{inner_indent}s_next = np.zeros(num_factors, dtype=int)")
        usage_lines.append(f"{inner_indent}if s_current is not None and B_gen_process is not None:")
        usage_lines.append(f"{inner_indent}{indent}for f_idx in range(num_factors):")
        usage_lines.append(f"{inner_indent}{indent}{indent}# B_gen_process[f_idx] shape: (num_states[f_idx], num_states[f_idx], num_actions_for_this_factor_or_1)")
        usage_lines.append(f"{inner_indent}{indent}{indent}action_for_factor = action_env[f_idx] if f_idx in control_fac_idx else 0")
        usage_lines.append(f"{inner_indent}{indent}{indent}s_next[f_idx] = utils.sample(B_gen_process[f_idx][:, s_current[f_idx], action_for_factor])")
        usage_lines.append(f"{inner_indent}s_current = s_next.tolist()")

        if print_states:
            usage_lines.append(f"{inner_indent}if s_current is not None:")
            usage_lines.append(f"{inner_indent}{indent}for f_idx, s_val in enumerate(s_current):")
            usage_lines.append(f"{inner_indent}{indent}{indent}print(f\"New true state ({{state_names[f_idx] if state_names else f'Factor {{f_idx}}'}}): {{s_val}}\")")


        usage_lines.append("")
        usage_lines.append(f"{inner_indent}# Generate next observation based on new true states")
        usage_lines.append(f"{inner_indent}o_next = np.zeros(num_modalities, dtype=int)")
        usage_lines.append(f"{inner_indent}if s_current is not None and A_gen_process is not None:")
        usage_lines.append(f"{inner_indent}{indent}for g_idx in range(num_modalities):")
        usage_lines.append(f"{inner_indent}{indent}{indent}# A_gen_process[g_idx] shape: (num_obs[g_idx], num_states[0], num_states[1], ...)")
        usage_lines.append(f"{inner_indent}{indent}{indent}# Construct index for A matrix: (outcome_idx, s_f0, s_f1, ...)")
        usage_lines.append(f"{inner_indent}{indent}{indent}prob_vector = A_gen_process[g_idx][:, " + ", ".join([f"s_current[{sf_i}]" for sf_i in range(self.num_factors)]) + "]")
        usage_lines.append(f"{inner_indent}{indent}{indent}o_next[g_idx] = utils.sample(prob_vector)")
        usage_lines.append(f"{inner_indent}o_current = o_next.tolist()")
        
        usage_lines.append("")
        usage_lines.append(f"{indent}print(f\"\\nSimulation finished after {{T}} timesteps.\")")

        return usage_lines


    def get_full_python_script(self, include_example_usage: bool = True) -> str:
        """Assembles all parts into a single Python script string."""
        
        # Pre-computation / matrix generation calls
        # These methods will now append to self.script_parts["matrix_definitions"]
        self.convert_A_matrix()
        self.convert_B_matrix()
        self.convert_C_vector()
        self.convert_D_vector()
        self.convert_E_vector() # E matrix for policy priors

        # Agent instantiation
        self.script_parts["agent_instantiation"].append(self.generate_agent_instantiation_code())
        
        # Example Usage (now more detailed)
        if include_example_usage:
            self.script_parts["example_usage"] = self.generate_example_usage_code()
        else:
            self.script_parts["example_usage"] = ["# Example usage block skipped as per options."]


        # Assemble the script
        script_content = []
        script_content.extend(self.script_parts["imports"])
        script_content.append("")
        
        # Add GNN to PyMDP Conversion Summary (from self.conversion_log)
        summary_header = ["# --- GNN to PyMDP Conversion Summary ---"]
        summary_lines = [f"# {log_entry}" for log_entry in self.conversion_log]
        summary_footer = ["# --- End of GNN to PyMDP Conversion Summary ---"]
        script_content.extend(summary_header)
        script_content.extend(summary_lines)
        script_content.extend(summary_footer)
        script_content.append("")
        script_content.append("")

        script_content.extend(self.script_parts["comments"]) # Model specific comments
        script_content.append("")

        script_content.extend(self.script_parts["preamble_vars"]) # obs_names, num_states etc.
        script_content.append("")

        script_content.append("# --- Matrix Definitions ---")
        script_content.extend(self.script_parts["matrix_definitions"])
        script_content.append("")
        
        script_content.append("# --- Agent Instantiation ---")
        script_content.extend(self.script_parts["agent_instantiation"])
        script_content.append("")

        script_content.extend(self.script_parts["example_usage"])
        
        # Add runtime debug block
        debug_block = [
            "print('--- PyMDP Runtime Debug ---')",
            "try:",
            "    import pymdp",
            "    print(f'AGENT_SCRIPT: Imported pymdp version: {pymdp.__version__}')",
            "    print(f'AGENT_SCRIPT: pymdp module location: {pymdp.__file__}')",
            "    from pymdp.agent import Agent",
            "    print(f'AGENT_SCRIPT: Imported Agent: {Agent}')",
            "    print(f'AGENT_SCRIPT: Agent module location: {inspect.getfile(Agent)}')",
            "    # Check if required variables are in global scope",
            "    required_vars = ['A', 'B', 'C', 'D', 'num_obs', 'num_states', 'num_controls', 'control_factor_idx', 'agent_params']",
            "    print('AGENT_SCRIPT: Checking for required variables in global scope:')",
            "    for var_name in required_vars:",
            "        if var_name in globals():",
            "            print(f'  AGENT_SCRIPT: {var_name} is defined. Value (first 100 chars): {str(globals()[var_name])[:100]}')",
            "        else:",
            "            print(f'  AGENT_SCRIPT: {var_name} is NOT defined.')",
            "    # Instantiate agent to catch initialization errors",
            "    print('AGENT_SCRIPT: Attempting to instantiate agent with defined parameters...')",
            "    temp_agent = Agent(**agent_params)",
            "    print(f'AGENT_SCRIPT: Agent successfully instantiated: {temp_agent}')",
            "except Exception as e_debug:",
            "    print(f'AGENT_SCRIPT: Error during PyMDP runtime debug: {e_debug}')", # Ensure this f-string is properly terminated
            "    print(f'AGENT_SCRIPT: Traceback:\\n{traceback.format_exc()}')",
            "print('--- End PyMDP Runtime Debug ---')",
        ]
        script_content.extend(debug_block)

        script_content.append(f"# --- GNN Model: {self.model_name} ---\n")

        return "\n".join(script_content)

def render_gnn_to_pymdp(
    gnn_spec: Dict[str, Any],
    output_script_path: Path,
    options: Optional[Dict[str, Any]] = None # e.g. {"include_example_usage": True}
) -> Tuple[bool, str, List[str]]:
    """
    Main function to render a GNN specification to a PyMDP Python script.

    Args:
        gnn_spec: The GNN specification as a Python dictionary.
        output_script_path: The path where the generated Python script will be saved.
        options: Dictionary of rendering options. 
                 Currently supports "include_example_usage" (bool, default True).

    Returns:
        A tuple (success: bool, message: str, artifact_uris: List[str]).
        `artifact_uris` will contain a file URI to the generated script on success.
    """
    options = options or {}
    include_example_usage = options.get("include_example_usage", True)

    try:
        logger.info(f"Initializing GNN to PyMDP converter for model: {gnn_spec.get('ModelName', 'UnknownModel')}")
        converter = GnnToPyMdpConverter(gnn_spec)
        
        logger.info("Generating PyMDP Python script content...")
        python_script_content = converter.get_full_python_script(
            include_example_usage=include_example_usage
        )
        
        logger.info(f"Writing PyMDP script to: {output_script_path}")
        with open(output_script_path, "w", encoding='utf-8') as f:
            f.write(python_script_content)
        
        success_msg = f"Successfully wrote PyMDP script: {output_script_path.name}"
        logger.info(success_msg)
        
        # Include conversion log in the final message for clarity, perhaps capped
        log_summary = "\n".join(converter.conversion_log[:20]) # First 20 log messages
        if len(converter.conversion_log) > 20:
            log_summary += "\n... (log truncated)"
            
        return True, f"{success_msg}\nConversion Log Summary:\n{log_summary}", [output_script_path.as_uri()]

    except Exception as e:
        error_msg = f"Failed to render GNN to PyMDP: {e}"
        logger.exception(error_msg) # Log full traceback
        return False, error_msg, []

# Placeholder for a more sophisticated GNN parser if this script were to load GNN files directly
# For now, it expects a pre-parsed gnn_spec dictionary.
def placeholder_gnn_parser_pymdp(gnn_file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Placeholder function to parse a GNN Markdown file into a dictionary.
    This would ideally live in a dedicated GNN parsing module.
    """
    logger.warning(f"Using placeholder GNN parser for {gnn_file_path}. This is for testing/dev only.")
    if not gnn_file_path.is_file():
        logger.error(f"GNN file not found: {gnn_file_path}")
        return None
    try:
        # This is a HACK: assumes GNN file is JSON for placeholder
        with open(gnn_file_path, 'r') as f:
            # A real parser would handle Markdown structure (StateSpaceBlock, Connections, etc.)
            # and convert to the structured dict `GnnToPyMdpConverter` expects.
            # For this placeholder, we assume it's a JSON that directly matches the expected dict structure.
            # This will likely fail for actual .md GNN files.
            data = json.load(f) 
        return data
    except Exception as e:
        logger.error(f"Error in placeholder GNN parser for {gnn_file_path}: {e}")
        return None

# Example of how this module might be called (e.g., from a main pipeline script)
if __name__ == '__main__':
    # This is for direct testing of this module.
    # Assumes a JSON file that has the structure GnnToPyMdpConverter expects.
    
    # Setup basic logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a dummy GNN spec dictionary for testing
    # This should reflect the structure that the GNN parser (src/gnn/gnn_parser.py) would produce.
    # And it should also reflect the new, more detailed GNN specification proposed.
    
    dummy_gnn_spec_path = Path(__file__).parent / "_test_dummy_gnn_spec_pymdp.json"

    # Generate a more complex dummy spec for testing
    test_spec = {
        "ModelName": "Test Agent Advanced",
        "StateSpaceBlock": {
            "ObservationModalities": [
                {"modality_name": "Visual", "num_outcomes": 3, "outcome_names": ["Red", "Green", "Blue"]},
                {"modality_name": "Audio", "num_outcomes": 2, "outcome_names": ["HighPitch", "LowPitch"]}
            ],
            "HiddenStateFactors": [
                {"factor_name": "Location", "num_states": 3, "controllable": True, "num_actions": 3, "action_names": ["Stay", "MoveLeft", "MoveRight"]},
                {"factor_name": "InternalContext", "num_states": 2, "controllable": False}
            ]
        },
        "InitialParameterization": {
            "A_Matrix": [ # List per modality
                { # Modality 0 (Visual)
                  "array": np.random.rand(3, 3, 2) # (n_obs_0, n_states_0, n_states_1)
                  # Could also be: "rule": "uniform"
                },
                { # Modality 1 (Audio)
                  "array": np.random.rand(2, 3, 2)
                }
            ],
            "B_Matrix": [ # List per factor
                { # Factor 0 (Location - controlled)
                  "array": np.random.rand(3,3,3) # (n_states_0, n_states_0, n_actions_0)
                },
                { # Factor 1 (InternalContext - uncontrolled)
                  "array": np.eye(2) # (n_states_1, n_states_1) -> will be expanded to (2,2,1)
                }
            ],
            "C_Vector": [ # List per modality (preferences)
                {"array": [1.0, 0.0, -1.0]}, # Pref for Visual
                {"array": [0.5, -0.5]}      # Pref for Audio
            ],
            "D_Vector": [ # List per factor (initial beliefs)
                {"array": [0.8, 0.1, 0.1]}, # Belief for Location
                {"array": [0.5, 0.5]}      # Belief for InternalContext
            ],
            "E_Vector": {"array": [0.25, 0.25, 0.25, 0.25]} # Example: Prior over 4 policies
        },
        "AgentHyperparameters": {
            "policy_len": 3,
            "use_utility": True,
            "action_selection": "stochastic"
        },
        "SimulationParameters": {
            "timesteps": 10,
            "initial_observations": [0,0],
            "initial_true_states": [0,0],
            "use_generative_process_copy": True,
            "print_observations": True,
            "print_belief_states": True,
            "print_actions": True,
            "print_true_states": True
        }
    }
    # Normalize A matrices in test_spec
    for mod_spec in test_spec["InitialParameterization"]["A_Matrix"]:
        if "array" in mod_spec and isinstance(mod_spec["array"], np.ndarray):
             mod_spec["array"] = pymdp_maths.norm_dist(mod_spec["array"]) if _PMDP_AVAILABLE else mod_spec["array"] / np.sum(mod_spec["array"], axis=0, keepdims=True)


    # Normalize B matrices
    for f_idx, fac_spec in enumerate(test_spec["InitialParameterization"]["B_Matrix"]):
         if "array" in fac_spec and isinstance(fac_spec["array"], np.ndarray):
            b_arr = fac_spec["array"]
            if f_idx in test_spec["StateSpaceBlock"]["HiddenStateFactors"][f_idx].get("controllable", False): # Controlled
                fac_spec["array"] = pymdp_maths.norm_dist(b_arr) if _PMDP_AVAILABLE else b_arr / np.sum(b_arr, axis=0, keepdims=True)

            else: # Uncontrolled
                # Ensure it's (Ns, Ns) then it will be expanded
                if b_arr.ndim == 2 and b_arr.shape[0] == b_arr.shape[1]:
                     fac_spec["array"] = pymdp_maths.norm_dist(b_arr) if _PMDP_AVAILABLE else b_arr / np.sum(b_arr, axis=0, keepdims=True)


    # Normalize D vectors
    for fac_spec in test_spec["InitialParameterization"]["D_Vector"]:
        if "array" in fac_spec and isinstance(fac_spec["array"], np.ndarray):
            fac_spec["array"] = pymdp_maths.norm_dist(fac_spec["array"]) if _PMDP_AVAILABLE else fac_spec["array"] / np.sum(fac_spec["array"])


    # Serialize the complex spec with numpy arrays if needed for placeholder_gnn_parser_pymdp
    # For direct use, it's fine.
    # with open(dummy_gnn_spec_path, 'w') as f:
    #    json.dump(test_spec, f, indent=2, cls=NumpyEncoder) # Needs a NumpyEncoder for json.dump

    output_dir = Path(__file__).parent.parent.parent / "output" / "test_render_pymdp" 
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / f"{test_spec['ModelName'].replace(' ','_')}_rendered.py"

    logger.info(f"Testing PyMDP rendering with dummy spec. Output to: {script_path}")
    
    # Use the test_spec directly
    success, msg, artifacts = render_gnn_to_pymdp(test_spec, script_path)

    if success:
        print(f"Dummy PyMDP script generated successfully: {artifacts}")
        print(f"Message: {msg}")
        # Try to run the generated script
        # import subprocess
        # try:
        #     print("\nRunning generated script...")
        #     result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True, check=True)
        #     print("Script Output:\n", result.stdout)
        # except subprocess.CalledProcessError as e:
        #     print("Error running script:")
        #     print("STDOUT:", e.stdout)
        #     print("STDERR:", e.stderr)
        # except FileNotFoundError:
        #     print(f"Could not find python interpreter {sys.executable} or script {script_path}")

    else:
        print(f"Dummy PyMDP script generation failed.")
        print(f"Message: {msg}")

