"""
PyMDP Converter Module for GNN Specifications

This module contains the core logic for converting GNN specifications
into PyMDP-compatible data structures and scripts.
"""

import logging
from pathlib import Path
import re
import ast
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Deferred numpy import to avoid recursion issues
def _get_numpy():
    """Safely import numpy with fallback."""
    try:
        import numpy as np
        return np
    except RecursionError:
        # Handle recursion error by using a minimal implementation
        logging.warning("Numpy recursion detected, using fallback implementation")
        return None
    except ImportError:
        logging.warning("Numpy not available, using fallback implementation")
        return None

# Use a lazy loading approach for numpy
_numpy_cache = None

def get_numpy():
    """Get numpy module with caching."""
    global _numpy_cache
    if _numpy_cache is None:
        _numpy_cache = _get_numpy()
    return _numpy_cache

class NumpySafeOperations:
    """Safe wrapper for numpy operations with fallbacks."""
    
    def __init__(self):
        self.np = get_numpy()
    
    def array(self, data):
        """Create array with fallback."""
        if self.np is not None:
            return self.np.array(data)
        return data  # Return raw data if numpy not available
    
    def ones(self, shape):
        """Create ones array with fallback."""
        if self.np is not None:
            return self.np.ones(shape)
        if isinstance(shape, (int, tuple)):
            if isinstance(shape, int):
                return [1.0] * shape
            # For multi-dimensional shapes, create nested lists
            return self._create_ones_list(shape)
        return [1.0]
    
    def zeros(self, shape):
        """Create zeros array with fallback."""
        if self.np is not None:
            return self.np.zeros(shape)
        if isinstance(shape, (int, tuple)):
            if isinstance(shape, int):
                return [0.0] * shape
            return self._create_zeros_list(shape)
        return [0.0]
    
    def eye(self, n):
        """Create identity matrix with fallback."""
        if self.np is not None:
            return self.np.eye(n)
        # Create identity matrix as nested lists
        matrix = []
        for i in range(n):
            row = [0.0] * n
            row[i] = 1.0
            matrix.append(row)
        return matrix
    
    def empty(self, shape, dtype=None):
        """Create empty array with fallback."""
        if self.np is not None:
            return self.np.empty(shape, dtype=dtype)
        # Return None for object dtype, appropriate structure for others
        if dtype == object:
            return [None] * shape if isinstance(shape, int) else None
        return self.zeros(shape)
    
    def sum(self, arr, axis=None, keepdims=False):
        """Sum with fallback."""
        if self.np is not None:
            return self.np.sum(arr, axis=axis, keepdims=keepdims)
        # Simple fallback sum
        if hasattr(arr, '__iter__'):
            return sum(arr)
        return arr
    
    def where(self, condition, x, y):
        """Where operation with fallback."""
        if self.np is not None:
            return self.np.where(condition, x, y)
        # Simple fallback
        if hasattr(condition, '__iter__'):
            return [x if c else y for c in condition]
        return x if condition else y
    
    def any(self, arr):
        """Any operation with fallback."""
        if self.np is not None:
            return self.np.any(arr)
        if hasattr(arr, '__iter__'):
            return any(arr)
        return bool(arr)
    
    def newaxis(self):
        """Get newaxis with fallback."""
        if self.np is not None:
            return self.np.newaxis
        return None  # Fallback
    
    def _create_ones_list(self, shape):
        """Create nested list of ones."""
        if len(shape) == 1:
            return [1.0] * shape[0]
        result = []
        for _ in range(shape[0]):
            result.append(self._create_ones_list(shape[1:]))
        return result
    
    def _create_zeros_list(self, shape):
        """Create nested list of zeros."""
        if len(shape) == 1:
            return [0.0] * shape[0]
        result = []
        for _ in range(shape[0]):
            result.append(self._create_zeros_list(shape[1:]))
        return result

# Create global instance
numpy_safe = NumpySafeOperations()

from .pymdp_utils import (
    _numpy_array_to_string,
    generate_pymdp_matrix_definition,
    generate_pymdp_agent_instantiation
)
from .pymdp_templates import (
    generate_file_header, 
    generate_conversion_summary,
    generate_debug_block,
    generate_example_usage_template,
    generate_placeholder_matrices
)

logger = logging.getLogger(__name__)

# Optional import of pymdp for validation if available
# Using modern API: from pymdp import Agent (inferactively-pymdp package)
_PYMDP_AVAILABLE = False
try:
    from pymdp import utils as pymdp_utils
    from pymdp import maths as pymdp_maths
    from pymdp.agent import Agent as PymdpAgent
    _PYMDP_AVAILABLE = True
except ImportError:
    pymdp_utils = None
    pymdp_maths = None
    PymdpAgent = None


class GnnToPyMdpConverter:
    """Converts a GNN specification dictionary to PyMDP-compatible format."""
    
    def __init__(self, gnn_spec: Dict[str, Any]):
        """
        Initialize the converter with a GNN specification.
        
        Args:
            gnn_spec: Dictionary containing the GNN specification
        """
        self.gnn_spec = gnn_spec
        
        # Initialize conversion log first before any _add_log calls
        self.conversion_log: List[str] = []
        
        raw_model_name = self.gnn_spec.get("name", self.gnn_spec.get("ModelName"))
        
        # Check if model name was found and log appropriately
        if raw_model_name is None:
            self._add_log("ModelName not found in GNN spec, using default 'pymdp_agent_model'.", "INFO")
            raw_model_name = "pymdp_agent_model"
        elif not isinstance(raw_model_name, str):
            self._add_log(f"ModelName from GNN spec is not a string (type: {type(raw_model_name)}), using default 'pymdp_agent_model'.", "WARNING")
            raw_model_name = "pymdp_agent_model"

        # Sanitize the model name to be a valid Python variable name
        self.model_name = self._sanitize_model_name(raw_model_name)
        
        # Script parts storage - will be filled during conversion
        self.script_parts: Dict[str, List[str]] = {
            "imports": [],
            "preamble_vars": [],
            "comments": [f"# --- GNN Model: {self.model_name} ---"],
            "matrix_definitions": [],
            "agent_instantiation": [],
            "example_usage": []
        }
        
        # Extracted and processed GNN data
        self.obs_names: List[str] = []
        self.state_names: List[str] = []
        self.action_names_per_control_factor: Dict[int, List[str]] = {}

        self.num_obs: List[int] = []  # num_outcomes per modality
        self.num_states: List[int] = [] # num_states per factor
        self.num_actions_per_control_factor: Dict[int, int] = {}

        self.num_modalities: int = 0
        self.num_factors: int = 0
        self.control_factor_indices: List[int] = []

        # Matrix specifications
        self.A_spec: Optional[Union[Dict, List[Dict]]] = None
        self.B_spec: Optional[Union[Dict, List[Dict]]] = None
        self.C_spec: Optional[Union[Dict, List[Dict]]] = None
        self.D_spec: Optional[Union[Dict, List[Dict]]] = None
        self.E_spec: Optional[Dict] = None

        self.agent_hyperparams: Dict[str, Any] = {}
        self.simulation_params: Dict[str, Any] = {}
        
        # Extract data from GNN spec
        self._extract_gnn_data()

    def _sanitize_model_name(self, name: str) -> str:
        """Sanitize model name to be a valid Python variable name."""
        # Replace spaces and hyphens with underscores
        temp_name = name.replace(" ", "_").replace("-", "_")
        # Remove other invalid characters (keep alphanumeric and underscores)
        temp_name = re.sub(r'[^0-9a-zA-Z_]', '', temp_name)
        # Ensure it doesn't start with a number
        if temp_name and temp_name[0].isdigit():
            temp_name = "_" + temp_name
        # Ensure it's not empty, default if it becomes empty after sanitization
        return temp_name if temp_name else "pymdp_agent_model"

    def _add_log(self, message: str, level: str = "INFO"):
        """Add message to conversion log and module logger."""
        level_upper = level.upper()
        self.conversion_log.append(f"{level_upper}: {message}")
        
        log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        actual_log_level = log_level_map.get(level_upper, logging.INFO)
        
        if actual_log_level >= logging.ERROR and sys.exc_info()[0] is not None:
            logger.log(actual_log_level, message, exc_info=True)
        else:
            logger.log(actual_log_level, message)

    def _parse_string_to_literal(self, data_str: Any, context_msg: str) -> Optional[Any]:
        """Parse a string representation of a Python literal."""
        if not isinstance(data_str, str):
            np = get_numpy()
            if np is not None:
                valid_types = (list, dict, tuple, int, float, bool, np.ndarray)
            else:
                valid_types = (list, dict, tuple, int, float, bool)
                
            if data_str is None or isinstance(data_str, valid_types):
                return data_str
            self._add_log(f"{context_msg}: Expected string for ast.literal_eval or a known pre-parsed type, but got {type(data_str)}. Returning None.", "ERROR")
            return None

        if not data_str.strip():
            self._add_log(f"{context_msg}: Received empty string data. Cannot parse. Returning None.", "WARNING")
            return None
        
        MAX_STRING_LEN_FOR_AST_EVAL = 1_000_000
        if len(data_str) > MAX_STRING_LEN_FOR_AST_EVAL:
            self._add_log(f"{context_msg}: Input string data is very long. Parsing might be slow.", "WARNING")

        # Check if it's a numpy expression with operations (like np.ones(3) / 3.0)
        if ("np." in data_str or "numpy." in data_str) and any(op in data_str for op in ["/", "*", "+", "-", "**"]):
            # For numpy expressions with operations, return as a special marker
            self._add_log(f"{context_msg}: Detected numpy expression with operations. Preserving as code.", "DEBUG")
            return f"__NUMPY_EXPRESSION__{data_str}"

        # First try standard ast.literal_eval
        try:
            return ast.literal_eval(data_str)
        except (ValueError, SyntaxError, TypeError) as e:
            # If that fails, check if it's a numpy array expression
            if "np.array(" in data_str or "numpy.array(" in data_str:
                try:
                    # Extract the array data from expressions like "np.array([[1,2],[3,4]])"
                    import re
                    # Match np.array(...) or numpy.array(...)
                    pattern = r'(?:np|numpy)\.array\s*\(\s*(.+)\s*\)'
                    match = re.search(pattern, data_str)
                    if match:
                        array_content = match.group(1)
                        # Try to parse the content as a literal
                        parsed_content = ast.literal_eval(array_content)
                        self._add_log(f"{context_msg}: Successfully parsed numpy array expression.", "DEBUG")
                        return parsed_content
                    else:
                        self._add_log(f"{context_msg}: Could not extract array content from numpy expression: {data_str[:100]}...", "ERROR")
                        return None
                except Exception as parse_error:
                    self._add_log(f"{context_msg}: Failed to parse numpy array expression: {parse_error}. String: {data_str[:100]}...", "ERROR")
                    return None
            else:
                error_message = f"ast.literal_eval failed. String '{data_str[:100]}...'. {e}"
                self._add_log(f"{context_msg}: {error_message}", "ERROR")
                return None

    def _extract_gnn_data(self):
        """Extract relevant data from the GNN specification."""
        self._add_log("Starting GNN data extraction.")
        
        # Handle parsed GNN data structure (new format)
        if "variables" in self.gnn_spec:
            self._add_log(f"Found parsed GNN data with {len(self.gnn_spec['variables'])} variables.")
            self._parse_variables_from_gnn_data()
        else:
            # Handle older raw text format
            self._extract_gnn_data_legacy()
        
        # Handle ModelParameters
        if "model_parameters" in self.gnn_spec:
            self.model_parameters = self.gnn_spec["model_parameters"]
            self._add_log(f"Found ModelParameters: {self.model_parameters}")
        else:
            self._add_log("ModelParameters not found or empty in GNN spec.")
            self.model_parameters = {}

        # Extract dimensions from ModelParameters if available
        self._extract_dimensions_from_model_params()
        
        # Validate that we have the necessary data
        if not self.num_modalities or not self.num_factors:
            self._add_log(f"Warning: Missing required data - modalities: {self.num_modalities}, factors: {self.num_factors}", "WARNING")
            # Try to infer from InitialParameterization if available
            self._infer_from_initial_parameterization()

        self._add_log("Finished GNN data extraction.")

    def _parse_variables_from_gnn_data(self):
        """Parse variables from the parsed GNN data structure."""
        variables = self.gnn_spec.get("variables", [])
        
        obs_modalities = {}
        state_factors = {}
        
        self._add_log(f"Processing {len(variables)} variables from GNN data")
        
        for var_data in variables:
            var_name = var_data.get("name", "")
            dimensions = var_data.get("dimensions", [])
            var_type = var_data.get("var_type", "")
            
            # Clean variable name - remove dimension brackets if present
            if '[' in var_name:
                var_name = var_name.split('[')[0]
            
            self._add_log(f"Processing variable: {var_name} with dimensions {dimensions} and type {var_type}")
            
            # Map GNN variable names to PyMDP structure
            if var_name == "A" and len(dimensions) >= 2:
                # A[3,3] -> A matrix with 3 observations, 3 states
                obs_modalities[0] = {
                    'name': 'state_observation',
                    'num_outcomes': dimensions[0]
                }
                state_factors[0] = {
                    'name': 'location',
                    'num_states': dimensions[1]
                }
                self._add_log(f"Recognized A matrix: {dimensions[0]} observations, {dimensions[1]} states")
            elif var_name == "B" and len(dimensions) >= 3:
                # B[3,3,3] -> B matrix with 3 states, 3 previous states, 3 actions
                if 0 not in state_factors:
                    state_factors[0] = {'name': 'location'}
                state_factors[0]['num_states'] = dimensions[0]
                state_factors[0]['num_actions'] = dimensions[2]
                self._add_log(f"Recognized B matrix: {dimensions[0]} states, {dimensions[2]} actions")
            elif var_name == "C" and len(dimensions) >= 1:
                # C[3] -> C vector with 3 observations
                if 0 not in obs_modalities:
                    obs_modalities[0] = {'name': 'state_observation'}
                obs_modalities[0]['num_outcomes'] = dimensions[0]
                self._add_log(f"Recognized C vector: {dimensions[0]} observations")
            elif var_name == "D" and len(dimensions) >= 1:
                # D[3] -> D vector with 3 states
                if 0 not in state_factors:
                    state_factors[0] = {'name': 'location'}
                state_factors[0]['num_states'] = dimensions[0]
                self._add_log(f"Recognized D vector: {dimensions[0]} states")
            elif var_name == "E" and len(dimensions) >= 1:
                # E[3] -> E vector with 3 actions
                if 0 not in state_factors:
                    state_factors[0] = {'name': 'location'}
                state_factors[0]['num_actions'] = dimensions[0]
                self._add_log(f"Recognized E vector: {dimensions[0]} actions")
            elif var_name == "o" and len(dimensions) >= 1:
                # o[3,1] -> observation with 3 outcomes
                if 0 not in obs_modalities:
                    obs_modalities[0] = {'name': 'state_observation'}
                obs_modalities[0]['num_outcomes'] = dimensions[0]
                self._add_log(f"Recognized o variable: {dimensions[0]} outcomes")
            elif var_name == "s" and len(dimensions) >= 1:
                # s[3,1] -> state with 3 states
                if 0 not in state_factors:
                    state_factors[0] = {'name': 'location'}
                state_factors[0]['num_states'] = dimensions[0]
                self._add_log(f"Recognized s variable: {dimensions[0]} states")
            else:
                self._add_log(f"Unrecognized variable: {var_name} with dimensions {dimensions}")
        
        # Update instance variables
        if obs_modalities:
            self.num_modalities = len(obs_modalities)
            self.num_obs = [obs_modalities[i].get('num_outcomes', 2) for i in range(self.num_modalities)]
            self.obs_names = [obs_modalities[i].get('name', f'modality_{i}') for i in range(self.num_modalities)]
            self._add_log(f"Extracted {self.num_modalities} observation modalities: {self.obs_names}")
        
        if state_factors:
            self.num_factors = len(state_factors)
            self.num_states = [state_factors[i].get('num_states', 2) for i in range(self.num_factors)]
            self.state_names = [state_factors[i].get('name', f'factor_{i}') for i in range(self.num_factors)]
            
            # Determine control factors (those with num_actions > 1)
            self.control_factor_indices = []
            for i in range(self.num_factors):
                num_actions = state_factors[i].get('num_actions', 1)
                if num_actions > 1:
                    self.control_factor_indices.append(i)
                    self.num_actions_per_control_factor[i] = num_actions
            
            self._add_log(f"Extracted {self.num_factors} state factors: {self.state_names}")
            self._add_log(f"Control factors: {self.control_factor_indices}")
        
        if not obs_modalities and not state_factors:
            self._add_log("No observation modalities or state factors found in variables", "WARNING")

    def _extract_gnn_data_legacy(self):
        """Method for extracting data from older raw text GNN format."""
        # Handle both old and new JSON export formats
        statespace_key = None
        if "StateSpaceBlock" in self.gnn_spec:
            statespace_key = "StateSpaceBlock"
        elif "statespaceblock" in self.gnn_spec:
            statespace_key = "statespaceblock"
        
        if statespace_key:
            self.state_space_data = self.gnn_spec[statespace_key]
            self._add_log(f"Found StateSpaceBlock data with {len(self.state_space_data)} items.")
            # Extract observation and state information from StateSpaceBlock
            self._parse_statespace_variables()
        else:
            self._add_log("StateSpaceBlock not found or empty in GNN spec.")
            self.state_space_data = []

        # Handle ModelParameters with similar flexibility
        model_params_key = None
        if "ModelParameters" in self.gnn_spec:
            model_params_key = "ModelParameters"
        elif "model_parameters" in self.gnn_spec:
            model_params_key = "model_parameters"
        elif "raw_sections" in self.gnn_spec and "ModelParameters" in self.gnn_spec["raw_sections"]:
            # Parse from raw sections if available
            raw_params = self.gnn_spec["raw_sections"]["ModelParameters"]
            self.model_parameters = self._parse_model_parameters_from_text(raw_params)
            self._add_log(f"Parsed ModelParameters from raw text: {self.model_parameters}")
        
        if model_params_key and not hasattr(self, 'model_parameters'):
            self.model_parameters = self.gnn_spec[model_params_key]
            self._add_log(f"Found ModelParameters: {self.model_parameters}")
        elif not hasattr(self, 'model_parameters') or not self.model_parameters:
            self._add_log("ModelParameters not found or empty in GNN spec.")
            self.model_parameters = {}

    def _parse_statespace_variables(self):
        """Parse StateSpaceBlock to extract observation modalities and state factors."""
        if not self.state_space_data:
            return
        
        obs_modalities = {}
        state_factors = {}
        
        # Handle both string and list formats for StateSpaceBlock
        lines_to_parse = []
        if isinstance(self.state_space_data, str):
            lines_to_parse = self.state_space_data.strip().split('\n')
        elif isinstance(self.state_space_data, list):
            lines_to_parse = self.state_space_data
        elif isinstance(self.state_space_data, dict):
            # Handle dictionary format - extract from values or convert to lines
            for key, value in self.state_space_data.items():
                if isinstance(value, str):
                    lines_to_parse.append(value)
                elif isinstance(value, dict):
                    # Convert dict representation to line format
                    # E.g., {'variable': 'A_m0', 'dimensions': [3,2,3], 'type': 'float'}
                    if 'variable' in value and 'dimensions' in value:
                        var_name = value['variable']
                        dims = value['dimensions']
                        type_info = value.get('type', 'float')
                        line = f"{var_name}[{','.join(map(str, dims))},type={type_info}]"
                        lines_to_parse.append(line)
            if not lines_to_parse:
                self._add_log("StateSpaceBlock is a dictionary but no parseable content found", "WARNING")
                return
        else:
            self._add_log(f"Unknown StateSpaceBlock format: {type(self.state_space_data)}", "WARNING")
            return
        
        for line in lines_to_parse:
            # Handle both string and non-string line types
            if not isinstance(line, str):
                if isinstance(line, dict):
                    # Handle individual dict entries
                    if 'variable' in line and 'dimensions' in line:
                        var_name = line['variable']
                        dims = line['dimensions']
                        self._parse_variable_definition(var_name, dims, obs_modalities, state_factors)
                continue
            
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse variable definitions like A_m0[3,2,3,type=float]
            import re
            # Match pattern like A_m0[3,2,3,type=float] or o_m0[3,1,type=float]
            match = re.match(r'([A-Za-z_]\w*)\[([^\]]+)\]', line)
            if match:
                var_name = match.group(1)
                dims_str = match.group(2)
                
                # Parse dimensions
                dims_parts = [part.strip() for part in dims_str.split(',')]
                dims = []
                for part in dims_parts:
                    if part.startswith('type='):
                        break
                    try:
                        dims.append(int(part))
                    except ValueError:
                        continue
                
                self._parse_variable_definition(var_name, dims, obs_modalities, state_factors)
        
        # Update instance variables
        if obs_modalities:
            self.num_modalities = len(obs_modalities)
            self.num_obs = [obs_modalities[i].get('num_outcomes', 2) for i in range(self.num_modalities)]
            self.obs_names = [obs_modalities[i].get('name', f'modality_{i}') for i in range(self.num_modalities)]
            self._add_log(f"Extracted {self.num_modalities} observation modalities: {self.obs_names}")
        
        if state_factors:
            self.num_factors = len(state_factors)
            self.num_states = [state_factors[i].get('num_states', 2) for i in range(self.num_factors)]
            self.state_names = [state_factors[i].get('name', f'factor_{i}') for i in range(self.num_factors)]
            
            # Determine control factors (those with num_actions > 1)
            self.control_factor_indices = []
            for i in range(self.num_factors):
                num_actions = state_factors[i].get('num_actions', 1)
                if num_actions > 1:
                    self.control_factor_indices.append(i)
                    self.num_actions_per_control_factor[i] = num_actions
            
            self._add_log(f"Extracted {self.num_factors} state factors: {self.state_names}")
            self._add_log(f"Control factors: {self.control_factor_indices}")

    def _parse_variable_definition(self, var_name: str, dims: List[int], obs_modalities: dict, state_factors: dict):
        """Parse a single variable definition into the appropriate data structure."""
        # Categorize variables
        if var_name.startswith('A_m'):
            # A_m0[3,2,3] means 3 outcomes, 2 states in factor 0, 3 states in factor 1
            mod_idx = int(var_name[3:]) if var_name[3:].isdigit() else 0
            if dims:
                obs_modalities[mod_idx] = {
                    'name': f'modality_{mod_idx}',
                    'num_outcomes': dims[0],
                    'state_dims': dims[1:] if len(dims) > 1 else []
                }
        elif var_name.startswith('o_m'):
            # o_m0[3,1] means 3 outcomes
            mod_idx = int(var_name[3:]) if var_name[3:].isdigit() else 0
            if dims:
                if mod_idx not in obs_modalities:
                    obs_modalities[mod_idx] = {'name': f'modality_{mod_idx}'}
                obs_modalities[mod_idx]['num_outcomes'] = dims[0]
        elif var_name.startswith('B_f'):
            # B_f0[2,2,1] means 2 states, 2 previous states, 1 action
            fac_idx = int(var_name[3:]) if var_name[3:].isdigit() else 0
            if dims and len(dims) >= 2:
                state_factors[fac_idx] = {
                    'name': f'factor_{fac_idx}',
                    'num_states': dims[0],
                    'num_actions': dims[2] if len(dims) > 2 else 1
                }
        elif var_name.startswith('s_f'):
            # s_f0[2,1] means 2 states
            fac_idx = int(var_name[3:]) if var_name[3:].isdigit() else 0
            if dims:
                if fac_idx not in state_factors:
                    state_factors[fac_idx] = {'name': f'factor_{fac_idx}'}
                state_factors[fac_idx]['num_states'] = dims[0]

    def _extract_dimensions_from_model_params(self):
        """Extract dimensions from ModelParameters section."""
        if not self.model_parameters:
            return
        
        # Handle num_hidden_states_factors
        if 'num_hidden_states_factors' in self.model_parameters:
            states_info = self.model_parameters['num_hidden_states_factors']
            if isinstance(states_info, list) and not self.num_states:
                self.num_states = states_info
                self.num_factors = len(states_info)
                self.state_names = [f'factor_{i}' for i in range(self.num_factors)]
                self._add_log(f"Extracted state factors from ModelParameters: {self.num_states}")
        
        # Handle num_obs_modalities
        if 'num_obs_modalities' in self.model_parameters:
            obs_info = self.model_parameters['num_obs_modalities']
            if isinstance(obs_info, list) and not self.num_obs:
                self.num_obs = obs_info
                self.num_modalities = len(obs_info)
                self.obs_names = [f'modality_{i}' for i in range(self.num_modalities)]
                self._add_log(f"Extracted observation modalities from ModelParameters: {self.num_obs}")
        
        # Handle num_control_factors
        if 'num_control_factors' in self.model_parameters:
            control_info = self.model_parameters['num_control_factors']
            if isinstance(control_info, list):
                for i, num_actions in enumerate(control_info):
                    if num_actions > 1:
                        if i not in self.control_factor_indices:
                            self.control_factor_indices.append(i)
                        self.num_actions_per_control_factor[i] = num_actions

    def _infer_from_initial_parameterization(self):
        """Try to infer dimensions from InitialParameterization section."""
        # Handle parsed GNN data structure
        if "parameters" in self.gnn_spec:
            self._parse_parameters_from_gnn_data()
        else:
            # Handle older raw text format
            self._infer_from_initial_parameterization_legacy()
        
        # NEW: Extract InitialParameterization matrices if available
        self._extract_initial_parameterization_matrices()

    def _extract_initial_parameterization_matrices(self):
        """Extract and assign matrices from InitialParameterization section."""
        # Check for InitialParameterization in the GNN spec
        initial_params = None
        if "InitialParameterization" in self.gnn_spec:
            initial_params = self.gnn_spec["InitialParameterization"]
        elif "initial_parameterization" in self.gnn_spec:
            initial_params = self.gnn_spec["initial_parameterization"]
        elif "parameters" in self.gnn_spec:
            parameters = self.gnn_spec["parameters"]
            if isinstance(parameters, list):
                # Convert list of parameter objects to dict
                initial_params = {}
                for param in parameters:
                    if hasattr(param, 'name') and hasattr(param, 'value'):
                        initial_params[param.name] = param.value
                    elif isinstance(param, dict):
                        initial_params.update(param)
        
        self._add_log(f"DEBUG: _extract_initial_parameterization_matrices: initial_params = {initial_params}")
        
        if initial_params and isinstance(initial_params, dict):
            # Extract A matrix
            if "A" in initial_params:
                a_raw = initial_params["A"]
                self._add_log(f"DEBUG: A raw value = {a_raw}")
                if isinstance(a_raw, str):
                    try:
                        self.A_spec = self._parse_gnn_matrix_string(a_raw)
                        self._add_log(f"DEBUG: A parsed = {self.A_spec}")
                    except Exception as e:
                        self._add_log(f"DEBUG: Failed to parse A matrix: {e}")
                        self.A_spec = None
                else:
                    self.A_spec = a_raw
            
            # Extract B matrix
            if "B" in initial_params:
                b_raw = initial_params["B"]
                self._add_log(f"DEBUG: B raw value = {b_raw}")
                if isinstance(b_raw, str):
                    try:
                        self.B_spec = self._parse_gnn_3d_matrix_string(b_raw)
                        self._add_log(f"DEBUG: B parsed = {self.B_spec}")
                    except Exception as e:
                        self._add_log(f"DEBUG: Failed to parse B matrix: {e}")
                        self.B_spec = None
                else:
                    self.B_spec = b_raw
            
            # Extract C, D, E vectors (these are working correctly)
            if "C" in initial_params:
                c_raw = initial_params["C"]
                self._add_log(f"DEBUG: C raw value = {c_raw}")
                if isinstance(c_raw, str):
                    try:
                        self.C_spec = self._parse_gnn_matrix_string(c_raw)
                        self._add_log(f"DEBUG: C parsed = {self.C_spec}")
                    except Exception as e:
                        self._add_log(f"DEBUG: Failed to parse C vector: {e}")
                        self.C_spec = None
                else:
                    self.C_spec = c_raw
            
            if "D" in initial_params:
                d_raw = initial_params["D"]
                self._add_log(f"DEBUG: D raw value = {d_raw}")
                if isinstance(d_raw, str):
                    try:
                        self.D_spec = self._parse_gnn_matrix_string(d_raw)
                        self._add_log(f"DEBUG: D parsed = {self.D_spec}")
                    except Exception as e:
                        self._add_log(f"DEBUG: Failed to parse D vector: {e}")
                        self.D_spec = None
                else:
                    self.D_spec = d_raw
            
            if "E" in initial_params:
                e_raw = initial_params["E"]
                self._add_log(f"DEBUG: E raw value = {e_raw}")
                if isinstance(e_raw, str):
                    try:
                        self.E_spec = self._parse_gnn_matrix_string(e_raw)
                        self._add_log(f"DEBUG: E parsed = {self.E_spec}")
                    except Exception as e:
                        self._add_log(f"DEBUG: Failed to parse E vector: {e}")
                        self.E_spec = None
                else:
                    self.E_spec = e_raw

    def _parse_gnn_matrix_string(self, matrix_str: str) -> List[List[float]]:
        """Parse GNN matrix string format into Python list of lists using robust parsing."""
        import re
        import ast
        
        # Remove comments and extra whitespace
        processed_str = self._strip_comments_from_multiline_str(matrix_str)
        
        if not processed_str:
            self._add_log(f"DEBUG: Matrix string was empty after comment stripping (original: '{matrix_str}')")
            return []
        
        # Convert GNN's array-like curly braces into valid Python list syntax for ast.literal_eval
        if '{' in processed_str and ':' not in processed_str:
            processed_str = processed_str.replace('{', '[').replace('}', ']')
        elif processed_str.startswith("{") and processed_str.endswith("}"):
            inner_content = processed_str[1:-1].strip()
            if ':' in inner_content and not (inner_content.startswith("(") and inner_content.endswith(")")):
                pass  # Likely a dictionary, leave as is
            else:
                # Assume GNN's { } means a list-like structure
                processed_str = "[" + inner_content + "]"
        
        try:
            parsed_value = ast.literal_eval(processed_str)
            
            # Convert sets to sorted lists
            def convert_structure(item):
                if isinstance(item, set):
                    try:
                        return sorted(list(item))
                    except TypeError:
                        return list(item)
                elif isinstance(item, list):
                    return [convert_structure(x) for x in item]
                elif isinstance(item, tuple):
                    return tuple(convert_structure(x) for x in item)
                elif isinstance(item, dict):
                    return {k: convert_structure(v) for k, v in item.items()}
                return item
            
            parsed_value = convert_structure(parsed_value)
            
            # Handle special case: single tuple in braces {(a,b)} -> convert to list
            if isinstance(parsed_value, list) and len(parsed_value) == 1 and isinstance(parsed_value[0], (list, tuple)):
                if processed_str.startswith('[(') and processed_str.endswith(')]'):
                    if processed_str.count('(') == 1 and processed_str.count(')') == 1:
                        parsed_value = list(parsed_value[0])
            
            # Ensure we have a 2D matrix (list of lists)
            if isinstance(parsed_value, list):
                if len(parsed_value) > 0 and isinstance(parsed_value[0], (list, tuple)):
                    # Already 2D
                    return [list(row) for row in parsed_value]
                else:
                    # 1D list, convert to 2D
                    return [parsed_value]
            elif isinstance(parsed_value, tuple):
                # Convert tuple to list
                return [list(parsed_value)]
            else:
                # Single value, wrap in list
                return [[parsed_value]]
                
        except Exception as e:
            self._add_log(f"DEBUG: Could not parse matrix string with ast.literal_eval: '{processed_str}'. Error: {e}. Trying fallback parsing.")
            
            # Fallback: manual parsing for complex cases
            return self._parse_matrix_fallback(matrix_str)
    
    def _parse_gnn_3d_matrix_string(self, matrix_str: str) -> List[List[List[float]]]:
        """Parse GNN 3D matrix string format into Python list of lists of lists."""
        import re
        import ast
        
        # Remove comments and extra whitespace
        processed_str = self._strip_comments_from_multiline_str(matrix_str)
        
        if not processed_str:
            self._add_log(f"DEBUG: 3D matrix string was empty after comment stripping (original: '{matrix_str}')")
            return []
        
        # Convert GNN's array-like curly braces into valid Python list syntax
        if '{' in processed_str and ':' not in processed_str:
            processed_str = processed_str.replace('{', '[').replace('}', ']')
        elif processed_str.startswith("{") and processed_str.endswith("}"):
            inner_content = processed_str[1:-1].strip()
            if ':' in inner_content and not (inner_content.startswith("(") and inner_content.endswith(")")):
                pass  # Likely a dictionary, leave as is
            else:
                # Assume GNN's { } means a list-like structure
                processed_str = "[" + inner_content + "]"
        
        try:
            parsed_value = ast.literal_eval(processed_str)
            
            # Convert sets to sorted lists
            def convert_structure(item):
                if isinstance(item, set):
                    try:
                        return sorted(list(item))
                    except TypeError:
                        return list(item)
                elif isinstance(item, list):
                    return [convert_structure(x) for x in item]
                elif isinstance(item, tuple):
                    return tuple(convert_structure(x) for x in item)
                elif isinstance(item, dict):
                    return {k: convert_structure(v) for k, v in item.items()}
                return item
            
            parsed_value = convert_structure(parsed_value)
            
            # Ensure we have a 3D matrix (list of lists of lists)
            if isinstance(parsed_value, list):
                if len(parsed_value) > 0:
                    if isinstance(parsed_value[0], list):
                        if len(parsed_value[0]) > 0 and isinstance(parsed_value[0][0], (list, tuple)):
                            # Already 3D
                            return [[list(row) for row in plane] for plane in parsed_value]
                        else:
                            # 2D list, wrap in outer list
                            return [[list(row) for row in parsed_value]]
                    else:
                        # 1D list, wrap in 2D then 3D
                        return [[parsed_value]]
                else:
                    return []
            else:
                # Single value, wrap in 3D
                return [[[parsed_value]]]
                
        except Exception as e:
            self._add_log(f"DEBUG: Could not parse 3D matrix string with ast.literal_eval: '{processed_str}'. Error: {e}. Trying fallback parsing.")
            
            # Fallback: manual parsing for complex cases
            return self._parse_3d_matrix_fallback(matrix_str)
    
    def _strip_comments_from_multiline_str(self, m_str: str) -> str:
        """Strip comments from a multiline string."""
        lines = m_str.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove comments (everything after #)
            if '#' in line:
                line = line.split('#')[0]
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        return ' '.join(cleaned_lines)
    
    def _parse_matrix_fallback(self, matrix_str: str) -> List[List[float]]:
        """Fallback parsing for complex matrix strings."""
        import re
        
        # Remove comments and extra whitespace
        lines = [line.strip() for line in matrix_str.split('\n') if line.strip() and not line.strip().startswith('#')]
        
        # Find the matrix content (between braces or parentheses)
        content = matrix_str.strip()
        if content.startswith('{') and content.endswith('}'):
            content = content[1:-1]
        elif content.startswith('(') and content.endswith(')'):
            content = content[1:-1]
        
        # Split into lines and parse each row
        matrix = []
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Remove outer parentheses if present
            if line.startswith('(') and line.endswith(')'):
                line = line[1:-1]
            
            # Parse the row values
            row_values = []
            
            # Handle the specific GNN format: (1.0, 0.0, 0.0)
            if line.startswith('(') and line.endswith(')'):
                # Extract values from tuple
                tuple_content = line[1:-1]
                tuple_values = [float(x.strip()) for x in tuple_content.split(',')]
                row_values = tuple_values
            else:
                # Split by comma, but handle nested parentheses
                parts = re.split(r',(?![^(]*\))', line)
                for part in parts:
                    part = part.strip()
                    if part:
                        try:
                            # Handle tuple format like (1.0, 0.0, 0.0)
                            if part.startswith('(') and part.endswith(')'):
                                # Extract values from tuple
                                tuple_content = part[1:-1]
                                tuple_values = [float(x.strip()) for x in tuple_content.split(',')]
                                row_values.extend(tuple_values)
                            else:
                                # Single value
                                row_values.append(float(part))
                        except ValueError:
                            continue
            
            if row_values:
                matrix.append(row_values)
        
        return matrix
    
    def _parse_3d_matrix_fallback(self, matrix_str: str) -> List[List[List[float]]]:
        """Fallback parsing for complex 3D matrix strings."""
        import re
        
        # Remove comments and extra whitespace
        lines = [line.strip() for line in matrix_str.split('\n') if line.strip() and not line.strip().startswith('#')]
        
        # Find the matrix content (between braces)
        content = matrix_str.strip()
        if content.startswith('{') and content.endswith('}'):
            content = content[1:-1]
        
        # Split into 2D matrices
        matrices = []
        current_matrix = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check if this line starts a new 2D matrix
            if line.startswith('((') and line.endswith(')'):
                # Save previous matrix if exists
                if current_matrix:
                    matrices.append(current_matrix)
                    current_matrix = []
                
                # Parse the new matrix row
                row_values = self._parse_matrix_row(line)
                if row_values:
                    current_matrix.append(row_values)
            elif line.startswith('(') and line.endswith(')'):
                # Continue current matrix
                row_values = self._parse_matrix_row(line)
                if row_values:
                    current_matrix.append(row_values)
        
        # Add the last matrix
        if current_matrix:
            matrices.append(current_matrix)
        
        return matrices
    
    def _parse_matrix_row(self, row_str: str) -> List[float]:
        """Parse a single matrix row string into list of floats."""
        import re
        
        # Remove outer parentheses
        if row_str.startswith('(') and row_str.endswith(')'):
            row_str = row_str[1:-1]
        
        # Parse the row values
        row_values = []
        
        # Handle the specific GNN format: (1.0, 0.0, 0.0)
        if row_str.startswith('(') and row_str.endswith(')'):
            # Extract values from tuple
            tuple_content = row_str[1:-1]
            tuple_values = [float(x.strip()) for x in tuple_content.split(',')]
            row_values = tuple_values
        else:
            # Split by comma, but handle nested parentheses
            parts = re.split(r',(?![^(]*\))', row_str)
            for part in parts:
                part = part.strip()
                if part:
                    try:
                        # Handle tuple format like (1.0, 0.0, 0.0)
                        if part.startswith('(') and part.endswith(')'):
                            # Extract values from tuple
                            tuple_content = part[1:-1]
                            tuple_values = [float(x.strip()) for x in tuple_content.split(',')]
                            row_values.extend(tuple_values)
                        else:
                            # Single value
                            row_values.append(float(part))
                    except ValueError:
                        continue
        
        return row_values

    def _parse_parameters_from_gnn_data(self):
        """Parse parameters from the parsed GNN data structure."""
        parameters = self.gnn_spec.get("parameters", [])
        
        for param_data in parameters:
            param_name = param_data.get("name", "")
            param_value = param_data.get("value")
            
            if param_name == "A" and param_value is not None:
                # Parse A matrix
                if isinstance(param_value, list) and len(param_value) >= 3:
                    # A matrix should be 3x3 based on the GNN spec
                    self.A_spec = param_value
                    if not self.num_obs:
                        self.num_obs = [len(param_value)]
                        self.num_modalities = 1
                        self.obs_names = ['state_observation']
                    if not self.num_states:
                        # Handle case where param_value[0] might be an integer or other non-iterable
                        if isinstance(param_value[0], (list, tuple)) and len(param_value[0]) > 0:
                            self.num_states = [len(param_value[0])]
                        else:
                            # Default to 3 states if we can't determine from A matrix
                            self.num_states = [3]
                        self.num_factors = 1
                        self.state_names = ['location']
                    self._add_log(f"Inferred dimensions from A matrix: {len(param_value)} observations, {len(param_value[0]) if isinstance(param_value[0], (list, tuple)) and param_value[0] else 3} states")
            
            elif param_name == "B" and param_value is not None:
                # Parse B matrix
                if isinstance(param_value, list) and len(param_value) >= 3:
                    self.B_spec = param_value
                    if not self.num_states:
                        self.num_states = [len(param_value)]
                        self.num_factors = 1
                        self.state_names = ['location']
                    # Infer number of actions from B matrix
                    if param_value and param_value[0] and param_value[0][0]:
                        num_actions = len(param_value[0][0])
                        self.control_factor_indices = [0]
                        self.num_actions_per_control_factor[0] = num_actions
                        self._add_log(f"Inferred {num_actions} actions from B matrix")
            
            elif param_name == "C" and param_value is not None:
                # Parse C vector
                if isinstance(param_value, list) and len(param_value) >= 1:
                    self.C_spec = param_value[0] if isinstance(param_value[0], list) else param_value
                    if not self.num_obs:
                        self.num_obs = [len(self.C_spec)]
                        self.num_modalities = 1
                        self.obs_names = ['state_observation']
                    self._add_log(f"Inferred {len(self.C_spec)} observations from C vector")
            
            elif param_name == "D" and param_value is not None:
                # Parse D vector
                if isinstance(param_value, list) and len(param_value) >= 1:
                    self.D_spec = param_value[0] if isinstance(param_value[0], list) else param_value
                    if not self.num_states:
                        self.num_states = [len(self.D_spec)]
                        self.num_factors = 1
                        self.state_names = ['location']
                    self._add_log(f"Inferred {len(self.D_spec)} states from D vector")
            
            elif param_name == "E" and param_value is not None:
                # Parse E vector
                if isinstance(param_value, list) and len(param_value) >= 1:
                    self.E_spec = param_value[0] if isinstance(param_value[0], list) else param_value
                    if not self.control_factor_indices:
                        self.control_factor_indices = [0]
                        self.num_actions_per_control_factor[0] = len(self.E_spec)
                    self._add_log(f"Inferred {len(self.E_spec)} actions from E vector")



    def _parse_model_parameters_from_text(self, text: str) -> Dict[str, Any]:
        """Parse ModelParameters from raw text format."""
        params = {}
        for line in text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Parse list values like [2, 3] or [3, 3, 3]
                if value.startswith('[') and value.endswith(']'):
                    # Remove brackets and comments
                    value_clean = value.split('#')[0].strip()[1:-1]
                    try:
                        params[key] = [int(x.strip()) for x in value_clean.split(',') if x.strip()]
                    except ValueError:
                        params[key] = value
                else:
                    params[key] = value
        return params

    def _numpy_array_to_string(self, arr, indent=0) -> str:
        """Convert numpy array to string representation for code generation."""
        return _numpy_array_to_string(arr, indent)

    def convert_A_matrix(self) -> str:
        self._add_log(f"DEBUG: convert_A_matrix: type(self.A_spec) = {type(self.A_spec)}, value = {self.A_spec}")
        self._extract_initial_parameterization_matrices()
        if not self.num_modalities:
            self._add_log("A_matrix: No observation modalities defined. 'A' will be None.", "INFO")
            self.script_parts["matrix_definitions"].append("A = None")
            return "# A matrix set to None due to no observation modalities."
        if not self.num_factors:
            self._add_log("A_matrix: No hidden state factors defined. Cannot form A. 'A' will be None.", "INFO")
            self.script_parts["matrix_definitions"].append("A = None")
            return "# A matrix set to None due to no hidden state factors."
        result_lines: List[str] = []
        matrix_assignments: List[str] = []
        if self.A_spec is not None and isinstance(self.A_spec, list):
            var_name = f"A_{self.obs_names[0] if self.obs_names else 'modality_0'}"
            np = get_numpy()
            if np is not None:
                array_str = self._numpy_array_to_string(np.array(self.A_spec), indent=0)
            else:
                array_str = str(self.A_spec)  # Fallback to string representation
            assignment = f"{var_name} = {array_str}"
            matrix_assignments.append(assignment)
            self._add_log(f"Injected A matrix from GNN InitialParameterization as {var_name}.")
        else:
            for mod_idx in range(self.num_modalities):
                modality_name = self.obs_names[mod_idx] if mod_idx < len(self.obs_names) else f"modality_{mod_idx}"
                var_name = f"A_{modality_name}"
                shape_A_mod = tuple([self.num_obs[mod_idx]] + self.num_states)
                default_assignment = f"{var_name} = utils.norm_dist(np.ones({shape_A_mod}))"
                matrix_assignments.append(default_assignment)
                self._add_log(f"A matrix for {var_name} set to default uniform.")
        result_lines.extend(matrix_assignments)
        init_code = f"A = np.empty({self.num_modalities}, dtype=object)"
        result_lines.append(init_code)
        for mod_idx in range(self.num_modalities):
            modality_name = self.obs_names[mod_idx] if mod_idx < len(self.obs_names) else f"modality_{mod_idx}"
            var_name = f"A_{modality_name}"
            result_lines.append(f"A[{mod_idx}] = {var_name}")
        for line in result_lines:
            self.script_parts["matrix_definitions"].append(line)
        return "\n".join(result_lines)

    def convert_B_matrix(self) -> str:
        self._add_log(f"DEBUG: convert_B_matrix: type(self.B_spec) = {type(self.B_spec)}, value = {self.B_spec}")
        self._extract_initial_parameterization_matrices()
        if not self.num_factors:
            self._add_log("B_matrix: No hidden state factors defined. 'B' will be None.", "INFO")
            self.script_parts["matrix_definitions"].append("B = None")
            return "# B matrix set to None due to no hidden state factors."
        result_lines: List[str] = []
        matrix_assignments: List[str] = []
        if self.B_spec is not None and isinstance(self.B_spec, list):
            var_name = f"B_{self.state_names[0] if self.state_names else 'factor_0'}"
            # Normalize B matrix to ensure proper PyMDP transition probabilities
            b_matrix = numpy_safe.array(self.B_spec)
            # Normalize along axis=0 (columns) for each action slice, handling zero columns
            column_sums = numpy_safe.sum(b_matrix, axis=0, keepdims=True)
            # Replace zero sums with 1 to avoid division by zero, and set those columns to uniform
            zero_cols = column_sums == 0
            column_sums = numpy_safe.where(zero_cols, 1.0, column_sums)
            b_matrix = b_matrix / column_sums
            # Set zero columns to uniform distribution
            if numpy_safe.any(zero_cols):
                uniform_prob = 1.0 / b_matrix.shape[0]
                b_matrix = numpy_safe.where(zero_cols, uniform_prob, b_matrix)
            array_str = self._numpy_array_to_string(b_matrix, indent=0)
            assignment = f"{var_name} = {array_str}"
            matrix_assignments.append(assignment)
            self._add_log(f"Injected and normalized B matrix from GNN InitialParameterization as {var_name}.")
        else:
            for f_idx in range(self.num_factors):
                factor_name = self.state_names[f_idx] if f_idx < len(self.state_names) else f"factor_{f_idx}"
                var_name = f"B_{factor_name}"
                is_controlled = f_idx in self.control_factor_indices
                if is_controlled:
                    num_actions = self.num_actions_per_control_factor.get(f_idx, self.num_states[f_idx])
                    shape_B_string = f"({self.num_states[f_idx]}, {self.num_states[f_idx]}, {num_actions})"
                    default_assignment = f"{var_name} = utils.norm_dist(np.ones{shape_B_string})"
                else:
                    default_assignment = f"{var_name} = utils.norm_dist(np.eye({self.num_states[f_idx]})[:, :, np.newaxis])"
                matrix_assignments.append(default_assignment)
                self._add_log(f"B matrix for {var_name} set to default uniform.")
        result_lines.extend(matrix_assignments)
        init_code = f"B = np.empty({self.num_factors}, dtype=object)"
        result_lines.append(init_code)
        for f_idx in range(self.num_factors):
            factor_name = self.state_names[f_idx] if f_idx < len(self.state_names) else f"factor_{f_idx}"
            var_name = f"B_{factor_name}"
            result_lines.append(f"B[{f_idx}] = {var_name}")
        for line in result_lines:
            self.script_parts["matrix_definitions"].append(line)
        return "\n".join(result_lines)

    def convert_C_vector(self) -> str:
        """Converts GNN's C vector (preferences over observations) to PyMDP format."""
        self._add_log(f"DEBUG: convert_C_vector: self.C_spec = {self.C_spec}")
        self._extract_initial_parameterization_matrices()
        if not self.num_modalities:
            self._add_log("C_vector: No observation modalities defined. 'C' will be None.", "INFO")
            self.script_parts["matrix_definitions"].append("C = None")
            return "# C vector set to None due to no observation modalities."
        result_lines: List[str] = []
        vector_assignments: List[str] = []
        if self.C_spec is not None:
            var_name = f"C_{self.obs_names[0] if self.obs_names else 'modality_0'}"
            # Ensure C vector is flattened for PyMDP compatibility
            c_vector = numpy_safe.array(self.C_spec)
            if c_vector.ndim > 1:
                c_vector = c_vector.flatten()
            array_str = self._numpy_array_to_string(c_vector, indent=0)
            assignment = f"{var_name} = {array_str}"
            vector_assignments.append(assignment)
            self._add_log(f"Injected C vector from GNN InitialParameterization as {var_name}.")
        else:
            for mod_idx in range(self.num_modalities):
                modality_name = self.obs_names[mod_idx] if mod_idx < len(self.obs_names) else f"modality_{mod_idx}"
                var_name = f"C_{modality_name}"
                default_assignment = f"{var_name} = np.zeros({self.num_obs[mod_idx]})"
                vector_assignments.append(default_assignment)
                self._add_log(f"C vector for {var_name} set to default zeros.")
        result_lines.extend(vector_assignments)
        init_code = f"C = np.empty({self.num_modalities}, dtype=object)"
        result_lines.append(init_code)
        for mod_idx in range(self.num_modalities):
            modality_name = self.obs_names[mod_idx] if mod_idx < len(self.obs_names) else f"modality_{mod_idx}"
            var_name = f"C_{modality_name}"
            result_lines.append(f"C[{mod_idx}] = {var_name}")
        for line in result_lines:
            self.script_parts["matrix_definitions"].append(line)
        return "\n".join(result_lines)

    def convert_D_vector(self) -> str:
        """Converts GNN's D vector (initial beliefs about hidden states) to PyMDP format."""
        self._add_log(f"DEBUG: convert_D_vector: self.D_spec = {self.D_spec}")
        self._extract_initial_parameterization_matrices()
        if not self.num_factors:
            self._add_log("D_vector: No hidden state factors defined. 'D' will be None.", "INFO")
            self.script_parts["matrix_definitions"].append("D = None")
            return "# D vector set to None due to no hidden state factors."
        result_lines: List[str] = []
        vector_assignments: List[str] = []
        if self.D_spec is not None:
            var_name = f"D_{self.state_names[0] if self.state_names else 'factor_0'}"
            # Ensure D vector is flattened for PyMDP compatibility
            d_vector = numpy_safe.array(self.D_spec)
            if d_vector.ndim > 1:
                d_vector = d_vector.flatten()
            array_str = self._numpy_array_to_string(d_vector, indent=0)
            assignment = f"{var_name} = {array_str}"
            vector_assignments.append(assignment)
            self._add_log(f"Injected D vector from GNN InitialParameterization as {var_name}.")
        else:
            for f_idx in range(self.num_factors):
                factor_name = self.state_names[f_idx] if f_idx < len(self.state_names) else f"factor_{f_idx}"
                var_name = f"D_{factor_name}"
                default_assignment = f"{var_name} = np.ones({self.num_states[f_idx]}) / {self.num_states[f_idx]}.0"
                vector_assignments.append(default_assignment)
                self._add_log(f"D vector for {var_name} set to default uniform.")
        result_lines.extend(vector_assignments)
        init_code = f"D = np.empty({self.num_factors}, dtype=object)"
        result_lines.append(init_code)
        for f_idx in range(self.num_factors):
            factor_name = self.state_names[f_idx] if f_idx < len(self.state_names) else f"factor_{f_idx}"
            var_name = f"D_{factor_name}"
            result_lines.append(f"D[{f_idx}] = {var_name}")
        for line in result_lines:
            self.script_parts["matrix_definitions"].append(line)
        return "\n".join(result_lines)

    def convert_E_vector(self) -> str:
        """Converts GNN's E vector (policy prior) to PyMDP format."""
        self._add_log(f"DEBUG: convert_E_vector: self.E_spec = {self.E_spec}")
        self._extract_initial_parameterization_matrices()
        result_lines: List[str] = []
        if self.E_spec is not None:
            var_name = "E"
            # Ensure E vector is flattened for PyMDP compatibility
            e_vector = numpy_safe.array(self.E_spec)
            if e_vector.ndim > 1:
                e_vector = e_vector.flatten()
            array_str = self._numpy_array_to_string(e_vector, indent=0)
            assignment = f"{var_name} = {array_str}"
            result_lines.append(assignment)
            self._add_log(f"Injected E vector from GNN InitialParameterization as {var_name}.")
        else:
            self._add_log("E_vector: No E (policy prior) specification found. Defaulting to None.", "INFO")
            result_lines.append("E = None")
        for line in result_lines:
            self.script_parts["matrix_definitions"].append(line)
        return "\n".join(result_lines)

    def extract_agent_hyperparameters(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Extract agent hyperparameters from GNN specification."""
        agent_params = self.agent_hyperparams if self.agent_hyperparams else {}
        
        # Extract general agent parameters
        general_params = {}
        for key in ("inference_horizon", "action_horizon", "planning_horizon",
                    "use_utility", "use_states_info_gain", "use_param_info_gain"):
            if key in agent_params:
                general_params[key] = agent_params[key]
        
        # Extract algorithm-specific parameters
        algo_params = {}
        for key in ("inference_algo", "num_iter", "dF", "dF_tol"):
            if key in agent_params:
                algo_params[key] = agent_params[key]
        
        # Extract learning parameters if available
        learning_params = {}
        for key in ("learn_a", "learn_b", "learn_d", "lr_pA", "lr_pB", "lr_pD"):
            if key in agent_params:
                learning_params[key] = agent_params[key]
                
        return general_params, algo_params, learning_params

    def generate_agent_instantiation_code(self, action_names: Optional[Dict[int, List[str]]] = None, qs_initial: Optional[Any] = None) -> str:
        """Generate code for instantiating a PyMDP agent with the model parameters."""
        if not self.num_modalities or not self.num_factors:
            self._add_log("Cannot generate agent instantiation with no observation modalities or hidden state factors.", "ERROR")
            return "# Cannot instantiate agent: missing modalities or state factors"
        
        # Extract hyperparameters from GNN spec
        general_params, algo_params, learning_params = self.extract_agent_hyperparameters()
        
        # Use provided action names or default to instance attribute
        if action_names is None:
            action_names = self.action_names_per_control_factor
        
        # Generate agent code using the utility function
        model_params = {
            "A": "A",
            "B": "B", 
            "C": "C",
            "D": "D",
            "E": "E"
        }
        
        # Combine all parameters into appropriate dictionaries
        control_params = {}
        if self.control_factor_indices:
            control_params["control_fac_idx"] = self.control_factor_indices
            
        algorithm_params = algo_params if algo_params else {}
        
        # Handle general_params safely
        general_params_safe = general_params if general_params else {}
        
        agent_code_lines = generate_pymdp_agent_instantiation(
            agent_name="agent",
            model_params=model_params,
            control_params=control_params,
            algorithm_params=algorithm_params,
            action_names=action_names,
            qs_initial=qs_initial,
            **general_params_safe
        )
        
        # Add the agent instantiation code to the script parts
        self.script_parts["agent_instantiation"].extend(agent_code_lines.split('\n'))
        
        # Return the actual generated code for testing/inspection
        return agent_code_lines

    def generate_example_usage_code(self) -> List[str]:
        """Generate example code that demonstrates how to use the PyMDP agent."""
        example_lines = []
        
        # Only generate example usage if we have observation modalities and state factors
        if not self.num_modalities or not self.num_factors:
            self._add_log("Skipping example usage code due to missing modalities or state factors.", "INFO")
            return ["# Example usage skipped due to missing modalities or state factors"]
        
        # Add comment for the example section
        example_lines.append("\n# Example usage of the agent")
        example_lines.append(f"if __name__ == \"__main__\":")
        
        # Generate the example usage template
        example_code = generate_example_usage_template(
            model_name=self.model_name,
            num_modalities=self.num_modalities,
            num_factors=self.num_factors,
            control_factor_indices=self.control_factor_indices
        )
        
        # Add indented example code lines
        for line in example_code:
            example_lines.append(f"    {line}")
        
        # Save the example usage code in the script parts
        self.script_parts["example_usage"] = example_lines
        
        return example_lines

    def get_full_python_script(self, include_example_usage: bool = True) -> str:
        """Generate the complete Python script for the PyMDP agent."""
        # First, ensure all necessary matrix conversions have been performed
        if not self.script_parts["matrix_definitions"]:
            self._add_log("Matrix definitions not generated yet. Generating matrices...", "INFO")
            self.convert_A_matrix()
            self.convert_B_matrix()
            self.convert_C_vector()
            self.convert_D_vector()
            self.convert_E_vector()
        
        # Generate agent instantiation code if not already generated
        if not self.script_parts["agent_instantiation"]:
            self._add_log("Agent instantiation code not generated yet. Generating...", "INFO")
            self.generate_agent_instantiation_code()
        
        # Generate example usage code if requested and not already generated
        if include_example_usage and not self.script_parts["example_usage"]:
            self._add_log("Example usage code not generated yet. Generating...", "INFO")
            self.generate_example_usage_code()
        
        # Generate file header with imports and docstring
        file_header = generate_file_header(model_name=self.model_name)
        
        # Generate conversion summary from the log
        conversion_summary = generate_conversion_summary(self.conversion_log)
        
        # Generate debug section to help with troubleshooting
        action_names_dict_str = str(self.action_names_per_control_factor) if self.action_names_per_control_factor else "{}"
        qs_initial_str = "None"  # Default since we don't have qs_initial in this context
        agent_hyperparams_dict_str = str(self.agent_hyperparams) if self.agent_hyperparams else "{}"
        
        debug_section = generate_debug_block(
            action_names_dict_str=action_names_dict_str,
            qs_initial_str=qs_initial_str,
            agent_hyperparams_dict_str=agent_hyperparams_dict_str
        )
        
        # Assemble the final script
        script_sections = [
            file_header,
            conversion_summary,
            "\n".join(self.script_parts["preamble_vars"]),
            "\n".join(self.script_parts["matrix_definitions"]),
        ]
        
        # Add the placeholder matrices section for missing data
        if (not self.num_modalities or not self.num_factors):
            placeholder_matrices_dict = generate_placeholder_matrices(
                num_modalities=self.num_modalities,
                num_states=self.num_states
            )
            # Convert the dictionary to a string representation
            placeholder_lines = []
            for matrix_name, lines in placeholder_matrices_dict.items():
                placeholder_lines.extend(lines)
            placeholder_matrices = "\n".join(placeholder_lines)
            script_sections.append(placeholder_matrices)
        
        # Add the agent instantiation code
        script_sections.append("\n".join(self.script_parts["agent_instantiation"]))
        
        # Add the example usage code if requested
        if include_example_usage:
            script_sections.append("\n".join(self.script_parts["example_usage"]))
        
        # Add the debug section at the end
        script_sections.append(debug_section)
        
        # Combine all sections into a single script
        full_script = "\n\n".join(section for section in script_sections if section)
        
        self._add_log(f"Generated full Python script ({len(full_script)} characters)", "INFO")
        return full_script 