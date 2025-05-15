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
import sys # for sys.modules
import inspect # for inspect.signature
import traceback # for traceback.format_exc()
import ast # Added for ast.literal_eval
import re # Added for sanitizing model names

# Import from the new utils module
from .pymdp_utils import (
    _numpy_array_to_string,
    format_list_recursive,
    generate_pymdp_matrix_definition,
    generate_pymdp_agent_instantiation
)

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


class GnnToPyMdpConverter:
    """Converts a GNN specification dictionary to a PyMDP compatible Python script."""

    def __init__(self, gnn_spec: Dict[str, Any]):
        self.gnn_spec = gnn_spec
        raw_model_name = self.gnn_spec.get("name", self.gnn_spec.get("ModelName", "pymdp_agent_model"))
        # Sanitize the model name to be a valid Python variable name
        # Replace spaces and hyphens with underscores
        temp_name = raw_model_name.replace(" ", "_").replace("-", "_")
        # Remove other invalid characters (keep alphanumeric and underscores)
        temp_name = re.sub(r'[^0-9a-zA-Z_]', '', temp_name)
        # Ensure it doesn't start with a number
        if temp_name and temp_name[0].isdigit():
            temp_name = "_" + temp_name
        # Ensure it's not empty, default if it becomes empty after sanitization
        self.model_name = temp_name if temp_name else "pymdp_agent_model"
        
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

    def _parse_string_to_literal(self, data_str: Any, context_msg: str) -> Optional[Any]:
        """Attempts to parse a string representation of a Python literal (e.g., list, tuple, dict)."""
        if not isinstance(data_str, str):
            if data_str is None or isinstance(data_str, (list, dict, tuple, int, float, bool)):
                 return data_str
            self._add_log(f"{context_msg}: Expected string for ast.literal_eval or pre-parsed object, but got {type(data_str)}. Value: '{str(data_str)[:100]}...'. Returning as is.", "WARNING")
            return data_str

        if not data_str.strip():
            self._add_log(f"{context_msg}: Received empty string data. Cannot parse. Returning None.", "WARNING")
            return None
        try:
            return ast.literal_eval(data_str)
        except (ValueError, SyntaxError, TypeError) as e:
            self._add_log(f"{context_msg}: Failed to parse string data '{data_str[:100]}...' with ast.literal_eval: {e}. Returning None.", "ERROR")
            return None

    def _extract_gnn_data(self):
        """Parses the raw gnn_spec and populates structured attributes."""
        self._add_log("Starting GNN data extraction.")
        # Ensure ast is imported if not already at module level
        import ast

        # Prioritize top-level keys from gnn_spec (populated by the exporter)
        direct_num_obs = self.gnn_spec.get("num_obs_modalities")
        direct_obs_names = self.gnn_spec.get("obs_modality_names") # New potential key

        direct_num_states = self.gnn_spec.get("num_hidden_states_factors")
        direct_state_names = self.gnn_spec.get("hidden_state_factor_names") # New potential key

        # Attempt to parse stringified lists for direct_num_obs
        if isinstance(direct_num_obs, str) and direct_num_obs.startswith('[') and direct_num_obs.endswith(']'):
            try:
                parsed_val = ast.literal_eval(direct_num_obs)
                if isinstance(parsed_val, list) and all(isinstance(n, int) for n in parsed_val):
                    direct_num_obs = parsed_val
                    self._add_log(f"Successfully parsed stringified direct_num_obs: {direct_num_obs}", "DEBUG")
            except (ValueError, SyntaxError, TypeError):
                self._add_log(f"Failed to parse stringified direct_num_obs: {direct_num_obs}", "WARNING")
        
        # Attempt to parse stringified lists for direct_num_states
        if isinstance(direct_num_states, str) and direct_num_states.startswith('[') and direct_num_states.endswith(']'):
            try:
                parsed_val = ast.literal_eval(direct_num_states)
                if isinstance(parsed_val, list) and all(isinstance(n, int) for n in parsed_val):
                    direct_num_states = parsed_val
                    self._add_log(f"Successfully parsed stringified direct_num_states: {direct_num_states}", "DEBUG")
            except (ValueError, SyntaxError, TypeError):
                self._add_log(f"Failed to parse stringified direct_num_states: {direct_num_states}", "WARNING")

        direct_num_control_dims = self.gnn_spec.get("num_control_factors") # This should be a list of action dimensions for each factor
        direct_control_action_names = self.gnn_spec.get("control_action_names_per_factor") # Dict: factor_idx -> list of names

        ss_block = self.gnn_spec.get("StateSpaceBlock", {})
        model_params_spec = self.gnn_spec.get("ModelParameters", {})

        # Observation Modalities Dimensions & Names
        processed_obs_directly = False
        if isinstance(direct_num_obs, list) and all(isinstance(n, int) for n in direct_num_obs):
            self.num_obs = direct_num_obs
            self.num_modalities = len(self.num_obs)
            self._add_log(f"Observation dimensions (num_obs) derived directly from gnn_spec.num_obs_modalities: {self.num_obs}", "INFO")
            if isinstance(direct_obs_names, list) and len(direct_obs_names) == self.num_modalities:
                self.obs_names = direct_obs_names
                self._add_log(f"Observation names derived directly from gnn_spec.obs_modality_names: {self.obs_names}", "INFO")
            else:
                self.obs_names = [f"modality_{i}" for i in range(self.num_modalities)]
                self._add_log(f"Observation names generated as defaults (gnn_spec.obs_modality_names not found or mismatched).", "INFO")
            processed_obs_directly = True

        if not processed_obs_directly:
            num_obs_from_model_params = model_params_spec.get("num_obs_modalities")
            if isinstance(num_obs_from_model_params, list) and all(isinstance(n, int) for n in num_obs_from_model_params):
                self.num_obs = num_obs_from_model_params
                self.num_modalities = len(self.num_obs)
                self._add_log(f"Observation dimensions (num_obs) derived from ModelParameters: {self.num_obs}", "INFO")
                obs_modalities_spec = ss_block.get("ObservationModalities", [])
                if obs_modalities_spec and len(obs_modalities_spec) == self.num_modalities:
                    self.obs_names = [mod.get("modality_name", f"modality_{i}") for i, mod in enumerate(obs_modalities_spec)]
                else:
                    self.obs_names = [f"modality_{i}" for i in range(self.num_modalities)]
            else:
                obs_modalities_spec = ss_block.get("ObservationModalities", [])
                if not obs_modalities_spec:
                    self._add_log("No 'ObservationModalities' in StateSpaceBlock and no 'num_obs_modalities' in ModelParameters or gnn_spec.", "WARNING")
                for i, mod_spec in enumerate(obs_modalities_spec):
                    name = mod_spec.get("modality_name", f"modality_{i}")
                    num_outcomes = mod_spec.get("num_outcomes")
                    if num_outcomes is None:
                        self._add_log(f"Modality '{name}' missing 'num_outcomes'. Skipping.", "ERROR")
                        continue
                    self.obs_names.append(name)
                    self.num_obs.append(int(num_outcomes))
                self.num_modalities = len(self.num_obs)

        self.script_parts["preamble_vars"].append(f"obs_names = {self.obs_names if self.num_modalities > 0 else []}")
        self.script_parts["preamble_vars"].append(f"num_obs = {self.num_obs if self.num_modalities > 0 else []}")
        self.script_parts["preamble_vars"].append(f"num_modalities = {self.num_modalities}")

        # Hidden State Factors Dimensions & Names
        processed_states_directly = False
        if isinstance(direct_num_states, list) and all(isinstance(n, int) for n in direct_num_states):
            self.num_states = direct_num_states
            self.num_factors = len(self.num_states)
            self._add_log(f"Hidden state dimensions (num_states) derived directly from gnn_spec.num_hidden_states_factors: {self.num_states}", "INFO")
            if isinstance(direct_state_names, list) and len(direct_state_names) == self.num_factors:
                self.state_names = direct_state_names
                self._add_log(f"Hidden state names derived directly from gnn_spec.hidden_state_factor_names: {self.state_names}", "INFO")
            else:
                self.state_names = [f"factor_{i}" for i in range(self.num_factors)]
                self._add_log(f"Hidden state names generated as defaults (gnn_spec.hidden_state_factor_names not found or mismatched).", "INFO")
            processed_states_directly = True

        if not processed_states_directly:
            num_states_from_model_params = model_params_spec.get("num_hidden_states_factors")
            if isinstance(num_states_from_model_params, list) and all(isinstance(n, int) for n in num_states_from_model_params):
                self.num_states = num_states_from_model_params
                self.num_factors = len(self.num_states)
                self._add_log(f"Hidden state dimensions (num_states) derived from ModelParameters: {self.num_states}", "INFO")
                hidden_factors_spec_mp_fallback = ss_block.get("HiddenStateFactors", [])
                if hidden_factors_spec_mp_fallback and len(hidden_factors_spec_mp_fallback) == self.num_factors:
                    self.state_names = [fac.get("factor_name", f"factor_{i}") for i, fac in enumerate(hidden_factors_spec_mp_fallback)]
                else:
                    self.state_names = [f"factor_{i}" for i in range(self.num_factors)]
            else:
                hidden_factors_spec = ss_block.get("HiddenStateFactors", [])
                if not hidden_factors_spec:
                    self._add_log("No 'HiddenStateFactors' in StateSpaceBlock and no 'num_hidden_states_factors' in ModelParameters or gnn_spec.", "WARNING")
                for i, fac_spec in enumerate(hidden_factors_spec):
                    name = fac_spec.get("factor_name", f"factor_{i}")
                    num_states_val = fac_spec.get("num_states")
                    if num_states_val is None:
                        self._add_log(f"Factor '{name}' missing 'num_states'. Skipping.", "ERROR")
                        continue
                    self.state_names.append(name)
                    self.num_states.append(int(num_states_val))
                self.num_factors = len(self.num_states)
            
        # Controllable factors and actions - try direct gnn_spec keys first
        processed_control_directly = False
        if isinstance(direct_num_control_dims, list) and self.num_factors > 0 and len(direct_num_control_dims) == self.num_factors:
            self._add_log(f"Control dimensions derived directly from gnn_spec.num_control_factors: {direct_num_control_dims}", "INFO")
            for f_idx, num_actions_for_factor_val in enumerate(direct_num_control_dims):
                num_actions_for_factor = int(num_actions_for_factor_val)
                if num_actions_for_factor > 1: # Assumes >1 action means controllable
                    self.control_factor_indices.append(f_idx)
                    self.num_actions_per_control_factor[f_idx] = num_actions_for_factor
                    # Get action names if provided directly
                    if isinstance(direct_control_action_names, dict) and f_idx in direct_control_action_names and \
                       isinstance(direct_control_action_names[f_idx], list) and len(direct_control_action_names[f_idx]) == num_actions_for_factor:
                        self.action_names_per_control_factor[f_idx] = direct_control_action_names[f_idx]
                        self._add_log(f"Action names for control factor {f_idx} derived directly from gnn_spec.control_action_names_per_factor.", "INFO")
                    else:
                        factor_name_for_action = self.state_names[f_idx] if f_idx < len(self.state_names) else f"factor_{f_idx}"
                        self.action_names_per_control_factor[f_idx] = [f"{factor_name_for_action}_action_{j}" for j in range(num_actions_for_factor)]
            processed_control_directly = True

        if not processed_control_directly:
            num_control_factors_dims_mp = model_params_spec.get("num_control_factors", model_params_spec.get("num_control_action_dims"))
            if isinstance(num_control_factors_dims_mp, list) and self.num_factors > 0 and len(num_control_factors_dims_mp) == self.num_factors:
                self._add_log(f"Control dimensions derived from ModelParameters.num_control_factors/dims: {num_control_factors_dims_mp}", "INFO")
                for f_idx, num_actions_for_factor_val in enumerate(num_control_factors_dims_mp):
                    num_actions_for_factor = int(num_actions_for_factor_val)
                    if num_actions_for_factor > 1:
                        self.control_factor_indices.append(f_idx)
                        self.num_actions_per_control_factor[f_idx] = num_actions_for_factor
                        hsf_block = ss_block.get("HiddenStateFactors", [])
                        factor_name_for_action = self.state_names[f_idx] if f_idx < len(self.state_names) else f"factor_{f_idx}"
                        action_names_from_hsf = None
                        if f_idx < len(hsf_block) and hsf_block[f_idx].get("factor_name") == factor_name_for_action:
                            action_names_from_hsf = hsf_block[f_idx].get("action_names")
                        
                        if action_names_from_hsf and len(action_names_from_hsf) == num_actions_for_factor:
                           self.action_names_per_control_factor[f_idx] = action_names_from_hsf
                        else:
                           self.action_names_per_control_factor[f_idx] = [f"{factor_name_for_action}_action_{j}" for j in range(num_actions_for_factor)]
            else:
                hidden_factors_spec_ctrl = ss_block.get("HiddenStateFactors", [])
                if self.num_factors > 0 and len(hidden_factors_spec_ctrl) == self.num_factors:
                    for i, fac_spec in enumerate(hidden_factors_spec_ctrl):
                        if fac_spec.get("controllable", False):
                            self.control_factor_indices.append(i)
                            num_actions = fac_spec.get("num_actions")
                            factor_name_for_action = self.state_names[i] if i < len(self.state_names) else f"factor_{i}"
                            if num_actions is None:
                                self._add_log(f"Controllable factor '{factor_name_for_action}' missing 'num_actions'. Defaulting to num_states ({self.num_states[i]}).", "WARNING")
                                num_actions = int(self.num_states[i])
                            self.num_actions_per_control_factor[i] = int(num_actions)
                            action_names = fac_spec.get("action_names")
                            if action_names and len(action_names) == num_actions:
                                self.action_names_per_control_factor[i] = action_names
                            else:
                                self.action_names_per_control_factor[i] = [f"{factor_name_for_action}_action_{j}" for j in range(num_actions)]
                else:
                     self._add_log(f"Could not definitively determine control structure from gnn_spec, ModelParameters or StateSpaceBlock. control_fac_idx might be empty.", "WARNING")

        self.script_parts["preamble_vars"].append(f"state_names = {self.state_names if self.num_factors > 0 else []}")
        self.script_parts["preamble_vars"].append(f"num_states = {self.num_states if self.num_factors > 0 else []}")
        self.script_parts["preamble_vars"].append(f"num_factors = {self.num_factors}")
        self.script_parts["preamble_vars"].append(f"control_fac_idx = {self.control_factor_indices if self.control_factor_indices else []}")

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
        
        # A_spec (Likelihoods)
        a_matrix_generic_spec = param_block.get("A_Matrix", param_block.get("A"))
        if a_matrix_generic_spec:
            self.A_spec = a_matrix_generic_spec
            self._add_log("A_spec: Loaded from generic 'A_Matrix' or 'A' key.", "DEBUG")
        else:
            temp_A_list = []
            found_A_m_keys = False
            for mod_idx in range(self.num_modalities):
                mod_a_data = param_block.get(f"A_m{mod_idx}")
                if mod_a_data is not None:
                    temp_A_list.append({"array": mod_a_data}) # Wrap data in dict
                    found_A_m_keys = True
                else:
                    # If a specific A_m<idx> is missing, but others might exist,
                    # we add a None placeholder. convert_A_matrix should handle it
                    # (e.g. by defaulting that modality or logging an error).
                    temp_A_list.append(None) 
            if found_A_m_keys:
                self.A_spec = temp_A_list
                self._add_log(f"A_spec: Constructed from individual 'A_m<idx>' keys. Found specs: {[s is not None for s in temp_A_list]}", "DEBUG")
            else:
                self.A_spec = None # No generic A_Matrix and no A_m<idx> keys found
                self._add_log("A_spec: No 'A_Matrix' or 'A_m<idx>' keys found in InitialParameterization.", "INFO")

        # B_spec (Transitions)
        b_matrix_generic_spec = param_block.get("B_Matrix", param_block.get("B"))
        if b_matrix_generic_spec:
            self.B_spec = b_matrix_generic_spec
            self._add_log("B_spec: Loaded from generic 'B_Matrix' or 'B' key.", "DEBUG")
        else:
            temp_B_list = []
            found_B_f_keys = False
            for f_idx in range(self.num_factors):
                fac_b_data = param_block.get(f"B_f{f_idx}")
                if fac_b_data is not None:
                    temp_B_list.append({"array": fac_b_data})
                    found_B_f_keys = True
                else:
                    temp_B_list.append(None)
            if found_B_f_keys:
                self.B_spec = temp_B_list
                self._add_log(f"B_spec: Constructed from individual 'B_f<idx>' keys. Found specs: {[s is not None for s in temp_B_list]}", "DEBUG")
            else:
                self.B_spec = None
                self._add_log("B_spec: No 'B_Matrix' or 'B_f<idx>' keys found in InitialParameterization.", "INFO")

        # C_spec (Preferences over outcomes)
        c_vector_generic_spec = param_block.get("C_Vector", param_block.get("C"))
        if c_vector_generic_spec:
            self.C_spec = c_vector_generic_spec
            self._add_log("C_spec: Loaded from generic 'C_Vector' or 'C' key.", "DEBUG")
        else:
            temp_C_list = []
            found_C_m_keys = False
            for mod_idx in range(self.num_modalities):
                mod_c_data = param_block.get(f"C_m{mod_idx}")
                if mod_c_data is not None:
                    temp_C_list.append({"array": mod_c_data})
                    found_C_m_keys = True
                else:
                    temp_C_list.append(None)
            if found_C_m_keys:
                self.C_spec = temp_C_list
                self._add_log(f"C_spec: Constructed from individual 'C_m<idx>' keys. Found specs: {[s is not None for s in temp_C_list]}", "DEBUG")
            else:
                self.C_spec = None
                self._add_log("C_spec: No 'C_Vector' or 'C_m<idx>' keys found in InitialParameterization.", "INFO")

        # D_spec (Initial hidden states)
        d_vector_generic_spec = param_block.get("D_Vector", param_block.get("D"))
        if d_vector_generic_spec:
            self.D_spec = d_vector_generic_spec
            self._add_log("D_spec: Loaded from generic 'D_Vector' or 'D' key.", "DEBUG")
        else:
            temp_D_list = []
            found_D_f_keys = False
            for f_idx in range(self.num_factors):
                fac_d_data = param_block.get(f"D_f{f_idx}")
                if fac_d_data is not None:
                    temp_D_list.append({"array": fac_d_data})
                    found_D_f_keys = True
                else:
                    temp_D_list.append(None)
            if found_D_f_keys:
                self.D_spec = temp_D_list
                self._add_log(f"D_spec: Constructed from individual 'D_f<idx>' keys. Found specs: {[s is not None for s in temp_D_list]}", "DEBUG")
            else:
                self.D_spec = None
                self._add_log("D_spec: No 'D_Vector' or 'D_f<idx>' keys found in InitialParameterization.", "INFO")
        
        # E_spec (Prior preferences over policies) - usually a single spec
        self.E_spec = param_block.get("E_Vector", param_block.get("E"))
        if self.E_spec:
            self._add_log("E_spec: Loaded from 'E_Vector' or 'E' key.", "DEBUG")
        else:
            self._add_log("E_spec: No 'E_Vector' or 'E' key found.", "INFO")


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
            self._add_log("A_matrix: No observation modalities defined. 'A' will be None.", "INFO")
            self.script_parts["matrix_definitions"].append("A = None")
            return "# A matrix set to None due to no observation modalities."

        if not self.num_factors: # A multi-factor likelihood depends on states
            self._add_log("A_matrix: No hidden state factors defined. Cannot form A. 'A' will be None.", "INFO")
            self.script_parts["matrix_definitions"].append("A = None")
            return "# A matrix set to None due to no hidden state factors."

        init_code = f"A = utils.obj_array({self.num_modalities})"
        self.script_parts["matrix_definitions"].append(init_code)

        # Default to uniform if no A_spec
        for mod_idx in range(self.num_modalities):
            shape_A_mod = tuple([self.num_obs[mod_idx]] + self.num_states)
            self.script_parts["matrix_definitions"].append(f"A[{mod_idx}] = utils.norm_dist(np.ones({shape_A_mod})) # Defaulted to uniform")

        if self.A_spec:
            if isinstance(self.A_spec, list): # List of specs per modality
                for mod_idx, mod_a_spec in enumerate(self.A_spec):
                    if mod_a_spec is None: # Modality spec might be missing
                        self._add_log(f"A_matrix (modality {mod_idx}): Spec is None. Using default uniform A[{mod_idx}].", "INFO")
                        continue

                    array_data_input = mod_a_spec.get("array")
                    rule = mod_a_spec.get("rule")
                    context_msg = f"A_matrix (modality {self.obs_names[mod_idx] if mod_idx < len(self.obs_names) else mod_idx})"

                    if array_data_input is not None:
                        parsed_array_data = self._parse_string_to_literal(array_data_input, context_msg)
                        if parsed_array_data is not None:
                            try:
                                np_array = np.array(parsed_array_data)
                                expected_shape = tuple([self.num_obs[mod_idx]] + self.num_states)
                                if np_array.shape == expected_shape:
                                    assign_str = f"utils.norm_dist({_numpy_array_to_string(np_array, indent=4)})"
                                    self.script_parts["matrix_definitions"].append(f"A[{mod_idx}] = {assign_str}")
                                else:
                                    self._add_log(f"{context_msg}: Shape mismatch. Expected {expected_shape}, got {np_array.shape}. Using default uniform A[{mod_idx}].", "ERROR")
                            except Exception as e:
                                self._add_log(f"{context_msg}: Error processing parsed array data to NumPy: {e}. Using default uniform A[{mod_idx}].", "ERROR")
                        else:
                            self._add_log(f"{context_msg}: Failed to parse array data string. Using default uniform A[{mod_idx}].", "INFO")
                    elif rule:
                        self._add_log(f"{context_msg}: Rule '{rule}' not fully implemented yet. Using default uniform A[{mod_idx}].", "WARNING")
                    else:
                        self._add_log(f"{context_msg}: No 'array' or 'rule' found. Using default uniform A[{mod_idx}].", "INFO")
            # Handling for A_spec as single dict if num_modalities == 1
            elif isinstance(self.A_spec, dict) and self.num_modalities == 1:
                mod_idx = 0
                array_data_input = self.A_spec.get("array")
                rule = self.A_spec.get("rule")
                context_msg = f"A_matrix (modality {mod_idx})"
                if array_data_input is not None:
                    parsed_array_data = self._parse_string_to_literal(array_data_input, context_msg)
                    if parsed_array_data is not None:
                        try:
                            np_array = np.array(parsed_array_data)
                            expected_shape = tuple([self.num_obs[mod_idx]] + self.num_states)
                            if np_array.shape == expected_shape:
                                assign_str = f"utils.norm_dist({_numpy_array_to_string(np_array, indent=4)})"
                                self.script_parts["matrix_definitions"].append(f"A[{mod_idx}] = {assign_str}")
                            else:
                                self._add_log(f"{context_msg}: Shape mismatch. Expected {expected_shape}, got {np_array.shape}. Using default uniform A[{mod_idx}].", "ERROR")
                        except Exception as e:
                            self._add_log(f"{context_msg}: Error processing parsed array data to NumPy: {e}. Using default uniform A[{mod_idx}].", "ERROR")
                    else:
                        self._add_log(f"{context_msg}: Failed to parse array data string. Using default uniform A[{mod_idx}].", "INFO")
                elif rule:
                    self._add_log(f"{context_msg}: Rule '{rule}' not fully implemented. Using default uniform A[{mod_idx}].", "WARNING")
                else:
                    self._add_log(f"{context_msg}: No 'array' or 'rule'. Using default uniform A[{mod_idx}].", "INFO")
            else:
                self._add_log("A_matrix: A_spec format not recognized for multiple modalities. All A modalities will be default uniform.", "WARNING")
        else: # No A_spec
            self._add_log("A_matrix: No A_spec provided in GNN. All modalities of A defaulted to uniform.", "INFO")
            return "# A matrix modalities remain default initialized (uniform)."
        
        return "# A matrix construction based on GNN spec."


    def convert_B_matrix(self) -> str:
        """Converts GNN's B matrix (transition) to PyMDP format."""
        if not self.num_factors:
            self._add_log("B_matrix: No hidden state factors defined. 'B' will be None.", "INFO")
            self.script_parts["matrix_definitions"].append("B = None")
            return "# B matrix set to None due to no hidden state factors."

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

        def get_b_assignment_string(spec_value_input, num_states_val, is_controlled_val, num_actions_val, factor_idx_for_log, indent_level=4) -> Optional[str]:
            np_array = None
            context_msg_b_assign = f"B_matrix (factor {self.state_names[factor_idx_for_log] if factor_idx_for_log < len(self.state_names) else factor_idx_for_log}) internal assignment"
            
            parsed_spec_value = self._parse_string_to_literal(spec_value_input, context_msg_b_assign)

            if parsed_spec_value is None:
                self._add_log(f"{context_msg_b_assign}: Parsed spec value from string is None. Cannot create NumPy array.", "ERROR")
                return None
            try:
                np_array = np.array(parsed_spec_value)
            except Exception as e:
                self._add_log(f"{context_msg_b_assign}: Error converting parsed data to NumPy array: {e}. Value was: '{str(parsed_spec_value)[:100]}...'", "ERROR")
                return None

            expected_shape_controlled = (num_states_val, num_states_val, num_actions_val)
            expected_shape_uncontrolled_2d = (num_states_val, num_states_val)
            expected_shape_uncontrolled_3d = (num_states_val, num_states_val, 1)

            assign_str_val = _numpy_array_to_string(np_array, indent=indent_level)

            if is_controlled_val:
                if np_array.shape == expected_shape_controlled:
                    return f"utils.norm_dist({assign_str_val})"
                else:
                    self._add_log(f"{context_msg_b_assign}: Shape mismatch for controlled factor. Expected {expected_shape_controlled}, got {np_array.shape}. Not assigning.", "ERROR")
                    return None
            else: # Uncontrolled
                if np_array.shape == expected_shape_uncontrolled_2d:
                    return f"utils.norm_dist({assign_str_val})[:, :, np.newaxis]"
                elif np_array.shape == expected_shape_uncontrolled_3d:
                    return f"utils.norm_dist({assign_str_val})"
                else:
                    self._add_log(f"{context_msg_b_assign}: Shape mismatch for uncontrolled factor. Expected {expected_shape_uncontrolled_2d} or {expected_shape_uncontrolled_3d}, got {np_array.shape}. Not assigning.", "ERROR")
                    return None
        
        if isinstance(self.B_spec, list):
            for f_idx, fac_b_spec in enumerate(self.B_spec):
                if fac_b_spec is None: 
                    self._add_log(f"B_matrix (factor {f_idx}): Spec is None. Using default B[{f_idx}].", "INFO")
                    continue
                
                array_data_input = fac_b_spec.get("array")
                rule = fac_b_spec.get("rule")
                is_controlled = f_idx in self.control_factor_indices
                num_actions = self.num_actions_per_control_factor.get(f_idx, 1) if is_controlled else 1
                context_msg_b = f"B_matrix (factor {self.state_names[f_idx] if f_idx < len(self.state_names) else f_idx})"

                if array_data_input is not None:
                    assign_str = get_b_assignment_string(array_data_input, self.num_states[f_idx], is_controlled, num_actions, f_idx)
                    if assign_str:
                        self.script_parts["matrix_definitions"].append(f"B[{f_idx}] = {assign_str}")
                    else:
                        self._add_log(f"{context_msg_b}: Failed to get assignment string from array data. Using default B[{f_idx}].", "WARNING")
                elif rule:
                    self._add_log(f"{context_msg_b}: Rule '{rule}' for B not fully implemented. Using default B[{f_idx}].", "WARNING")
                else:
                    self._add_log(f"{context_msg_b}: No 'array' or 'rule' found. Using default B[{f_idx}].", "INFO")

        elif isinstance(self.B_spec, dict) and self.num_factors == 1:
            f_idx = 0
            array_data_input = self.B_spec.get("array")
            rule = self.B_spec.get("rule")
            is_controlled = f_idx in self.control_factor_indices
            num_actions = self.num_actions_per_control_factor.get(f_idx, 1) if is_controlled else 1
            context_msg_b = f"B_matrix (factor {self.state_names[f_idx] if f_idx < len(self.state_names) else f_idx})"

            if array_data_input is not None:
                assign_str = get_b_assignment_string(array_data_input, self.num_states[f_idx], is_controlled, num_actions, f_idx)
                if assign_str:
                    self.script_parts["matrix_definitions"].append(f"B[{f_idx}] = {assign_str}")
                else:
                    self._add_log(f"{context_msg_b}: Failed to get assignment string from array data for single factor. Using default B[{f_idx}].", "WARNING")
            elif rule:
                self._add_log(f"{context_msg_b}: Rule '{rule}' for B (single factor) not fully implemented. Using default B[{f_idx}].", "WARNING")
            else:
                self._add_log(f"{context_msg_b}: No 'array' or 'rule' for B (single factor). Using default B[{f_idx}].", "INFO")
        else:
            self._add_log("B_matrix: B_spec format not recognized. B factors will be default initialized.", "WARNING")
            
        return "# B matrix construction based on GNN spec."

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
                    
                    array_data_input = mod_c_spec.get("array") # Expects 1D array of length num_obs[mod_idx]
                    if array_data_input:
                        try:
                            np_array = np.array(array_data_input)
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
        """Converts GNN's D vector (initial beliefs about hidden states) to PyMDP format."""
        if not self.num_factors:
            self._add_log("D_vector: No hidden state factors defined. 'D' will be None.", "INFO")
            self.script_parts["matrix_definitions"].append("D = None")
            return "# D vector set to None due to no hidden state factors."

        init_code = f"D = utils.obj_array({self.num_factors})"
        self.script_parts["matrix_definitions"].append(init_code)
        
        # Default to uniform if no D_spec
        for f_idx in range(self.num_factors):
            self.script_parts["matrix_definitions"].append(f"D[{f_idx}] = utils.norm_dist(np.ones({self.num_states[f_idx]})) # Default: uniform D for factor {f_idx}")

        if self.D_spec:
            if isinstance(self.D_spec, list): 
                for f_idx, fac_d_spec in enumerate(self.D_spec):
                    if fac_d_spec is None: 
                        self._add_log(f"D_vector (factor {f_idx}): Spec is None. Using default uniform D[{f_idx}].", "INFO")
                        continue

                    array_data_input = fac_d_spec.get("array")
                    rule = fac_d_spec.get("rule")
                    context_msg = f"D_vector (factor {self.state_names[f_idx] if f_idx < len(self.state_names) else f_idx})"

                    if array_data_input is not None:
                        parsed_array_data = self._parse_string_to_literal(array_data_input, context_msg)
                        if parsed_array_data is not None:
                            try:
                                np_array = np.array(parsed_array_data)
                                expected_shape = (self.num_states[f_idx],)
                                if np_array.shape == expected_shape:
                                    if np.isclose(np.sum(np_array), 1.0) and np.all(np_array >= 0):
                                         assign_str = _numpy_array_to_string(np_array, indent=4)
                                    else: 
                                         assign_str = f"utils.norm_dist({_numpy_array_to_string(np_array, indent=4)})"
                                    self.script_parts["matrix_definitions"].append(f"D[{f_idx}] = {assign_str}")
                                else:
                                    self._add_log(f"{context_msg}: Shape mismatch. Expected {expected_shape}, got {np_array.shape}. Using default uniform.", "ERROR")
                            except Exception as e:
                                 self._add_log(f"{context_msg}: Error processing parsed array data to NumPy: {e}. Using default uniform.", "ERROR")
                        else:
                            self._add_log(f"{context_msg}: Failed to parse array data string. Using default uniform D[{f_idx}].", "INFO")
                    elif rule:
                        self._add_log(f"{context_msg}: Rule '{rule}' for D not implemented. Using default uniform D[{f_idx}].", "WARNING")
                    else:
                        self._add_log(f"{context_msg}: No 'array' or 'rule' found. Using default uniform D[{f_idx}].", "INFO")
            elif isinstance(self.D_spec, dict) and self.num_factors == 1:
                f_idx = 0
                array_data_input = self.D_spec.get("array")
                rule = self.D_spec.get("rule")
                context_msg = f"D_vector (factor {f_idx})"
                if array_data_input is not None:
                    parsed_array_data = self._parse_string_to_literal(array_data_input, context_msg)
                    if parsed_array_data is not None:
                        try:
                            np_array = np.array(parsed_array_data)
                            expected_shape = (self.num_states[f_idx],)
                            if np_array.shape == expected_shape:
                                if np.isclose(np.sum(np_array), 1.0) and np.all(np_array >= 0):
                                     assign_str = _numpy_array_to_string(np_array, indent=4)
                                else: 
                                     assign_str = f"utils.norm_dist({_numpy_array_to_string(np_array, indent=4)})"
                                self.script_parts["matrix_definitions"].append(f"D[{f_idx}] = {assign_str}")
                            else:
                                self._add_log(f"{context_msg}: Shape mismatch for single factor D. Expected {expected_shape}, got {np_array.shape}. Using default uniform.", "ERROR")
                        except Exception as e:
                            self._add_log(f"{context_msg}: Error processing parsed array data for single factor D: {e}. Using default uniform.", "ERROR")
                    else:
                         self._add_log(f"{context_msg}: Failed to parse array data string for single factor D. Using default uniform D[{f_idx}].", "INFO")
                elif rule:
                    self._add_log(f"{context_msg}: Rule '{rule}' for D (single factor) not implemented. Using default uniform D[{f_idx}].", "WARNING")
                else:
                    self._add_log(f"{context_msg}: No 'array' or 'rule' for D (single factor). Using default uniform D[{f_idx}].", "INFO")
            else: # This 'else' aligns with 'if isinstance(self.D_spec, list)' and 'elif isinstance(self.D_spec, dict)'
                 self._add_log("D_vector: D_spec format not recognized. D factors will be default uniform.", "WARNING")
        else: # This 'else' aligns with 'if self.D_spec:'
            self._add_log("D_vector: No D_spec. All factors of D defaulted to uniform.", "INFO")
            return "# D vector factors remain default initialized (uniform)."
        
        return "# D vector construction based on GNN spec."

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
        
        # Return None for control_params_dict if it's not meant for direct Agent constructor args here.
        # Or ensure it only contains parameters that pymdp.Agent would accept via **kwargs if that's supported.
        # For now, assume it's not directly used for fixed Agent constructor args.
        return None, learning_params_dict, algorithm_params_dict

    def generate_agent_instantiation_code(self, action_names: Optional[Dict[int,List[str]]] = None, qs_initial: Optional[Any] = None) -> str: # Added args
        model_matrix_params = {"A": "A", "B": "B", "C": "C"} # A and B are always included
        
        # Add D if there are factors, otherwise Agent handles D=None if not provided and factors exist
        if self.num_factors > 0: 
            model_matrix_params["D"] = "D"
        
        # E is added only if explicitly provided and non-empty in the spec
        if self.E_spec and self.E_spec.get("array") is not None and len(self.E_spec.get("array")) > 0 :
             model_matrix_params["E"] = "E"

        _unused_control_dict, learning_params, algorithm_params = self.extract_agent_hyperparameters()
        
        policy_len = self.agent_hyperparams.get("policy_len")
        control_fac_idx_to_pass = self.control_factor_indices if self.control_factor_indices else None
        
        use_utility = self.agent_hyperparams.get("use_utility")
        use_states_info_gain = self.agent_hyperparams.get("use_states_info_gain")
        use_param_info_gain = self.agent_hyperparams.get("use_param_info_gain")
        action_selection = self.agent_hyperparams.get("action_selection")

        # qs_initial can be a string like "np.array(...)" or None (if not specified)
        # The generate_pymdp_agent_instantiation function handles this.
        # If qs_initial is actual data (list of arrays), it should be formatted by the caller or handled in generate_...
        # For now, assume if qs_initial (the variable) is passed, it's a string name of a var or direct data.

        return generate_pymdp_agent_instantiation(
            self.model_name, 
            model_params=model_matrix_params,
            control_fac_idx_list=control_fac_idx_to_pass,
            policy_len=policy_len,
            use_utility=use_utility,
            use_states_info_gain=use_states_info_gain,
            use_param_info_gain=use_param_info_gain,
            action_selection=action_selection,
            action_names=action_names, # Pass parsed action_names
            qs_initial=qs_initial,     # Pass parsed qs_initial
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
        usage_lines.append(f"{indent}print(f\"Agent '{self.model_name}' initialized with {{agent.num_factors if hasattr(agent, 'num_factors') else 'N/A'}} factors and {{agent.num_modalities if hasattr(agent, 'num_modalities') else 'N/A'}} modalities.\")")

        # Initial observation
        if init_o_raw and isinstance(init_o_raw, list) and len(init_o_raw) == self.num_modalities:
            usage_lines.append(f"{indent}o_current = {init_o_raw} # Initial observation from GNN spec")
        else: # Default initial observation (e.g., first outcome for each modality or a placeholder)
            default_o = [0] * self.num_modalities if self.num_modalities > 0 else []
            usage_lines.append(f"{indent}o_current = {default_o} # Example initial observation (e.g. first outcome for each modality)")
            if not init_o_raw and self.num_modalities > 0 : self._add_log("Simulation: No 'initial_observations' in GNN, using default.", "INFO")
        
        # Initial true state (for simulation purposes, not agent's belief D)
        if init_s_raw and isinstance(init_s_raw, list) and len(init_s_raw) == self.num_factors:
             usage_lines.append(f"{indent}s_current = {init_s_raw} # Initial true states from GNN spec")
        else:
            default_s = [0] * self.num_factors if self.num_factors > 0 else [] # Example: first state for each factor
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
        usage_lines.append(f"{inner_indent}if hasattr(agent, 'q_pi') and agent.q_pi is not None:") # Check if q_pi is available
        usage_lines.append(f"{inner_indent}{indent}print(f\"Posterior over policies (q_pi): {{agent.q_pi}}\")")
        usage_lines.append(f"{inner_indent}if efe_current is not None:")
        usage_lines.append(f"{inner_indent}{indent}print(f\"Expected Free Energy (EFE): {{efe_current}}\")")
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
        self.convert_A_matrix()
        self.convert_B_matrix()
        self.convert_C_vector()
        self.convert_D_vector()
        self.convert_E_vector()

        # Agent instantiation
        self.script_parts["agent_instantiation"].append(self.generate_agent_instantiation_code())
        
        if include_example_usage:
            self.script_parts["example_usage"] = self.generate_example_usage_code()
        else:
            self.script_parts["example_usage"] = ["# Example usage block skipped as per options."]

        script_content = []
        script_content.extend(self.script_parts["imports"])
        script_content.append("")
        
        summary_header = ["# --- GNN to PyMDP Conversion Summary ---"]
        summary_lines = [f"# {log_entry}" for log_entry in self.conversion_log]
        summary_footer = ["# --- End of GNN to PyMDP Conversion Summary ---"]
        script_content.extend(summary_header)
        script_content.extend(summary_lines)
        script_content.extend(summary_footer)
        script_content.append("")
        script_content.append("")

        script_content.extend(self.script_parts["comments"])
        script_content.append("")

        script_content.extend(self.script_parts["preamble_vars"])
        script_content.append("")

        script_content.append("# --- Matrix Definitions ---")
        script_content.extend(self.script_parts["matrix_definitions"])
        script_content.append("")
        
        script_content.append("# --- Agent Instantiation ---")
        script_content.extend(self.script_parts["agent_instantiation"])
        script_content.append("")

        # Define agent_params_for_debug dictionary for the debug block
        # It should capture the actual arguments intended for the Agent constructor
        agent_params_lines = ["agent_params_for_debug = {"]
        # Collect args from how generate_agent_instantiation_code structures them
        # Based on model_matrix_params and other specific args in generate_agent_instantiation_code:
        agent_params_lines.append("    'A': A if 'A' in globals() else None,")
        agent_params_lines.append("    'B': B if 'B' in globals() else None,")
        agent_params_lines.append("    'C': C if 'C' in globals() else None,")
        if self.num_factors > 0: # D is only passed if there are factors
             agent_params_lines.append("    'D': D if 'D' in globals() else None,")
        if self.E_spec and self.E_spec.get("array") is not None:
             agent_params_lines.append("    'E': E if 'E' in globals() else None,")

        # Add other specific Agent constructor parameters that generate_agent_instantiation_code handles
        # These should ideally be sourced from the same place generate_agent_instantiation_code gets them (self.agent_hyperparams)
        # or directly from the variables defined in the preamble.
        agent_params_lines.append("    'control_fac_idx': (control_fac_idx if control_fac_idx else None) if 'control_fac_idx' in globals() else None,")
        if self.agent_hyperparams.get("policy_len") is not None:
             agent_params_lines.append(f"    'policy_len': {self.agent_hyperparams['policy_len']},")
        if self.agent_hyperparams.get("use_utility") is not None:
             agent_params_lines.append(f"    'use_utility': {self.agent_hyperparams['use_utility']},")
        if self.agent_hyperparams.get("use_states_info_gain") is not None:
            agent_params_lines.append(f"    'use_states_info_gain': {self.agent_hyperparams['use_states_info_gain']},")
        if self.agent_hyperparams.get("use_param_info_gain") is not None:
            agent_params_lines.append(f"    'use_param_info_gain': {self.agent_hyperparams['use_param_info_gain']},")
        if self.agent_hyperparams.get("action_selection") is not None:
            agent_params_lines.append(f"    'action_selection': '{self.agent_hyperparams['action_selection']}',")
        
        # Learning and algorithm params are passed as dicts
        _, learning_params_dict, algorithm_params_dict = self.extract_agent_hyperparameters()
        if learning_params_dict:
            agent_params_lines.append(f"    'learning_params': {learning_params_dict},")
        if algorithm_params_dict:
            agent_params_lines.append(f"    'algorithm_params': {algorithm_params_dict},")

        if agent_params_lines[-1].endswith(','): # Remove trailing comma if any
            agent_params_lines[-1] = agent_params_lines[-1][:-1]
        agent_params_lines.append("}")
        script_content.extend(agent_params_lines)
        script_content.append("")

        script_content.extend(self.script_parts["example_usage"]) # This should come after agent_params_for_debug definition
        
        # Add runtime debug block
        # The temp_agent instantiation should use agent_params_for_debug
        debug_block = [
            "print('--- PyMDP Runtime Debug ---')",
            "try:",
            "    import pymdp",
            "    print(f'AGENT_SCRIPT: Imported pymdp version: {pymdp.__version__}')",
            "    print(f'AGENT_SCRIPT: pymdp module location: {pymdp.__file__}')",
            "    from pymdp.agent import Agent",
            "    print(f'AGENT_SCRIPT: Imported Agent: {Agent}')",
            "    print(f'AGENT_SCRIPT: Agent module location: {inspect.getfile(Agent)}')",
            "    print('AGENT_SCRIPT: Checking for required variables in global scope:')",
            "    # Check defined parameters for the main agent",
            "    print(f\"  AGENT_SCRIPT: A = {{A if 'A' in globals() else 'Not Defined'}}\")",
            "    print(f\"  AGENT_SCRIPT: B = {{B if 'B' in globals() else 'Not Defined'}}\")",
            "    print(f\"  AGENT_SCRIPT: C = {{C if 'C' in globals() else 'Not Defined'}}\")",
            "    print(f\"  AGENT_SCRIPT: D = {{D if 'D' in globals() else 'Not Defined'}}\")",
            "    print(f\"  AGENT_SCRIPT: E = {{E if 'E' in globals() else 'Not Defined'}}\")",
            "    print(f\"  AGENT_SCRIPT: control_fac_idx = {{control_fac_idx if 'control_fac_idx' in globals() else 'Not Defined'}}\")",
            "    print(f\"  AGENT_SCRIPT: action_names = {{action_names_dict_str if action_names_dict_str != '{{}}' else 'Not Defined'}}\")",
            "    print(f\"  AGENT_SCRIPT: qs_initial = {{qs_initial_str if qs_initial_str != 'None' else 'Not Defined'}}\")",
            "    print(f\"  AGENT_SCRIPT: agent_hyperparams = {{{agent_hyperparams_dict_str}}}\")",
            "    print('AGENT_SCRIPT: Attempting to instantiate agent with defined parameters for debug...')",
            "    # Filter out None hyperparams from agent_params_for_debug if it was originally None",
            "    # The ** unpacking handles empty dicts correctly if agent_hyperparams_dict_str was \"{}\"",
            "    debug_params_copy = {k: v for k, v in agent_params_for_debug.items() if not (isinstance(v, str) and v == 'None')}",
            "    temp_agent = Agent(**debug_params_copy)",
            "    print(f'AGENT_SCRIPT: Debug agent successfully instantiated: {temp_agent}')",
            "except Exception as e_debug:",
            "    print(f'AGENT_SCRIPT: Error during PyMDP runtime debug: {e_debug}')", 
            "    print(f\"AGENT_SCRIPT: Traceback:\\n{traceback.format_exc()}\")", # Keep f\" for this multi-line
            "print('--- End PyMDP Runtime Debug ---')",
        ]
        script_content.extend(debug_block)

        script_content.append(f"# --- GNN Model: {self.model_name} ---\\n")

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

