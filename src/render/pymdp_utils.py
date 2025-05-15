import numpy as np
import re
from typing import Any, Dict, List, Optional, Union

# It's good practice to have a logger for utils too, if they might log errors/warnings
import logging
logger = logging.getLogger(__name__)

def _numpy_array_to_string(arr: np.ndarray, indent=8) -> str:
    """Converts a NumPy array to a string representation for Python script, with proper indentation."""
    if arr is None:
        return "None"
    if arr.ndim == 0: # Scalar
        return str(arr.item())
    if arr.ndim == 1:
        list_str = np.array2string(arr, separator=', ', prefix=' ' * indent)
    else:
        list_str = np.array2string(arr, separator=', ', prefix=' ' * indent)
        list_str = list_str.replace('\n', '\n' + ' ' * indent)
    list_str = re.sub(r'\[\s+', '[', list_str)
    list_str = re.sub(r'\s+\]', ']', list_str)
    list_str = re.sub(r'\s+,', ',', list_str)
    return f"np.array({list_str})"

def format_list_recursive(data_list: list, current_indent: int, item_formatter: callable) -> str:
    """Formats a potentially nested list of items (like NumPy arrays) into a string for Python script."""
    lines = []
    base_indent_str = ' ' * current_indent
    item_indent_str = ' ' * (current_indent + 4)
    
    if not any(isinstance(item, list) for item in data_list):
        formatted_items = [item_formatter(item, current_indent + 4) for item in data_list]
        if len(formatted_items) > 3: 
            lines.append("[")
            for fi in formatted_items:
                lines.append(item_indent_str + fi + ",")
            lines.append(base_indent_str + "]")
        else:
            lines.append("[" + ", ".join(formatted_items) + "]")
    else: 
        lines.append("[")
        for item in data_list:
            if isinstance(item, np.ndarray):
                lines.append(item_indent_str + item_formatter(item, current_indent + 4) + ",")
        lines.append(base_indent_str + "]")
    return '\n'.join(lines)

def generate_pymdp_matrix_definition(
    matrix_name: str,
    data: Any, 
    is_object_array: bool = False,
    num_modalities_or_factors: Optional[int] = None, 
    is_vector: bool = False
) -> str:
    """
    Generates Python code for a PyMDP matrix (A, B, C, D, etc.).
    Handles single matrices, lists of matrices (object arrays), and vectors.
    If data is already a string (e.g. "pymdp.utils.get_A_likelihood_identity(...)"), use it directly.
    """
    lines = []
    indent_str = "    " 

    if data is None:
        if is_object_array: 
            logger.debug(f"Data for object array {matrix_name} is None, defaulting to np.array([], dtype=object).")
            lines.append(f"{matrix_name} = np.array([], dtype=object)")
        else: 
            logger.debug(f"Data for non-object array {matrix_name} is None, setting to None.")
            lines.append(f"{matrix_name} = None")
        return '\n'.join(lines)
    
    if isinstance(data, str) and ("pymdp." in data or "np." in data or "utils." in data or "maths." in data):
        lines.append(f"{matrix_name} = {data}")
        return '\n'.join(lines)

    if is_object_array and isinstance(data, list):
        valid_arr_items = [item for item in data if item is not None]

        if not valid_arr_items:
            logger.debug(f"All items for object array {matrix_name} were None or filtered out. Defaulting {matrix_name} to np.array([], dtype=object).")
            lines.append(f"{matrix_name} = np.array([], dtype=object)")
            return '\n'.join(lines)

        array_strs = []
        for i, arr_item in enumerate(valid_arr_items):
            if not isinstance(arr_item, np.ndarray):
                try:
                    arr_item = np.array(arr_item)
                except Exception as e:
                    logger.error(f"Object array {matrix_name}: Could not convert non-None item at index {i} (value: {arr_item}) to np.array: {e}. Skipping this item.")
                    continue 
            array_strs.append(_numpy_array_to_string(arr_item, indent=8))
        
        if not array_strs: 
            logger.debug(f"All non-None items for object array {matrix_name} failed string conversion or were skipped. Defaulting {matrix_name} to np.array([], dtype=object).")
            lines.append(f"{matrix_name} = np.array([], dtype=object)")
            return '\n'.join(lines)

        actual_num_elements = len(array_strs)
        lines.append(f"{matrix_name} = [ # Object array for {actual_num_elements} modalities/factors")
        for arr_s in array_strs:
            lines.append(indent_str + indent_str + arr_s + ",")
        lines.append(indent_str + "]")
        
        if actual_num_elements > 0:
             lines.append(f"{matrix_name} = np.array({matrix_name}, dtype=object)")

    elif isinstance(data, (list, tuple)) and not is_object_array:
        try:
            np_array = np.array(data)
            lines.append(f"{matrix_name} = {_numpy_array_to_string(np_array, indent=4)}")
        except ValueError as e:
            logger.error(f"Could not convert data for matrix '{matrix_name}' to numpy array: {e}. Assigning None.", exc_info=True)
            lines.append(f"# ERROR: Could not convert {matrix_name} data to numpy array: {e}")
            lines.append(f"# Raw data: {data}")
            lines.append(f"{matrix_name} = None")
            
    elif isinstance(data, np.ndarray) and not is_object_array:
        lines.append(f"{matrix_name} = {_numpy_array_to_string(data, indent=4)}")
    else:
        logger.warning(f"Data for matrix '{matrix_name}' is of unexpected type: {type(data)}. Assigning as is or None.")
        lines.append(f"# Note: Data for {matrix_name} is of unexpected type: {type(data)}. Assigning as is or None.")
        lines.append(f"{matrix_name} = {data if data is not None else 'None'}")

    return '\n'.join(lines)

def generate_pymdp_agent_instantiation(
    agent_name: str,
    model_params: Dict[str, str], 
    control_params: Optional[Dict[str, Any]] = None, 
    learning_params: Optional[Dict[str, Any]] = None, 
    algorithm_params: Optional[Dict[str, Any]] = None,
    policy_len: Optional[int] = None,
    control_fac_idx_list: Optional[List[int]] = None, 
    use_utility: Optional[bool] = None, 
    use_states_info_gain: Optional[bool] = None, 
    use_param_info_gain: Optional[bool] = None, 
    action_selection: Optional[str] = None, 
    action_names: Optional[Dict[int, List[str]]] = None, 
    qs_initial: Optional[Union[List[np.ndarray], str]] = None
) -> str:
    """Generates the Agent instantiation code string."""
    # Note: This function assumes 'Agent' is available in the scope where the generated code runs.
    lines = [f"{agent_name} = Agent("]
    indent = "    "

    if model_params:
        for key, value_var_name in model_params.items():
            lines.append(f"{indent}{key}={value_var_name},")

    if control_fac_idx_list is not None:
        lines.append(f"{indent}control_fac_idx={control_fac_idx_list},")
    if policy_len is not None: lines.append(f"{indent}policy_len={policy_len},")
    if use_utility is not None: lines.append(f"{indent}use_utility={use_utility},")
    if use_states_info_gain is not None: lines.append(f"{indent}use_states_info_gain={use_states_info_gain},")
    if use_param_info_gain is not None: lines.append(f"{indent}use_param_info_gain={use_param_info_gain},")
    if action_selection is not None: lines.append(f"{indent}action_selection='{action_selection}',")
    if action_names is not None: lines.append(f"{indent}action_names={action_names},")
    if qs_initial is not None:
        if isinstance(qs_initial, str):
            lines.append(f"{indent}qs_initial={qs_initial},")
        else: 
            lines.append(f"{indent}qs_initial={repr(qs_initial)},")

    if learning_params:
        for key, value in learning_params.items():
            lines.append(f"{indent}{key}={repr(value)},")

    if algorithm_params:
        for key, value in algorithm_params.items():
            lines.append(f"{indent}{key}={repr(value)},")

    if lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]
    
    lines.append(")")
    return "\n".join(lines) 