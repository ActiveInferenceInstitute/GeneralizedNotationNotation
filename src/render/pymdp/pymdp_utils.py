import numpy as np
import re
from typing import Any, Dict, List, Optional, Union, Callable

# It's good practice to have a logger for utils too, if they might log errors/warnings
import logging
logger = logging.getLogger(__name__)

def _numpy_array_to_string(arr: np.ndarray, indent=0) -> str:
    """Converts a NumPy array to a string representation for Python script.
    The `indent` parameter specifies the indentation for lines *after the first* if the array string is multi-line.
    """
    if arr is None:
        return "None"
    if arr.ndim == 0: # Scalar
        # Ensure float scalars like np.array(1.0) are "1.0" not "1"
        item = arr.item()
        if isinstance(item, float):
            return f"{item:.1f}" if item.is_integer() else str(item)
        return str(item)
    
    # Ensure floats are like "1.0", "2.5"
    def float_format_func(x):
        if isinstance(x, float):
            return f"{x:.1f}" if x.is_integer() else str(x)
        return str(x)

    float_formatter = {'float_kind': float_format_func, 'int_kind': str}

    array_str_raw = np.array2string(arr, separator=',', 
                                 formatter=float_formatter,
                                 prefix=' '*indent)

    cleaned_str = re.sub(r'\[\s+', '[', array_str_raw)
    cleaned_str = re.sub(r'\s+\]', ']', cleaned_str)
    
    cleaned_str = re.sub(r',\s*', ',', cleaned_str)
    cleaned_str = re.sub(r'\s+,', ',', cleaned_str)
    
    cleaned_str = cleaned_str.strip()

    # Force 1D and small 2D arrays onto a single line by removing newlines
    if (arr.ndim == 1) or (arr.ndim == 2 and arr.shape[0] <= 3 and arr.shape[1] <= 5): # Heuristic for small 2D
        cleaned_str = re.sub(r'\s*\n\s*', '', cleaned_str)

    suffix = ""
    if arr.dtype == object:
        # Check if dtype=object is already in the string (np.array2string might add it for complex objects)
        if 'dtype=object' not in cleaned_str and 'dtype=np.object_' not in cleaned_str:
            suffix = ",dtype=object"

    return f"np.array({cleaned_str}{suffix})"

def format_list_recursive(data_list: list, current_indent: int, item_formatter: Callable[[Any, int], str]) -> str:
    """Formats a potentially nested list of items (like NumPy arrays) into a string for Python script."""
    lines = []
    base_indent_str = ' ' * current_indent
    # Indentation for items within the list, relative to the list's own indentation
    item_block_indent = current_indent + 4 
    item_indent_str = ' ' * item_block_indent

    if not data_list:
        return "[]"

    # Heuristic: make multi-line if list is long, or contains complex items (arrays, sublists)
    # that are likely to be multi-line themselves.
    is_complex_list = any(isinstance(item, (np.ndarray, list)) for item in data_list)
    make_multiline = len(data_list) > 2 or is_complex_list

    if make_multiline:
        lines.append("[")
        for i, item in enumerate(data_list):
            # The item_formatter should handle its own internal indentation based on the indent it's passed.
            # Here, we pass item_block_indent, meaning the *start* of the formatted item string
            # (if it were single line) would be at item_block_indent.
            # If item_formatter returns a multi-line string, its subsequent lines should be relative to that start.
            formatted_item_str = item_formatter(item, item_block_indent)

            # Add the item, properly indented, with a comma
            item_lines = formatted_item_str.split('\n')
            first_item_line = item_lines[0]
            lines.append(item_indent_str + first_item_line) # First line of item
            for sub_line in item_lines[1:]: # Subsequent lines of item, already indented by formatter relative to its start
                lines.append(item_indent_str + sub_line) # So, just add the item_indent_str for this block

            if i < len(data_list) - 1: # Add comma if not the last item
                lines[-1] += ","
        lines.append(base_indent_str + "]")
    else: # Short list, simple items, try single line
        # For single-line list, items themselves are formatted with 0 relative indent from _numpy_array_to_string
        formatted_items = [item_formatter(item, 0) for item in data_list]
        lines.append("[" + ",".join(formatted_items) + "]") # Use join with just comma
        
    return '\n'.join(lines)

def generate_pymdp_matrix_definition(
    matrix_name: str,
    data: Any, 
    is_object_array: bool = False,
    num_modalities_or_factors: Optional[int] = None, # This param is mostly for comment generation
    is_vector: bool = False
) -> str:
    """
    Generates Python code for a PyMDP matrix (A, B, C, D, etc.).
    Handles single matrices, lists of matrices (object arrays), and vectors.
    If data is already a string (e.g. "pymdp.utils.get_A_likelihood_identity(...)"), use it directly.
    """
    lines = []
    base_indent_str = "    " # Standard base indent for matrix definition lines

    if data is None:
        if is_object_array: 
            logger.debug(f"Data for object array {matrix_name} is None, defaulting to np.array([], dtype=object).")
            lines.append(f"{matrix_name} = np.array([], dtype=object)")
        else: 
            logger.debug(f"Data for non-object array {matrix_name} is None, setting to None.")
            lines.append(f"{matrix_name} = None")
        return '\n'.join(lines)
    
    if isinstance(data, str) and any(util_str in data for util_str in ["pymdp.", "np.", "utils.", "maths."]):
        lines.append(f"{matrix_name} = {data}")
        return '\n'.join(lines)

    if is_object_array and isinstance(data, list):
        valid_item_strings = []
        for item_val in data:
            if item_val is None: # Explicitly skip None items from the list
                continue
            
            item_str = None
            if isinstance(item_val, np.ndarray):
                # Indent for np.array content itself (lines after first if multi-line)
                item_str = _numpy_array_to_string(item_val, indent=len(base_indent_str * 2)) 
            elif isinstance(item_val, (list, tuple)): # Try to convert to np.array
                try:
                    np_equiv = np.array(item_val)
                    item_str = _numpy_array_to_string(np_equiv, indent=len(base_indent_str * 2))
                except Exception as e:
                    logger.warning(f"Object array {matrix_name}: Skipping unconvertible list/tuple item: {item_val} ({e})")
                    continue
            else: # Non-array, non-list/tuple item: skip as per test expectations
                logger.warning(f"Object array {matrix_name}: Skipping item of type {type(item_val)}: {item_val}")
                continue
            
            if item_str:
                valid_item_strings.append(item_str)

        if not valid_item_strings:
            lines.append(f"{matrix_name} = np.array([], dtype=object)")
        else:
            # Use a temporary list name for construction to avoid conflict if matrix_name is 'A', 'B', etc.
            temp_list_name = f"_{matrix_name}_items" 
            # Comment should reflect the number of actual items being put into the list
            actual_items_count = len(valid_item_strings)
            lines.append(f"{temp_list_name} = [ # Object array for {actual_items_count} modalities/factors")
            for i, s_arr in enumerate(valid_item_strings):
                # Each s_arr is a string like "np.array(...)". Indent this whole string.
                s_arr_lines = s_arr.split('\n')
                lines.append(base_indent_str + s_arr_lines[0] + ",") # First line of "np.array(...)"
                for sub_line in s_arr_lines[1:]:
                    lines.append(base_indent_str + sub_line + ",") # Subsequent lines of "np.array(...)"
            
            # Remove trailing comma from the last actual item in the list
            if lines and valid_item_strings: # Ensure there was at least one item
                for i_l in range(len(lines) -1, -1, -1):
                    stripped_line = lines[i_l].rstrip()
                    if stripped_line.endswith(','):
                        lines[i_l] = stripped_line[:-1]
                        break # Comma removed, stop searching
                    if stripped_line == f"{temp_list_name} = [": # Stop if we hit list start
                        break
            
            lines.append("]") # Close the list
            lines.append(f"{matrix_name} = np.array({temp_list_name}, dtype=object)")

    elif isinstance(data, (list, np.ndarray)) and not is_object_array:
        np_array_data = data if isinstance(data, np.ndarray) else None
        if np_array_data is None: # If it was a list/tuple, try to convert
            try:
                np_array_data = np.array(data)
            except ValueError as e:
                logger.error(f"Matrix {matrix_name}: Could not convert list/tuple to np.array: {e}. Assigning None.", exc_info=True)
                lines.append(f"# ERROR: Data for {matrix_name} not convertible: {e}")
                lines.append(f"{matrix_name} = None")
                return '\n'.join(lines)
        
        # Indent for the content of _numpy_array_to_string
        matrix_str = _numpy_array_to_string(np_array_data, indent=len(base_indent_str))
        lines.append(f"{matrix_name} = {matrix_str}")
    else: # Data is not a list/array or pre-formatted string, assign as is (or repr for strings)
        logger.warning(f"Matrix {matrix_name}: Unexpected data type {type(data)}. Assigning directly or via repr.")
        assign_val = repr(data) if isinstance(data, str) else str(data)
        lines.append(f"{matrix_name} = {assign_val}")

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

    all_params: Dict[str, Any] = {}

    # Model params are variable names
    for key, value_var_name in model_params.items():
        all_params[key] = value_var_name # Store as string, it's a var name

    # Other explicit agent parameters (excluding action_names which is not a constructor parameter)
    if control_fac_idx_list is not None: all_params["control_fac_idx"] = control_fac_idx_list
    if policy_len is not None: all_params["policy_len"] = policy_len
    if use_utility is not None: all_params["use_utility"] = use_utility
    if use_states_info_gain is not None: all_params["use_states_info_gain"] = use_states_info_gain
    if use_param_info_gain is not None: all_params["use_param_info_gain"] = use_param_info_gain
    if action_selection is not None: all_params["action_selection"] = action_selection # String literal for action_selection value
    
    if qs_initial is not None:
        all_params["qs_initial"] = qs_initial # String if var name, or direct list

    if learning_params: all_params.update(learning_params)
    if algorithm_params: all_params.update(algorithm_params)
    
    # Deprecated control_params - merge if present
    if control_params: 
        logger.warning("Usage of 'control_params' in generate_pymdp_agent_instantiation is deprecated. Merge into other relevant parameter groups.")
        all_params.update(control_params)


    param_lines = []
    for key, value in all_params.items():
        value_str = ""
        if key in model_params or (key == "qs_initial" and isinstance(value, str)):
            # These are variable names, so pass them directly
            value_str = str(value)
        elif key == "action_selection": # This specific one needs to be a string literal
             value_str = repr(str(value))
        else: # For others, use repr for safety (booleans, numbers, lists, dicts)
            value_str = repr(value)
        param_lines.append(f"{indent}{key}={value_str}")

    if param_lines:
        lines.extend([line + "," for line in param_lines[:-1]])
        lines.append(param_lines[-1]) 
    
    lines.append(")")
    
    # Set action_names as an attribute after instantiation if provided
    if action_names is not None:
        lines.append("")
        lines.append(f"# Set action names as agent attribute")
        lines.append(f"{agent_name}.action_names = {repr(action_names)}")
    
    return '\n'.join(lines) 