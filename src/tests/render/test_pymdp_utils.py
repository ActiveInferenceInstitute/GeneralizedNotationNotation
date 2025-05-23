import unittest
import numpy as np
from typing import Any, Dict, List, Callable

# Adjust sys.path to ensure src can be imported
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.render.pymdp_utils import (
    _numpy_array_to_string,
    format_list_recursive,
    generate_pymdp_matrix_definition,
    generate_pymdp_agent_instantiation
)

class TestNumpyArrayToString(unittest.TestCase):
    """Tests for _numpy_array_to_string function."""

    def test_none_input(self):
        self.assertEqual(_numpy_array_to_string(None), "None")

    def test_scalar_array(self):
        arr = np.array(5)
        self.assertEqual(_numpy_array_to_string(arr), "5")
        arr_float = np.array(5.0)
        self.assertEqual(_numpy_array_to_string(arr_float), "5.0")
        arr_float_precise = np.array(5.5)
        self.assertEqual(_numpy_array_to_string(arr_float_precise), "5.5")

    def test_1d_array(self):
        arr_int = np.array([1, 2, 3])
        expected_int = "np.array([1,2,3])"
        self.assertEqual(_numpy_array_to_string(arr_int, indent=0), expected_int)

        arr_float = np.array([1.0, 2.5, 3.1])
        expected_float = "np.array([1.0,2.5,3.1])"
        self.assertEqual(_numpy_array_to_string(arr_float, indent=0), expected_float)
        
        arr_float_trailing_zero = np.array([1.0, 2.0, 3.0])
        expected_float_trailing = "np.array([1.0,2.0,3.0])"
        self.assertEqual(_numpy_array_to_string(arr_float_trailing_zero, indent=0), expected_float_trailing)

        # Test with indentation (should not affect 1D array output string much if it's single line)
        self.assertEqual(_numpy_array_to_string(arr_int, indent=4), expected_int)


    def test_2d_array(self):
        arr = np.array([[1, 2], [3, 4]])
        expected_str = "np.array([[1,2],[3,4]])" # With linewidth=np.inf, should be single line
        self.assertEqual(_numpy_array_to_string(arr), expected_str)
        
        arr_float = np.array([[1.0, 2.0], [3.5, 4.1]])
        expected_float_str = "np.array([[1.0,2.0],[3.5,4.1]])"
        self.assertEqual(_numpy_array_to_string(arr_float), expected_float_str)

    def test_3d_array(self):
        arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        # linewidth=np.inf should try to make this a single line
        expected_3d_str = "np.array([[[1,2],[3,4]],[[5,6],[7,8]]])"
        self.assertEqual(_numpy_array_to_string(arr), expected_3d_str)

    def test_empty_array(self):
        arr = np.array([])
        expected = "np.array([])"
        self.assertEqual(_numpy_array_to_string(arr), expected)

    def test_array_with_object_dtype(self):
        arr_obj = np.array([1, "text", None], dtype=object)
        arr_str = _numpy_array_to_string(arr_obj)
        # Now expecting ,dtype=object to be appended
        expected_obj_str = "np.array([1,text,None],dtype=object)"
        self.assertEqual(arr_str.replace(" ", "").replace("\'", ""), expected_obj_str)


class TestFormatListRecursive(unittest.TestCase):
    """Tests for format_list_recursive function."""

    def _dummy_item_formatter(self, item: Any, indent: int) -> str:
        if isinstance(item, np.ndarray):
            return _numpy_array_to_string(item, indent=indent)
        return repr(item)

    def test_flat_list_simple_items(self):
        data = [1, "hello", True]
        expected = "[1,\'hello\',True]" # No spaces after comma due to _numpy_array_to_string changes for items
        self.assertEqual(format_list_recursive(data, 0, self._dummy_item_formatter).replace(" ", ""), expected.replace(" ", ""))

        data_long = [1, 2, 3, 4, 5] # > 2 items, should be multi-line
        expected_long_lines = [
            "[",
            "    1,",
            "    2,",
            "    3,",
            "    4,",
            "    5", # Last item no comma
            "]"
        ]
        self.assertEqual(format_list_recursive(data_long, 0, self._dummy_item_formatter), '\n'.join(expected_long_lines))


    def test_flat_list_numpy_arrays(self):
        data = [np.array([1,2]), np.array([3,4])] # 2 items, simple, might be single line
        expected = "[np.array([1,2]),np.array([3,4])]"
        result = format_list_recursive(data, 0, self._dummy_item_formatter)
        self.assertEqual(result.replace(" ", ""), expected.replace(" ", ""))

    def test_nested_list_numpy_arrays_multiline(self):
        data = [
            np.array([[1, 0], [0, 1]]),
            np.array([[0.9, 0.1], [0.1, 0.9]]),
            np.array([5,6]) # Add a third to force multi-line list
        ]
        result = format_list_recursive(data, 0, _numpy_array_to_string)
        self.assertTrue(result.startswith("["))
        self.assertTrue(result.endswith("]"))
        self.assertIn("    np.array([[1,0],[0,1]]),", result) # Note the comma
        self.assertIn("    np.array([[0.9,0.1],[0.1,0.9]]),", result)
        self.assertIn("    np.array([5,6])", result) # Last item, no comma
        self.assertGreater(len(result.split('\n')), 3) # Should be multi-line

    def test_empty_list(self):
        self.assertEqual(format_list_recursive([], 0, self._dummy_item_formatter), "[]")

    def test_indentation_multiline(self):
        data = [1, 2, 3, 4] # > 2 items -> multi-line list
        expected_lines = [
            "[",                # Base indent for list is 4 (current_indent for format_list_recursive)
            "        1,",       # item_indent_str = base_indent + 4 = 8 spaces
            "        2,",
            "        3,",
            "        4",
            "    ]"             # base_indent_str for closing bracket
        ]
        self.assertEqual(format_list_recursive(data, 4, self._dummy_item_formatter), '\n'.join(expected_lines))


class TestGeneratePyMDPMatrixDefinition(unittest.TestCase):
    """Tests for generate_pymdp_matrix_definition function."""

    def test_data_is_none(self):
        self.assertEqual(generate_pymdp_matrix_definition("A", None), "A = None")
        self.assertEqual(generate_pymdp_matrix_definition("B", None, is_object_array=True), "B = np.array([], dtype=object)")

    def test_data_is_preformatted_string(self):
        s = "np.array([1,2,3])"
        self.assertEqual(generate_pymdp_matrix_definition("C", s), "C = np.array([1,2,3])")
        s_util = "pymdp.utils.random_A_matrix(2,3)"
        self.assertEqual(generate_pymdp_matrix_definition("A_rand", s_util), f"A_rand = {s_util}")

    def test_object_array_list_of_numpy(self):
        data = [np.array([1,0]), np.array([0,1])]
        result = generate_pymdp_matrix_definition("A_obj", data, is_object_array=True)
        self.assertIn("_A_obj_items = [ # Object array for 2 modalities/factors", result)
        # Check for the items within the list, allowing for formatting from _numpy_array_to_string
        self.assertIn("    np.array([1,0]),", result) # Note: indent is from generate_pymdp_matrix_definition
        self.assertIn("    np.array([0,1])", result) # Last item, no comma after it in the list
        self.assertIn("]", result) # Closing the list
        self.assertIn("A_obj = np.array(_A_obj_items, dtype=object)", result)

    def test_object_array_list_with_none(self):
        data = [np.array([1,0]), None, np.array([0,1])]
        result = generate_pymdp_matrix_definition("B_obj_none", data, is_object_array=True)
        self.assertIn("_B_obj_none_items = [ # Object array for 2 modalities/factors", result)
        self.assertIn("    np.array([1,0]),", result)
        self.assertIn("    np.array([0,1])", result)
        self.assertNotIn("None,", result) 
        self.assertIn("B_obj_none = np.array(_B_obj_none_items, dtype=object)", result)

    def test_object_array_empty_or_all_none_list(self):
        self.assertEqual(generate_pymdp_matrix_definition("C_empty_obj", [], is_object_array=True), "C_empty_obj = np.array([], dtype=object)")
        self.assertEqual(generate_pymdp_matrix_definition("D_all_none_obj", [None, None], is_object_array=True), "D_all_none_obj = np.array([], dtype=object)")

    def test_object_array_non_convertible_item(self):
        data = [np.array([1,0]), "not_an_array_str", np.array([0,1])]
        # num_modalities_or_factors hint is 3, but only 2 are valid.
        # The comment in generate_pymdp_matrix_definition now reflects actual items found.
        result = generate_pymdp_matrix_definition("E_obj_bad", data, is_object_array=True, num_modalities_or_factors=3)
        self.assertIn("_E_obj_bad_items = [ # Object array for 2 modalities/factors", result)
        self.assertIn("np.array([1,0])", result.replace(" ",""))
        self.assertNotIn("not_an_array_str", result) 
        self.assertIn("np.array([0,1])", result.replace(" ",""))
        self.assertIn("E_obj_bad = np.array(_E_obj_bad_items, dtype=object)", result)

    def test_normal_array_from_list(self):
        data = [[1,0],[0,1]]
        result = generate_pymdp_matrix_definition("F_normal", data)
        expected_str = _numpy_array_to_string(np.array(data), indent=4) # indent for matrix content
        self.assertEqual(result, f"F_normal = {expected_str}")

    def test_normal_array_from_numpy(self):
        data = np.array([[1.0, 0.5],[0.5, 1.0]])
        result = generate_pymdp_matrix_definition("G_np", data)
        # The indent passed to _numpy_array_to_string from generate_pymdp_matrix_definition is len(base_indent_str) = 4
        expected_str = _numpy_array_to_string(data, indent=4)
        self.assertEqual(result, f"G_np = {expected_str}")
        
    def test_normal_array_conversion_error(self):
        data = [[1,0], [0,1,2]] 
        result = generate_pymdp_matrix_definition("H_jagged", data)
        self.assertIn("# ERROR: Data for H_jagged not convertible", result)
        self.assertIn("H_jagged = None", result)

    def test_unexpected_data_type_handled_as_string_or_repr(self):
        data_dict = {'a': 1} 
        result_dict = generate_pymdp_matrix_definition("I_dict", data_dict)
        self.assertIn("# Note: Data for I_dict is of unexpected type <class 'dict'>", result_dict)
        self.assertIn(f"I_dict = {str(data_dict)}", result_dict) # Default to str()
        
        data_int = 123
        result_int = generate_pymdp_matrix_definition("J_int", data_int)
        self.assertIn("# Note: Data for J_int is of unexpected type <class 'int'>", result_int)
        self.assertIn(f"J_int = {str(data_int)}", result_int)

        data_str_literal = "some_string_literal"
        result_str_lit = generate_pymdp_matrix_definition("K_str", data_str_literal)
        self.assertIn("# Note: Data for K_str is of unexpected type <class 'str'>", result_str_lit)
        self.assertIn(f"K_str = {repr(data_str_literal)}", result_str_lit) # Strings should be repr'd


class TestGeneratePyMDPAgentInstantiation(unittest.TestCase):
    """Tests for generate_pymdp_agent_instantiation function."""

    def test_basic_instantiation(self):
        model_params = {"A": "A_matrix", "B": "B_matrix"}
        result = generate_pymdp_agent_instantiation("my_agent", model_params)
        expected_lines = [
            "my_agent = Agent(",
            "    A=A_matrix,",
            "    B=B_matrix", # No comma for last param
            ")"
        ]
        self.assertEqual(result, '\n'.join(expected_lines))

    def test_all_parameters(self):
        model_params = {"A": "A_m", "B": "B_m", "C": "C_m", "D": "D_m"}
        action_names_val = {0: ['L', 'R'], 1: ['U', 'D']}
        qs_initial_val_str = "initial_qs_var"

        result = generate_pymdp_agent_instantiation(
            agent_name="complex_agent",
            model_params=model_params,
            control_fac_idx_list=[0, 1],
            policy_len=5,
            use_utility=True,
            use_states_info_gain=False,
            use_param_info_gain=True,
            action_selection="bayesian_model_reduction",
            action_names=action_names_val,
            qs_initial=qs_initial_val_str,
            learning_params={'lr_pA': 0.1, 'lr_pB': 0.2, 'use_BMA': False},
            algorithm_params={'num_iter': 10, 'policy_sep_prior': True}
        )

        expected_param_order_independent_check = {
            "A": "A_m", "B": "B_m", "C": "C_m", "D": "D_m",
            "control_fac_idx": repr([0, 1]),
            "policy_len": repr(5),
            "use_utility": repr(True),
            "use_states_info_gain": repr(False),
            "use_param_info_gain": repr(True),
            "action_selection": repr("bayesian_model_reduction"), # action_selection value is a string
            "action_names": repr(action_names_val),
            "qs_initial": qs_initial_val_str, # This is a variable name
            "lr_pA": repr(0.1),
            "lr_pB": repr(0.2),
            "use_BMA": repr(False),
            "num_iter": repr(10),
            "policy_sep_prior": repr(True)
        }

        self.assertTrue(result.startswith("complex_agent = Agent("))
        self.assertTrue(result.endswith(")"))
        # Check if all expected parameter assignments are present
        for key, val_repr in expected_param_order_independent_check.items():
            self.assertIn(f"{key}={val_repr}", result.replace(" ",""))
        
        # Count commas to ensure correct formatting (N-1 commas for N parameters)
        num_params = len(expected_param_order_independent_check)
        self.assertEqual(result.count(','), num_params - 1)


    def test_qs_initial_as_list_of_arrays(self):
        model_params = {"A": "A_mat"}
        qs_list = [np.array([0.5, 0.5]), np.array([1.0, 0.0])]
        result = generate_pymdp_agent_instantiation("agent_qs_list", model_params, qs_initial=qs_list)
        self.assertIn(f"qs_initial={repr(qs_list)}", result)
    
    def test_no_optional_params(self):
        model_params = {"A": "A_var"}
        result = generate_pymdp_agent_instantiation("simple_agent", model_params)
        expected = "simple_agent = Agent(\n    A=A_var\n)"
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 