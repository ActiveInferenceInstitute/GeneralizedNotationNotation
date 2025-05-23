import unittest
import numpy as np
from typing import List, Dict, Any

# Adjust sys.path to ensure src can be imported
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.render.pymdp_templates import (
    generate_file_header,
    generate_conversion_summary,
    generate_debug_block,
    generate_example_usage_template,
    generate_placeholder_matrices,
    IMPORTS_TEMPLATE, # For checking content
    CONVERSION_SUMMARY_TEMPLATE, # For checking content
    DEBUG_BLOCK_TEMPLATE # For checking content
)

class TestGenerateFileHeader(unittest.TestCase):
    def test_header_generation(self):
        model_name = "TestAgent123"
        header = generate_file_header(model_name)
        self.assertIn(f"PyMDP Agent Script - {model_name}", header)
        self.assertIn("import numpy as np", header)
        self.assertIn("from pymdp.agent import Agent", header)
        self.assertTrue(header.startswith("#!/usr/bin/env python3"))

class TestGenerateConversionSummary(unittest.TestCase):
    def test_summary_with_entries(self):
        log_entries = ["INFO: Step 1 successful", "WARNING: Step 2 had issues"]
        summary = generate_conversion_summary(log_entries)
        expected_summary_lines = "\n".join([f"# {entry}" for entry in log_entries])
        self.assertIn(CONVERSION_SUMMARY_TEMPLATE.split("{summary_lines}")[0].strip(), summary)
        self.assertIn(expected_summary_lines, summary)

    def test_summary_empty_entries(self):
        summary = generate_conversion_summary([])
        # Expecting the template structure with an empty summary_lines part
        self.assertIn("# --- GNN to PyMDP Conversion Summary ---", summary)
        self.assertIn("# --- End of GNN to PyMDP Conversion Summary ---", summary)
        # Ensure nothing substantial is between the start and end summary lines if entries are empty
        content_between = summary.split("# --- GNN to PyMDP Conversion Summary ---")[1].split("# --- End of GNN to PyMDP Conversion Summary ---")[0]
        self.assertEqual(content_between.strip(), "")

class TestGenerateDebugBlock(unittest.TestCase):
    def test_debug_block_generation_all_defined(self):
        action_names_str = "{'factor_0': ['action_A', 'action_B']}"
        qs_initial_str = "initial_qs_variable"
        agent_hyperparams_str = "{'param1': True, 'param2': 10}"
        
        debug_block = generate_debug_block(action_names_str, qs_initial_str, agent_hyperparams_str)
        
        self.assertIn("print(f'AGENT_SCRIPT: Imported pymdp version:", debug_block)
        self.assertIn(f"action_names = {action_names_str}", debug_block)
        self.assertIn(f"qs_initial = {qs_initial_str}", debug_block)
        self.assertIn(f"agent_hyperparams = {agent_hyperparams_str}", debug_block)
        self.assertIn("temp_agent = Agent(**debug_params_copy)", debug_block)

    def test_debug_block_generation_none_values(self):
        action_names_str = "None" # Representing it as the string "None"
        qs_initial_str = "None"
        agent_hyperparams_str = "{}" # Empty dict string

        debug_block = generate_debug_block(action_names_str, qs_initial_str, agent_hyperparams_str)
        self.assertIn(f"action_names = Not Defined", debug_block) # Template logic for None/empty
        self.assertIn(f"qs_initial = Not Defined", debug_block)
        self.assertIn(f"agent_hyperparams = {agent_hyperparams_str}", debug_block)


class TestGenerateExampleUsageTemplate(unittest.TestCase):
    def test_basic_usage_generation(self):
        model_name = "MySimAgent"
        lines = generate_example_usage_template(model_name, 1, 1, [0])
        text = "\n".join(lines)

        self.assertIn(f"if __name__ == '__main__':", text)
        self.assertIn(f"agent = {model_name}", text)
        self.assertIn("o_current = [0]", text) # 1 modality
        self.assertIn("s_current = [0]", text) # 1 factor
        self.assertIn("T = 5", text) # Default timesteps are set as T
        self.assertIn("for t_step in range(T):", text) # Loop uses T
        self.assertIn("qs_current = agent.infer_states(o_current)", text)
        self.assertIn("action_agent = agent.sample_action()", text)
        self.assertIn("action_env[cf_idx] = int(action_agent[i])", text)

    def test_usage_no_modalities_no_factors(self):
        lines = generate_example_usage_template("ZeroAgent", 0, 0, [])
        text = "\n".join(lines)
        self.assertIn("o_current = []", text)
        self.assertIn("s_current = []", text)
        self.assertIn("action_env = np.zeros(num_factors, dtype=int)", text)
        # The template currently includes infer_states regardless of num_modalities
        self.assertIn("qs_current = agent.infer_states(o_current)", text)

    def test_usage_multiple_modalities_factors(self):
        lines = generate_example_usage_template("MultiAgent", 2, 3, [0, 2], sim_timesteps=3)
        text = "\n".join(lines)
        self.assertIn("o_current = [0, 0]", text) # 2 modalities
        self.assertIn("s_current = [0, 0, 0]", text) # 3 factors
        self.assertIn("T = 3", text) # sim_timesteps sets T
        self.assertIn("for t_step in range(T):", text) # Loop uses T
        self.assertIn("if control_fac_idx and action_agent is not None:", text)
        self.assertIn("for i, cf_idx in enumerate(control_fac_idx):", text)

    def test_usage_boolean_flags(self):
        lines_no_gp_copy = generate_example_usage_template("Test1", 1, 1, [0], use_gp_copy=False)
        self.assertIn("A_gen_process = A", "\n".join(lines_no_gp_copy))
        self.assertNotIn("copy.deepcopy(A)", "\n".join(lines_no_gp_copy))

        lines_no_print_obs = generate_example_usage_template("Test2", 1, 1, [0], print_obs=False)
        self.assertNotIn("print(f\"Observation", "\n".join(lines_no_print_obs))

        lines_no_print_beliefs = generate_example_usage_template("Test3", 1, 1, [0], print_beliefs=False)
        self.assertNotIn("print(f\"Beliefs about", "\n".join(lines_no_print_beliefs))
        
        lines_no_print_actions = generate_example_usage_template("Test4", 1, 1, [0], print_actions=False)
        self.assertNotIn("print(f\"Action selected by agent", "\n".join(lines_no_print_actions))

        lines_no_print_states = generate_example_usage_template("Test5", 1, 1, [0], print_states=False)
        self.assertNotIn("print(f\"True state (s_current)", "\n".join(lines_no_print_states))


class TestGeneratePlaceholderMatrices(unittest.TestCase):
    def test_zero_modalities_factors(self):
        placeholders = generate_placeholder_matrices(0, [])
        self.assertEqual(placeholders["A"], ["A = None # No modalities or states defined"]) 
        self.assertEqual(placeholders["B"], ["B = None # No state factors defined"])   
        self.assertEqual(placeholders["C"], ["C = None # No modalities defined"])   
        self.assertEqual(placeholders["D"], ["D = None # No state factors defined"])   

    def test_single_modality_single_factor(self):
        num_states_list = [3] # One factor with 3 states
        placeholders = generate_placeholder_matrices(1, num_states_list)

        self.assertIn("A = utils.obj_array(1)", "\n".join(placeholders["A"]))
        self.assertIn("A[0] = utils.norm_dist(np.ones((1, 1))) # Placeholder for modality 0", "\n".join(placeholders["A"]))

        self.assertIn("B = utils.obj_array(1)", "\n".join(placeholders["B"]))
        self.assertIn("B[0] = utils.norm_dist(np.ones((1, 1, 2)))", "\n".join(placeholders["B"]))

        self.assertIn("C = utils.obj_array_zeros(1)", "\n".join(placeholders["C"]))
        
        self.assertIn("D = utils.obj_array(1)", "\n".join(placeholders["D"]))
        self.assertIn("D[0] = utils.norm_dist(np.ones(3))", "\n".join(placeholders["D"]))

    def test_multiple_modalities_factors(self):
        num_states_list = [2, 3] # Factor 0: 2 states, Factor 1: 3 states
        num_modalities = 2
        placeholders = generate_placeholder_matrices(num_modalities, num_states_list)

        self.assertIn(f"A = utils.obj_array({num_modalities})", "\n".join(placeholders["A"]))
        self.assertIn("A[0] = utils.norm_dist(np.ones((1, 1))) # Placeholder for modality 0", "\n".join(placeholders["A"]))
        self.assertIn("A[1] = utils.norm_dist(np.ones((1, 1))) # Placeholder for modality 1", "\n".join(placeholders["A"]))
        
        self.assertIn(f"B = utils.obj_array({len(num_states_list)})", "\n".join(placeholders["B"]))
        self.assertIn("B[0] = utils.norm_dist(np.ones((1, 1, 2)))", "\n".join(placeholders["B"])) 
        self.assertIn("B[1] = utils.norm_dist(np.ones((1, 1, 2)))", "\n".join(placeholders["B"])) 

        self.assertIn(f"C = utils.obj_array_zeros({num_modalities})", "\n".join(placeholders["C"]))

        self.assertIn(f"D = utils.obj_array({len(num_states_list)})", "\n".join(placeholders["D"]))
        self.assertIn("D[0] = utils.norm_dist(np.ones(2))", "\n".join(placeholders["D"]))
        self.assertIn("D[1] = utils.norm_dist(np.ones(3))", "\n".join(placeholders["D"]))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 