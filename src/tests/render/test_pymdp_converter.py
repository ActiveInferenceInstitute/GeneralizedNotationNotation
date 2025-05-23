import unittest
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

# Attempt to import the class to be tested
# This might require adjusting sys.path if the test runner doesn't handle it.
# For now, let's assume it's discoverable or we'll adjust later.
try:
    from src.render.pymdp_converter import GnnToPyMdpConverter
    # If pymdp_utils or pymdp_templates are also directly tested or needed for setup:
    # from src.render.pymdp_utils import (
    #     _numpy_array_to_string,
    #     generate_pymdp_matrix_definition,
    #     generate_pymdp_agent_instantiation
    # )
    # from src.render.pymdp_templates import (
    #     generate_file_header, 
    #     generate_conversion_summary,
    #     generate_debug_block,
    #     generate_example_usage_template,
    #     generate_placeholder_matrices
    # )
except ImportError:
    # This is a fallback for local execution if PYTHONPATH isn't set up.
    # It's better if the test runner handles this.
    import sys
    # Assuming the script is run from the project root
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) 
    from src.render.pymdp_converter import GnnToPyMdpConverter
    # from src.render.pymdp_utils import (...) # if needed
    # from src.render.pymdp_templates import (...) # if needed


# --- Test Fixtures and Helper Functions ---

def create_basic_gnn_spec(
    model_name: str = "TestModel",
    num_obs_modalities: Optional[List[int]] = None,
    num_hidden_states_factors: Optional[List[int]] = None,
    obs_modality_names: Optional[List[str]] = None,
    hidden_state_factor_names: Optional[List[str]] = None,
    A_spec: Optional[Union[Dict, List[Dict]]] = None,
    B_spec: Optional[Union[Dict, List[Dict]]] = None,
    C_spec: Optional[Union[Dict, List[Dict]]] = None,
    D_spec: Optional[Union[Dict, List[Dict]]] = None,
    E_spec: Optional[Dict] = None,
    agent_hyperparameters: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """Helper to create a GNN spec for testing, focusing on key fields."""
    spec: Dict[str, Any] = {
        "ModelName": model_name,
        "StateSpaceBlock": { # Old way, converter should still handle some of this
            "ObservationModalities": {}, # Fill based on names/dims
            "HiddenStateFactors": {} # Fill based on names/dims
        },
        "ModelParameters": { # Old way for matrices
            "A": A_spec if A_spec is not None else {}, # Likelihood mapping
            "B": B_spec if B_spec is not None else {}, # Transition dynamics
            "C": C_spec if C_spec is not None else {}, # Preferred outcomes
            "D": D_spec if D_spec is not None else {}, # Initial hidden states
            "E": E_spec if E_spec is not None else {}  # Policy priors
        },
        "AgentHyperparameters": agent_hyperparameters if agent_hyperparameters is not None else {}
    }

    # Newer, direct fields take precedence if provided
    if num_obs_modalities is not None:
        spec["num_obs_modalities"] = num_obs_modalities
    if obs_modality_names is not None:
        spec["obs_modality_names"] = obs_modality_names
        if not num_obs_modalities and obs_modality_names: # Try to infer num_obs if not given
             spec["StateSpaceBlock"]["ObservationModalities"] = {name: {"Dimension": 1} for name in obs_modality_names} # Dummy dim for now

    if num_hidden_states_factors is not None:
        spec["num_hidden_states_factors"] = num_hidden_states_factors
    if hidden_state_factor_names is not None:
        spec["hidden_state_factor_names"] = hidden_state_factor_names
        if not num_hidden_states_factors and hidden_state_factor_names:
            spec["StateSpaceBlock"]["HiddenStateFactors"] = {name: {"Dimension": 1} for name in hidden_state_factor_names} # Dummy dim

    # Populate StateSpaceBlock from names if dimensions are not directly provided
    # This logic can be refined as per converter's actual parsing strategy
    if obs_modality_names and not spec["StateSpaceBlock"]["ObservationModalities"]:
        spec["StateSpaceBlock"]["ObservationModalities"] = {
            name: {"Dimension": (num_obs_modalities[i] if num_obs_modalities and i < len(num_obs_modalities) else 2)} 
            for i, name in enumerate(obs_modality_names)
        }
    
    if hidden_state_factor_names and not spec["StateSpaceBlock"]["HiddenStateFactors"]:
         spec["StateSpaceBlock"]["HiddenStateFactors"] = {
            name: {"Dimension": (num_hidden_states_factors[i] if num_hidden_states_factors and i < len(num_hidden_states_factors) else 2)} 
            for i, name in enumerate(hidden_state_factor_names)
        }
         
    # Add any other kwargs directly to the spec for flexibility
    spec.update(kwargs)
    return spec

# --- Test Classes ---

class TestGnnToPyMdpConverterInitialization(unittest.TestCase):
    """Tests for the __init__ method of GnnToPyMdpConverter."""

    def test_initialization_with_minimal_spec(self):
        """Test basic initialization with a very simple GNN spec."""
        gnn_spec = create_basic_gnn_spec(model_name="MinimalModel")
        converter = GnnToPyMdpConverter(gnn_spec)
        self.assertEqual(converter.model_name, "MinimalModel")
        self.assertIsInstance(converter.script_parts, dict)
        self.assertIsInstance(converter.conversion_log, list)
        # Check default values for essential attributes if not in spec
        self.assertEqual(converter.num_modalities, 0) # Assuming it tries to parse from empty blocks
        self.assertEqual(converter.num_factors, 0)

    def test_initialization_with_model_name_sanitization(self):
        """Test that model names are correctly sanitized."""
        test_cases = [
            ("My Model Name", "My_Model_Name"),
            ("model-with-hyphens", "model_with_hyphens"),
            ("123model", "_123model"),
            ("model_!@#$", "model_"),
            ("", "pymdp_agent_model"), # Empty name
            (None, "pymdp_agent_model"), # None name (handled by create_basic_gnn_spec or constructor)
            (123, "pymdp_agent_model") # Non-string name
        ]
        for original_name, expected_sanitized_name in test_cases:
            with self.subTest(original_name=original_name):
                if original_name is None: # None needs to be passed as GNN value directly
                    gnn_spec = {"ModelName": None}
                elif isinstance(original_name, int):
                     gnn_spec = {"ModelName": original_name}
                else:
                    gnn_spec = create_basic_gnn_spec(model_name=original_name)
                
                converter = GnnToPyMdpConverter(gnn_spec)
                self.assertEqual(converter.model_name, expected_sanitized_name)

    def test_initialization_state_and_obs_dimensions_direct(self):
        """Test parsing of num_obs_modalities and num_hidden_states_factors directly."""
        gnn_spec = create_basic_gnn_spec(
            num_obs_modalities=[2, 3],
            obs_modality_names=["ModalityA", "ModalityB"],
            num_hidden_states_factors=[4, 5],
            hidden_state_factor_names=["FactorX", "FactorY"]
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        self.assertEqual(converter.num_obs, [2, 3])
        self.assertEqual(converter.obs_names, ["ModalityA", "ModalityB"])
        self.assertEqual(converter.num_modalities, 2)
        self.assertEqual(converter.num_states, [4, 5])
        self.assertEqual(converter.state_names, ["FactorX", "FactorY"])
        self.assertEqual(converter.num_factors, 2)
        self.assertTrue(any("Successfully parsed stringified direct_num_obs" not in log for log in converter.conversion_log))


    def test_initialization_state_and_obs_dimensions_from_statespaceblock(self):
        """Test parsing from StateSpaceBlock if direct keys are absent."""
        gnn_spec = {
            "ModelName": "SSBlockModel",
            "StateSpaceBlock": {
                "ObservationModalities": {
                    "Visual": {"Dimension": 2, "Comment": "Screen pixels"},
                    "Audio": {"Dimension": 5, "Comment": "Sound frequency bands"}
                },
                "HiddenStateFactors": {
                    "Location": {"Dimension": 10, "Comment": "Grid cells"},
                    "Emotion": {"Dimension": 3, "Comment": "Basic emotions"}
                }
            },
            "ModelParameters": {},
            "AgentHyperparameters": {}
        }
        converter = GnnToPyMdpConverter(gnn_spec)
        self.assertEqual(converter.num_obs, [2, 5]) # Order might depend on dict iteration, ensure it's consistent or names are used
        self.assertEqual(sorted(converter.obs_names), sorted(["Visual", "Audio"]))
        self.assertEqual(converter.num_modalities, 2)
        
        self.assertEqual(converter.num_states, [10, 3]) # Order might depend on dict iteration
        self.assertEqual(sorted(converter.state_names), sorted(["Location", "Emotion"]))
        self.assertEqual(converter.num_factors, 2)

    def test_initialization_with_stringified_direct_dimensions(self):
        """Test parsing of stringified num_obs_modalities and num_hidden_states_factors."""
        gnn_spec = {
            "ModelName": "StringifiedDims",
            "num_obs_modalities": "[2, 3]", # Stringified list
            "obs_modality_names": ["ModA", "ModB"],
            "num_hidden_states_factors": "[4, 5]", # Stringified list
            "hidden_state_factor_names": ["FactorX", "FactorY"],
            "StateSpaceBlock": {},
            "ModelParameters": {},
            "AgentHyperparameters": {}
        }
        converter = GnnToPyMdpConverter(gnn_spec)
        self.assertEqual(converter.num_obs, [2, 3])
        self.assertEqual(converter.obs_names, ["ModA", "ModB"])
        self.assertEqual(converter.num_modalities, 2)
        self.assertEqual(converter.num_states, [4, 5])
        self.assertEqual(converter.state_names, ["FactorX", "FactorY"])
        self.assertEqual(converter.num_factors, 2)
        self.assertTrue(any("Successfully parsed stringified direct_num_obs: [2, 3]" in log for log in converter.conversion_log))
        self.assertTrue(any("Successfully parsed stringified direct_num_states: [4, 5]" in log for log in converter.conversion_log))

    def test_initialization_matrix_specs_stored(self):
        """Test that matrix specifications from GNN spec are stored."""
        a_spec_data = {"Modality1": "np.array([[0.8, 0.2], [0.1, 0.9]])"}
        b_spec_data = {"Factor1": "np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.5,0.5], [0.5,0.5]]])"} # B needs action dim
        c_spec_data = {"Modality1": "[0.0, 1.0]"}
        d_spec_data = {"Factor1": "[0.9, 0.1]"}
        e_spec_data = {"policy_prior": "[0.5, 0.5]"}

        gnn_spec = create_basic_gnn_spec(
            A_spec=a_spec_data, B_spec=b_spec_data, C_spec=c_spec_data, D_spec=d_spec_data, E_spec=e_spec_data,
            num_obs_modalities=[2], obs_modality_names=["Modality1"],
            num_hidden_states_factors=[2], hidden_state_factor_names=["Factor1"],
            # For B matrix to be processed, we need control factors and actions
            num_control_factors=[2], # num_actions for Factor1 (the controlled one)
            control_action_names_per_factor={0: ["Action1", "Action2"]}, # Factor 0 is Factor1
            ControlFactorsBlock={"Factor1": {"Actions": ["Action1", "Action2"]}} # Old way to define actions
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        self.assertEqual(converter.A_spec, a_spec_data)
        self.assertEqual(converter.B_spec, b_spec_data)
        self.assertEqual(converter.C_spec, c_spec_data)
        self.assertEqual(converter.D_spec, d_spec_data)
        self.assertEqual(converter.E_spec, e_spec_data)

    def test_initialization_empty_gnn_spec(self):
        """Test initialization with a completely empty GNN spec."""
        gnn_spec: Dict[str, Any] = {}
        converter = GnnToPyMdpConverter(gnn_spec)
        self.assertEqual(converter.model_name, "pymdp_agent_model") # Default model name
        self.assertEqual(converter.num_modalities, 0)
        self.assertEqual(converter.num_factors, 0)
        self.assertEqual(converter.num_obs, [])
        self.assertEqual(converter.num_states, [])
        self.assertIsNone(converter.A_spec) # Should be none if not in spec
        self.assertIsNone(converter.B_spec)
        self.assertTrue(any("Using default model name" in log or "ModelName not found" in log for log in converter.conversion_log), "Expected log for default model name")

    def test_initialization_missing_essential_blocks(self):
        """Test initialization when StateSpaceBlock or ModelParameters are missing."""
        gnn_spec_no_ss = {"ModelName": "NoSS"}
        converter_no_ss = GnnToPyMdpConverter(gnn_spec_no_ss)
        self.assertEqual(converter_no_ss.num_modalities, 0)
        self.assertEqual(converter_no_ss.num_factors, 0)
        # Check for logs indicating missing blocks
        self.assertTrue(any("StateSpaceBlock not found or empty" in log for log in converter_no_ss.conversion_log))

        gnn_spec_no_mp = {"ModelName": "NoMP", "StateSpaceBlock": {"ObservationModalities": {"O1": {"Dimension": 2}}}}
        converter_no_mp = GnnToPyMdpConverter(gnn_spec_no_mp)
        self.assertIsNone(converter_no_mp.A_spec) # A_spec comes from ModelParameters
        self.assertTrue(any("ModelParameters not found or empty" in log for log in converter_no_mp.conversion_log))

    def test_initialization_mixed_direct_and_statespaceblock_precedence(self):
        """Test that direct num_obs/states keys take precedence over StateSpaceBlock."""
        gnn_spec = {
            "ModelName": "MixedPrecedence",
            "num_obs_modalities": [3], # Direct
            "obs_modality_names": ["DirectObs"],
            "num_hidden_states_factors": [4], # Direct
            "hidden_state_factor_names": ["DirectFactor"],
            "StateSpaceBlock": {
                "ObservationModalities": {"SSObs": {"Dimension": 22}}, # Should be ignored for num_obs
                "HiddenStateFactors": {"SSFactor": {"Dimension": 33}}  # Should be ignored for num_states
            },
            "ModelParameters": {},
            "AgentHyperparameters": {}
        }
        converter = GnnToPyMdpConverter(gnn_spec)
        self.assertEqual(converter.num_obs, [3])
        self.assertEqual(converter.obs_names, ["DirectObs"])
        self.assertEqual(converter.num_modalities, 1)
        self.assertEqual(converter.num_states, [4])
        self.assertEqual(converter.state_names, ["DirectFactor"])
        self.assertEqual(converter.num_factors, 1)
        # Ensure logs show that direct values were used, and perhaps a warning about StateSpaceBlock mismatch if implemented
        self.assertTrue(any("Using direct 'num_obs_modalities' from GNN spec" in log for log in converter.conversion_log))
        self.assertTrue(any("Using direct 'num_hidden_states_factors' from GNN spec" in log for log in converter.conversion_log))


    def test_initialization_control_factor_parsing(self):
        """Test parsing of control factor information."""
        gnn_spec = create_basic_gnn_spec(
            hidden_state_factor_names=["Location", "Tool"],
            num_hidden_states_factors=[3, 2],
            num_control_factors=[2, 0],  # Location is controlled (2 actions), Tool is not
            control_action_names_per_factor={
                0: ["Stay", "Move"], # Actions for Location (factor 0)
                # Factor 1 (Tool) is not controlled, so no entry or empty list expected
            },
            ControlFactorsBlock={ # Old way, for compatibility testing if converter supports it
                "Location": {"Actions": ["Stay_Old", "Move_Old"]}
            }
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        self.assertEqual(converter.num_actions_per_control_factor, {0: 2}) # Only factor 0 is controlled
        self.assertEqual(converter.action_names_per_control_factor, {0: ["Stay", "Move"]})
        self.assertEqual(converter.control_factor_indices, [0]) # Index of the 'Location' factor

        # Test with only ControlFactorsBlock (if direct num_control_factors etc. are missing)
        gnn_spec_old_control = {
            "ModelName": "OldControl",
            "StateSpaceBlock": {
                "HiddenStateFactors": {
                    "S1": {"Dimension": 2}, "S2": {"Dimension": 3}
                }
            },
            "ControlFactorsBlock": {
                "S1": {"Actions": ["a1", "a2"]} # S1 is controlled
            },
            "ModelParameters": {}
        }
        converter_old = GnnToPyMdpConverter(gnn_spec_old_control)
        self.assertEqual(converter_old.action_names_per_control_factor.get(0), ["a1", "a2"])
        self.assertEqual(converter_old.num_actions_per_control_factor.get(0), 2)
        self.assertIn(0, converter_old.control_factor_indices)


    def test_initialization_agent_hyperparameter_storage(self):
        """Test that agent hyperparameters are stored correctly from the spec."""
        hyperparams = {
            "agent_params": {"use_param_info_gain": True, "planning_horizon": 2},
            "policy_params": {"initial_action_selection": "deterministic"},
            "qs_initial_params": {"method": "uniform"}
        }
        gnn_spec = create_basic_gnn_spec(agent_hyperparameters=hyperparams)
        converter = GnnToPyMdpConverter(gnn_spec)
        # The converter primarily stores the raw block for later processing by extract_agent_hyperparameters
        self.assertEqual(converter.gnn_spec.get("AgentHyperparameters"), hyperparams)
        # Test that the internal attribute for them is also populated (if it is during __init__)
        # Or confirm they are correctly extracted later. For now, check storage.
        self.assertEqual(converter.agent_hyperparams, {}) # Assuming these are populated by extract_agent_hyperparameters, not init


class TestMatrixConversion(unittest.TestCase):
    """Tests for A, B, C, D, E matrix conversion methods."""

    def test_convert_A_matrix_single_modality_single_factor(self):
        """Test A matrix conversion for a simple case: 1 obs modality, 1 state factor."""
        gnn_spec = create_basic_gnn_spec(
            obs_modality_names=["Visual"], num_obs_modalities=[2],
            hidden_state_factor_names=["Location"], num_hidden_states_factors=[3],
            A_spec={"Visual": "np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])"} # 2 obs_outcomes x 3 states
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        a_matrix_str = converter.convert_A_matrix()
        
        self.assertIn("A_Visual = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])", a_matrix_str)
        self.assertIn("A = np.empty(1, dtype=object)", a_matrix_str) # num_modalities = 1
        self.assertIn("A[0] = A_Visual", a_matrix_str)

    def test_convert_A_matrix_multiple_modalities_single_factor(self):
        gnn_spec = create_basic_gnn_spec(
            obs_modality_names=["Visual", "Audio"], num_obs_modalities=[2, 3],
            hidden_state_factor_names=["Location"], num_hidden_states_factors=[4],
            A_spec={
                "Visual": "np.array([[0.8,0.1,0.0,0.1], [0.1,0.8,0.1,0.0]])", # 2x4
                "Audio": "np.array([[0.5,0.3,0.1,0.1], [0.3,0.4,0.2,0.1], [0.1,0.1,0.6,0.2]])" # 3x4
            }
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        a_matrix_str = converter.convert_A_matrix()

        self.assertIn("A_Visual = np.array([[0.8,0.1,0.0,0.1], [0.1,0.8,0.1,0.0]])", a_matrix_str)
        self.assertIn("A_Audio = np.array([[0.5,0.3,0.1,0.1], [0.3,0.4,0.2,0.1], [0.1,0.1,0.6,0.2]])", a_matrix_str)
        self.assertIn("A = np.empty(2, dtype=object)", a_matrix_str) # num_modalities = 2
        self.assertIn("A[0] = A_Visual", a_matrix_str) # Assuming Visual is first
        self.assertIn("A[1] = A_Audio", a_matrix_str) # Assuming Audio is second

    def test_convert_A_matrix_single_modality_multiple_factors(self):
        gnn_spec = create_basic_gnn_spec(
            obs_modality_names=["Sensor"], num_obs_modalities=[2], # 2 outcomes
            hidden_state_factor_names=["FactorA", "FactorB"], num_hidden_states_factors=[2, 3], # 2 states, 3 states
            A_spec={"Sensor": "np.random.rand(2, 2, 3)"} # Shape (num_outcomes, num_states_fA, num_states_fB)
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        a_matrix_str = converter.convert_A_matrix()
        
        self.assertIn("A_Sensor = np.random.rand(2, 2, 3)", a_matrix_str)
        self.assertIn("A = np.empty(1, dtype=object)", a_matrix_str)
        self.assertIn("A[0] = A_Sensor", a_matrix_str)

    def test_convert_A_matrix_invalid_spec(self):
        gnn_spec_mismatch = create_basic_gnn_spec(
            obs_modality_names=["Visual"], num_obs_modalities=[2],
            hidden_state_factor_names=["Location"], num_hidden_states_factors=[3],
            A_spec={"Visual": "np.array([[0.8, 0.1], [0.1, 0.8]])"} # 2x2, but expected 2x3
        )
        converter_mismatch = GnnToPyMdpConverter(gnn_spec_mismatch)
        a_matrix_str_mismatch = converter_mismatch.convert_A_matrix()
        self.assertTrue(any("Error processing A matrix for modality Visual" in log for log in converter_mismatch.conversion_log) or 
                        any("Could not parse or validate matrix A_Visual" in log for log in converter_mismatch.conversion_log) or 
                        any("shapes (2,2) and (2,3) not aligned" in log.lower() for log in converter_mismatch.conversion_log))
        self.assertIn("# Placeholder for A_Visual due to error", a_matrix_str_mismatch)

        gnn_spec_invalid_str = create_basic_gnn_spec(
            obs_modality_names=["Visual"], num_obs_modalities=[2],
            hidden_state_factor_names=["Location"], num_hidden_states_factors=[3],
            A_spec={"Visual": "not_a_valid_array(("}
        )
        converter_invalid_str = GnnToPyMdpConverter(gnn_spec_invalid_str)
        a_matrix_str_invalid = converter_invalid_str.convert_A_matrix()
        self.assertTrue(any("ast.literal_eval failed" in log for log in converter_invalid_str.conversion_log) or 
                        any("Could not parse matrix string for A_Visual" in log for log in converter_invalid_str.conversion_log))
        self.assertIn("# Placeholder for A_Visual due to error", a_matrix_str_invalid)

    def test_convert_A_matrix_numpy_array_input(self):
        """Test A matrix conversion when the spec provides an actual np.ndarray."""
        gnn_spec = create_basic_gnn_spec(
            obs_modality_names=["Sensor"], num_obs_modalities=[2],
            hidden_state_factor_names=["Internal"], num_hidden_states_factors=[2],
            A_spec={"Sensor": np.array([[0.9, 0.1], [0.2, 0.8]])} 
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        a_matrix_str = converter.convert_A_matrix()
        self.assertIn("A_Sensor = np.array([[0.9, 0.1], [0.2, 0.8]])", a_matrix_str)

    # --- B Matrix Tests ---
    def test_convert_B_matrix_single_factor_controlled(self):
        gnn_spec = create_basic_gnn_spec(
            hidden_state_factor_names=["Position"], num_hidden_states_factors=[3],
            num_control_factors=[2], # 2 actions for Position factor
            control_action_names_per_factor={0: ["Stay", "Move"]},
            B_spec={"Position": "np.array([[[1,0,0],[0,1,0],[0,0,1]], [[0,1,0],[0,0,1],[1,0,0]]])"} # 2 actions x 3 states x 3 states
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        b_matrix_str = converter.convert_B_matrix()

        self.assertIn("B_Position = np.array([[[1,0,0],[0,1,0],[0,0,1]], [[0,1,0],[0,0,1],[1,0,0]]])", b_matrix_str)
        self.assertIn("B = np.empty(1, dtype=object)", b_matrix_str) # num_factors = 1
        self.assertIn("B[0] = B_Position", b_matrix_str)

    def test_convert_B_matrix_multiple_factors_mixed_control(self):
        gnn_spec = create_basic_gnn_spec(
            hidden_state_factor_names=["Location", "Tool"], num_hidden_states_factors=[2, 3], # Loc:2 states, Tool:3 states
            num_control_factors=[2, 0], # Location controlled (2 actions), Tool not controlled
            control_action_names_per_factor={0: ["Left", "Right"]}, # Actions for Location
            B_spec={
                "Location": "np.array([[[0.9,0.1],[0.1,0.9]], [[0.1,0.9],[0.9,0.1]]])", # 2 actions x 2 states x 2 states
            }
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        b_matrix_str = converter.convert_B_matrix()

        self.assertIn("B_Location = np.array([[[0.9,0.1],[0.1,0.9]], [[0.1,0.9],[0.9,0.1]]])", b_matrix_str)
        self.assertIn("B_Tool = np.array([[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]])", b_matrix_str.replace(" ", "")) 
        self.assertIn("B = np.empty(2, dtype=object)", b_matrix_str) # num_factors = 2
        self.assertIn("B[0] = B_Location", b_matrix_str)
        self.assertIn("B[1] = B_Tool", b_matrix_str)
        
    def test_convert_B_matrix_numpy_array_input(self):
        b_val = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.5,0.5], [0.5,0.5]]]) # 2 actions, 2 states, 2 states
        gnn_spec = create_basic_gnn_spec(
            hidden_state_factor_names=["Mode"], num_hidden_states_factors=[2],
            num_control_factors=[2], control_action_names_per_factor={0: ["A1", "A2"]},
            B_spec={"Mode": b_val}
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        b_matrix_str = converter.convert_B_matrix()
        self.assertIn(f"B_Mode = {converter._numpy_array_to_string(b_val)}", b_matrix_str)
        self.assertIn("B = np.empty(1, dtype=object)", b_matrix_str)
        self.assertIn("B[0] = B_Mode", b_matrix_str)

    def test_convert_B_matrix_invalid_spec(self):
        gnn_spec_mismatch = create_basic_gnn_spec(
            hidden_state_factor_names=["Position"], num_hidden_states_factors=[3],
            num_control_factors=[2], control_action_names_per_factor={0: ["Stay", "Move"]},
            B_spec={"Position": "np.array([[1,0],[0,1]])"} # Incorrect shape for B
        )
        converter_mismatch = GnnToPyMdpConverter(gnn_spec_mismatch)
        b_matrix_str_mismatch = converter_mismatch.convert_B_matrix()
        self.assertTrue(any("Error processing B matrix for factor Position" in log for log in converter_mismatch.conversion_log) or 
                        any("Could not parse or validate matrix B_Position" in log for log in converter_mismatch.conversion_log))
        self.assertIn("# Placeholder for B_Position due to error", b_matrix_str_mismatch)

        gnn_spec_invalid_str = create_basic_gnn_spec(
            hidden_state_factor_names=["Position"], num_hidden_states_factors=[3],
            num_control_factors=[2], control_action_names_per_factor={0: ["Stay", "Move"]},
            B_spec={"Position": "invalid_array(("}
        )
        converter_invalid_str = GnnToPyMdpConverter(gnn_spec_invalid_str)
        b_matrix_str_invalid = converter_invalid_str.convert_B_matrix()
        self.assertTrue(any("ast.literal_eval failed" in log for log in converter_invalid_str.conversion_log) or 
                        any("Could not parse matrix string for B_Position" in log for log in converter_invalid_str.conversion_log))
        self.assertIn("# Placeholder for B_Position due to error", b_matrix_str_invalid)

    # --- C Vector Tests ---
    def test_convert_C_vector_single_modality(self):
        gnn_spec = create_basic_gnn_spec(
            obs_modality_names=["Reward"], num_obs_modalities=[3], # 3 outcomes for Reward modality
            C_spec={"Reward": "np.array([0.0, 1.0, -1.0])"}
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        c_vector_str = converter.convert_C_vector()
        self.assertIn("C_Reward = np.array([0.0, 1.0, -1.0])", c_vector_str)
        self.assertIn("C = np.empty(1, dtype=object)", c_vector_str)
        self.assertIn("C[0] = C_Reward", c_vector_str)

    def test_convert_C_vector_multiple_modalities(self):
        gnn_spec = create_basic_gnn_spec(
            obs_modality_names=["Feedback", "Score"], num_obs_modalities=[2, 5],
            C_spec={
                "Feedback": "np.array([1.0, -1.0])",
                "Score": "np.array([0, 10, 20, 30, 0])"
            }
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        c_vector_str = converter.convert_C_vector()
        self.assertIn("C_Feedback = np.array([1.0, -1.0])", c_vector_str)
        self.assertIn("C_Score = np.array([0, 10, 20, 30, 0])", c_vector_str)
        self.assertIn("C = np.empty(2, dtype=object)", c_vector_str)
        self.assertIn("C[0] = C_Feedback", c_vector_str) # Assuming Feedback is first
        self.assertIn("C[1] = C_Score", c_vector_str)

    def test_convert_C_vector_invalid_spec(self):
        gnn_spec_mismatch = create_basic_gnn_spec(
            obs_modality_names=["Reward"], num_obs_modalities=[3],
            C_spec={"Reward": "np.array([0.0, 1.0])"} # Mismatched size (2 vs 3)
        )
        converter_mismatch = GnnToPyMdpConverter(gnn_spec_mismatch)
        c_str_mismatch = converter_mismatch.convert_C_vector()
        self.assertTrue(any("Error processing C vector for modality Reward" in log for log in converter_mismatch.conversion_log))
        self.assertIn("# Placeholder for C_Reward due to error", c_str_mismatch)
        
        gnn_spec_none = create_basic_gnn_spec(
            obs_modality_names=["Reward"], num_obs_modalities=[3],
            C_spec=None # No C spec provided
        )
        converter_none = GnnToPyMdpConverter(gnn_spec_none)
        c_str_none = converter_none.convert_C_vector()
        self.assertIn("C_Reward = np.zeros(3)", c_str_none) # Default to zeros
        self.assertTrue(any("No C specification found for modality Reward. Defaulting to zeros." in log for log in converter_none.conversion_log))


    # --- D Vector Tests ---
    def test_convert_D_vector_single_factor(self):
        gnn_spec = create_basic_gnn_spec(
            hidden_state_factor_names=["Belief"], num_hidden_states_factors=[4],
            D_spec={"Belief": "np.array([0.1, 0.2, 0.3, 0.4])"}
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        d_vector_str = converter.convert_D_vector()
        self.assertIn("D_Belief = np.array([0.1, 0.2, 0.3, 0.4])", d_vector_str)
        self.assertIn("D = np.empty(1, dtype=object)", d_vector_str)
        self.assertIn("D[0] = D_Belief", d_vector_str)

    def test_convert_D_vector_multiple_factors(self):
        gnn_spec = create_basic_gnn_spec(
            hidden_state_factor_names=["Location", "Topic"], num_hidden_states_factors=[2, 3],
            D_spec={
                "Location": "np.array([0.8, 0.2])",
                "Topic": "np.array([0.5, 0.25, 0.25])"
            }
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        d_vector_str = converter.convert_D_vector()
        self.assertIn("D_Location = np.array([0.8, 0.2])", d_vector_str)
        self.assertIn("D_Topic = np.array([0.5, 0.25, 0.25])", d_vector_str)
        self.assertIn("D = np.empty(2, dtype=object)", d_vector_str)
        self.assertIn("D[0] = D_Location", d_vector_str)
        self.assertIn("D[1] = D_Topic", d_vector_str)
        
    def test_convert_D_vector_invalid_spec(self):
        gnn_spec_mismatch = create_basic_gnn_spec(
            hidden_state_factor_names=["Belief"], num_hidden_states_factors=[4],
            D_spec={"Belief": "np.array([0.1, 0.9])"} # Mismatched size (2 vs 4)
        )
        converter_mismatch = GnnToPyMdpConverter(gnn_spec_mismatch)
        d_str_mismatch = converter_mismatch.convert_D_vector()
        self.assertTrue(any("Error processing D vector for factor Belief" in log for log in converter_mismatch.conversion_log))
        self.assertIn("# Placeholder for D_Belief due to error", d_str_mismatch)

        gnn_spec_none = create_basic_gnn_spec(
            hidden_state_factor_names=["Belief"], num_hidden_states_factors=[4],
            D_spec=None # No D spec provided
        )
        converter_none = GnnToPyMdpConverter(gnn_spec_none)
        d_str_none = converter_none.convert_D_vector()
        self.assertIn("D_Belief = np.ones(4) / 4.0", d_str_none.replace(" ", "")) # Default to uniform
        self.assertTrue(any("No D specification found for factor Belief. Defaulting to uniform." in log for log in converter_none.conversion_log))

    # --- E Vector Tests ---
    def test_convert_E_vector(self):
        # E is policy prior, a single vector (num_policies)
        num_policies = 3
        gnn_spec = create_basic_gnn_spec(
            hidden_state_factor_names=["S1"], num_hidden_states_factors=[2],
            num_control_factors=[2], control_action_names_per_factor={0: ["a1", "a2"]},
            E_spec={"policy_prior": f"np.ones({num_policies}) / {num_policies}.0"}
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        e_vector_str = converter.convert_E_vector()
        expected_e_str = f"E_policy_prior = np.ones({num_policies}) / {num_policies}.0"
        self.assertIn(expected_e_str.replace(" ", ""), e_vector_str.replace(" ", ""))
        self.assertIn("E = E_policy_prior", e_vector_str) 

    def test_convert_E_vector_invalid_spec(self):
        gnn_spec_no_E = create_basic_gnn_spec(
             hidden_state_factor_names=["S1"], num_hidden_states_factors=[2],
             num_control_factors=[2], control_action_names_per_factor={0: ["a1", "a2"]},
             E_spec=None
        )
        converter_no_E = GnnToPyMdpConverter(gnn_spec_no_E)
        e_str_no_E = converter_no_E.convert_E_vector()
        self.assertIn("E = None", e_str_no_E) 
        self.assertTrue(any("No E (policy prior) specification found. Defaulting to None." in log for log in converter_no_E.conversion_log))

        gnn_spec_invalid_E_str = create_basic_gnn_spec(
             hidden_state_factor_names=["S1"], num_hidden_states_factors=[2],
             num_control_factors=[2], control_action_names_per_factor={0: ["a1", "a2"]},
             E_spec={"policy_prior": "not_a_vector(("}
        )
        converter_invalid_E = GnnToPyMdpConverter(gnn_spec_invalid_E_str)
        e_str_invalid = converter_invalid_E.convert_E_vector()
        self.assertIn("E = None", e_str_invalid) 
        self.assertTrue(any("Error processing E vector (policy_prior)" in log for log in converter_invalid_E.conversion_log))


class TestAgentAndUsageCodeGeneration(unittest.TestCase):
    """Tests for agent instantiation and example usage code generation."""

    def test_extract_agent_hyperparameters(self):
        base_spec = create_basic_gnn_spec() # Minimal spec
        
        # Scenario 1: No hyperparameters specified
        converter_no_params = GnnToPyMdpConverter(base_spec)
        agent_p, policy_p, qs_init_p = converter_no_params.extract_agent_hyperparameters()
        self.assertIsNone(agent_p)
        self.assertIsNone(policy_p)
        self.assertIsNone(qs_init_p)
        self.assertTrue(any("No AgentHyperparameters block found or it's empty" in log for log in converter_no_params.conversion_log))

        # Scenario 2: Only agent_params
        agent_params_data = {"use_param_info_gain": True, "planning_horizon": 3}
        spec_agent_only = create_basic_gnn_spec(agent_hyperparameters={"agent_params": agent_params_data})
        converter_agent_only = GnnToPyMdpConverter(spec_agent_only)
        agent_p, policy_p, qs_init_p = converter_agent_only.extract_agent_hyperparameters()
        self.assertEqual(agent_p, agent_params_data)
        self.assertIsNone(policy_p)
        self.assertIsNone(qs_init_p)

        # Scenario 3: All params specified
        full_hyperparams = {
            "agent_params": {"planning_horizon": 5},
            "policy_params": {"initial_action_selection": "deterministic", "policy_len": 5},
            "qs_initial_params": {"method": "uniform_random", "factor_idx": 0}
        }
        spec_full_params = create_basic_gnn_spec(agent_hyperparameters=full_hyperparams)
        converter_full_params = GnnToPyMdpConverter(spec_full_params)
        agent_p, policy_p, qs_init_p = converter_full_params.extract_agent_hyperparameters()
        self.assertEqual(agent_p, full_hyperparams["agent_params"])
        self.assertEqual(policy_p, full_hyperparams["policy_params"])
        self.assertEqual(qs_init_p, full_hyperparams["qs_initial_params"])

        # Scenario 4: Stringified parameters (if converter supports this for hyperparameters)
        # Current converter seems to expect dicts directly for hyperparams.
        # If string parsing was added here, tests would go here.
        # For now, assume direct dicts.

        # Scenario 5: Empty sub-dictionaries
        empty_sub_hyperparams = {
            "agent_params": {},
            "policy_params": {},
            "qs_initial_params": {}
        }
        spec_empty_sub = create_basic_gnn_spec(agent_hyperparameters=empty_sub_hyperparams)
        converter_empty_sub = GnnToPyMdpConverter(spec_empty_sub)
        agent_p, policy_p, qs_init_p = converter_empty_sub.extract_agent_hyperparameters()
        self.assertEqual(agent_p, {})
        self.assertEqual(policy_p, {})
        self.assertEqual(qs_init_p, {})


    def test_generate_agent_instantiation_code_basic(self):
        gnn_spec = create_basic_gnn_spec(
            obs_modality_names=["Obs1"], num_obs_modalities=[2],
            hidden_state_factor_names=["State1"], num_hidden_states_factors=[3]
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        # Manually ensure matrices are processed to populate script_parts (or mock them)
        # A, B, C, D must be defined before agent instantiation for the template to work
        converter.script_parts["matrix_definitions"].append(converter.convert_A_matrix())
        converter.script_parts["matrix_definitions"].append(converter.convert_B_matrix())
        converter.script_parts["matrix_definitions"].append(converter.convert_C_vector())
        converter.script_parts["matrix_definitions"].append(converter.convert_D_vector())
        
        agent_code = converter.generate_agent_instantiation_code()
        self.assertIn(f"agent = Agent(", agent_code)
        self.assertIn("A=A,", agent_code)
        self.assertIn("B=B,", agent_code)
        self.assertIn("C=C,", agent_code)
        self.assertIn("D=D,", agent_code)
        self.assertNotIn("agent_params=", agent_code) # No hyperparams in basic spec
        self.assertNotIn("policy_params=", agent_code)
        self.assertNotIn("initial_qs=", agent_code)

    def test_generate_agent_instantiation_code_with_hyperparams(self):
        hyperparams = {
            "agent_params": {"planning_horizon": 1, "use_param_info_gain": False},
            "policy_params": {"initial_action_selection": "random"},
            "qs_initial_params": {"method": "fixed", "values": "np.array([0.1,0.9])"} # qs_initial will be handled by qs_initial arg
        }
        gnn_spec = create_basic_gnn_spec(
            obs_modality_names=["O"], num_obs_modalities=[2],
            hidden_state_factor_names=["S"], num_hidden_states_factors=[2],
            agent_hyperparameters=hyperparams
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        # Process matrices and extract hyperparams
        _ = converter.convert_A_matrix(); _ = converter.convert_B_matrix(); _ = converter.convert_C_vector(); _ = converter.convert_D_vector()
        extracted_agent_params, extracted_policy_params, extracted_qs_initial_params = converter.extract_agent_hyperparameters()
        
        # The qs_initial_params are used to generate a qs_initial numpy array string
        # which is then passed to generate_agent_instantiation_code
        qs_initial_str = None
        if extracted_qs_initial_params and extracted_qs_initial_params.get("method") == "fixed":
            qs_initial_val_str = extracted_qs_initial_params.get("values")
            if qs_initial_val_str:
                qs_initial_str = (
                    f"initial_qs_S = {qs_initial_val_str}\n"
                    f"initial_qs = np.empty(1, dtype=object)\n"
                    f"initial_qs[0] = initial_qs_S"
                )
                converter.script_parts["preamble_vars"].append(qs_initial_str) 
                qs_initial_arg_for_agent = "initial_qs" 
            else:
                qs_initial_arg_for_agent = None

        agent_code = converter.generate_agent_instantiation_code(
            agent_params=extracted_agent_params, 
            policy_params=extracted_policy_params,
            qs_initial=qs_initial_arg_for_agent # Pass the name of the variable holding initial_qs
        )
        
        self.assertIn("agent = Agent(", agent_code)
        self.assertIn("agent_params={'planning_horizon': 1, 'use_param_info_gain': False}", agent_code.replace(" ", ""))
        self.assertIn("policy_params={'initial_action_selection': 'random'}", agent_code.replace(" ", ""))
        if qs_initial_arg_for_agent:
            self.assertIn(f"initial_qs={qs_initial_arg_for_agent}", agent_code.replace(" ", ""))
        
    def test_generate_agent_instantiation_code_with_action_names(self):
        gnn_spec = create_basic_gnn_spec(
            hidden_state_factor_names=["S1"], num_hidden_states_factors=[2],
            num_control_factors=[2], 
            control_action_names_per_factor={0: ["Up", "Down"]} # Factor 0 is S1
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        _ = converter.convert_A_matrix(); _ = converter.convert_B_matrix(); _ = converter.convert_C_vector(); _ = converter.convert_D_vector()

        agent_code = converter.generate_agent_instantiation_code() 

        self.assertIn("agent = Agent(", agent_code)
        self.assertIn("control_names={0:['Up','Down']}", agent_code.replace(" ", "").replace("\n",""))


    def test_generate_example_usage_code(self):
        gnn_spec = create_basic_gnn_spec(
            obs_modality_names=["ObsModality"], num_obs_modalities=[3],
            hidden_state_factor_names=["StateFactor"], num_hidden_states_factors=[2],
            num_control_factors=[2], control_action_names_per_factor={0: ["Action1", "Action2"]}
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        # Ensure necessary attributes are set (as if matrices were converted)
        converter.num_modalities = 1
        converter.num_factors = 1
        # converter.num_obs = [3] # Already set by create_basic_gnn_spec -> __init__
        # converter.num_states = [2] # Already set

        example_code_lines = converter.generate_example_usage_code()
        
        self.assertIsInstance(example_code_lines, list)
        self.assertTrue(len(example_code_lines) > 0)
        
        example_code_str = "\n".join(example_code_lines)
        self.assertIn(f"# --- Example Usage for {converter.model_name} ---", example_code_str)
        self.assertIn("obs = [np.random.randint(3)]", example_code_str) # num_obs[0] = 3
        self.assertIn("agent.infer_states(obs)", example_code_str)
        self.assertIn("agent.infer_policies()", example_code_str)
        self.assertIn("chosen_action = agent.sample_action()", example_code_str)


class TestFullScriptGeneration(unittest.TestCase):
    """Tests for the get_full_python_script method."""

    def test_get_full_python_script_minimal(self):
        gnn_spec = create_basic_gnn_spec(
            model_name="MinimalAgent",
            obs_modality_names=["O"], num_obs_modalities=[2],
            hidden_state_factor_names=["S"], num_hidden_states_factors=[2]
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        script_content = converter.get_full_python_script(include_example_usage=False)

        self.assertIsInstance(script_content, str)
        self.assertIn("import numpy as np", script_content)
        self.assertIn("from pymdp.agent import Agent", script_content)
        self.assertIn("# --- GNN Model: MinimalAgent ---", script_content)
        self.assertIn("A = np.empty(1, dtype=object)", script_content) # Check for A matrix setup
        self.assertIn("B = np.empty(1, dtype=object)", script_content) # Check for B matrix setup
        self.assertIn("agent = Agent(A=A, B=B, C=C, D=D)", script_content)
        self.assertNotIn("# --- Example Usage for MinimalAgent ---", script_content)

    def test_get_full_python_script_with_all_elements(self):
        hyperparams = {
            "agent_params": {"planning_horizon": 2},
            "policy_params": {"initial_action_selection": "boltzmann"}
        }
        gnn_spec = create_basic_gnn_spec(
            model_name="ComplexAgent",
            obs_modality_names=["Visual", "Audio"], num_obs_modalities=[2,3],
            hidden_state_factor_names=["Location", "Context"], num_hidden_states_factors=[4,2],
            num_control_factors=[3,0], control_action_names_per_factor={0:["L","R","S"]},
            A_spec={
                "Visual": "np.random.rand(2,4,2)", 
                "Audio": "np.random.rand(3,4,2)"
            },
            B_spec={
                "Location": "np.random.rand(3,4,4)", # Controlled
                # Context is not controlled, will get identity
            },
            C_spec={"Visual": "np.zeros(2)", "Audio": "np.zeros(3)"},
            D_spec={"Location": "np.ones(4)/4.0", "Context": "np.ones(2)/2.0"},
            E_spec={"policy_prior": "np.ones(10)/10.0"}, # Assuming 10 policies for test
            agent_hyperparameters=hyperparams
        )
        converter = GnnToPyMdpConverter(gnn_spec)
        script_content = converter.get_full_python_script(include_example_usage=True)

        self.assertIn("import numpy as np", script_content)
        self.assertIn("from pymdp.agent import Agent", script_content)
        self.assertIn("# --- GNN Model: ComplexAgent ---", script_content)
        
        # Check for matrix definitions
        self.assertIn("A_Visual = np.random.rand(2,4,2)", script_content)
        self.assertIn("B_Location = np.random.rand(3,4,4)", script_content)
        self.assertIn("B_Context = np.array(", script_content) # Identity for Context
        self.assertIn("C_Visual = np.zeros(2)", script_content)
        self.assertIn("D_Location = np.ones(4)/4.0", script_content)
        self.assertIn("E_policy_prior = np.ones(10)/10.0", script_content)
        
        # Check for agent instantiation with params
        self.assertIn("agent = Agent(", script_content)
        self.assertIn("agent_params={'planning_horizon': 2}", script_content.replace(" ", ""))
        self.assertIn("policy_params={'initial_action_selection': 'boltzmann'}", script_content.replace(" ", ""))
        self.assertIn("control_names={0: ['L','R','S']}", script_content.replace(" ", "").replace("\n", "")) # Check for control_names
        self.assertIn("E=E", script_content)


        # Check for example usage
        self.assertIn("# --- Example Usage for ComplexAgent ---", script_content)
        self.assertIn("obs = [np.random.randint(2), np.random.randint(3)]", script_content) # Visual (2), Audio (3)

    def test_get_full_python_script_no_example_usage(self):
        gnn_spec = create_basic_gnn_spec(model_name="NoExample")
        converter = GnnToPyMdpConverter(gnn_spec)
        script_content = converter.get_full_python_script(include_example_usage=False)
        
        self.assertIn("# --- GNN Model: NoExample ---", script_content)
        self.assertNotIn("# --- Example Usage for NoExample ---", script_content)


class TestHelperMethodsIndirectly(unittest.TestCase):
    """ Focus on testing _parse_string_to_literal and _sanitize_model_name if not covered enough by init tests."""
    
    def test_parse_string_to_literal_valid_inputs(self):
        converter = GnnToPyMdpConverter(create_basic_gnn_spec()) # Dummy converter for method access
        test_cases = [
            ("[1, 2, 3]", [1, 2, 3]),
            ("{'a': 1, 'b': 2}", {'a': 1, 'b': 2}),
            ("'hello'", "hello"),
            ("123", 123),
            ("123.45", 123.45),
            ("True", True),
            ("None", None),
            ("(1, 'a')", (1, 'a')),
            # Already parsed types
            ([1,2], [1,2]),
            (None, None),
            (np.array([1]), np.array([1])) # Should return as is
        ]
        for input_str, expected_output in test_cases:
            with self.subTest(input_str=input_str):
                if isinstance(expected_output, np.ndarray):
                     np.testing.assert_array_equal(converter._parse_string_to_literal(input_str, "test"), expected_output)
                else:
                    self.assertEqual(converter._parse_string_to_literal(input_str, "test"), expected_output)

    def test_parse_string_to_literal_invalid_inputs(self):
        converter = GnnToPyMdpConverter(create_basic_gnn_spec())
        invalid_cases = [
            "np.array([1, 2])", # Not a literal
            "some_function()",   # Not a literal
            "[1, 2",             # Syntax error
            "{'a': 1",           # Syntax error
            "1.2.3",             # Not valid Python
            "\"unclosed string",
            "",                  # Empty string
            "  "                 # Whitespace only
        ]
        for invalid_input in invalid_cases:
            with self.subTest(invalid_input=invalid_input):
                result = converter._parse_string_to_literal(invalid_input, "test_invalid")
                self.assertIsNone(result, f"Expected None for invalid input: {invalid_input}")
                self.assertTrue(any("ERROR" in log or "WARNING" in log for log in converter.conversion_log), f"No error/warning logged for {invalid_input}")
                converter.conversion_log.clear() # Clear log for next subtest


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 