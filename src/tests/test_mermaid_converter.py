"""
Unit tests for mermaid_converter module

Tests GNN to Mermaid conversion functions:
- Node shape inference
- Edge style mapping
- Metadata generation
- Styling generation
"""

import pytest
from oxdraw.mermaid_converter import (
    gnn_to_mermaid,
    _generate_node_definition,
    _generate_edge_definition,
    _generate_node_styles,
    _classify_variable,
    generate_mermaid_metadata
)
from oxdraw.utils import infer_node_shape, infer_edge_style


class TestNodeShapeInference:
    """Test node shape inference from GNN variables."""
    
    def test_matrix_shape(self):
        """Test matrix variables use rectangles."""
        var_data = {
            "dimensions": [3, 3],
            "data_type": "float",
            "ontology_mapping": "LikelihoodMatrix"
        }
        
        open_b, close_b = infer_node_shape("A", var_data)
        assert open_b == "["
        assert close_b == "]"
    
    def test_state_shape(self):
        """Test state variables use stadium shape."""
        var_data = {
            "dimensions": [3, 1],
            "data_type": "float",
            "ontology_mapping": "HiddenState"
        }
        
        open_b, close_b = infer_node_shape("s", var_data)
        assert open_b == "(["
        assert close_b == "])"
    
    def test_observation_shape(self):
        """Test observation variables use circles."""
        var_data = {
            "dimensions": [3, 1],
            "data_type": "int",
            "ontology_mapping": "Observation"
        }
        
        open_b, close_b = infer_node_shape("o", var_data)
        assert open_b == "(("
        assert close_b == "))"
    
    def test_action_shape(self):
        """Test action variables use hexagons."""
        var_data = {
            "dimensions": [1],
            "data_type": "int",
            "ontology_mapping": "Action"
        }
        
        open_b, close_b = infer_node_shape("u", var_data)
        assert open_b == "{{"
        assert close_b == "}}"
    
    def test_policy_shape(self):
        """Test policy variables use diamonds."""
        var_data = {
            "dimensions": [3],
            "data_type": "float",
            "ontology_mapping": "PolicyVector"
        }
        
        open_b, close_b = infer_node_shape("π", var_data)
        assert open_b == "{"
        assert close_b == "}"
    
    def test_free_energy_shape(self):
        """Test free energy variables use trapezoids."""
        var_data = {
            "dimensions": [],
            "data_type": "float",
            "ontology_mapping": "VariationalFreeEnergy"
        }
        
        open_b, close_b = infer_node_shape("F", var_data)
        assert open_b == "[/"
        assert close_b == "\\]"


class TestEdgeStyleMapping:
    """Test edge style mapping from GNN symbols."""
    
    def test_generative_style(self):
        """Test generative connections use thick arrows."""
        style = infer_edge_style(">")
        assert style == "==>"
    
    def test_inference_style(self):
        """Test inference connections use dashed arrows."""
        style = infer_edge_style("-")
        assert style == "-.->"
    
    def test_modulation_style(self):
        """Test modulation connections use dotted arrows."""
        style = infer_edge_style("*")
        assert style == "-..->"    
    def test_coupling_style(self):
        """Test coupling connections use normal arrows."""
        style = infer_edge_style("~")
        assert style == "-->"
    
    def test_default_style(self):
        """Test unknown symbols default to normal arrows."""
        style = infer_edge_style("?")
        assert style == "-->"


class TestNodeDefinitionGeneration:
    """Test Mermaid node definition generation."""
    
    def test_node_with_dimensions(self):
        """Test node definition includes dimensions."""
        var_data = {
            "dimensions": [3, 3],
            "data_type": "float",
            "ontology_mapping": "LikelihoodMatrix"
        }
        
        node_def = _generate_node_definition("A", var_data)
        
        assert "A[" in node_def
        assert "3x3" in node_def
        assert "float" in node_def
    
    def test_node_without_dimensions(self):
        """Test node definition without dimensions."""
        var_data = {
            "dimensions": [],
            "data_type": "float",
            "ontology_mapping": "VariationalFreeEnergy"
        }
        
        node_def = _generate_node_definition("F", var_data)
        
        assert "F" in node_def
        assert "float" in node_def


class TestEdgeDefinitionGeneration:
    """Test Mermaid edge definition generation."""
    
    def test_edge_without_label(self):
        """Test edge definition without label."""
        conn = {
            "source": "D",
            "target": "s",
            "symbol": ">",
            "description": ""
        }
        
        edge_def = _generate_edge_definition(conn)
        
        assert "D" in edge_def
        assert "==>" in edge_def
        assert "s" in edge_def
    
    def test_edge_with_label(self):
        """Test edge definition with label."""
        conn = {
            "source": "s",
            "target": "A",
            "symbol": "-",
            "description": "inference"
        }
        
        edge_def = _generate_edge_definition(conn)
        
        assert "s" in edge_def
        assert "-.->" in edge_def
        assert "inference" in edge_def
        assert "A" in edge_def


class TestVariableClassification:
    """Test variable classification for styling."""
    
    def test_classify_matrix(self):
        """Test matrix classification."""
        var_data = {"dimensions": [3, 3], "ontology_mapping": ""}
        var_type = _classify_variable("A", var_data)
        assert var_type == "matrix"
    
    def test_classify_state(self):
        """Test state classification."""
        var_data = {"dimensions": [3, 1], "ontology_mapping": "HiddenState"}
        var_type = _classify_variable("s", var_data)
        assert var_type == "state"
    
    def test_classify_observation(self):
        """Test observation classification."""
        var_data = {"dimensions": [3], "ontology_mapping": "Observation"}
        var_type = _classify_variable("o", var_data)
        assert var_type == "observation"
    
    def test_classify_action(self):
        """Test action classification."""
        var_data = {"dimensions": [1], "ontology_mapping": "Action"}
        var_type = _classify_variable("u", var_data)
        assert var_type == "action"
    
    def test_classify_policy(self):
        """Test policy classification."""
        var_data = {"dimensions": [3], "ontology_mapping": "PolicyVector"}
        var_type = _classify_variable("π", var_data)
        assert var_type == "policy"
    
    def test_classify_free_energy(self):
        """Test free energy classification."""
        var_data = {"dimensions": [], "ontology_mapping": "VariationalFreeEnergy"}
        var_type = _classify_variable("F", var_data)
        assert var_type == "free_energy"
    
    def test_classify_vector_default(self):
        """Test default vector classification."""
        var_data = {"dimensions": [3], "ontology_mapping": ""}
        var_type = _classify_variable("v", var_data)
        assert var_type == "vector"


class TestStyleGeneration:
    """Test style directive generation."""
    
    def test_generate_styles(self):
        """Test style generation for various variable types."""
        variables = {
            "A": {"dimensions": [3, 3], "ontology_mapping": "LikelihoodMatrix"},
            "s": {"dimensions": [3, 1], "ontology_mapping": "HiddenState"},
            "o": {"dimensions": [3, 1], "ontology_mapping": "Observation"},
            "u": {"dimensions": [1], "ontology_mapping": "Action"}
        }
        
        styles = _generate_node_styles(variables)
        
        assert isinstance(styles, list)
        assert len(styles) > 0
        
        # Check for class definitions
        style_str = " ".join(styles)
        assert "classDef matrixStyle" in style_str
        assert "classDef stateStyle" in style_str
        assert "classDef observationStyle" in style_str
        assert "classDef actionStyle" in style_str


class TestMetadataGeneration:
    """Test metadata dictionary generation."""
    
    def test_metadata_includes_all_sections(self):
        """Test metadata includes all required sections."""
        gnn_model = {
            "model_name": "Test Model",
            "version": "1.0",
            "variables": {
                "A": {"dimensions": [3, 3], "data_type": "float", "ontology_mapping": "LikelihoodMatrix"}
            },
            "connections": [
                {"source": "D", "target": "s", "symbol": ">", "connection_type": "generative"}
            ],
            "parameters": {"num_states": 3}
        }
        
        metadata = generate_mermaid_metadata(gnn_model)
        
        assert "model_name" in metadata
        assert "version" in metadata
        assert "variables" in metadata
        assert "connections" in metadata
        assert "parameters" in metadata
        assert "ontology_mappings" in metadata
    
    def test_metadata_serialization(self):
        """Test metadata can be JSON serialized."""
        gnn_model = {
            "model_name": "Test Model",
            "version": "1.0",
            "variables": {"A": {"dimensions": [3, 3]}},
            "connections": [],
            "parameters": {}
        }
        
        metadata = generate_mermaid_metadata(gnn_model)
        
        # Should be JSON serializable
        import json
        json_str = json.dumps(metadata)
        assert isinstance(json_str, str)
        
        # Should be deserializable
        recovered = json.loads(json_str)
        assert recovered["model_name"] == "Test Model"


class TestFullConversion:
    """Test complete GNN to Mermaid conversion."""
    
    def test_complete_conversion(self):
        """Test full conversion with all features."""
        gnn_model = {
            "model_name": "Complete Test Model",
            "version": "1.0",
            "variables": {
                "A": {
                    "dimensions": [3, 3],
                    "data_type": "float",
                    "ontology_mapping": "LikelihoodMatrix",
                    "description": "Likelihood"
                },
                "s": {
                    "dimensions": [3, 1],
                    "data_type": "float",
                    "ontology_mapping": "HiddenState",
                    "description": "State"
                }
            },
            "connections": [
                {"source": "s", "target": "A", "symbol": "-", "connection_type": "inference", "description": ""}
            ],
            "parameters": {"num_states": 3},
            "ontology_mappings": [
                {"variable": "A", "ontology_term": "LikelihoodMatrix"},
                {"variable": "s", "ontology_term": "HiddenState"}
            ]
        }
        
        mermaid_content = gnn_to_mermaid(gnn_model, include_metadata=True, include_styling=True)
        
        # Check structure
        assert "flowchart TD" in mermaid_content
        assert "Complete Test Model" in mermaid_content
        
        # Check nodes
        assert "A[" in mermaid_content
        assert "s([" in mermaid_content  # Stadium shape
        
        # Check edges
        assert "s -.->" in mermaid_content
        
        # Check metadata
        assert "GNN_METADATA" in mermaid_content
        
        # Check styling
        assert "classDef" in mermaid_content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

