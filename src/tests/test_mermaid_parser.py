"""
Unit tests for mermaid_parser module

Tests Mermaid to GNN conversion functions:
- Metadata extraction
- Node parsing
- Edge parsing
- Variable merging
- Connection merging
"""

import pytest
from gui.oxdraw.mermaid_parser import (
    mermaid_to_gnn,
    extract_gnn_metadata,
    _extract_nodes,
    _extract_edges,
    _infer_dimensions_from_label,
    _infer_type_from_label,
    _merge_variables,
    _merge_connections,
    _reconstruct_ontology_mappings,
    _gnn_model_to_markdown
)


# Sample Mermaid content for testing
SAMPLE_MERMAID = """flowchart TD
    %% GNN Model: Test Model
    %% GNN Version: 1.0
    %% GNN_METADATA_START
    %% {"model_name":"Test Model","version":"1.0","variables":{"A":{"dimensions":[3,3],"data_type":"float","ontology_mapping":"LikelihoodMatrix","description":"Likelihood matrix"},"s":{"dimensions":[3,1],"data_type":"float","ontology_mapping":"HiddenState","description":"Hidden state"}},"connections":[{"source":"D","target":"s","symbol":">","connection_type":"generative"}],"parameters":{"num_states":3},"ontology_mappings":{"A":"LikelihoodMatrix","s":"HiddenState"}}
    %% GNN_METADATA_END
    
    A[A<br/>3x3<br/>float]
    s([s<br/>3x1<br/>float])
    
    D ==> s
    s -.-> A
    
    classDef matrixStyle fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    class A matrixStyle
"""


class TestMetadataExtraction:
    """Test metadata extraction from Mermaid comments."""
    
    def test_extract_metadata_multiline(self):
        """Test extraction of multi-line metadata format."""
        metadata = extract_gnn_metadata(SAMPLE_MERMAID)
        
        assert isinstance(metadata, dict)
        assert metadata["model_name"] == "Test Model"
        assert metadata["version"] == "1.0"
        assert "variables" in metadata
        assert "connections" in metadata
    

    
    def test_extract_metadata_not_found(self):
        """Test handling when no metadata present."""
        no_metadata = """flowchart TD
        A[A]
        B[B]
        A --> B
        """
        
        metadata = extract_gnn_metadata(no_metadata)
        
        assert isinstance(metadata, dict)
        assert len(metadata) == 0


class TestNodeExtraction:
    """Test node extraction from Mermaid diagrams."""
    
    def test_extract_rectangle_nodes(self):
        """Test extraction of rectangle nodes."""
        mermaid = "A[Label]"
        nodes = _extract_nodes(mermaid)
        
        assert "A" in nodes
        assert nodes["A"]["shape"] == "rectangle"
        assert nodes["A"]["label"] == "Label"
    
    def test_extract_rounded_nodes(self):
        """Test extraction of rounded nodes."""
        mermaid = "C(Label)"
        nodes = _extract_nodes(mermaid)
        
        assert "C" in nodes
        assert nodes["C"]["shape"] == "rounded"
    
    def test_extract_stadium_nodes(self):
        """Test extraction of stadium nodes."""
        mermaid = "s([Label])"
        nodes = _extract_nodes(mermaid)
        
        assert "s" in nodes
        assert nodes["s"]["shape"] == "stadium"
    
    def test_extract_circle_nodes(self):
        """Test extraction of circle nodes."""
        mermaid = "o((Label))"
        nodes = _extract_nodes(mermaid)
        
        assert "o" in nodes
        assert nodes["o"]["shape"] == "circle"
    
    def test_extract_hexagon_nodes(self):
        """Test extraction of hexagon nodes."""
        mermaid = "u{{Label}}"
        nodes = _extract_nodes(mermaid)
        
        assert "u" in nodes
        assert nodes["u"]["shape"] == "hexagon"
    
    def test_extract_diamond_nodes(self):
        """Test extraction of diamond nodes."""
        mermaid = "π{Label}"
        nodes = _extract_nodes(mermaid)
        
        assert "π" in nodes
        assert nodes["π"]["shape"] == "diamond"
    

    
    def test_extract_nodes_with_multipart_labels(self):
        """Test nodes with complex labels."""
        mermaid = "A[A<br/>3x3<br/>float]"
        nodes = _extract_nodes(mermaid)
        
        assert "A" in nodes
        assert "3x3" in nodes["A"]["label"]
        assert "float" in nodes["A"]["label"]


class TestEdgeExtraction:
    """Test edge extraction from Mermaid diagrams."""
    
    def test_extract_thick_arrow(self):
        """Test extraction of thick arrows (generative)."""
        mermaid = "D ==> s"
        edges = _extract_edges(mermaid)
        
        assert len(edges) == 1
        assert edges[0]["source"] == "D"
        assert edges[0]["target"] == "s"
        assert edges[0]["symbol"] == ">"
    
    def test_extract_dashed_arrow(self):
        """Test extraction of dashed arrows (inference)."""
        mermaid = "s -.-> A"
        edges = _extract_edges(mermaid)
        
        assert len(edges) == 1
        assert edges[0]["symbol"] == "-"
    
    def test_extract_dotted_arrow(self):
        """Test extraction of dotted arrows (modulation)."""
        mermaid = "γ -..-> F"
        edges = _extract_edges(mermaid)
        
        assert len(edges) == 1
        assert edges[0]["symbol"] == "*"
    
    def test_extract_normal_arrow(self):
        """Test extraction of normal arrows (coupling)."""
        mermaid = "x --> y"
        edges = _extract_edges(mermaid)
        
        assert len(edges) == 1
        assert edges[0]["symbol"] == "~"
    
    def test_extract_edge_with_label(self):
        """Test extraction of labeled edges."""
        mermaid = "A ==>|inference| B"
        edges = _extract_edges(mermaid)
        
        assert len(edges) == 1
        assert edges[0]["description"] == "inference"


class TestLabelInference:
    """Test inference of dimensions and types from labels."""
    
    def test_infer_dimensions_2d(self):
        """Test 2D dimension inference."""
        label_parts = ["A", "3x3", "float"]
        dims = _infer_dimensions_from_label(label_parts)
        
        assert dims == [3, 3]
    
    def test_infer_dimensions_3d(self):
        """Test 3D dimension inference."""
        label_parts = ["B", "3x3x3", "float"]
        dims = _infer_dimensions_from_label(label_parts)
        
        assert dims == [3, 3, 3]
    
    def test_infer_dimensions_1d(self):
        """Test 1D dimension inference."""
        label_parts = ["C", "3", "float"]
        dims = _infer_dimensions_from_label(label_parts)
        
        assert dims == [3]
    
    def test_infer_type_float(self):
        """Test float type inference."""
        label_parts = ["A", "3x3", "float"]
        dtype = _infer_type_from_label(label_parts)
        
        assert dtype == "float"
    
    def test_infer_type_int(self):
        """Test int type inference."""
        label_parts = ["o", "3x1", "int"]
        dtype = _infer_type_from_label(label_parts)
        
        assert dtype == "int"
    
    def test_infer_type_default(self):
        """Test default type inference."""
        label_parts = ["x", "3"]
        dtype = _infer_type_from_label(label_parts)
        
        assert dtype == "float"


class TestVariableMerging:
    """Test variable merging logic."""
    
    def test_merge_preserves_metadata(self):
        """Test merging preserves metadata."""
        metadata_vars = {
            "A": {
                "dimensions": [3, 3],
                "data_type": "float",
                "ontology_mapping": "LikelihoodMatrix"
            }
        }
        
        visual_nodes = {
            "A": {
                "shape": "rectangle",
                "label": "A<br/>3x3<br/>float",
                "label_parts": ["A", "3x3", "float"],
                "inferred_dimensions": [3, 3],
                "inferred_type": "float"
            }
        }
        
        merged = _merge_variables(metadata_vars, visual_nodes)
        
        assert "A" in merged
        assert merged["A"]["ontology_mapping"] == "LikelihoodMatrix"
        assert merged["A"]["dimensions"] == [3, 3]
    
    def test_merge_adds_new_variables(self):
        """Test merging adds new variables from visual structure."""
        metadata_vars = {
            "A": {"dimensions": [3, 3]}
        }
        
        visual_nodes = {
            "A": {"label_parts": ["A"], "inferred_dimensions": [3, 3], "inferred_type": "float"},
            "B": {"label_parts": ["B"], "inferred_dimensions": [3], "inferred_type": "float"}
        }
        
        merged = _merge_variables(metadata_vars, visual_nodes)
        
        assert "A" in merged
        assert "B" in merged


class TestConnectionMerging:
    """Test connection merging logic."""
    
    def test_merge_connections_visual_precedence(self):
        """Test visual structure takes precedence."""
        metadata_conns = [
            {"source": "A", "target": "B", "symbol": ">", "connection_type": "generative"}
        ]
        
        visual_edges = [
            {"source": "A", "target": "B", "symbol": "-", "description": ""}
        ]
        
        merged = _merge_connections(metadata_conns, visual_edges)
        
        assert len(merged) == 1
        # Visual symbol should override
        assert merged[0]["symbol"] == "-"
        # But metadata connection_type preserved
        assert merged[0]["connection_type"] == "generative"
    
    def test_merge_adds_new_connections(self):
        """Test new visual connections are added."""
        metadata_conns = [
            {"source": "A", "target": "B", "symbol": ">"}
        ]
        
        visual_edges = [
            {"source": "A", "target": "B", "symbol": ">", "description": ""},
            {"source": "B", "target": "C", "symbol": "-", "description": ""}
        ]
        
        merged = _merge_connections(metadata_conns, visual_edges)
        
        assert len(merged) == 2


class TestOntologyReconstruction:
    """Test ontology mapping reconstruction."""
    
    def test_reconstruct_ontology_mappings(self):
        """Test ontology mapping reconstruction."""
        variables = {
            "A": {"ontology_mapping": "LikelihoodMatrix"},
            "s": {"ontology_mapping": "HiddenState"}
        }
        
        ontology_map = {}
        
        mappings = _reconstruct_ontology_mappings(variables, ontology_map)
        
        assert len(mappings) == 2
        
        # Check mappings present
        var_terms = {m["variable"]: m["ontology_term"] for m in mappings}
        assert var_terms["A"] == "LikelihoodMatrix"
        assert var_terms["s"] == "HiddenState"


class TestGNNMarkdownGeneration:
    """Test GNN markdown generation from model."""
    
    def test_markdown_generation(self):
        """Test basic markdown generation."""
        gnn_model = {
            "model_name": "Test Model",
            "version": "1.0",
            "variables": {
                "A": {"dimensions": [3, 3], "data_type": "float", "description": "Matrix"},
                "s": {"dimensions": [3, 1], "data_type": "float", "description": "State"}
            },
            "connections": [
                {"source": "s", "target": "A", "symbol": "-"}
            ],
            "ontology_mappings": [
                {"variable": "A", "ontology_term": "LikelihoodMatrix"}
            ],
            "parameters": {"num_states": 3}
        }
        
        markdown = _gnn_model_to_markdown(gnn_model)
        
        assert isinstance(markdown, str)
        assert "Test Model" in markdown
        assert "## ModelName" in markdown
        assert "## StateSpaceBlock" in markdown
        assert "## Connections" in markdown
        assert "## ActInfOntologyAnnotation" in markdown
        assert "A[3,3,type=float]" in markdown
        assert "s-A" in markdown


class TestFullParsing:
    """Test complete Mermaid to GNN parsing."""
    
    def test_full_parsing(self):
        """Test complete parsing pipeline."""
        gnn_model = mermaid_to_gnn(SAMPLE_MERMAID, validate_ontology=False)
        
        assert isinstance(gnn_model, dict)
        assert "model_name" in gnn_model
        assert gnn_model["model_name"] == "Test Model"
        
        assert "variables" in gnn_model
        assert "A" in gnn_model["variables"]
        assert "s" in gnn_model["variables"]
        
        assert "connections" in gnn_model
        assert len(gnn_model["connections"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

