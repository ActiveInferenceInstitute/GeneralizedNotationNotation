"""
Integration tests for oxdraw module

Tests end-to-end workflows:
- GNN file discovery and conversion
- Mermaid generation with metadata
- oxdraw integration (headless mode)
- Round-trip conversion validation
"""

import pytest
from pathlib import Path
import json
import tempfile
import shutil

from oxdraw.processor import (
    process_oxdraw,
    check_oxdraw_installed,
    get_module_info
)
from oxdraw.mermaid_converter import (
    convert_gnn_file_to_mermaid,
    gnn_to_mermaid,
    generate_mermaid_metadata
)
from oxdraw.mermaid_parser import (
    convert_mermaid_file_to_gnn,
    mermaid_to_gnn,
    extract_gnn_metadata
)


# Sample GNN content for testing
SAMPLE_GNN_CONTENT = """# GNN Example: Simple Active Inference Agent
# GNN Version: 1.0

## ModelName
Simple Active Inference Agent

## StateSpaceBlock
A[3,3,type=float]   # Likelihood matrix
B[3,3,3,type=float] # Transition matrix
C[3,type=float]     # Preference vector
D[3,type=float]     # Prior vector
s[3,1,type=float]   # Hidden state
o[3,1,type=int]     # Observation
u[1,type=int]       # Action

## Connections
D>s
s-A
A-o
s-B
u>B

## ActInfOntologyAnnotation
A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates
s=HiddenState
o=Observation
u=Action

## ModelParameters
num_states: 3
num_obs: 3
num_actions: 3
"""


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_gnn_file(temp_dir):
    """Create sample GNN file for testing."""
    gnn_file = temp_dir / "test_model.md"
    gnn_file.write_text(SAMPLE_GNN_CONTENT)
    return gnn_file


@pytest.fixture
def sample_gnn_model():
    """Create sample parsed GNN model."""
    return {
        "model_name": "Simple Active Inference Agent",
        "version": "1.0",
        "variables": {
            "A": {
                "dimensions": [3, 3],
                "data_type": "float",
                "ontology_mapping": "LikelihoodMatrix",
                "description": "Likelihood matrix"
            },
            "B": {
                "dimensions": [3, 3, 3],
                "data_type": "float",
                "ontology_mapping": "TransitionMatrix",
                "description": "Transition matrix"
            },
            "C": {
                "dimensions": [3],
                "data_type": "float",
                "ontology_mapping": "LogPreferenceVector",
                "description": "Preference vector"
            },
            "D": {
                "dimensions": [3],
                "data_type": "float",
                "ontology_mapping": "PriorOverHiddenStates",
                "description": "Prior vector"
            },
            "s": {
                "dimensions": [3, 1],
                "data_type": "float",
                "ontology_mapping": "HiddenState",
                "description": "Hidden state"
            },
            "o": {
                "dimensions": [3, 1],
                "data_type": "int",
                "ontology_mapping": "Observation",
                "description": "Observation"
            },
            "u": {
                "dimensions": [1],
                "data_type": "int",
                "ontology_mapping": "Action",
                "description": "Action"
            }
        },
        "connections": [
            {"source": "D", "target": "s", "symbol": ">", "connection_type": "generative"},
            {"source": "s", "target": "A", "symbol": "-", "connection_type": "inference"},
            {"source": "A", "target": "o", "symbol": "-", "connection_type": "inference"},
            {"source": "s", "target": "B", "symbol": "-", "connection_type": "inference"},
            {"source": "u", "target": "B", "symbol": ">", "connection_type": "generative"}
        ],
        "parameters": {
            "num_states": 3,
            "num_obs": 3,
            "num_actions": 3
        },
        "ontology_mappings": [
            {"variable": "A", "ontology_term": "LikelihoodMatrix"},
            {"variable": "B", "ontology_term": "TransitionMatrix"},
            {"variable": "C", "ontology_term": "LogPreferenceVector"},
            {"variable": "D", "ontology_term": "PriorOverHiddenStates"},
            {"variable": "s", "ontology_term": "HiddenState"},
            {"variable": "o", "ontology_term": "Observation"},
            {"variable": "u", "ontology_term": "Action"}
        ]
    }


class TestModuleInfo:
    """Test module information and capabilities."""
    
    def test_get_module_info(self):
        """Test module info retrieval."""
        info = get_module_info()
        
        assert isinstance(info, dict)
        assert "name" in info
        assert info["name"] == "oxdraw"
        assert "version" in info
        assert "capabilities" in info
        assert isinstance(info["capabilities"], list)
        assert len(info["capabilities"]) > 0
    
    def test_check_oxdraw_installed(self):
        """Test oxdraw CLI availability check."""
        result = check_oxdraw_installed()
        assert isinstance(result, bool)
        # Test should work regardless of whether oxdraw is installed


class TestGNNToMermaidConversion:
    """Test GNN to Mermaid conversion functionality."""
    
    def test_gnn_to_mermaid_basic(self, sample_gnn_model):
        """Test basic GNN to Mermaid conversion."""
        mermaid_content = gnn_to_mermaid(sample_gnn_model, include_metadata=True)
        
        assert isinstance(mermaid_content, str)
        assert "flowchart TD" in mermaid_content
        assert "Simple Active Inference Agent" in mermaid_content
        
        # Check for variables
        for var_name in ["A", "B", "C", "D", "s", "o", "u"]:
            assert var_name in mermaid_content
        
        # Check for connections
        assert "D ==>" in mermaid_content  # Generative
        assert "s -.->" in mermaid_content  # Inference
    
    def test_gnn_to_mermaid_with_metadata(self, sample_gnn_model):
        """Test metadata embedding in Mermaid."""
        mermaid_content = gnn_to_mermaid(sample_gnn_model, include_metadata=True)
        
        assert "GNN_METADATA_START" in mermaid_content
        assert "GNN_METADATA_END" in mermaid_content
        
        # Extract and parse metadata
        metadata = extract_gnn_metadata(mermaid_content)
        assert isinstance(metadata, dict)
        assert metadata["model_name"] == "Simple Active Inference Agent"
        assert "variables" in metadata
        assert "connections" in metadata
    
    def test_gnn_to_mermaid_without_metadata(self, sample_gnn_model):
        """Test conversion without metadata."""
        mermaid_content = gnn_to_mermaid(sample_gnn_model, include_metadata=False)
        
        assert "flowchart TD" in mermaid_content
        assert "GNN_METADATA" not in mermaid_content
    
    def test_convert_gnn_file_to_mermaid(self, sample_gnn_file, temp_dir):
        """Test file-based GNN to Mermaid conversion."""
        output_file = temp_dir / "test_model.mmd"
        
        mermaid_content = convert_gnn_file_to_mermaid(
            sample_gnn_file,
            output_file
        )
        
        assert output_file.exists()
        assert isinstance(mermaid_content, str)
        assert len(mermaid_content) > 0
        
        # Verify file content
        file_content = output_file.read_text()
        assert file_content == mermaid_content


class TestMermaidToGNNConversion:
    """Test Mermaid to GNN conversion functionality."""
    
    def test_mermaid_to_gnn_basic(self, sample_gnn_model):
        """Test basic Mermaid to GNN conversion."""
        # First convert to Mermaid
        mermaid_content = gnn_to_mermaid(sample_gnn_model, include_metadata=True)
        
        # Then convert back
        gnn_model = mermaid_to_gnn(mermaid_content, validate_ontology=False)
        
        assert isinstance(gnn_model, dict)
        assert "model_name" in gnn_model
        assert "variables" in gnn_model
        assert "connections" in gnn_model
        
        # Check variables preserved
        assert len(gnn_model["variables"]) == len(sample_gnn_model["variables"])
        
        # Check connections preserved
        assert len(gnn_model["connections"]) == len(sample_gnn_model["connections"])
    
    def test_extract_gnn_metadata(self, sample_gnn_model):
        """Test metadata extraction from Mermaid."""
        mermaid_content = gnn_to_mermaid(sample_gnn_model, include_metadata=True)
        metadata = extract_gnn_metadata(mermaid_content)
        
        assert isinstance(metadata, dict)
        assert metadata["model_name"] == sample_gnn_model["model_name"]
        assert "variables" in metadata
        assert len(metadata["variables"]) == len(sample_gnn_model["variables"])
    
    def test_convert_mermaid_file_to_gnn(self, sample_gnn_file, temp_dir):
        """Test file-based Mermaid to GNN conversion."""
        # First create Mermaid file
        mermaid_file = temp_dir / "test_model.mmd"
        convert_gnn_file_to_mermaid(sample_gnn_file, mermaid_file)
        
        # Then convert back
        output_gnn = temp_dir / "test_model_from_mermaid.md"
        gnn_model = convert_mermaid_file_to_gnn(mermaid_file, output_gnn)
        
        assert output_gnn.exists()
        assert isinstance(gnn_model, dict)
        assert "variables" in gnn_model
        assert len(gnn_model["variables"]) > 0


class TestRoundTripConversion:
    """Test round-trip conversion: GNN → Mermaid → GNN."""
    
    def test_round_trip_preserves_structure(self, sample_gnn_model):
        """Test that round-trip conversion preserves model structure."""
        # GNN → Mermaid
        mermaid_content = gnn_to_mermaid(sample_gnn_model, include_metadata=True)
        
        # Mermaid → GNN
        recovered_model = mermaid_to_gnn(mermaid_content, validate_ontology=False)
        
        # Compare structures
        assert len(recovered_model["variables"]) == len(sample_gnn_model["variables"])
        assert len(recovered_model["connections"]) == len(sample_gnn_model["connections"])
        
        # Check variable names preserved
        original_vars = set(sample_gnn_model["variables"].keys())
        recovered_vars = set(recovered_model["variables"].keys())
        assert original_vars == recovered_vars
    
    def test_round_trip_preserves_ontology(self, sample_gnn_model):
        """Test that ontology mappings are preserved."""
        # GNN → Mermaid
        mermaid_content = gnn_to_mermaid(sample_gnn_model, include_metadata=True)
        
        # Mermaid → GNN
        recovered_model = mermaid_to_gnn(mermaid_content, validate_ontology=False)
        
        # Check ontology mappings
        assert "ontology_mappings" in recovered_model
        original_ontology = {
            m["variable"]: m["ontology_term"]
            for m in sample_gnn_model["ontology_mappings"]
        }
        recovered_ontology = {
            m["variable"]: m["ontology_term"]
            for m in recovered_model["ontology_mappings"]
        }
        
        for var_name in original_ontology:
            assert var_name in recovered_ontology
            assert original_ontology[var_name] == recovered_ontology[var_name]


class TestProcessOxdraw:
    """Test main processing function."""
    
    def test_process_oxdraw_headless(self, sample_gnn_file, temp_dir, capsys):
        """Test headless processing mode."""
        import logging
        logger = logging.getLogger(__name__)
        
        output_dir = temp_dir / "output"
        
        success = process_oxdraw(
            target_dir=sample_gnn_file.parent,
            output_dir=output_dir,
            logger=logger,
            mode="headless",
            auto_convert=True,
            validate_on_save=False,
            launch_editor=False
        )
        
        assert success
        assert output_dir.exists()
        
        # Check for results file
        results_file = output_dir / "oxdraw_processing_results.json"
        assert results_file.exists()
        
        # Parse results
        with open(results_file) as f:
            results = json.load(f)
        
        assert "gnn_to_mermaid_conversions" in results
        assert len(results["gnn_to_mermaid_conversions"]) > 0
    
    def test_process_oxdraw_no_files(self, temp_dir, capsys):
        """Test processing with no GNN files."""
        import logging
        logger = logging.getLogger(__name__)
        
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        output_dir = temp_dir / "output"
        
        success = process_oxdraw(
            target_dir=empty_dir,
            output_dir=output_dir,
            logger=logger,
            mode="headless"
        )
        
        assert not success  # Should fail with no files


class TestMetadataGeneration:
    """Test metadata generation utilities."""
    
    def test_generate_mermaid_metadata(self, sample_gnn_model):
        """Test metadata generation."""
        metadata = generate_mermaid_metadata(sample_gnn_model)
        
        assert isinstance(metadata, dict)
        assert "model_name" in metadata
        assert "variables" in metadata
        assert "connections" in metadata
        assert "ontology_mappings" in metadata
        
        # Verify variables serialized correctly
        assert len(metadata["variables"]) == len(sample_gnn_model["variables"])
        
        # Verify connections serialized correctly
        assert len(metadata["connections"]) == len(sample_gnn_model["connections"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

