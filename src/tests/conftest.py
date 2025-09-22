#!/usr/bin/env python3
"""Minimal test configuration for GNN pipeline tests."""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

@pytest.fixture
def isolated_temp_dir():
    """Provide isolated temporary directory for tests."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)

@pytest.fixture
def project_root():
    """Provide the project root directory for tests."""
    return Path(__file__).parent.parent.parent

@pytest.fixture
def sample_gnn_files():
    """Provide sample GNN files for testing."""
    # Return the actual sample GNN file from the input directory
    sample_files = {}
    gnn_dir = Path(__file__).parent.parent.parent / "input" / "gnn_files"
    if gnn_dir.exists():
        for file_path in gnn_dir.glob("*.md"):
            if file_path.is_file():
                sample_files[file_path.name] = file_path
    return sample_files

@pytest.fixture
def comprehensive_test_data():
    """Provide comprehensive test data for complex tests."""
    return {
        "test_models": ["model1", "model2", "model3"],
        "test_data": {"key": "value"},
        "test_config": {"setting": "test"},
        "test_files": ["file1.md", "file2.md"],
        "test_parameters": {"param1": "value1", "param2": "value2"}
    }

@pytest.fixture
def test_temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return tmp_path

@pytest.fixture
def test_config():
    """Provide test configuration data."""
    return {
        "target_dir": "test_input",
        "output_dir": "test_output",
        "verbose": True,
        "recursive": True
    }

@pytest.fixture
def test_gnn_content():
    """Provide sample GNN content for testing."""
    return """## ModelName
TestModel

## StateSpaceBlock
s[3,1,type=int]

## Connections
s -> o
"""

@pytest.fixture
def test_ontology_terms():
    """Provide sample ontology terms for testing."""
    return {
        "state_space": "The set of all possible states",
        "observation_space": "The set of all possible observations",
        "action_space": "The set of all possible actions"
    }

@pytest.fixture
def sample_markdown():
    """Provide sample markdown content for testing."""
    return """# Test Model

## StateSpaceBlock
s[3,1,type=int]

## ObservationSpaceBlock
o[2,1,type=float]

## ActionSpaceBlock
a[2,1,type=int]

## Connections
s -> o
a -> s
"""

@pytest.fixture
def sample_scala():
    """Provide sample Scala-like code for testing."""
    return """
object TestModel {
  val stateSpace = StateSpace(3, 1)
  val observationSpace = ObservationSpace(2, 1)
  val actionSpace = ActionSpace(2, 1)

  def policy(state: State): Action = {
    // Simple policy implementation
    Action(0)
  }
}
"""

@pytest.fixture
def sample_json():
    """Provide sample JSON data for testing."""
    return {
        "model_name": "TestModel",
        "state_space": [3, 1],
        "observation_space": [2, 1],
        "action_space": [2, 1],
        "connections": [
            {"from": "s", "to": "o"},
            {"from": "a", "to": "s"}
        ]
    }

@pytest.fixture
def sample_xml():
    """Provide sample XML data for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<model>
  <name>TestModel</name>
  <states>
    <state name="s" dimensions="[3,1]" type="int" />
  </states>
  <observations>
    <observation name="o" dimensions="[2,1]" type="float" />
  </observations>
  <actions>
    <action name="a" dimensions="[2,1]" type="int" />
  </actions>
  <connections>
    <connection from="s" to="o" />
    <connection from="a" to="s" />
  </connections>
</model>"""

# Register pytest markers to avoid warnings
def pytest_configure(config):
    """Register custom pytest markers."""
    markers = {
        "unit": "Unit tests for individual components",
        "integration": "Integration tests for component interactions",
        "performance": "Performance and resource usage tests",
        "slow": "Tests that take significant time to complete",
        "fast": "Quick tests for rapid feedback",
        "safe_to_fail": "Tests safe to run without side effects",
        "destructive": "Tests that may modify system state",
        "external": "Tests requiring external dependencies",
        "core": "Core module tests",
        "utilities": "Utility function tests",
        "environment": "Environment validation tests",
        "render": "Rendering and code generation tests",
        "export": "Export functionality tests",
        "parsers": "Parser and format tests",
        "thin_orchestrator": "Thin orchestrator pattern tests",
        "pipeline_modules": "Pipeline module tests",
        "mcp": "Model Context Protocol tests",
        "audio": "Audio generation and processing tests",
        "visualization": "Visualization and plotting tests",
        "ontology": "Ontology processing tests",
        "execute": "Simulation execution tests",
        "llm": "Large Language Model integration tests",
        "website": "Website generation tests",
        "security": "Security and access control tests",
        "research": "Research and experimental features",
        "ml_integration": "Machine learning integration tests",
        "advanced_visualization": "Advanced visualization tests",
        "comprehensive": "Comprehensive API and integration tests"
    }

    for name, description in markers.items():
        config.addinivalue_line("markers", f"{name}: {description}")

# Test markers dictionary for reference
PYTEST_MARKERS = {
    "unit": "Unit tests for individual components",
    "integration": "Integration tests for component interactions",
    "performance": "Performance and resource usage tests",
    "slow": "Tests that take significant time to complete",
    "fast": "Quick tests for rapid feedback",
    "safe_to_fail": "Tests safe to run without side effects",
    "destructive": "Tests that may modify system state",
    "external": "Tests requiring external dependencies",
    "core": "Core module tests",
    "utilities": "Utility function tests",
    "environment": "Environment validation tests",
    "render": "Rendering and code generation tests",
    "export": "Export functionality tests",
    "parsers": "Parser and format tests",
    "thin_orchestrator": "Thin orchestrator pattern tests",
    "pipeline_modules": "Pipeline module tests",
    "mcp": "Model Context Protocol tests",
    "audio": "Audio generation and processing tests",
    "visualization": "Visualization and plotting tests",
    "ontology": "Ontology processing tests",
    "execute": "Simulation execution tests",
    "llm": "Large Language Model integration tests",
    "website": "Website generation tests",
    "security": "Security and access control tests",
    "research": "Research and experimental features",
    "ml_integration": "Machine learning integration tests",
    "advanced_visualization": "Advanced visualization tests",
    "comprehensive": "Comprehensive API and integration tests"
}