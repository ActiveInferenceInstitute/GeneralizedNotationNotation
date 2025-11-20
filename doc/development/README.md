# Developer Documentation

> **📋 Document Metadata**  
> **Type**: Development Guide | **Audience**: Developers & Contributors | **Complexity**: Intermediate-Advanced  
> **Cross-References**: [Testing Guide](../testing/README.md) | [API Documentation](../api/README.md)

## Overview
This guide provides comprehensive information for developers contributing to the GNN project, including code organization, development workflows, and architecture patterns.

## Quick Start for Developers

### Environment Setup
```bash
# Clone repository
git clone https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation.git
cd GeneralizedNotationNotation

# Run setup (creates venv, installs dependencies)
python src/main.py --only-steps 2 --dev

# Activate virtual environment
source src/.venv/bin/activate  # Linux/Mac
# src/.venv/Scripts/activate    # Windows

# Run tests to verify setup
python src/main.py --only-steps 3
```

### Development Workflow
```bash
# Create feature branch
git checkout -b feature/my-new-feature

# Make changes, add tests
# ...

# Run quality checks
python src/main.py --only-steps 3,4 --strict

# Commit and push
git add .
git commit -m "feat: add new feature"
git push origin feature/my-new-feature
```

## Project Architecture

### Module Organization
```
src/
├── main.py                    # Pipeline orchestrator
├── pipeline/                  # Pipeline configuration
│   ├── config.py             # Step configuration, timeouts
│   └── dependency_validator.py # Dependency checking
├── gnn/                      # Core GNN processing
│   ├── parser.py             # GNN file parsing
│   ├── validator.py          # Syntax validation  
│   └── examples/             # Example GNN files
├── export/                   # Multi-format export
│   ├── format_exporters.py  # Export implementations
│   └── mcp.py               # Export MCP tools
├── visualization/            # Model visualization
│   ├── visualize_gnn.py     # Main visualization logic
│   ├── diagram_generators.py # Specific diagram types
│   └── mcp.py               # Visualization MCP tools
├── render/                   # Code generation
│   ├── pymdp/               # PyMDP code generation
│   ├── rxinfer/             # RxInfer.jl generation
│   └── render.py            # Main rendering interface
├── execute/                  # Simulator execution
│   ├── pymdp_runner.py      # PyMDP execution
│   └── rxinfer_runner.py    # RxInfer.jl execution
├── llm/                     # LLM integration
│   ├── providers/           # LLM provider implementations
│   ├── tasks/               # Specific LLM tasks
│   └── mcp.py              # LLM MCP tools
├── mcp/                     # Model Context Protocol
│   ├── mcp.py              # Core MCP instance
│   ├── server_http.py      # HTTP server
│   ├── server_stdio.py     # STDIO server
│   └── cli.py              # Command-line interface
├── ontology/                # Ontology processing
│   ├── ontology_processor.py # Core processing
│   ├── act_inf_ontology_terms.json # Ontology data
│   └── mcp.py              # Ontology MCP tools
├── discopy_translator_module/ # Category theory
│   ├── translator.py       # GNN to DisCoPy translation
│   └── visualize_jax_output.py # JAX visualization
├── utils/                   # Shared utilities
│   ├── logging_utils.py    # Logging configuration
│   └── file_utils.py       # File operations
├── setup/                   # Environment setup
│   ├── setup.py           # Main setup logic
│   └── utils.py           # Setup utilities
├── tests/                   # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test data
└── site/                   # Site generation
    └── site_generator.py  # HTML generation
```

### Design Patterns

#### Pipeline Step Pattern
Each pipeline step follows this pattern:

```python
#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step N: Description

Purpose description and usage information.
"""

import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)

def process_main_logic(args: argparse.Namespace) -> bool:
    """Main processing logic for this step."""
    # Implementation here
    return success

def main(args: argparse.Namespace) -> int:
    """Entry point called by main.py pipeline."""
    logger.info(f"Starting Step N: Description")
    
    if not process_main_logic(args):
        logger.error("Step N failed")
        return 1
        
    logger.info("Step N completed successfully")
    return 0

if __name__ == "__main__":
    # Standalone execution setup
    parser = argparse.ArgumentParser(description="Step N (Standalone)")
    # Add arguments
    args = parser.parse_args()
    
    # Setup logging for standalone
    from utils.logging_utils import setup_standalone_logging
    setup_standalone_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        logger_name=__name__
    )
    
    sys.exit(main(args))
```

#### MCP Tool Pattern
Each module can provide MCP tools:

```python
# In module_name/mcp.py
def tool_function(param1: str, param2: int = 10) -> dict:
    """
    Tool description for MCP schema generation.
    
    Args:
        param1: Parameter description
        param2: Optional parameter with default
        
    Returns:
        Result dictionary
    """
    # Implementation
    return {"result": result}

def register_tools():
    """Register all tools from this module."""
    from src.mcp import mcp_instance
    
    tools = {
        "module_tool_name": tool_function,
        # Add more tools
    }
    
    for name, func in tools.items():
        mcp_instance.register_tool(name, func)
```

## Development Guidelines

### Code Quality Standards

#### Type Hints
```python
# Required for all function signatures
def process_gnn_file(file_path: Path, strict: bool = False) -> Dict[str, Any]:
    """Process a GNN file and return parsed data."""
    pass

# Use Union types where appropriate
from typing import Union, Optional, List, Dict

def validate_model(model: Union[str, Path, Dict]) -> Optional[List[str]]:
    """Validate a model, return errors if any."""
    pass
```

#### Documentation Standards
```python
def complex_function(param1: str, param2: Optional[int] = None) -> Dict[str, Any]:
    """
    One-line summary of the function.
    
    More detailed description if needed. Explain the purpose,
    key behaviors, and any important considerations.
    
    Args:
        param1: Description of what this parameter does
        param2: Optional parameter, explain default behavior
        
    Returns:
        Dictionary containing:
            - key1: Description of what this contains
            - key2: Description of this field
            
    Raises:
        ValueError: When param1 is invalid
        FileNotFoundError: When required files are missing
        
    Example:
        >>> result = complex_function("test", 42)
        >>> print(result["key1"])
        expected_output
    """
    pass
```

#### Error Handling
```python
# Use specific exceptions
def parse_gnn_file(file_path: Path) -> Dict[str, Any]:
    """Parse GNN file with proper error handling."""
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"GNN file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Processing logic
        return parse_content(content)
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in GNN file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error parsing {file_path}: {e}")
        raise
```

### Testing Standards

#### Test Organization
```
tests/
├── unit/                    # Unit tests
│   ├── test_gnn_parser.py  # Test individual modules
│   ├── test_exporters.py   
│   └── test_visualization.py
├── integration/             # Integration tests  
│   ├── test_pipeline_steps.py # Test step interactions
│   ├── test_export_formats.py # Test full export workflows
│   └── test_mcp_integration.py # Test MCP functionality
├── fixtures/                # Test data
│   ├── valid_gnn_files/    # Valid test models
│   ├── invalid_gnn_files/  # Invalid models for error testing
│   └── expected_outputs/   # Expected results for validation
└── conftest.py             # Pytest configuration
```

#### Test Writing Patterns
```python
import pytest
from pathlib import Path
from src.gnn.parser import parse_gnn_file

class TestGNNParser:
    """Test suite for GNN file parsing."""
    
    @pytest.fixture
    def sample_gnn_file(self, tmp_path):
        """Create a sample GNN file for testing."""
        content = """
        # ModelName
        Test Model
        
        # StateSpaceBlock
        s_f0[3,1,type=float]
        """
        file_path = tmp_path / "test_model.md"
        file_path.write_text(content)
        return file_path
    
    def test_parse_valid_file(self, sample_gnn_file):
        """Test parsing a valid GNN file."""
        result = parse_gnn_file(sample_gnn_file)
        
        assert result["model_name"] == "Test Model"
        assert "s_f0" in result["state_space"]
        assert result["state_space"]["s_f0"]["dimensions"] == [3, 1]
    
    def test_parse_missing_file(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            parse_gnn_file(Path("nonexistent.md"))
            
    @pytest.mark.parametrize("invalid_content,expected_error", [
        ("# InvalidSection\nContent", "Unknown section"),
        ("# StateSpaceBlock\ninvalid[format", "Invalid dimension"),
    ])
    def test_parse_invalid_content(self, tmp_path, invalid_content, expected_error):
        """Test error handling for various invalid content."""
        file_path = tmp_path / "invalid.md"
        file_path.write_text(invalid_content)
        
        with pytest.raises(ValueError, match=expected_error):
            parse_gnn_file(file_path)
```

#### Running Tests
```bash
# Run all tests
python src/main.py --only-steps 3

# Run specific test categories
pytest src/tests/unit/ -v
pytest src/tests/integration/ -v

# Run with coverage
pytest --cov=src --cov-report=html:output/coverage

# Run performance tests
pytest src/tests/performance/ --benchmark-only
```

### Adding New Features

#### Adding a New Pipeline Step

1. **Create the script**: `src/N_description.py`
2. **Add configuration**: Update `src/pipeline/config.py`
3. **Add tests**: Create tests in `tests/unit/` and `tests/integration/`
4. **Add documentation**: Update pipeline documentation
5. **Add MCP tools**: Create `module/mcp.py` if exposing APIs

#### Adding New Export Formats

1. **Implement exporter**: Add to `src/export/format_exporters.py`
2. **Register format**: Update `AVAILABLE_EXPORT_FUNCTIONS`
3. **Add tests**: Test export functionality and output validation
4. **Update documentation**: Add format to documentation

#### Adding New MCP Tools

1. **Implement tools**: Create functions in appropriate `module/mcp.py`
2. **Register tools**: Call registration in module initialization
3. **Add schemas**: Ensure proper type hints for auto-schema generation
4. **Test integration**: Test via MCP CLI and HTTP server
5. **Document APIs**: Update MCP documentation

## Performance Considerations

### Memory Management
- Use generators for large file processing
- Implement streaming for export formats
- Monitor memory usage in resource estimation

### Optimization Guidelines
- Profile performance-critical paths
- Use caching appropriately
- Consider JAX for numerical computations
- Implement timeout handling for long operations

### Monitoring
- Log performance metrics
- Track memory usage per pipeline step
- Monitor export file sizes
- Measure visualization generation times

## Release Process

### Versioning
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Tag releases corresponding to paper submissions
- Maintain CHANGELOG.md with clear categories

### Quality Gates
1. All tests pass (`python src/main.py --only-steps 3`)
2. Type checking passes (`mypy src/`)
3. Code formatting (`black src/`, `isort src/`)
4. Documentation builds successfully
5. Example models validate correctly

### Release Checklist
- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create GitHub release
- [ ] Update citation information

## Troubleshooting Development Issues

### Common Problems

1. **Import Errors**
   - Check virtual environment activation
   - Verify `src/` is in Python path
   - Run setup step: `python src/main.py --only-steps 2`

2. **Test Failures**
   - Check test data fixtures
   - Verify dependencies installed
   - Run individual test files for debugging

3. **Pipeline Issues**
   - Check step configuration in `src/pipeline/config.py`
   - Verify argument passing between steps
   - Check timeout settings for slow operations

4. **MCP Integration Problems**
   - Verify tool registration in module `mcp.py` files
   - Check MCP server startup logs
   - Test tools via CLI before HTTP integration

### Debug Tools
```bash
# Verbose pipeline execution
python src/main.py --verbose --only-steps 1,4

# MCP tool debugging
python src/mcp/cli.py --debug list-tools

# Test debugging
pytest -vvv --pdb src/tests/unit/test_specific.py

# Type checking
mypy src/ --show-error-codes
```

## Contributing Guidelines

### Code Review Process
1. Create feature branch from `main`
2. Implement changes with tests
3. Submit pull request with clear description
4. Address review feedback
5. Merge after approval and CI passes

### Documentation Requirements
- Update relevant documentation for all changes
- Add docstrings for new functions/classes
- Include examples for new features
- Update troubleshooting guides for new error cases

### Testing Requirements
- Unit tests for all new functions
- Integration tests for new pipeline steps
- Performance tests for computationally intensive features
- Regression tests for bug fixes 