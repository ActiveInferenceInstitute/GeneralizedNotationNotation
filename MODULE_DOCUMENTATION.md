# GNN Pipeline Module Documentation

## Overview
This document provides comprehensive documentation for all modules within the GeneralizedNotationNotation (GNN) pipeline, focusing on POMDP (Partially Observable Markov Decision Process) integration and Active Inference capabilities.

## Core Pipeline Modules

### 1. **Type Checker Module** (`src/type_checker/`)
**Status**: ✅ Fully Functional

#### Components
- **`processor.py`**: Main type checking orchestration
- **`pomdp_analyzer.py`**: POMDP-specific analysis and validation
- **`mcp.py`**: Model Context Protocol tools for POMDP operations

#### Key Features
- **POMDP Structure Analysis**: Validates POMDP model components and dimensions
- **Ontology Compliance**: Checks Active Inference term compliance
- **Complexity Estimation**: Provides resource estimates and recommendations
- **Dimension Validation**: Ensures consistent state/observation/action dimensions

#### API
```python
from src.type_checker.processor import TypeChecker
from src.type_checker.pomdp_analyzer import POMDPAnalyzer

# Initialize with POMDP support
type_checker = TypeChecker(ontology_file="path/to/ontology.yaml")
pomdp_analyzer = POMDPAnalyzer(ontology_file="path/to/ontology.yaml")

# Analyze POMDP model
result = pomdp_analyzer.analyze_pomdp_structure(gnn_content)
validation = pomdp_analyzer.validate_pomdp_model(gnn_file_path)
complexity = pomdp_analyzer.estimate_pomdp_complexity(analysis_result)
```

### 2. **Test Framework** (`src/tests/`)
**Status**: ✅ Fully Functional

#### Components
- **`runner.py`**: Test execution orchestration
- **`test_pomdp_validation.py`**: POMDP-specific validation tests
- **`test_pomdp_integration.py`**: Integration tests for POMDP functionality

#### Test Categories
- **`pomdp`**: POMDP-specific tests
- **`validation`**: General validation tests
- **`comprehensive`**: Full integration tests
- **`fast_suite`**: Quick execution tests

#### Usage
```bash
# Run all POMDP tests
python -m src.tests.runner --category pomdp

# Run fast test suite
python -m src.tests.runner --category fast_suite

# Run comprehensive tests
python -m src.tests.runner --category comprehensive
```

### 3. **Configuration Management** (`src/utils/`)
**Status**: ✅ Fully Functional

#### Components
- **`argument_utils.py`**: Pipeline argument definitions
- **`configuration.py`**: Configuration loading and management
- **`pipeline_template.py`**: Template for pipeline steps

#### POMDP Configuration
```yaml
# config.yaml
validation:
  pomdp:
    enabled: true
    ontology_file: "input/ontology.yaml"
    strict_mode: false
```

#### API
```python
from src.utils.argument_utils import PipelineArguments
from src.utils.configuration import get_config

# Load configuration
config = get_config()
args = PipelineArguments(
    pomdp_mode=True,
    ontology_file="input/ontology.yaml"
)
```

## Execution Modules

### 4. **PyMDP Execution** (`src/execute/pymdp/`)
**Status**: ✅ Fully Functional

#### Components
- **`execute_pymdp.py`**: Main execution interface
- **`pymdp_simulation.py`**: PyMDP simulation implementation
- **`__init__.py`**: Module exports

#### Features
- **Graceful Degradation**: Works even when PyMDP is not installed
- **POMDP Support**: Full Active Inference POMDP simulation
- **Error Handling**: Comprehensive error reporting and fallback behavior
- **Result Export**: Detailed simulation results and metrics

#### Usage
```python
from src.execute.pymdp import execute_pymdp_simulation

# Execute PyMDP simulation
result = execute_pymdp_simulation(
    model_name="Classic Active Inference POMDP Agent v1",
    num_states=3,
    num_observations=3,
    num_actions=3,
    num_episodes=20,
    config_overrides={}
)
```

### 5. **DisCoPy Execution** (`src/execute/discopy/`)
**Status**: ✅ Fully Functional

#### Features
- **Categorical Diagrams**: Generates string diagrams for Active Inference models
- **Image Generation**: Creates actual PNG files (no popups)
- **JSON Export**: Structured data export for analysis
- **Quantum Circuit Types**: Uses `Digit` types for quantum circuit compatibility

#### Output Files
- `perception_action_loop.png`: Main perception-action loop diagram
- `generative_model.png`: Generative model diagram
- `model_components.png`: Individual component diagrams
- `circuit_analysis.json`: Structured analysis data
- `circuit_info.json`: Circuit metadata

#### Usage
```python
# DisCoPy scripts are generated automatically by the pipeline
# Output directory: output/11_render_output/.../discopy/
```

### 6. **JAX Execution** (`src/execute/jax/`)
**Status**: ✅ Fully Functional

#### Features
- **High-Performance Computing**: GPU-accelerated numerical operations
- **Active Inference**: Efficient implementation of Active Inference algorithms
- **Model Summaries**: Comprehensive model analysis and reporting

## Pending Modules

### 7. **RxInfer.jl Execution** (`src/execute/rxinfer/`)
**Status**: ❌ Complex API Issues

#### Issues
- **API Changes**: RxInfer.jl 4.x has significant API changes
- **Model Definition**: Complex model definition syntax
- **Data Passing**: Intricate data passing requirements
- **Expert Knowledge**: Requires specialized library expertise

#### Current Status
- **Version**: 4.5.2 (latest)
- **Error**: Model definition and data passing incompatibilities
- **Recommendation**: Consult with RxInfer.jl community for proper implementation

### 8. **ActiveInference.jl Execution** (`src/execute/activeinference_jl/`)
**Status**: ❌ API Compatibility Issues

#### Issues
- **E-vector Mismatch**: Length mismatch with number of policies
- **Matrix Creation**: Complex matrix creation and parameter passing
- **Library Requirements**: Specific understanding of library requirements needed

#### Current Status
- **Version**: 0.1.2
- **Error**: E-vector length and policy count mismatches
- **Recommendation**: Consult with ActiveInference.jl community for proper implementation

## MCP Integration

### Model Context Protocol Tools
**Status**: ✅ Partially Functional

#### Available Tools
- **`validate_pomdp_file`**: Validate POMDP model files
- **`analyze_pomdp_structure`**: Analyze POMDP model structure
- **`estimate_pomdp_complexity`**: Estimate computational complexity

#### Usage
```python
# MCP tools are automatically registered when modules are imported
# Available through MCP protocol for external tool integration
```

## Testing Framework

### Test Categories
1. **`pomdp`**: POMDP-specific functionality tests
2. **`validation`**: General validation and compliance tests
3. **`comprehensive`**: Full integration and end-to-end tests
4. **`fast_suite`**: Quick execution tests for development

### Test Execution
```bash
# Run specific test category
python -m src.tests.runner --category pomdp

# Run all tests
python -m src.tests.runner --all

# Run with verbose output
python -m src.tests.runner --category comprehensive --verbose
```

## Configuration Management

### Pipeline Arguments
```python
@dataclass
class PipelineArguments:
    # POMDP-specific arguments
    pomdp_mode: bool = False
    ontology_file: Optional[str] = None
    
    # General pipeline arguments
    input_dir: str = "input"
    output_dir: str = "output"
    verbose: bool = False
    # ... other arguments
```

### Configuration Files
- **`input/config.yaml`**: Main configuration file
- **`input/ontology.yaml`**: Active Inference ontology definitions
- **Environment Variables**: Override configuration settings

## Error Handling and Logging

### Logging Levels
- **DEBUG**: Detailed execution information
- **INFO**: General progress updates
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors requiring attention

### Error Codes
- **0**: Success
- **1**: Critical error
- **2**: Success with warnings

### Logging Format
```
[YYYY-MM-DD HH:MM:SS] [LEVEL] [MODULE] Message
```

## Performance Metrics

### Execution Times
- **PyMDP**: ~2-5 seconds for 20-step simulation
- **DisCoPy**: ~1-3 seconds for diagram generation
- **JAX**: ~1-2 seconds for model analysis
- **Type Checker**: ~0.5-1 second for POMDP analysis

### Memory Usage
- **PyMDP**: ~50-100MB for typical simulations
- **DisCoPy**: ~20-50MB for diagram generation
- **JAX**: ~100-200MB for numerical operations

## Future Development

### Immediate Priorities
1. **Julia Package Integration**: Resolve RxInfer.jl and ActiveInference.jl issues
2. **Documentation**: Expand user guides and API documentation
3. **Testing**: Increase test coverage for edge cases
4. **Performance**: Optimize execution times and memory usage

### Long-term Goals
1. **Advanced Visualization**: 3D state space visualization
2. **Interactive Tools**: GUI for model construction and editing
3. **Cloud Integration**: Cloud-based execution and storage
4. **Community Tools**: Enhanced MCP integration and external tool support

## Support and Maintenance

### Getting Help
1. **Documentation**: Check this documentation first
2. **Tests**: Run tests to verify functionality
3. **Logs**: Check execution logs for error details
4. **Community**: Engage with relevant package communities

### Contributing
1. **Code Style**: Follow established patterns and conventions
2. **Testing**: Add tests for new functionality
3. **Documentation**: Update documentation for changes
4. **Review**: Submit changes for review and testing

---
*Documentation generated: 2025-09-15*
*Pipeline version: GNN v1.0*
*Status: Core functionality working, Julia packages pending*
