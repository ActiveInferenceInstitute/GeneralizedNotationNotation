# POMDP-Specific Type Checker Guide

This guide provides comprehensive documentation for POMDP-specific features in the Type Checker module, designed for Active Inference models in the Generalized Notation Notation (GNN) framework.

## Overview

The POMDP-specific type checker extends the standard GNN type checking capabilities with specialized analysis for Partially Observable Markov Decision Process (POMDP) models in the Active Inference framework. This includes validation of POMDP-specific structures, ontology compliance checking, and computational complexity estimation.

## Key Features

### 1. POMDP Structure Validation
- **Required Components**: Validates presence of all essential POMDP components (A, B, C, D, E matrices, s, o, π, u vectors, etc.)
- **Dimension Consistency**: Ensures proper dimensional relationships between matrices and vectors
- **Ontology Compliance**: Verifies adherence to Active Inference ontology standards
- **Connection Validation**: Validates POMDP-specific connection patterns

### 2. Active Inference Specific Analysis
- **Likelihood Matrix (A)**: Validates observation-to-state mapping
- **Transition Matrix (B)**: Validates state transition dynamics
- **Preference Vector (C)**: Validates agent preferences over observations
- **Prior Vector (D)**: Validates initial state beliefs
- **Habit Vector (E)**: Validates policy priors
- **Free Energy (F)**: Validates variational free energy calculations
- **Expected Free Energy (G)**: Validates policy selection mechanisms

### 3. Computational Complexity Estimation
- **Inference Operations**: Estimates computational cost of state estimation
- **Policy Operations**: Estimates computational cost of action selection
- **Memory Requirements**: Calculates memory usage for POMDP matrices
- **Scalability Analysis**: Assesses how complexity scales with model size

## Configuration

### Basic POMDP Mode
```yaml
# In input/config.yaml
validation:
  pomdp:
    enabled: true
    ontology_file: "input/ontology_terms.json"
    strict_ontology: true
    strict_dimensions: true
    analyze_complexity: true
    estimate_resources: true
    generate_summary: true
```

### Command Line Usage
```bash
# Enable POMDP mode
python3 src/5_type_checker.py --pomdp-mode

# With custom ontology file
python3 src/5_type_checker.py --pomdp-mode --ontology-file custom_ontology.json

# Strict POMDP validation
python3 src/5_type_checker.py --pomdp-mode --strict
```

## API Usage

### Basic POMDP Analysis
```python
from src.type_checker.processor import GNNTypeChecker
from pathlib import Path

# Initialize with POMDP mode
checker = GNNTypeChecker(
    pomdp_mode=True,
    ontology_file=Path("input/ontology_terms.json")
)

# Validate a POMDP file
result = checker.validate_pomdp_file(Path("input/gnn_files/actinf_pomdp_agent.md"))

# Check validation results
if result["validation_results"]["overall_valid"]:
    print("POMDP model is valid!")
    print(f"State space size: {result['pomdp_specific']['state_space_size']}")
    print(f"Observation space size: {result['pomdp_specific']['observation_space_size']}")
    print(f"Action space size: {result['pomdp_specific']['action_space_size']}")
```

### POMDP Structure Analysis
```python
from src.type_checker.pomdp_analyzer import POMDPAnalyzer

# Create analyzer
analyzer = POMDPAnalyzer()

# Analyze POMDP structure
with open("input/gnn_files/actinf_pomdp_agent.md", "r") as f:
    content = f.read()

analysis = analyzer.analyze_pomdp_structure(content)

# Check component validation
print(f"Structure valid: {analysis['validation_results']['structure_valid']}")
print(f"Missing components: {analysis['validation_results']['missing_components']}")
print(f"Dimension consistent: {analysis['validation_results']['dimension_consistency']}")
print(f"Ontology compliant: {analysis['validation_results']['ontology_compliance']}")
```

### Complexity Estimation
```python
# Estimate computational complexity
complexity = analyzer.estimate_pomdp_complexity(analysis)

print(f"Inference operations: {complexity['inference_complexity']['total_inference_ops']}")
print(f"Policy operations: {complexity['policy_complexity']['total_policy_ops']}")
print(f"Memory required: {complexity['memory_requirements']['total_memory_mb']:.2f} MB")
print(f"Scaling: {complexity['scalability']['overall_scaling']}")
```

## POMDP Model Requirements

### Required Components
A valid POMDP model must include:

1. **Likelihood Matrix (A)**: `A[obs_dim, state_dim, type=float]`
2. **Transition Matrix (B)**: `B[state_dim, state_dim, action_dim, type=float]`
3. **Preference Vector (C)**: `C[obs_dim, type=float]`
4. **Prior Vector (D)**: `D[state_dim, type=float]`
5. **Habit Vector (E)**: `E[action_dim, type=float]`
6. **Hidden State (s)**: `s[state_dim, 1, type=float]`
7. **Observation (o)**: `o[obs_dim, 1, type=int]`
8. **Policy (π)**: `π[action_dim, type=float]`
9. **Action (u)**: `u[1, type=int]`
10. **Free Energy (F)**: `F[π, type=float]`
11. **Expected Free Energy (G)**: `G[π, type=float]`
12. **Time (t)**: `t[1, type=int]`

### Dimension Consistency Rules
- `A[obs_dim, state_dim]` - observation × state dimensions
- `B[state_dim, state_dim, action_dim]` - state × state × action dimensions
- `C[obs_dim]` - observation dimension
- `D[state_dim]` - state dimension
- `E[action_dim]` - action dimension
- `s[state_dim, 1]` - state dimension
- `o[obs_dim, 1]` - observation dimension
- `π[action_dim]` - action dimension
- `u[1]` - single action
- `F[π]` - policy dimension
- `G[π]` - policy dimension
- `t[1]` - single time step

### Required Connections
Essential POMDP connections:
- `D>s` - Prior to hidden state
- `s-A` - Hidden state to likelihood
- `A-o` - Likelihood to observation
- `s-B` - Hidden state to transition
- `C>G` - Preference to expected free energy
- `E>π` - Habit to policy
- `G>π` - Expected free energy to policy
- `π>u` - Policy to action
- `B>u` - Transition to action
- `u>s_prime` - Action to next state

### Ontology Compliance
Must include `## ActInfOntologyAnnotation` section with:
```
A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates
E=Habit
F=VariationalFreeEnergy
G=ExpectedFreeEnergy
s=HiddenState
s_prime=NextHiddenState
o=Observation
π=PolicyVector
u=Action
t=Time
```

## Output Files

### Standard Output
- `type_check_results.json` - Complete validation results
- `type_check_summary.json` - Summary statistics
- `global_type_analysis.json` - Global analysis results

### POMDP-Specific Output
- `pomdp_analysis_summary.md` - Detailed POMDP analysis report
- POMDP metrics in summary files
- Complexity estimation results
- Ontology compliance reports

## Error Handling

### Common POMDP Errors
1. **Missing Components**: Required POMDP components not found
2. **Dimension Mismatch**: Inconsistent dimensions between matrices
3. **Ontology Violations**: Missing or incorrect ontology annotations
4. **Connection Errors**: Invalid or missing POMDP connections

### Error Recovery
- Graceful degradation for missing optional components
- Detailed error messages with suggestions
- Partial validation results for incomplete models
- Warning system for non-critical issues

## Performance Considerations

### Memory Usage
- POMDP matrices can be memory-intensive for large state spaces
- Memory estimation provided for resource planning
- Efficient algorithms for dimension validation

### Computational Complexity
- O(n²) for state space operations
- O(n³) for transition matrix operations
- Linear scaling for vector operations
- Parallel processing support for large models

## Testing

### Running POMDP Tests
```bash
# Run all POMDP-specific tests
python3 -m pytest src/tests/test_type_checker_pomdp.py -v

# Run specific test categories
python3 -m pytest src/tests/test_type_checker_pomdp.py::TestPOMDPAnalyzer -v
python3 -m pytest src/tests/test_type_checker_pomdp.py::TestPOMDPTypeChecker -v
python3 -m pytest src/tests/test_type_checker_pomdp.py::TestPOMDPIntegration -v
```

### Test Coverage
- POMDP structure validation
- Dimension consistency checking
- Ontology compliance validation
- Complexity estimation
- Error handling and recovery
- Integration with main type checker
- Performance testing with large models

## MCP Integration

### Available POMDP Tools
1. `validate_pomdp_file` - Validate POMDP GNN file
2. `analyze_pomdp_structure` - Analyze POMDP structure
3. `estimate_pomdp_complexity` - Estimate computational complexity

### MCP Usage Example
```python
from src.type_checker.mcp import execute_mcp_tool

# Validate POMDP file via MCP
result = execute_mcp_tool("validate_pomdp_file", {
    "file_path": "input/gnn_files/actinf_pomdp_agent.md",
    "ontology_file": "input/ontology_terms.json",
    "strict_ontology": True
})

if result["success"]:
    print("POMDP validation successful!")
    pomdp_data = result["result"]
    print(f"Model complexity: {pomdp_data['pomdp_specific']['model_complexity']}")
```

## Troubleshooting

### Common Issues

1. **POMDP mode not enabled**
   - Check configuration file settings
   - Verify command line arguments
   - Ensure ontology file exists

2. **Dimension validation failures**
   - Verify matrix dimensions match POMDP requirements
   - Check for typos in dimension specifications
   - Ensure consistent state/observation/action dimensions

3. **Ontology compliance errors**
   - Include required `## ActInfOntologyAnnotation` section
   - Verify all required mappings are present
   - Check for typos in ontology terms

4. **Performance issues**
   - Use complexity estimation to identify bottlenecks
   - Consider reducing state space size for testing
   - Enable parallel processing for large models

### Debug Mode
```bash
# Enable verbose output for debugging
python3 src/5_type_checker.py --pomdp-mode --verbose

# Enable strict mode for comprehensive validation
python3 src/5_type_checker.py --pomdp-mode --strict
```

## Advanced Usage

### Custom Ontology Terms
```python
custom_ontology = {
    "likelihood_matrix": "Custom description for A matrix",
    "transition_matrix": "Custom description for B matrix",
    # ... other terms
}

analyzer = POMDPAnalyzer(custom_ontology)
```

### Batch Processing
```python
# Process multiple POMDP files
checker = GNNTypeChecker(pomdp_mode=True)
success = checker.validate_gnn_files(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/pomdp_analysis"),
    verbose=True
)
```

### Integration with Pipeline
The POMDP type checker integrates seamlessly with the GNN processing pipeline:
- Automatic POMDP detection based on file content
- Configuration-driven POMDP mode activation
- Comprehensive reporting and analysis
- MCP tool integration for external access

## References

- [Active Inference Framework](https://www.activeinference.org/)
- [POMDP Theory](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process)
- [GNN Specification](https://github.com/GeneralizedNotationNotation)
- [Type Checker Module Documentation](README.md)
