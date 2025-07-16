# Enhanced Ontology Processing

This directory contains the GNN processing pipeline step for comprehensive ontology-related operations, primarily managed by the `8_ontology.py` script located in the `src/` directory.

## Overview

The enhanced ontology processing module significantly improves the **accessibility, rigor, reproducibility, and flexibility** of GNN models by leveraging explicit ontological annotations with advanced validation and analysis capabilities.

GNN files can include an `## ActInfOntologyAnnotation` section, which provides a mapping from model-specific variable names (defined in `StateSpaceBlock` or other sections) to standardized terms from a relevant ontology (e.g., an Active Inference Ontology).

### Example Annotation Section:
```
## ActInfOntologyAnnotation
s_t=HiddenState
A=TransitionMatrix
o_t=Observation
B_matrix=TransitionMatrix  # This could be flagged as a case mismatch
unknownTerm=InvalidTerm    # This would be flagged as invalid
```

## Enhanced Features

### 🔍 **Advanced Validation Modes**
- **Strict Mode**: All terms must be valid, processing fails on invalid terms
- **Lenient Mode** (default): Warnings for invalid terms, processing continues
- **Permissive Mode**: No validation, just parsing and reporting

### 🔧 **Smart Term Matching**
- **Case Sensitivity Control**: Configure case-sensitive or case-insensitive matching
- **Fuzzy Matching**: Suggests corrections for mistyped terms using similarity algorithms
- **Case Mismatch Detection**: Identifies terms with incorrect capitalization

### 📊 **Comprehensive Reporting**
- **Enhanced Reports**: Detailed analysis with validation results, suggestions, and warnings
- **Summary Statistics**: Success rates, validation rates, unique terms analysis
- **Export Capabilities**: JSON mappings export for integration with other tools
- **Processing Metrics**: Timing and performance analysis

### ⚡ **Improved Error Handling**
- **Graceful Failures**: Individual file processing errors don't stop the entire pipeline
- **Detailed Error Messages**: Clear reporting of what went wrong and where
- **Performance Tracking**: Processing time analysis and bottleneck identification

## Processing Configuration

The enhanced processor supports various configuration options:

```python
# Available validation modes
ValidationMode.STRICT      # Fail on any invalid terms
ValidationMode.LENIENT     # Warn on invalid terms (default)
ValidationMode.PERMISSIVE  # No validation, just parsing

# Fuzzy matching configuration
fuzzy_matching = True      # Enable fuzzy term suggestions
fuzzy_threshold = 0.8     # Similarity threshold (0.0-1.0)

# Case sensitivity
case_sensitive = True     # Exact case matching (default)
```

## Benefits of Enhanced Ontological Annotation

### **Accessibility**
By linking potentially opaque variable names (e.g., `A`, `B`, `s1`) to well-defined ontological concepts (e.g., `RecognitionMatrix`, `TransitionMatrix`, `HiddenState`), models become easier to understand. The fuzzy matching feature helps users discover correct terms even with typos.

### **Rigor** 
Enhanced validation ensures terms are used consistently and correctly. Multiple validation modes allow different levels of strictness based on project requirements.

### **Reproducibility & Comparability**
Standardized semantic meaning allows accurate comparison between different GNN models. Enhanced reporting provides detailed analysis for research documentation.

### **Flexibility**
Users can choose their preferred validation level while maintaining their preferred variable naming within models. Smart suggestions help correct common mistakes without being overly restrictive.

## Enhanced Script Functionality

The enhanced `8_ontology.py` script provides:

### 1. **Advanced Parsing**
- Robust parsing of `ActInfOntologyAnnotation` sections
- Comment handling and malformed line detection
- Unicode support and encoding detection

### 2. **Multi-Mode Validation**
- Configurable validation strictness
- Fuzzy term matching with similarity scoring
- Case mismatch detection and correction suggestions
- Comprehensive validation reporting

### 3. **Enhanced Reporting**
- Individual file reports with detailed analysis
- Summary reports with statistics and metrics
- JSON export for programmatic access
- Markdown reports for human readability

### 4. **Performance & Reliability**
- Individual file error handling
- Processing time tracking
- Memory usage optimization
- Graceful degradation on missing dependencies

## Usage Examples

### Basic Usage
```bash
# Process with default settings (lenient validation)
python 8_ontology.py --target-dir path/to/gnn_files --output-dir path/to/output

# Use specific ontology terms file
python 8_ontology.py --target-dir path/to/gnn_files --ontology-terms-file src/ontology/act_inf_ontology_terms.json
```

### Advanced Configuration
```bash
# Strict validation mode
python 8_ontology.py --target-dir path/to/gnn_files --validation-mode strict

# Case-insensitive matching with fuzzy suggestions
python 8_ontology.py --target-dir path/to/gnn_files --case-sensitive false --fuzzy-matching true

# Permissive mode for exploration
python 8_ontology.py --target-dir path/to/gnn_files --validation-mode permissive
```

### Pipeline Integration
```bash
# As part of main pipeline with enhanced options
python main.py --target-dir path/to/gnn_files --ontology-terms-file src/ontology/act_inf_ontology_terms.json
```

## Output Structure

The enhanced processor generates:

```
output/ontology_processing/
├── ontology_processing_summary.json     # Overall statistics
├── ontology_summary_report.md           # Human-readable summary
├── all_ontology_mappings.json          # Exported mappings
└── [file_name]/
    ├── ontology_report.md               # Individual file report
    └── ontology_results.json            # Individual file results
```

## Enhanced MCP Integration

The `mcp.py` file provides comprehensive Model Context Protocol integration with:
- Advanced parsing capabilities
- Validation result structures
- Error handling and recovery
- Performance optimizations

## Configuration Files

### Default Ontology Terms
The system uses `act_inf_ontology_terms.json` which contains structured ontology definitions:

```json
{
    "HiddenState": {
        "description": "A state of the environment or agent that is not directly observable.",
        "uri": "obo:ACTO_000001"
    },
    "TransitionMatrix": {
        "description": "A probabilistic mapping defining the dynamics of hidden states over time.",
        "uri": "obo:ACTO_000009"
    }
}
```

## Troubleshooting

### Common Issues

1. **Argument parsing errors**: The system now uses standardized argument parsing with both `--ontology-terms-file` and `--ontology_terms_file` support.

2. **Missing ontology terms**: The system provides helpful default path resolution and clear error messages.

3. **Validation failures**: Use different validation modes and check fuzzy suggestions for corrections.

4. **Performance issues**: The enhanced system includes performance tracking and optimization.

### Validation Modes Guide

- Use **strict mode** for production models requiring perfect ontology compliance
- Use **lenient mode** (default) for development and testing with helpful warnings
- Use **permissive mode** for exploratory analysis without validation constraints

## Integration with Other Tools

The enhanced ontology processor integrates seamlessly with:
- **Type Checker**: Validates ontological consistency with type definitions
- **Visualization**: Generates ontology-aware visualizations
- **Export Tools**: Includes ontology metadata in exported formats
- **LLM Analysis**: Provides ontological context for language model analysis

For complete integration examples, see the main pipeline documentation and the API reference. 